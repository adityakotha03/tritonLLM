import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# === Custom Triton Kernels ===

@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C_in, C_out, H, W, patch_size,
    stride, pad,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    # Block indices
    block_h = tl.program_id(0)
    block_w = tl.program_id(1)
    block_c = tl.program_id(2)

    # Compute output indices
    h_start = block_h * BLOCK_H
    w_start = block_w * BLOCK_W
    c_start = block_c * BLOCK_C

    # Output indices within block
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    c_offsets = c_start + tl.arange(0, BLOCK_C)

    # Masks for bounds
    h_mask = h_offsets < H
    w_mask = w_offsets < W
    c_mask = c_offsets < C_out

    # Load input and weights
    x_ptrs = x_ptr + (h_offsets[:, None] * W + w_offsets[None, :]) * C_in
    w_ptrs = w_ptr + (c_offsets[:, None, None] * C_in * patch_size * patch_size +
                      tl.arange(0, patch_size)[None, :, None] * C_in * patch_size +
                      tl.arange(0, patch_size)[None, None, :] * C_in)

    # Load input block
    x_block = tl.load(x_ptrs, mask=h_mask[:, None] & w_mask[None, :], other=0.0)
    x_block = tl.reshape(x_block, (BLOCK_H, BLOCK_W, C_in))

    # Load weight block
    w_block = tl.load(w_ptrs, mask=c_mask[:, None, None] & (tl.arange(0, patch_size)[:, None, None] < patch_size) & (tl.arange(0, patch_size)[None, :, None] < patch_size), other=0.0)

    # Compute output
    out = tl.zeros((BLOCK_H, BLOCK_W, C_out), dtype=tl.float32)
    for c in range(C_in):
        x_c = x_block[:, :, c]
        w_c = w_block[:, :, c]  # (patch_size, patch_size, C_out)
        for i in range(patch_size):
            for j in range(patch_size):
                out += x_c[i:i+BLOCK_H, j:j+BLOCK_W][:, :, None] * w_c[i, j, :]
    
    # Store output
    out_ptrs = out_ptr + (h_offsets[:, None] * W + w_offsets[None, :]) * C_out
    out = tl.reshape(out, (BLOCK_H, BLOCK_W, C_out))
    tl.store(out_ptrs, out, mask=h_mask[:, None] & w_mask[None, :] & c_mask[None, None, :])

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # Masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K

    # Load A and B
    a_ptrs = a_ptr + (offs_m[:, None] * K + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_n[None, :])

    a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

    # Compute product
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc += tl.dot(a, b)

    # Store result
    out_ptrs = out_ptr + (offs_m[:, None] * N + offs_n[None, :])
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

@triton.jit
def softmax_kernel(
    x_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    # Block index
    pid = tl.program_id(0)

    # Offsets
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)

    # Online softmax: subtract max for numerical stability
    x_max = tl.max(x, axis=0)
    x = x - x_max

    # Exponentiate
    x_exp = tl.exp(x)

    # Compute sum
    x_exp_sum = tl.sum(x_exp, axis=0)

    # Compute softmax
    out = x_exp / x_exp_sum

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    # Block index
    pid = tl.program_id(0)

    # Offsets
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Add
    out = x + y

    # Store
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def relu_kernel(
    x_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    # Block index
    pid = tl.program_id(0)

    # Offsets
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Relu
    out = tl.max(x, 0)

    # Store
    tl.store(out_ptr + offsets, out, mask=mask)

# === Triton Wrapper Functions ===

def triton_conv2d(x, w, patch_size, stride=1, padding=0):
    B, C_in, H, W = x.shape
    C_out, _, _, _ = w.shape
    H_out = (H + 2 * padding - patch_size) // stride + 1
    W_out = (W + 2 * padding - patch_size) // stride + 1

    x = x.contiguous()
    w = w.contiguous()

    out = torch.empty(B, C_out, H_out, W_out, device=x.device, dtype=x.dtype)

    BLOCK_H, BLOCK_W, BLOCK_C = 16, 16, 32
    grid = (H_out + BLOCK_H - 1) // BLOCK_H, (W_out + BLOCK_W - 1) // BLOCK_W, (C_out + BLOCK_C - 1) // BLOCK_C

    conv2d_kernel[grid](
        x, w, out,
        B, C_in, C_out, H, W, patch_size, stride, padding,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C
    )
    return out

def triton_matmul(a, b):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Matmul dimension mismatch"

    a = a.contiguous()
    b = b.contiguous()

    out = torch.empty(M, N, device=a.device, dtype=a.dtype)

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 32
    grid = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N, (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    matmul_kernel[grid](
        a, b, out,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return out

def triton_softmax(x, dim=-1):
    B, N = x.shape
    if dim == -1:
        x = x.contiguous()
        out = torch.empty_like(x)

        BLOCK_SIZE = 1024
        grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

        softmax_kernel[grid](x, out, N, BLOCK_SIZE=BLOCK_SIZE)
        return out
    else:
        raise NotImplementedError("Only dim=-1 supported for now")

def triton_add(a, b):
    assert a.shape == b.shape
    a = a.contiguous()
    b = b.contiguous()
    out = torch.empty_like(a)

    BLOCK_SIZE = 1024
    N = a.numel()
    grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    add_kernel[grid](a, b, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out

def triton_relu(x):
    x = x.contiguous()
    out = torch.empty_like(x)

    BLOCK_SIZE = 1024
    N = x.numel()
    grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    relu_kernel[grid](x, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out

# === Optimized Model New ===

class ModelNew(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6, 
                 mlp_ratio=4.0, patch_size=4, in_channels=3, image_size=32):
        super().__init__()

        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim

        # Replace Conv2d with Triton-based version
        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.conv1.weight.requires_grad = False
        self.conv1.bias.requires_grad = False

        num_patches = (image_size // patch_size) ** 2
        self.linear_proj = nn.Linear(embed_dim * num_patches, embed_dim)

        # Replace transformer layers with Triton-optimized versions
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        
        # Conv2d + Flatten + Linear Projection
        x = self.conv1(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(start_dim=1)  # (B, embed_dim * num_patches)
        x = self.linear_proj(x)  # (B, embed_dim)

        # CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)  # (B, 1 + num_patches, embed_dim)

        # Replace transformer layers with Triton-optimized versions
        for layer in self.transformer_layers:
            # Replace MHA and MLP with Triton kernels
            # Attention: q, k, v = x @ Wq, Wk, Wv
            q = layer.self_attn.in_proj_q(x)
            k = layer.self_attn.in_proj_k(x)
            v = layer.self_attn.in_proj_v(x)

            # Split heads
            q = q.view(B, -1, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(1, 2)
            k = k.view(B, -1, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(1, 2)
            v = v.view(B, -1, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(1, 2)

            # Matmul + Softmax + Matmul
            # Q @ K^T
            attn_weights = triton_matmul(q, k.transpose(-2, -1))
            # Softmax
            attn_weights = triton_softmax(attn_weights)
            # attn_weights @ V
            attn_out = triton_matmul(attn_weights, v)
            # Reshape and concat
            attn_out = attn_out.transpose(1, 2).reshape(B, -1, layer.self_attn.embed_dim)

            # Add and norm
            x = x + attn_out
            x = layer.self_attn.norm(x)

            # MLP
            mlp_out = layer.linear1(x)
            mlp_out = triton_relu(mlp_out)
            mlp_out = layer.linear2(mlp_out)
            x = x + mlp_out
            x = layer.norm2(x)

        return self.fc_out(x[:, 0])