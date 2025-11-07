import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr, 
    y_ptr, 
    n_rows, 
    n_cols, 
    BLOCK_SIZE: tl.constexpr
):
    # Each program processes a block of rows
    row_idx = tl.program_id(0)
    # Offset into the row
    offsets = tl.arange(0, BLOCK_SIZE)  # [0, 1, ..., BLOCK_SIZE - 1]
    mask = offsets < n_cols  # Only valid elements in the row

    # Load row of x
    x = tl.load(x_ptr + row_idx * n_cols + offsets, mask=mask, other=-float('inf'))

    # Online softmax: subtract max and exponentiate
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    y = x_exp / x_sum

    # Store the result
    tl.store(y_ptr + row_idx * n_cols + offsets, y, mask=mask)


@triton.jit
def matmul_kernel(
    a_ptr, 
    b_ptr, 
    c_ptr, 
    m, 
    n, 
    k, 
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # Get block indices
    pid = tl.program_id(0)  # block row
    pid_k = tl.program_id(1)  # block col
    # Calculate row and col indices
    block_row = pid // GROUP_SIZE_M
    block_col = pid % GROUP_SIZE_M

    # Compute offsets
    offs_m = block_row * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_col * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # Create mask for boundaries
    mask_m = offs_m < m
    mask_n = offs_n < n
    mask_k = offs_k < k

    # Load A and B
    a = tl.load(a_ptr + offs_m[:, None] * k + offs_k[None, :], 
                mask=mask_m[:, None] & mask_k[None, :], 
                other=0.0)
    b = tl.load(b_ptr + offs_k[:, None] * n + offs_n[None, :], 
                mask=mask_k[:, None] & mask_n[None, :], 
                other=0.0)

    # Perform matmul
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc += tl.dot(a, b, allow_tf32=True)

    # Store result
    tl.store(c_ptr + offs_m[:, None] * n + offs_n[None, :], 
             acc, 
             mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    eps: tl.float32,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / n_elements
    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / n_elements
    # Normalize
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * inv_std

    # Apply weight and bias
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    out = x_norm * weight + bias

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_softmax(x: torch.Tensor):
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()
    B, H, S = x.shape
    assert x.is_contiguous(), "Input tensor must be contiguous"
    out = torch.empty_like(x)

    # Launch kernel
    grid = lambda meta: (B * H,)

    # Use BLOCK_SIZE as 512 for optimal coalescing
    softmax_kernel[grid](x, out, B * H, S, BLOCK_SIZE=512)
    return out


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    assert a.is_contiguous() and b.is_contiguous(), "Tensors must be contiguous"
    assert a.shape[1] == b.shape[0], "Matmul dimensions must match"
    M, K = a.shape
    K, N = b.shape
    # Output tensor
    out = torch.empty(M, N, device=a.device, dtype=a.dtype)

    # Configure kernel
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    # Grid definition
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]) // GROUP_SIZE_M,
                         triton.cdiv(K, meta["BLOCK_SIZE_K"]))

    # Launch kernel
    matmul_kernel[grid](
        a, b, out,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    return out


def triton_add(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA"
    assert x.shape == y.shape, "Shapes must match"
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 128

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 128

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    layer_norm_kernel[grid](x, weight, bias, out, n_elements, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Initialize weights for Q, K, V projections
        self.q_proj_weight = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.k_proj_weight = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.v_proj_weight = nn.Parameter(torch.randn(embed_dim, embed_dim))
        # Output projection
        self.out_proj_weight = nn.Parameter(torch.randn(embed_dim, embed_dim))
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        seq_len = H * W
        # Reshape to (seq_len, batch_size, embed_dim)
        x = x.view(B, C, seq_len).permute(2, 0, 1)

        # Project to Q, K, V
        q = triton_matmul(x, self.q_proj_weight)
        k = triton_matmul(x, self.k_proj_weight)
        v = triton_matmul(x, self.v_proj_weight)

        # Reshape for multi-head attention: (seq_len, B, num_heads, head_dim)
        head_dim = self.embed_dim // self.num_heads
        q = q.view(seq_len, B, self.num_heads, head_dim).permute(1, 2, 0, 3)
        k = k.view(seq_len, B, self.num_heads, head_dim).permute(1, 2, 0, 3)
        v = v.view(seq_len, B, self.num_heads, head_dim).permute(1, 2, 0, 3)

        # Reshape to (B * num_heads, seq_len, head_dim) for matmul
        q = q.reshape(B * self.num_heads, seq_len, head_dim)
        k = k.reshape(B * self.num_heads, seq_len, head_dim)
        v = v.reshape(B * self.num_heads, seq_len, head_dim)

        # Compute attention scores: (B * num_heads, seq_len, seq_len)
        attn_scores = triton_matmul(q, k.transpose(1, 2))  # (B * num_heads, seq_len, seq_len)
        # Scale attention scores
        attn_scores = attn_scores / (head_dim ** 0.5)

        # Apply softmax
        attn_weights = triton_softmax(attn_scores)

        # Apply attention weights to values
        attn_output = triton_matmul(attn_weights, v)  # (B * num_heads, seq_len, head_dim)

        # Reshape back to (B, num_heads, seq_len, head_dim)
        attn_output = attn_output.view(B, self.num_heads, seq_len, head_dim)

        # Combine heads
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, seq_len, C)

        # Project output
        output = triton_matmul(attn_output, self.out_proj_weight)

        # Add residual and norm
        output = triton_add(output, x.permute(2, 0, 1))  # Residual connection
        output = triton_layer_norm(output, self.norm.weight, self.norm.bias, self.norm.eps)

        # Reshape back to (B, C, H, W)
        output = output.permute(1, 0, 2).view(B, C, H, W)
        return output