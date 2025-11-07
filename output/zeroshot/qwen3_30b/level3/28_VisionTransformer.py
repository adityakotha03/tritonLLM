import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K

    offs_am = offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    offs_bk = offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    a_ptrs = a_ptr + offs_am
    b_ptrs = b_ptr + offs_bk

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        accumulator += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator

    if ACTIVATION == 1:  # GELU
        c = c * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (c + 0.044715 * c * c * c)))

    offs_cm = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_ptrs = c_ptr + offs_cm
    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def softmax_kernel(
    x_ptr, y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x = tl.exp(x * SOFTMAX_SCALE)
    x_sum = tl.sum(x, axis=0)
    y = x / x_sum
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def gelu_kernel(
    x_ptr, out_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    out = x * cdf
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def dropout_kernel(
    x_ptr, mask_ptr, out_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr,
    dropout_prob: tl.constexpr, scale: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mask = tl.load(mask_ptr + offsets, mask=mask, other=0.0)
    out = x * mask * scale
    tl.store(out_ptr + offsets, out, mask=mask)


class TritonMatmul(nn.Module):
    def __init__(self, shape, activation=0):
        super().__init__()
        self.shape = shape
        self.activation = activation

    def forward(self, a, b):
        assert a.shape[1] == b.shape[0], "Matmul dimension mismatch"
        M, K = a.shape
        K, N = b.shape

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
        num_stages = 1
        num_warps = 8

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']), triton.cdiv(K, meta['BLOCK_K']))

        out = torch.empty(M, N, device=a.device, dtype=a.dtype)

        matmul_kernel[grid](
            a, b, out,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            ACC_TYPE=tl.float32 if a.dtype == torch.float32 else tl.bfloat16,
            ACTIVATION=self.activation
        )
        return out


class TritonAdd(nn.Module):
    def forward(self, a, b):
        assert a.shape == b.shape
        n_elements = a.numel()
        BLOCK_SIZE = 128
        grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

        out = torch.empty_like(a)
        add_kernel[grid](a, b, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out


class TritonGelu(nn.Module):
    def forward(self, x):
        n_elements = x.numel()
        BLOCK_SIZE = 128
        grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

        out = torch.empty_like(x)
        gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out


class TritonSoftmax(nn.Module):
    def forward(self, x, dim=-1):
        assert dim == -1
        M, N = x.shape
        BLOCK_SIZE = 128
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

        out = torch.empty_like(x)
        softmax_kernel[grid](x, out, N, BLOCK_SIZE=BLOCK_SIZE, SOFTMAX_SCALE=1.0)
        return out


class TritonDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
        self.scale = 1.0 / (1.0 - p)

    def forward(self, x, rng):
        assert x.is_contiguous()
        n_elements = x.numel()
        BLOCK_SIZE = 128
        grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

        out = torch.empty_like(x)
        mask = torch.empty(n_elements, device=x.device, dtype=torch.float32).uniform_(0, 1) > self.p
        mask = mask.view_as(x)
        dropout_kernel[grid](x, mask, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, dropout_prob=self.p, scale=self.scale)
        return out


class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # Use Triton kernels
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.matmul1 = TritonMatmul((dim, dim), activation=0)  # For attention QK^T
        self.matmul2 = TritonMatmul((dim, dim), activation=0)  # For attention output
        self.matmul3 = TritonMatmul((dim, dim), activation=0)  # For MLP
        self.matmul4 = TritonMatmul((dim, dim), activation=0)  # For MLP
        self.add = TritonAdd()
        self.gelu = TritonGelu()
        self.softmax = TritonSoftmax()
        self.dropout_layer = TritonDropout(p=dropout)
        self.dropout_emb = TritonDropout(p=emb_dropout)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

        self.to_cls_token = nn.Identity()
        self.depth = depth
        self.heads = heads
        self.dim = dim

    def forward(self, img):
        p = self.patch_size
        B, C, H, W = img.shape
        N = (H // p) * (W // p)

        # Patchify and embed
        x = img.unfold(2, p, p).unfold(3, p, p).reshape(B, C * p * p, N)
        x = self.patch_to_embedding(x)

        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
        x += self.pos_embedding
        x = self.dropout_emb(x, torch.randn(1))

        # Transformer layers
        for _ in range(self.depth):
            # Self-attention: QKV
            q = self.matmul1(x, x.transpose(-2, -1))  # (B, H, N+1, N+1)
            q = self.softmax(q)  # Softmax over last dim
            v = x
            x = self.matmul2(q, v)  # (B, H, N+1, D)
            x = self.add(x, cls_tokens)  # Residual connection

            # MLP
            x = self.matmul3(x, x)  # First linear
            x = self.gelu(x)
            x = self.matmul4(x, x)  # Second linear
            x = self.add(x, x)  # Residual connection

        # Classification head
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)