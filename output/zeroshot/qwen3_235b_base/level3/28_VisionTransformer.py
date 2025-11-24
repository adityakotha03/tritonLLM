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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_mask = (offs_am < M)[:, None] & (offs_k < K)[None, :]
        b_mask = (offs_k < K)[:, None] & (offs_bn < N)[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am < M)[:, None] & (offs_bn < N)[None, :]

    if ACTIVATION == "gelu":
        accumulator = accumulator.to(tl.float32)
        c = tl.math.erf(accumulator / 1.41421356) + 1.0
        accumulator = accumulator * c * 0.5
    elif ACTIVATION == "none":
        pass

    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_matmul_gelu(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda
    assert a.shape[-1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    def grid(META): return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        ACTIVATION="gelu",
    )
    return c


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda
    assert a.shape[-1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    def grid(META): return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        ACTIVATION="none",
    )
    return c


@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out


@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows_per_program = N // tl.num_programs(0)
    row_start = pid * rows_per_program
    row_end = (pid + 1) * rows_per_program
    row_end = tl.minimum(row_end, N)
    offs_cols = tl.arange(0, BLOCK_SIZE)
    for row in range(row_start, row_end):
        row_start_ptr = x_ptr + row * N
        mask = offs_cols < N
        row_vals = tl.load(row_start_ptr + offs_cols, mask=mask, other=0.0)
        mean = tl.sum(row_vals, axis=0) / N
        row_minus_mean = row_vals - mean
        var = tl.sum(row_minus_mean * row_minus_mean, axis=0) / N
        inv_var = tl.math.rsqrt(var + eps)
        normed = row_minus_mean * inv_var
        weight = tl.load(weight_ptr + offs_cols, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + offs_cols, mask=mask, other=0.0)
        output = normed * weight + bias
        out_row_ptr = out_ptr + row * N
        tl.store(out_row_ptr + offs_cols, output, mask=mask)


def triton_layer_norm(x: torch.Tensor, weight: torch.nn.Parameter, bias: torch.nn.Parameter, eps: float = 1e-5):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    N = x.shape[-1]
    grid = lambda meta: (x.shape[0],)
    layer_norm_kernel[grid](
        x, weight, bias, out,
        N, eps,
        BLOCK_SIZE=1024,
    )
    return out


class TritonLinearGELU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        w = self.weight.to(dtype=x.dtype)
        b = self.bias.to(dtype=x.dtype)
        x = triton_matmul_gelu(x, w.t()) + b
        return x


class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        w = self.weight.to(dtype=x.dtype)
        b = self.bias.to(dtype=x.dtype)
        x = triton_matmul(x, w.t()) + b
        return x


class TritonLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        return triton_layer_norm(x, self.weight, self.bias, self.eps)


class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super(ModelNew, self).__init__()
        
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = TritonLinear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            TritonLinearGELU(dim, mlp_dim),
            nn.Dropout(dropout),
            TritonLinear(mlp_dim, num_classes)
        )
    
    def forward(self, img):
        p = self.patch_size
        
        x = img.unfold(2, p, p).unfold(3, p, p)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], p, p * img.shape[1])
        x = x.reshape(x.shape[0], -1, p * p * img.shape[1])
        x = self.patch_to_embedding(x)
        
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = triton_add(x, self.pos_embedding)
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)