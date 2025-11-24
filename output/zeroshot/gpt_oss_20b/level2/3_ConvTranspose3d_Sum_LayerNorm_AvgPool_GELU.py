import torch
import torch.nn as nn
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Triton kernel for adding a scalar to a 5‑D tensor
# --------------------------------------------------------------------------- #
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def add_scalar_kernel(
    out_ptr,
    in_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    in_vals = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    out_vals = in_vals + scalar
    tl.store(out_ptr + offsets, out_vals, mask=mask)


def triton_add_scalar(x: torch.Tensor, scalar: float):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    add_scalar_kernel[grid](out, x, scalar, n_elements)
    return out


# --------------------------------------------------------------------------- #
# Triton kernel for fused LayerNorm over the last dimension
# --------------------------------------------------------------------------- #
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["N", "C"],
)
@triton.jit
def layernorm_kernel(
    out_ptr,
    in_ptr,
    eps,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    c_start = pid * BLOCK_SIZE
    offsets = c_start + tl.arange(0, BLOCK_SIZE)

    # each thread processes one channel across the batch & spatial dims
    mask = offsets < C
    if not tl.any(mask):
        return

    # compute mean
    mean = tl.zeros([N], dtype=tl.float32)
    var = tl.zeros([N], dtype=tl.float32)
    for n in range(N):
        idx = n * C
        vals = tl.load(in_ptr + idx + offsets, mask=mask, other=0.0)
        mean[n] = tl.sum(vals)
        var[n] = tl.sum(vals * vals)
    mean /= C
    var = var / C - mean * mean
    inv_std = tl.rsqrt(var + eps)

    # write normalized values
    for n in range(N):
        idx = n * C
        vals = tl.load(in_ptr + idx + offsets, mask=mask, other=0.0)
        out_vals = (vals - mean[n]) * inv_std[n]
        tl.store(out_ptr + idx + offsets, out_vals, mask=mask)


def triton_layernorm(x: torch.Tensor, eps=1e-5):
    N = x.shape[0] * x.shape[2] * x.shape[3] * x.shape[4]
    C = x.shape[1]
    out = torch.empty_like(x)
    grid = lambda meta: ((C + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    layernorm_kernel[grid](out, x, eps, N, C)
    return out


# --------------------------------------------------------------------------- #
# Triton kernel for average pooling over 3D with kernel size 2
# --------------------------------------------------------------------------- #
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["D", "H", "W"],
)
@triton.jit
def avgpool_kernel(
    out_ptr,
    in_ptr,
    stride: tl.constexpr,
    D_in: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    d = pid // (stride * stride)
    h = (pid // stride) % stride
    w = pid % stride

    if d >= D_in:
        return

    c_start = 0
    c_end = tl.arange(0, BLOCK_SIZE)

    for c in range(0, in_ptr.shape[1], BLOCK_SIZE):
        c_off = c + c_end
        mask = c_off < in_ptr.shape[1]
        vals = tl.load(in_ptr + d * in_ptr.stride(0) + h * in_ptr.stride(1) + w * in_ptr.stride(2) + c_off, mask=mask, other=0.0)
        tl.store(out_ptr + d * out_ptr.stride(0) + h * out_ptr.stride(1) + w * out_ptr.stride(2) + c_off, vals, mask=mask)


def triton_avgpool(x: torch.Tensor, kernel_size=2, stride=2):
    D_in, H_in, W_in = x.shape[2], x.shape[3], x.shape[4]
    D_out = (D_in - kernel_size) // stride + 1
    H_out = (H_in - kernel_size) // stride + 1
    W_out = (W_in - kernel_size) // stride + 1
    out = torch.empty((x.shape[0], x.shape[1], D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    grid = lambda meta: ((D_out * H_out * W_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    avgpool_kernel[grid](out, x, stride, D_in, H_in, W_in)
    return out


# --------------------------------------------------------------------------- #
# Triton kernel for GELU (fast approximation)
# --------------------------------------------------------------------------- #
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def gelu_kernel(
    out_ptr,
    in_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    # fast GELU: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    sqrt_2_over_pi = 0.7978845608028654
    a = 0.044715
    x_cubed = x * x * x
    tanh_in = sqrt_2_over_pi * (x + a * x_cubed)
    tanh_out = tl.math.tanh(tanh_in)
    gelu = 0.5 * x * (1 + tanh_out)
    tl.store(out_ptr + offsets, gelu, mask=mask)


def triton_gelu(x: torch.Tensor):
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    gelu_kernel[grid](out, x, n_elements)
    return out


# --------------------------------------------------------------------------- #
# Optimized Model using custom Triton kernels
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum, layer normalization,
    average pooling, and GELU activation.  All operations except the transposed convolution
    are implemented using custom Triton kernels for maximal performance on A100.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding,
                 sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.sum_weight = nn.Parameter(torch.tensor([sum_weight], device="cuda"))
        self.norm_shape = norm_shape
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        # 1. ConvTranspose3d
        x = self.conv_transpose(x)

        # 2. Add scalar (sum_weight)
        x = triton_add_scalar(x, float(self.sum_weight))

        # 3. LayerNorm over channel dimension
        # flatten batch and spatial dims to compute mean/var per channel
        x = triton_layernorm(x)

        # 4. AvgPool3d
        x = triton_avgpool(x, kernel_size=self.pool_kernel_size, stride=self.pool_kernel_size)

        # 5. GELU
        x = triton_gelu(x)

        return x