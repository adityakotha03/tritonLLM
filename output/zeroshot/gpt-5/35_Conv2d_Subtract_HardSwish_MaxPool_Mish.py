import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_hswish_maxpool_mish_kernel(
    x_ptr,  # input tensor ptr (after conv), NCHW
    out_ptr,  # output tensor ptr (after pooling)
    N, C, H, W,
    OH, OW,
    stride_n, stride_c, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    subtract_value,  # scalar
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,  # pooling kernel size, stride = K
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * OH * OW
    mask = offs < total

    # Decompose flat index into n, c, oh, ow
    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp = tmp // OH
    c = tmp % C
    n = tmp // C

    # Compute input base offset for the top-left of each pooling window
    ih0 = oh * K
    iw0 = ow * K

    in_base = (
        n * stride_n
        + c * stride_c
        + ih0 * stride_h
        + iw0 * stride_w
    )

    NEG_INF = tl.full([BLOCK_SIZE], -1.0e20, dtype=tl.float32)

    # initialize max with very negative number
    maxv = NEG_INF

    # Iterate over KxK window
    for ky in range(0, K):
        for kx in range(0, K):
            ptr = x_ptr + (in_base + ky * stride_h + kx * stride_w)
            val = tl.load(ptr, mask=mask, other=0.0)
            z = val - subtract_value
            # HardSwish: z * clamp(z + 3, 0, 6) / 6
            t = tl.minimum(tl.maximum(z + 3.0, 0.0), 6.0)
            hsw = z * t * (1.0 / 6.0)
            hsw = tl.where(mask, hsw, NEG_INF)
            maxv = tl.maximum(maxv, hsw)

    # Mish: x * tanh(softplus(x)), with stable softplus
    # softplus(x) = log(1 + exp(-x)) + x if x > 0 else log(1 + exp(x))
    zero = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    pos_mask = maxv > zero
    sp_pos = maxv + tl.log(1.0 + tl.exp(-maxv))
    sp_neg = tl.log(1.0 + tl.exp(maxv))
    softplus = tl.where(pos_mask, sp_pos, sp_neg)
    mish = maxv * tl.tanh(softplus)

    # Compute output pointer offset and store
    out_offs = (
        n * out_stride_n
        + c * out_stride_c
        + oh * out_stride_h
        + ow * out_stride_w
    )
    tl.store(out_ptr + out_offs, mish, mask=mask)


def fused_hswish_maxpool_mish(x: torch.Tensor, subtract_value: float, kernel_size: int):
    """
    Fuses: (x - subtract_value) -> HardSwish -> MaxPool2d(kernel_size, stride=kernel_size) -> Mish
    x: NCHW tensor on CUDA
    """
    assert x.is_cuda, "Input must be on CUDA for Triton kernels."
    assert x.ndim == 4, "Expected NCHW input."
    N, C, H, W = x.shape
    K = int(kernel_size)
    assert K >= 1, "kernel_size must be >= 1"
    # Compute output spatial dims (no padding, stride=K)
    OH = (H - K) // K + 1
    OW = (W - K) // K + 1
    assert OH > 0 and OW > 0, "kernel_size too large for input size."

    x_contig = x.contiguous()

    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)

    # Strides in elements
    stride_n, stride_c, stride_h, stride_w = x_contig.stride()
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = out.stride()

    total = N * C * OH * OW

    # Choose BLOCK_SIZE based on problem size
    BLOCK_SIZE = 256
    grid = lambda meta: ((total + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Ensure subtract_value is of correct dtype
    sub_val = torch.tensor(subtract_value, dtype=out.dtype, device=out.device)

    fused_hswish_maxpool_mish_kernel[grid](
        x_contig,
        out,
        N, C, H, W,
        OH, OW,
        stride_n, stride_c, stride_h, stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        sub_val,
        BLOCK_SIZE=BLOCK_SIZE,
        K=K,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model: keep cuDNN convolution, then fuse subtract + HardSwish + MaxPool2d + Mish into a single Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = float(subtract_value)
        self.pool_kernel_size = int(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        # Fused: (x - subtract_value) -> HardSwish -> MaxPool2d(K, stride=K) -> Mish
        x = fused_hswish_maxpool_mish(x, self.subtract_value, self.pool_kernel_size)
        return x


# Keep the same interfaces for initialization and inputs
batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
subtract_value = 0.5
pool_kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]