import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def conv2d_igemm_fwd_kernel(
    x_ptr,  # (B, C, H, W)
    w_ptr,  # (OC, C, KH, KW)
    b_ptr,  # (OC,) or nullptr if no bias
    y_ptr,  # (B, OC, HO, WO)

    B, C, H, W,
    OC, KH, KW,
    HO, WO,

    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,

    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_woc, stride_wc, stride_wkh, stride_wkw,
    stride_yb, stride_yc, stride_yh, stride_yw,

    M, N, K,
    has_bias: tl.constexpr,

    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Decode m -> (b, oh, ow)
    HW_out = HO * WO
    b_idx = offs_m // HW_out
    tmp = offs_m % HW_out
    oh = tmp // WO
    ow = tmp % WO

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k0 = 0
    while k0 < K:
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Decode k -> (ci, kh, kw)
        KH_KW = KH * KW
        ci = offs_k // KH_KW
        rem = offs_k % KH_KW
        kh = rem // KW
        kw = rem % KW

        # Compute input coordinates for A tile
        in_h = oh[:, None] * stride_h + kh[None, :] * dil_h - pad_h
        in_w = ow[:, None] * stride_w + kw[None, :] * dil_w - pad_w

        # Bounds mask for input
        mask_in_h = (in_h >= 0) & (in_h < H)
        mask_in_w = (in_w >= 0) & (in_w < W)
        a_mask = mask_in_h & mask_in_w & mask_m[:, None] & mask_k[None, :]

        # Pointers to A (input)
        a_ptrs = (
            x_ptr
            + b_idx[:, None] * stride_xb
            + ci[None, :] * stride_xc
            + in_h * stride_xh
            + in_w * stride_xw
        )

        # Load A tile
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Pointers to B (weights)
        b_ptrs = (
            w_ptr
            + offs_n[None, :] * stride_woc
            + ci[:, None] * stride_wc
            + kh[:, None] * stride_wkh
            + kw[:, None] * stride_wkw
        )
        b_mask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

        k0 += BLOCK_K

    # Add bias if present
    if has_bias:
        bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias[None, :]

    # Store results
    y_ptrs = (
        y_ptr
        + b_idx[:, None] * stride_yb
        + offs_n[None, :] * stride_yc
        + oh[:, None] * stride_yh
        + ow[:, None] * stride_yw
    )
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=out_mask)


def triton_conv2d_implicit_gemm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                                stride=(1, 1), padding=(0, 0), dilation=(1, 1)) -> torch.Tensor:
    """
    x: (B, C, H, W)
    weight: (OC, C, KH, KW)
    bias: (OC,) or None
    stride, padding, dilation: 2-tuples
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors."
    assert x.ndim == 4 and weight.ndim == 4
    B, C, H, W = x.shape
    OC, Cw, KH, KW = weight.shape
    assert C == Cw, "Input channels must match weight's in_channels"
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    # Compute output spatial size
    HO = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    WO = (W + 2 * pw - dw * (KW - 1) - 1) // sw + 1

    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Dtype handling: compute in input dtype, accumulate in fp32 in kernel; output same dtype as input
    # Kernel pointers are specialized to dtype at runtime.
    y = torch.empty((B, OC, HO, WO), device=x.device, dtype=x.dtype)

    # Shapes for implicit GEMM: M x N = (B*HO*WO) x OC, K = C*KH*KW
    M = B * HO * WO
    N = OC
    K = C * KH * KW

    # Strides in elements
    stride_xb, stride_xc, stride_xh, stride_xw = x.stride()
    stride_woc, stride_wc, stride_wkh, stride_wkw = weight.stride()
    stride_yb, stride_yc, stride_yh, stride_yw = y.stride()

    has_bias = 1 if bias is not None else 0
    b_ptr = bias if bias is not None else weight  # dummy non-null pointer if has_bias == 0

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    conv2d_igemm_fwd_kernel[grid](
        x, weight, b_ptr, y,
        B, C, H, W,
        OC, KH, KW,
        HO, WO,
        sh, sw,
        ph, pw,
        dh, dw,
        stride_xb, stride_xc, stride_xh, stride_xw,
        stride_woc, stride_wc, stride_wkh, stride_wkw,
        stride_yb, stride_yc, stride_yh, stride_yw,
        M, N, K,
        has_bias=has_bias,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2, bias=True)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor):
        # CPU fallback for correctness
        if not x.is_cuda:
            return self.conv1(x)
        # Extract conv params
        kh, kw = self.conv1.kernel_size
        sh, sw = self.conv1.stride
        ph, pw = self.conv1.padding
        dh, dw = self.conv1.dilation

        # Align weight/bias dtype with input
        w = self.conv1.weight.to(dtype=x.dtype, device=x.device)
        b = self.conv1.bias.to(dtype=x.dtype, device=x.device) if self.conv1.bias is not None else None

        out = triton_conv2d_implicit_gemm(
            x, w, b, stride=(sh, sw), padding=(ph, pw), dilation=(dh, dw)
        )
        return out


# Test code helper functions (as per original interface)
batch_size = 256
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]