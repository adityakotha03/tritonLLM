import torch
import torch.nn as nn
import triton
import triton.language as tl

# -------------------------------------------------------------
# Triton kernels for im2col + GEMM (transposed conv) fusion
# -------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_a, stride_b, stride_c,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_offset = pid * BLOCK_M
    col_offset = tl.program_id(axis=1) * BLOCK_N

    row_indices = row_offset + tl.arange(0, BLOCK_M)
    col_indices = col_offset + tl.arange(0, BLOCK_N)

    # mask to avoid out-of-bounds accesses
    mask_a = row_indices < M
    mask_b = col_indices < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_block = tl.load(
            a_ptr + k * stride_a + row_indices[:, None] * stride_a,
            mask=mask_a[:, None],
            other=0.0,
        )
        b_block = tl.load(
            b_ptr + k * stride_b + col_indices[None, :] * stride_b,
            mask=mask_b[None, :],
            other=0.0,
        )
        acc += tl.dot(a_block, b_block)

    if row_offset < M and col_offset < N:
        tl.store(c_ptr + row_offset * stride_c + col_offset * stride_c,
                 acc, mask=mask_a[:, None] & mask_b[None, :])


def matmul_torch_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication using the Triton kernel above."""
    assert a.is_cuda and b.is_cuda
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
    _matmul_kernel[grid](
        a,
        b,
        out,
        M,
        N,
        K,
        a.stride(0),
        b.stride(0),
        out.stride(0),
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
    )
    return out


# -------------------------------------------------------------
# Helper to perform im2col for transposed convolution
# -------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4),
    ],
    key=["N", "K"],
)
@triton.jit
def _im2col_kernel(
    input_ptr,
    output_ptr,
    N, K,
    stride_h, stride_w,
    pad_h, pad_w,
    kernel_h, kernel_w,
    stride_input_h, stride_input_w,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    col_offset = pid * BLOCK_N

    col_indices = col_offset + tl.arange(0, BLOCK_N)
    kernel_indices = tl.arange(0, BLOCK_K)

    mask_col = col_indices < N
    mask_k = kernel_indices < K

    for n in range(col_indices[mask_col].max() + 1):
        # compute the corresponding (oh, ow) in the output
        oh = n // (kernel_w * stride_w)
        ow = n % (kernel_w * stride_w)
        # compute input coordinates
        ih = oh * stride_h - pad_h
        iw = ow * stride_w - pad_w
        for k in kernel_indices[mask_k]:
            kh = k // kernel_w
            kw = k % kernel_w
            h_idx = ih + kh
            w_idx = iw + kw
            # bounds check
            mask_h = (h_idx >= 0) & (h_idx < stride_input_h)
            mask_w = (w_idx >= 0) & (w_idx < stride_input_w)
            idx = (h_idx * stride_input_w + w_idx) * stride_input_h
            val = tl.load(input_ptr + idx, mask=mask_h & mask_w, other=0.0)
            tl.store(output_ptr + n * K + k, val, mask=mask_col & mask_k)


def im2col_torch_triton(input: torch.Tensor,
                       kernel_size: int,
                       stride: int,
                       padding: int) -> torch.Tensor:
    """Perform im2col on input for transposed convolution."""
    b, c, h, w = input.shape
    oh = (h - 1) * stride + kernel_size - 2 * padding
    ow = (w - 1) * stride + kernel_size - 2 * padding
    # each column corresponds to one output pixel
    N = oh * ow
    K = c * kernel_size * kernel_size
    col = torch.empty((N, K), dtype=input.dtype, device=input.device)
    grid = (triton.cdiv(N, 128),)
    _im2col_kernel[grid](
        input.view(-1),
        col,
        N,
        K,
        stride,
        stride,
        padding,
        padding,
        kernel_size,
        kernel_size,
        h,
        w,
        BLOCK_N=128,
        BLOCK_K=32,
    )
    return col


# -------------------------------------------------------------
# ModelNew using Triton kernels
# -------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with asymmetric input and a square kernel,
    implemented entirely with custom Triton kernels.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Weight shape: (in_channels, out_channels, kernel_h, kernel_w)
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kernel_size, kernel_size, device="cuda")
        )
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_channels, device="cuda"))
        else:
            self.bias_param = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements transposed convolution using im2col + GEMM + bias addition.
        """
        b, c, h, w = x.shape
        kh, kw = self.kernel_size, self.kernel_size
        oh = (h - 1) * self.stride + kh - 2 * self.padding + self.output_padding
        ow = (w - 1) * self.stride + kw - 2 * self.padding + self.output_padding
        out = torch.empty((b, self.out_channels, oh, ow), dtype=x.dtype, device=x.device)

        # Reshape weight to (out_channels, in_channels * kh * kw)
        weight_reshaped = self.weight.view(self.in_channels, self.out_channels, -1).transpose(0, 1).reshape(self.out_channels, -1)

        for i in range(b):
            # im2col: shape (oh*ow, in_channels*kh*kw)
            col = im2col_torch_triton(x[i], self.kernel_size, self.stride, self.padding)
            # matrix multiplication: (out_channels, in_channels*kh*kw) @ (in_channels*kh*kw, oh*ow).T
            # result shape: (out_channels, oh*ow)
            matmul_out = matmul_torch_triton(weight_reshaped, col.t())
            # reshape to (out_channels, oh, ow)
            out[i] = matmul_out.view(self.out_channels, oh, ow)

        if self.bias:
            out += self.bias_param.view(1, -1, 1, 1)

        return out