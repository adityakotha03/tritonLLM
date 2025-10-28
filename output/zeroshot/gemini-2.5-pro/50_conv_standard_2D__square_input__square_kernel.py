An optimized version of the provided `Model` architecture is presented below, named `ModelNew`.

The primary performance bottleneck in the original model is the `nn.Conv2d` layer. While PyTorch's convolution implementation (often backed by the highly-optimized cuDNN library) is very fast, a custom Triton kernel can offer speedups for specific shapes and hardware by reducing overhead and enabling fusion opportunities.

In this optimized version, the `nn.Conv2d` operation is replaced by a custom Triton kernel, `conv_kernel`.

### Optimization Strategy:

1.  **Direct Convolution Kernel**: Instead of using a two-step `im2col` + `GEMM` approach which would materialize a large intermediate tensor, we implement a "direct" or "implicit GEMM" convolution. This approach computes the convolution directly from the input tensor.
2.  **GEMM Formulation**: The kernel is structured like a matrix multiplication (`GEMM`) problem, where the output feature map is treated as the result matrix `C`, the convolution weights as matrix `A`, and the corresponding input patches as a virtual matrix `B`. This allows us to leverage Triton's `tl.dot` operator, which efficiently maps to the A100's Tensor Cores for high throughput.
3.  **On-the-fly Indexing**: The main complexity lies in loading the correct input patches (matrix `B`) from the input tensor `X`. The kernel calculates the memory addresses of the required input elements on-the-fly, effectively performing the `im2col` transformation implicitly without any global memory overhead.
4.  **Tiling and Parallelism**: The output tensor is divided into tiles. Each Triton program instance (which maps to a CUDA thread block) is responsible for computing one output tile. This strategy maximizes parallelism and data locality.
5.  **Autotuning**: The kernel is decorated with `@triton.autotune` to automatically test different tiling sizes (`BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`), pipeline stages, and warp configurations. This ensures that the kernel achieves optimal performance for the specific shapes and hardware without manual tuning.

The `ModelNew` class maintains the `nn.Conv2d` layer to store the model's parameters (weights and biases), but its `forward` method bypasses the PyTorch implementation and calls our high-performance `triton_conv_op` wrapper instead.

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=2, num_warps=8),
    ],
    key=['C_out', 'N', 'H_out', 'W_out', 'C_in', 'K_h', 'K_w', 'stride_h', 'pad_h'],
)
@triton.jit
def conv_kernel(
    # Pointers to Tensors
    X, W, B, Y,
    # Input/Output Strides
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    # Dimensions
    N, C_in, H_in, W_in,
    C_out, K_h, K_w,
    H_out, W_out,
    # Convolution Parameters
    stride_h, stride_w,
    pad_h, pad_w,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel for 2D convolution.
    This kernel implements a direct convolution using an implicit GEMM approach.
    It computes a tile of the output tensor Y for each program instance.
    - M dimension: C_out
    - N dimension: N * H_out * W_out
    - K dimension: C_in * K_h * K_w
    """
    # Program IDs for the 2D grid
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for the output block
    # offs_m are indices along the C_out dimension
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # offs_n are linear indices along the spatial (N, H_out, W_out) dimensions
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Accumulator for the output tile
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # K dimension of the GEMM
    K_GEMM = C_in * K_h * K_w
    
    # Pointers to the reshaped weight matrix W (C_out, K_GEMM)
    w_base_ptr = W + offs_m[:, None] * K_GEMM

    # Inner loop over the K dimension of the GEMM
    for k_start in range(0