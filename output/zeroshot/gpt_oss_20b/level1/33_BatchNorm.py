import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel that performs BatchNorm forward with fused affine transform
@triton.jit
def batchnorm_forward_kernel(
    input_ptr,          # input tensor pointer
    weight_ptr,         # weight (gamma) pointer
    bias_ptr,           # bias (beta) pointer
    mean_ptr,           # per‑channel mean pointer
    inv_std_ptr,        # per‑channel 1/sqrt(var + eps) pointer
    out_ptr,            # output tensor pointer
    N,                  # total number of elements in input
    C: tl.constexpr,    # number of channels
    H: tl.constexpr,    # spatial height
    W: tl.constexpr,    # spatial width
    BLOCK_SIZE: tl.constexpr,
):
    # Global block offset
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets