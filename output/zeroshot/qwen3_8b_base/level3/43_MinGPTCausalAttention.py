import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    q_ptr, k_ptr, v_ptr,
    out_ptr,
    B, T, C, H, HS,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    # Compute the block offset in the batch dimension
    b_idx = pid // (T // GROUP_SIZE)
    # Compute the block offset in the sequence dimension
    t_idx = (pid % (T // GROUP_SIZE)) * GROUP_SIZE
    # Compute the block offset in the head dimension
    h_idx = tl.program_id(1)
    # Compute the block offset in the head size dimension
    hs_idx = tl.program_id(2)
    # Compute the block offset in the sequence dimension
    t_start = t_idx
    # Compute the block offset in the head dimension
    h_start = h_idx * H
    # Compute the block offset in the head size dimension
    hs_start = hs_idx * HS
    # Compute the block offset in the batch dimension
    b_start = b_idx * B
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_start + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_offsets + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t_offsets = t_start + tl.arange(0, GROUP_SIZE)
    # Compute the block offset in the head dimension
    h_offsets = h_start + tl.arange(0, H)
    # Compute the block offset in the head size dimension
    hs_offsets = hs_start + tl.arange(0, HS)
    # Compute the block offset in the batch dimension
    b_offsets = b_idx * B + tl.arange(0, B)
    # Compute the block offset in the sequence dimension
    t