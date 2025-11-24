import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def diag_matmul_kernel(
    diag_ptr,  # Pointer to diagonal vector A (shape: N,)
    mat_ptr,   # Pointer to matrix B (shape: N, M)
    output_ptr,  # Pointer to output matrix C (shape: N, M)
    N,
    M,
    stride_mat,  # Stride of matrix B (usually M)
    stride_out,  # Stride of output (usually M)
    BLOCK_SIZE_M: tl.constexpr,
):
    # Compute program ids
    pid_n = tl.program_id(0)  # Row block index
    pid_m = tl.program_id(1)  # Column block index

    # Define offsets for blocks
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M

    # Load diagonal element for row pid_n
    diag_val = tl.load(diag_ptr + pid_n)

    # Compute pointers to matrix B and output for this block
    mat_block_ptr = mat_ptr + pid_n * stride_mat + offs_m
    out_block_ptr = output_ptr + pid_n * stride_out + offs_m

    # Load block of B
    b_vals = tl.load(mat_block_ptr, mask=mask_m, other=0.0)

    # Multiply by diagonal element
    result = diag_val * b_vals

    # Store result
    tl.store(out_block_ptr, result, mask=mask_m)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        # Ensure inputs are on GPU and contiguous
        A = A.contiguous().cuda()
        B = B.contiguous().cuda()

        N = A.shape[0]
        M = B.shape[1]

        # Allocate output
        C = torch.empty((N, M), device="cuda", dtype=B.dtype)

        # Launch kernel
        # Each block handles one row of N, and a block of columns of size BLOCK_SIZE_M
        BLOCK_SIZE_M = 128
        grid = (N, triton.cdiv(M, BLOCK_SIZE_M))

        diag_matmul_kernel[grid](
            A,
            B,
            C,
            N,
            M,
            B.stride(0),
            C.stride(0),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
        )

        return C