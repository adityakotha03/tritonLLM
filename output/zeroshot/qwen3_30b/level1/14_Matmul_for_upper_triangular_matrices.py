import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triangular_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,  # Size of the square matrices
    stride_A, stride_B, stride_C,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ENABLE_TENSOR_CORES: tl.constexpr,
):
    # Each program handles a block of the output matrix C
    pid_n = tl.program_id(0)  # Row block ID
    pid_k = tl.program_id(1)  # Column block ID

    # Calculate the starting row and column indices for this block
    row_start = pid_n * BLOCK_SIZE_N
    col_start = pid_k * BLOCK_SIZE_K

    # Create offsets for the current block
    offs_n = row_start + tl.arange(0, BLOCK_SIZE_N)
    offs_k = col_start + tl.arange(0, BLOCK_SIZE_K)

    # Mask for valid rows and columns (only upper triangular part)
    mask_n = offs_n < N
    mask_k = offs_k < N

    # Load A block (only upper triangular elements)
    A = tl.load(
        A_ptr + (offs_n[:, None] * stride_A) + offs_k[None, :],
        mask=(mask_n[:, None] & mask_k[None, :] & (offs_n[:, None] <= offs_k[None, :])),
        other=0.0
    )

    # Load B block (only upper triangular elements)
    B = tl.load(
        B_ptr + (offs_k[:, None] * stride_B) + offs_n[None, :],
        mask=(mask_k[:, None] & mask_n[None, :] & (offs_k[:, None] <= offs_n[None, :])),
        other=0.0
    )

    # Perform the matmul (FMA) using tensor cores if enabled
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
    if ENABLE_TENSOR_CORES:
        acc += tl.dot(A, B, allow_tf32=True)
    else:
        acc += tl.dot(A, B)

    # Final store: only write upper triangular part of output
    # Output C should also be upper triangular
    offs_c_n = row_start + tl.arange(0, BLOCK_SIZE_N)
    offs_c_k = col_start + tl.arange(0, BLOCK_SIZE_K)
    mask_c_n = offs_c_n[:, None] < N
    mask_c_k = offs_c_k[None, :] < N
    mask_upper = offs_c_n[:, None] <= offs_c_k[None, :]

    mask = mask_c_n & mask_c_k & mask_upper
    tl.store(
        C_ptr + (offs_c_n[:, None] * stride_C) + offs_c_k[None, :],
        acc,
        mask=mask
    )


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 1024, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 1024, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=8),
    ],
    key=['N'],
    nearest_power_of_2=True
)
def triton_triangular_matmul(A, B, C, N, BLOCK_SIZE_N, BLOCK_SIZE_K):
    # Ensure inputs are on GPU and contiguous
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA"
    assert A.is_contiguous() and B.is_contiguous(), "Inputs must be contiguous"

    # Initialize output tensor
    C.zero_()

    # Grid configuration: 2D grid (N/BLOCK_SIZE_N, N/BLOCK_SIZE_K)
    grid = (triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(N, BLOCK_SIZE_K))

    # Enable tensor cores by using fp16 or bf16, but promote to fp32 for accumulation
    # We use fp16 to leverage tensor cores and TF32 if available
    dtype = A.dtype
    if dtype == torch.float32:
        # Use tf32 if available, but we'll use bf16 to better leverage tensor cores
        A = A.to(torch.bfloat16)
        B = B.to(torch.bfloat16)
        C = C.to(torch.bfloat16)
        ENABLE_TENSOR_CORES = True
    else:
        A = A.to(torch.float16)
        B = B.to(torch.float16)
        C = C.to(torch.float16)
        ENABLE_TENSOR_CORES = True

    # Launch kernel
    triangular_matmul_kernel[grid](
        A, B, C,
        N,
        A.stride(0), B.stride(0), C.stride(0),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        ENABLE_TENSOR_CORES=ENABLE_TENSOR_CORES
    )

    # Final conversion to float32 if original input was float32
    if dtype == torch.float32:
        C = C.to(torch.float32)

    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        # Ensure input matrices are square and upper triangular
        N = A.size(0)
        assert A.size() == B.size() == (N, N), "Matrices must be square and of the same size"
        
        # Create output tensor
        C = torch.empty_like(A)
        
        # Use Triton kernel with autotuned block sizes
        C = triton_triangular_matmul(A, B, C, N)
        
        # Return only upper triangular part of the result
        return torch.triu(C)