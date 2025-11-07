import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def matmul_gelu_kernel(
    inp_ptr,  # Input pointer
    w_ptr,    # Weight pointer
    out_ptr,  # Output pointer
    n_rows,   # Number of rows in input (batch_size)
    n_cols,   # Number of columns in weight (out_features)
    n_k,      # Number of columns in input (in_features), i.e. k-dimension
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program ID (block index)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets
    row_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create mask for valid rows and columns
    row_mask = row_offset < n_rows
    col_mask = col_offset < n_cols
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, n_k, BLOCK_SIZE_K):
        # Load input block (BLOCK_SIZE_M x BLOCK_SIZE_K)
        k_offset = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offset < n_k
        inp_block = tl.load(
            inp_ptr + row_offset[:, None] * n_k + k_offset[None, :],
            mask=tl.expand_mask(row_mask[:, None] & mask_k[None, :]),
            other=0.0
        )
        
        # Load weight block (BLOCK_SIZE_K x BLOCK_SIZE_N)
        w_block = tl.load(
            w_ptr + k_offset[:, None] * n_cols + col_offset[None, :],
            mask=tl.expand_mask(mask_k[:, None] & col_mask[None, :]),
            other=0.0
        )
        
        # Perform matrix multiplication
        acc += tl.dot(inp_block, w_block)
    
    # Convert to float16 if needed (for GELU computation)
    acc = acc.to(tl.float16)
    
    # Apply GELU activation
    # GELU = 0.5 * x * (1 + erf(x / sqrt(2)))
    # Approximate with tanh: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x = acc
    sqrt_2_over_pi = tl.constexpr(0.7978845608028654)
    coeff = tl.constexpr(0.044715)
    tanh_input = sqrt_2_over_pi * (x + coeff * x * x * x)
    tanh_out = tl.tanh(tanh_input)
    gelu_out = 0.5 * x * (1.0 + tanh_out)
    
    # Store GELU output
    tl.store(
        out_ptr + row_offset[:, None] * n_cols + col_offset[None, :],
        gelu_out,
        mask=tl.expand_mask(row_mask[:, None] & col_mask[None, :])
    )


@triton.jit
def softmax_kernel(
    inp_ptr,  # Input pointer
    out_ptr,  # Output pointer
    n_rows,   # Number of rows (batch_size)
    n_cols,   # Number of columns (out_features)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Get program ID
    pid_m = tl.program_id(0)
    row_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offset < n_rows
    
    # Shared memory for storing max values and exponential sums
    shmem = tl.make_block_ptr(
        base=tl.zeros((1,), dtype=tl.float32),
        shape=(2 * BLOCK_SIZE_N,),
        strides=(1,),
        offsets=(0, 0),
        block_shape=(2 * BLOCK_SIZE_N,),
        order=(0,)
    )
    
    # Compute maximum value per row (for numerical stability)
    for i in range(0, n_cols, BLOCK_SIZE_N):
        col_offset = i + tl.arange(0, BLOCK_SIZE_N)
        mask = row_mask[:, None] & (col_offset[None, :] < n_cols)
        x = tl.load(inp_ptr + row_offset[:, None] * n_cols + col_offset[None, :], mask=mask, other=-float('inf'))
        x_max = tl.max(x, axis=1)
        tl.store(shmem, x_max, mask=tl.expand_mask(row_mask))
        
        # Load max and compute exponential
        x_max = tl.load(shmem, mask=row_mask)
        x = x - x_max[:, None]
        exp_x = tl.exp(x)
        exp_sum = tl.sum(exp_x, axis=1)
        tl.store(shmem + BLOCK_SIZE_N, exp_sum, mask=row_mask)
        
        # Store softmax output
        softmax = exp_x / exp_sum[:, None]
        tl.store(
            out_ptr + row_offset[:, None] * n_cols + col_offset[None, :],
            softmax,
            mask=mask
        )


def triton_matmul_gelu_softmax(x: torch.Tensor, w: torch.Tensor):
    # Ensure inputs are contiguous
    x = x.contiguous()
    w = w.contiguous()
    
    # Output tensor
    out = torch.empty(x.shape[0], w.shape[0], device=x.device, dtype=torch.float16)
    
    # Parameters
    n_rows = x.shape[0]
    n_cols = w.shape[0]
    n_k = x.shape[1]
    
    # Block sizes (optimized for A100)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    
    # Grid for matmul kernel
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE_M"]),
                         triton.cdiv(n_cols, meta["BLOCK_SIZE_N"]))
    
    # Launch matmul + gelu kernel
    matmul_gelu_kernel[grid](
        x, w, out,
        n_rows, n_cols, n_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    # Grid for softmax kernel
    grid_softmax = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE_M"]),)
    
    # Launch softmax kernel
    softmax_kernel[grid_softmax](
        out, out,
        n_rows, n_cols,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        # Initialize weights with appropriate scaling
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        # Use Triton kernel for matmul + gelu + softmax
        return triton_matmul_gelu_softmax(x, self.linear.weight.T)