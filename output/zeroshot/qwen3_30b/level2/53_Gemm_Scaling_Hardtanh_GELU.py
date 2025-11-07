import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_scale_hardtanh_gelu_kernel(
    x_ptr,      # Input matrix pointer
    w_ptr,      # Weight matrix pointer
    out_ptr,    # Output pointer
    n_elements, # Total number of elements in output
    batch_size, # Number of batches
    in_features, # Input feature dimension
    out_features, # Output feature dimension
    scaling_factor, # Scaling factor for the output
    hardtanh_min, # Min value for hardtanh
    hardtanh_max, # Max value for hardtanh
    BLOCK_SIZE_M: tl.constexpr,  # Block size along M (output features)
    BLOCK_SIZE_N: tl.constexpr,  # Block size along N (input features)
    BLOCK_SIZE_K: tl.constexpr,  # Block size along K (inner dimension)
):
    # Program ID: each block computes a tile of the output
    pid = tl.program_id(0)  # Block ID in the grid
    block_id = tl.program_id(1)  # Block ID in the batch dimension

    # Calculate the starting row and column indices for this block
    row_start = pid * BLOCK_SIZE_M
    col_start = block_id * BLOCK_SIZE_N

    # Create row and column offsets for this block
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)

    # Mask to ensure we don't go out of bounds
    row_mask = row_offsets < out_features
    col_mask = col_offsets < in_features

    # Load the input tile (BLOCK_SIZE_M x BLOCK_SIZE_N)
    x = tl.load(x_ptr + row_offsets[:, None] * in_features + col_offsets[None, :],
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0)

    # Load the weight tile (BLOCK_SIZE_N x BLOCK_SIZE_M) in transposed form
    w = tl.load(w_ptr + col_offsets[:, None] * out_features + row_offsets[None, :],
                mask=col_mask[:, None] & row_mask[None, :],
                other=0.0)

    # Initialize accumulator for the dot product
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Perform the matrix multiplication in blocks
    for k in range(0, in_features, BLOCK_SIZE_K):
        # Calculate k offsets
        k_offset = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offset < in_features

        # Load x block (BLOCK_SIZE_M x BLOCK_SIZE_K)
        x_block = tl.load(x_ptr + row_offsets[:, None] * in_features + k_offset[None, :],
                          mask=row_mask[:, None] & k_mask[None, :],
                          other=0.0)

        # Load w block (BLOCK_SIZE_K x BLOCK_SIZE_M) (transposed weight)
        w_block = tl.load(w_ptr + k_offset[:, None] * out_features + row_offsets[None, :],
                          mask=k_mask[:, None] & row_mask[None, :],
                          other=0.0)

        # Perform the dot product
        acc += tl.dot(x_block, w_block)

    # Scale the result
    acc = acc * scaling_factor

    # Apply hardtanh in-place
    acc = tl.clip(acc, hardtanh_min, hardtanh_max)

    # Apply GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Use fused approximation for better performance
    pi_sqrt = 0.7978845608028654
    x3 = acc * acc * acc
    tanh_input = pi_sqrt * (acc + 0.044715 * x3)
    tanh_val = tl.tanh(tanh_input)
    gelu_out = 0.5 * acc * (1.0 + tanh_val)

    # Write the output back
    output_ptr = out_ptr + row_offsets[:, None] * out_features + col_offsets[None, :]
    tl.store(output_ptr,
             gelu_out,
             mask=row_mask[:, None] & col_mask[None, :])


def triton_gemm_scale_hardtanh_gelu(x: torch.Tensor, w: torch.Tensor, scaling_factor: float,
                                   hardtanh_min: float, hardtanh_max: float) -> torch.Tensor:
    # Ensure inputs are contiguous and on GPU
    x = x.contiguous()
    w = w.contiguous()
    
    # Input dimensions
    batch_size, in_features = x.shape
    out_features = w.shape[0]

    # Output tensor
    out = torch.empty_like(x)

    # Block sizes
    BLOCK_SIZE_M = 128  # Tuneable: 128, 256, 512
    BLOCK_SIZE_N = 128  # Tuneable: 128, 256
    BLOCK_SIZE_K = 128  # Tuneable: 128, 256

    # Grid: one block per M x N tile, and one block per batch
    # Use a 2D grid: (M_blocks, N_blocks) for tiles, but we flatten batch into grid
    # Total blocks = ceil(batch_size / BLOCK_SIZE_N) * ceil(out_features / BLOCK_SIZE_M)
    # But we need to tile across batches, so we use 2D grid with:
    #   grid(0) = total M blocks (along out_features)
    #   grid(1) = total N blocks (along in_features), but we also need to cover batches
    # Instead, we do per-batch tiled computation across M and N

    # We'll use a 1D grid, with each block computing one tile across batch, M, and N
    # But better to use 2D: (out_features // BLOCK_SIZE_M) * (in_features // BLOCK_SIZE_N) * batch_size
    # Use a 2D grid: (num_M_tiles, num_N_tiles), and then use program_id(1) to iterate over batches

    # Compute number of M and N tiles
    num_m_tiles = triton.cdiv(out_features, BLOCK_SIZE_M)
    num_n_tiles = triton.cdiv(in_features, BLOCK_SIZE_N)

    # Grid is (num_m_tiles * num_n_tiles, batch_size)
    # Each (m_tile, n_tile) tile is computed for each batch
    grid = lambda meta: (num_m_tiles * num_n_tiles, batch_size)

    # Launch kernel
    matmul_scale_hardtanh_gelu_kernel[grid](
        x_ptr=x,
        w_ptr=w,
        out_ptr=out,
        n_elements=out.numel(),
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
        scaling_factor=scaling_factor,
        hardtanh_min=hardtanh_min,
        hardtanh_max=hardtanh_max,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

        # Use torch's linear layer but keep weights in FP16 or BF16 for better tensor core utilization
        # But we will override forward with Triton
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        # We can optionally cast to BF16 to leverage tensor cores
        # But we'll let Triton handle the dtype

    def forward(self, x):
        # Convert to BF16 if needed for tensor core usage
        # Use bf16 to enable BF16 Tensor Core acceleration in Triton
        x = x.to(torch.bfloat16)
        w = self.gemm.weight.to(torch.bfloat16)

        # Call the fused Triton kernel
        return triton_gemm_scale_hardtanh_gelu(x, w, self.scaling_factor,
                                               self.hardtanh_min, self.hardtanh_max)
