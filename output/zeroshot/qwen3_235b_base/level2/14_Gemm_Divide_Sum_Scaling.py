import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_div_sum_scale_kernel(
    x_ptr, weight_ptr, output_ptr,
    batch_size, input_size, hidden_size,
    scaling_factor,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program IDs
    pid_m = tl.program_id(0)

    # Pointers for the current batch row
    offset_x = pid_m * input_size
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(batch_size, input_size),
        strides=(input_size, 1),
        offsets=(pid_m, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0)
    )
    # We'll compute a reduced output: (batch_size, 1), so each row reduces over hidden_size
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # Iterate over input_size in blocks
    for k in range(0, input_size, BLOCK_SIZE_K):
        # Load input tile (BLOCK_SIZE_M x BLOCK_SIZE_K)
        x = tl.load(x_block_ptr, boundary_check=(0,1), padding_option="zero")
        
        # Weight is (hidden_size, input_size), we need to load a block of (BLOCK_SIZE_K, hidden_size)
        weight_block_ptr = tl.make_block_ptr(
            base=weight_ptr,
            shape=(input_size, hidden_size),
            strides=(1, input_size),
            offsets=(k, 0),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
            order=(0, 1)
        )
        weight = tl.load(weight_block_ptr, boundary_check=(0,1), padding_option="zero")

        # Perform matmul tile: (BLOCK_SIZE_M x BLOCK_SIZE_K) @ (BLOCK_SIZE_K x BLOCK_SIZE_N)
        # But we only need the sum over hidden_size, so we can accumulate per-row sum
        # Instead of computing full (M, N), we compute partial sum of (x @ weight.T) per row
        # x: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # weight: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # x @ weight -> (BLOCK_SIZE_M, BLOCK_SIZE_N), then sum over N -> (BLOCK_SIZE_M,)
        # But we can fuse: sum_{k,n} x[m,k] * weight[k,n] = sum_k x[m,k] * (sum_n weight[k,n])
        weight_rowsum = tl.sum(weight, axis=1)  # (BLOCK_SIZE_K,)
        # Multiply and accumulate
        acc += tl.dot(x, weight_rowsum)  # (BLOCK_SIZE_M,)

        # Update pointers
        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_SIZE_K))
        weight_block_ptr = tl.advance(weight_block_ptr, (BLOCK_SIZE_K, 0))

    # Now divide by 2 and scale by scaling_factor
    acc = acc / 2.0
    acc = acc * scaling_factor

    # Store result (output is (batch_size, 1))
    output_offset = pid_m
    tl.store(output_ptr + output_offset, acc, mask=(tl.arange(0, BLOCK_SIZE_M) < batch_size))


class ModelNew(nn.Module):
    """
    Optimized version of Model using a fused Triton kernel that combines:
    - matmul(x, weight.T)
    - divide by 2
    - sum over hidden dimension (dim=1)
    - scale by scaling_factor
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        batch_size, input_size = x.shape
        hidden_size = self.weight.shape[0]

        # Output is (batch_size, 1)
        out = torch.empty((batch_size, 1), dtype=torch.float32, device=x.device)

        # Ensure contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()

        # Kernel launch parameters
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 64

        # Grid
        grid = (batch_size,)

        # Launch kernel
        matmul_div_sum_scale_kernel[grid](
            x, weight, out,
            batch_size, input_size, hidden_size,
            self.scaling_factor,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K
        )

        return out