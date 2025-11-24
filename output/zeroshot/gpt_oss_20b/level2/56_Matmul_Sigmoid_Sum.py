import torch
import torch.nn as nn
import triton
import triton.language as tl

# ---------- Triton kernels ----------

# Matrix multiplication: output = input @ weight.T + bias
# We fuse bias addition, sigmoid activation, and row‑wise sum into a single kernel.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_W": 256, "BLOCK_SIZE_K": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE_W": 256, "BLOCK_SIZE_K": 64},  num_warps=2),
        triton.Config({"BLOCK_SIZE_W": 128, "BLOCK_SIZE_K": 64},  num_warps=2),
    ],
    key=["N", "M", "K"],
)
@triton.jit
def matmul_sigmoid_sum_kernel(
    input_ptr,          # (N, K)
    weight_ptr,         # (M, K)  -- note: we store weight in row major for easy load
    bias_ptr,           # (M,)
    out_ptr,            # (N, 1)
    N, M, K,
    BLOCK_SIZE_W: tl.constexpr,  # output columns per block
    BLOCK_SIZE_K: tl.constexpr,  # input dim per block
):
    """
    Compute y = sigmoid( x @ weight.T + bias ), then sum over the output
    dimension and write a scalar per batch sample to out_ptr.
    """
    batch_id = tl.program_id(0)          # which row of the input
    col_id   = tl.program_id(1) * BLOCK_SIZE_W

    # Load the input vector for this batch sample
    # We load in chunks of BLOCK_SIZE_K
    acc = 0.0
    for k in range(0, K, BLOCK_SIZE_K):
        # Load a chunk of the input
        k_offset = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offset < K
        x = tl.load(input_ptr + batch_id * K + k_offset, mask=mask_k, other=0.0)

        # Load a chunk of weight rows
        # Each row of weight is of size K; we load BLOCK_SIZE_W rows at a time
        w_base = (col_id + tl.arange(0, BLOCK_SIZE_W)) * K + k_offset[None, :]
        w = tl.load(weight_ptr + w_base, mask=mask_k[None, :], other=0.0)

        # Matrix‑vector multiplication chunk
        acc += tl.dot(w, x)

    # Add bias (one per row)
    bias = tl.load(bias_ptr + col_id + tl.arange(0, BLOCK_SIZE_W))

    # Sigmoid
    acc = acc + bias
    acc = 1.0 / (1.0 + tl.exp(-acc))

    # Reduce over the columns of this block
    block_sum = tl.sum(acc)

    # Store partial sums to a temp buffer (shared memory via atomic add)
    # We use a global atomic add per batch sample
    tl.atomic_add(out_ptr + batch_id, block_sum)

# ---------- Triton wrapper ----------

def triton_matmul_sigmoid_sum(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    x: (N, K)
    weight: (M, K)  (row major)
    bias: (M,)
    returns: (N, 1)
    """
    N, K = x.shape
    M = weight.shape[0]
    # Allocate output and zero it
    out = torch.zeros((N, 1), dtype=x.dtype, device=x.device)

    # Launch grid
    grid = lambda meta: (
        (N,),                            # one block per input row
        ( (M + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"], ),
    )

    matmul_sigmoid_sum_kernel[grid](
        x, weight, bias, out,
        N, M, K,
        BLOCK_SIZE_W=meta["BLOCK_SIZE_W"],
        BLOCK_SIZE_K=meta["BLOCK_SIZE_K"],
    )

    return out

# ---------- Model ----------

class ModelNew(nn.Module):
    """
    Optimized model that replaces the linear layer, sigmoid, and sum
    with a fused Triton kernel.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super(ModelNew, self).__init__()
        # Store weights in fp16 for tensor‑core speed; convert at init
        self.weight = nn.Parameter(
            torch.randn(hidden_size, input_size, dtype=torch.float16, device="cuda")
        )
        self.bias = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float16, device="cuda")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, input_size)  -- dtype float16 expected
        Returns: (batch_size, 1)  -- dtype float16
        """
        # Ensure input is fp16
        if x.dtype != torch.float16:
            x = x.to(torch.float16)
        out = triton_matmul_sigmoid_sum(x, self.weight, self.bias)
        return out

# Example usage (commented out – do not include in the final code file)
# batch_size, input_size, hidden_size = 128, 32768, 32768
# model = ModelNew(input_size, hidden_size)
# x = torch.rand(batch_size, input_size, dtype=torch.float16, device="cuda")
# out = model(x)