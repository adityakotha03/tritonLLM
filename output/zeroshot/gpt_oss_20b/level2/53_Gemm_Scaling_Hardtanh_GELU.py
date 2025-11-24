import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# -----------------------------
# Triton kernel: GEMV + scaling + hardtanh + gelu
# -----------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256, "TILE_K": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512, "TILE_K": 256}, num_warps=8),
    ],
    key=["n_out"],
)
@triton.jit
def gemv_fused_kernel(
    weight_ptr,          # [n_out, n_in]
    input_ptr,           # [n_in]
    out_ptr,             # [n_out]
    n_in: tl.constexpr,
    n_out: tl.constexpr,
    scaling: tl.constexpr,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    TILE_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_SIZE
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)

    # Mask for handling the tail rows
    mask = row_offsets < n_out

    # Accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over the input dimension in tiles
    for k in range(0, n_in, TILE_K):
        # Load a tile of the weight matrix
        weight_offsets = row_offsets[:, None] * n_in + k + tl.arange(0, TILE_K)[None, :]
        weight_tile = tl.load(weight_ptr + weight_offsets, mask=mask[:, None], other=0.0)

        # Load a tile of the input vector
        input_offsets = k + tl.arange(0, TILE_K)
        input_tile = tl.load(input_ptr + input_offsets, mask=input_offsets < n_in, other=0.0)

        # dot product for each row in the block
        acc += tl.dot(weight_tile, input_tile)

    # Apply scaling
    acc *= scaling

    # Hardtanh
    acc = tl.where(acc < min_val, min_val, acc)
    acc = tl.where(acc > max_val, max_val, acc)

    # GELU (approximate)
    sqrt_2_over_pi = 0.7978845608028654
    x_cubed = acc * acc * acc
    inner = sqrt_2_over_pi * (acc + 0.044715 * x_cubed)
    tanh_inner = tl.tanh(inner)
    acc = 0.5 * acc * (1.0 + tanh_inner)

    # Store result
    tl.store(out_ptr + row_offsets, acc, mask=mask)


def fused_linear_activation(
    weight: torch.Tensor,  # [n_out, n_in]
    input_vec: torch.Tensor,  # [n_in]
    scaling_factor: float,
    hardtanh_min: float,
    hardtanh_max: float,
) -> torch.Tensor:
    """
    Performs: y = GELU(Hardtanh((x @ W.T) * scaling_factor))
    """
    assert weight.is_cuda and input_vec.is_cuda
    weight = weight.contiguous()
    input_vec = input_vec.contiguous()

    n_out, n_in = weight.shape

    out = torch.empty(n_out, device=weight.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(n_out, meta["BLOCK_SIZE"]),)

    gemv_fused_kernel[grid](
        weight_ptr=weight.data_ptr(),
        input_ptr=input_vec.data_ptr(),
        out_ptr=out.data_ptr(),
        n_in=n_in,
        n_out=n_out,
        scaling=scaling_factor,
        min_val=hardtanh_min,
        max_val=hardtanh_max,
        TILE_K=256,
        BLOCK_SIZE=256,
    )
    return out


# -----------------------------
# New Model definition
# -----------------------------
class ModelNew(nn.Module):
    """
    Optimized model that fuses GEMM, scaling, hardtanh, and GELU into a single Triton kernel.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device="cuda"))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        # x: [batch, in_features]
        # Apply fused operation per batch element
        batch = x.shape[0]
        out = []
        for i in range(batch):
            out.append(fused_linear_activation(
                self.weight,
                x[i],
                self.scaling_factor,
                self.hardtanh_min,
                self.hardtanh_max
            ))
        return torch.stack(out, dim=0)


# -----------------------------
# Helper functions (unchanged)
# -----------------------------
def get_inputs():
    return [torch.rand(2048, 8192, device="cuda")]

def get_init_inputs():
    return [8192, 8192, 0.5, -2.0, 2.0]