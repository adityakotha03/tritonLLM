import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel that computes the sum of squared differences
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["n"],
)
@triton.jit
def _mse_sum_kernel(
    pred_ptr,
    target_ptr,
    out_ptr,
    n,  # number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # each program processes a contiguous block of size BLOCK_SIZE
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # load, compute squared diff, reduce within block
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    diff = pred - target
    sq = diff * diff

    # reduction within the block
    smem = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    smem[tl.arange(0, BLOCK_SIZE)] = sq
    tl.sync()
    i = BLOCK_SIZE // 2
    while i > 0:
        smem[:i] += smem[i : i * 2]
        tl.sync()
        i //= 2

    # store partial sum
    if mask[0]:
        tl.store(out_ptr + pid, smem[0])


def mse_loss_triton(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute mean squared error loss using a Triton kernel.
    """
    assert pred.is_cuda and target.is_cuda, "Tensors must be on CUDA."
    assert pred.shape == target.shape, "Shape mismatch between predictions and targets."

    n_elements = pred.numel()
    # Allocate output for partial sums
    partial_sums = torch.empty(( (n_elements + 256 - 1) // 256, ), device=pred.device, dtype=torch.float32)

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    _mse_sum_kernel[grid](pred, target, partial_sums, n_elements)

    # Sum partial sums on GPU
    total = torch.sum(partial_sums, dtype=torch.float32)
    # Mean
    return total / n_elements


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return mse_loss_triton(predictions, targets)