import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mse_kernel(
    pred_ptr,     # Pointer to predictions
    target_ptr,   # Pointer to targets
    output_ptr,   # Pointer to output (reduction result)
    n_elements,   # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the starting index for this block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load predictions and targets
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

    # Compute squared difference
    diff = pred - target
    sq_diff = diff * diff

    # Block-local sum using reduce
    sum_sq_diff = tl.sum(sq_diff, axis=0)

    # Store partial sum in output via atomic add
    tl.atomic_add(output_ptr, sum_sq_diff)

@triton.jit
def fill_kernel(
    output_ptr,
    value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, value, mask=mask)

def triton_mse_loss(predictions: torch.Tensor, targets: torch.Tensor):
    assert predictions.is_cuda and targets.is_cuda, "Inputs must be on CUDA"
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    n_elements = predictions.numel()
    assert targets.numel() == n_elements, "Predictions and targets must have same number of elements"

    # Use a single-element tensor for atomic accumulation
    sum_sq_diff = torch.zeros(1, dtype=torch.float32, device='cuda')

    # Kernel launch grid
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch reduction kernel with atomic add
    mse_kernel[grid](predictions, targets, sum_sq_diff, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Compute mean
    mean_sq_diff = sum_sq_diff / n_elements

    return mean_sq_diff

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_mse_loss(predictions, targets)