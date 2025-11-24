import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hinge_loss_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)

    product = predictions * targets
    clipped = tl.clamp(1.0 - product, 0.0, float('inf'))

    sum = tl.sum(clipped)
    
    tl.atomic_add(output_ptr, sum)

@triton.jit
def sum_to_mean_kernel(
    sum_ptr,
    output_ptr,
    n_elements: tl.constexpr,
):
    total_sum = tl.load(sum_ptr)
    mean = total_sum / n_elements
    tl.store(output_ptr, mean)

def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    n_elements = predictions.numel()
    assert n_elements == targets.numel(), "Predictions and targets must have the same number of elements."

    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    sum_tensor = torch.zeros(1, dtype=torch.float32, device='cuda')
    hinge_loss_kernel[grid](predictions, targets, sum_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    output = torch.empty(1, dtype=torch.float32, device='cuda')
    sum_to_mean_kernel[(1,)](sum_tensor, output, n_elements)

    return output.squeeze()

class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for Hinge Loss computation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)