import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hinge_loss_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)

    # Compute 1 - predictions * targets
    term = 1.0 - predictions * targets
    # Clamp to min=0
    clamped = tl.where(term < 0.0, 0.0, term)
    # Sum all elements
    sum_val = tl.sum(clamped, axis=0)
    # Store the sum (we'll compute mean outside)
    tl.store(output_ptr + pid, sum_val)


def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    n_elements = predictions.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Output tensor to store sum of clamped values
    output = torch.zeros((n_elements // BLOCK_SIZE + 1,), device=predictions.device)

    # Grid size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the kernel
    hinge_loss_kernel[grid](predictions, targets, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Compute mean
    return torch.mean(output)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)