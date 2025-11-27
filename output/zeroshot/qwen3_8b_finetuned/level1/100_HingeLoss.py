import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hinge_loss_kernel(
    predictions_ptr,  # Pointer to predictions (shape (B,))
    targets_ptr,     # Pointer to targets (shape (B,))
    loss_ptr,        # Pointer to output loss (scalar)
    n_elements,      # Total number of elements (B)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a contiguous block of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load predictions and targets
    p = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    t = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    # Compute hinge term: max(0, 1 - p*t)
    pt = p * t
    term = 1.0 - pt
    clamped = tl.maximum(term, 0.0)
    # Sum the clamped values
    sum_clamped = tl.sum(clamped, axis=0)
    # Store the sum (used by the mean reduction)
    tl.store(loss_ptr + 0, sum_clamped, mask=mask)
    tl.store(loss_ptr + 1, n_elements, mask=mask)  # Store element count for mean


def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Triton implementation of hinge loss = mean(max(0, 1 - predictions * targets)).

    Args:
        predictions: Tensor of shape (B,)
        targets: Tensor of shape (B,) with values in {-1, 1}

    Returns:
        Scalar tensor containing the hinge loss
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.dim() == 1 and targets.dim() == 1, "Only 1D tensors supported."
    assert predictions.size(0) == targets.size(0), "Batch sizes must match."

    B = predictions.size(0)
    # Allocate a temporary buffer for the sum and element count
    temp = torch.empty(2, dtype=torch.float32, device=predictions.device)
    # Launch the fused kernel
    grid = lambda meta: ((B + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    hinge_loss_kernel[grid](predictions, targets, temp, B, BLOCK_SIZE=256)
    # Compute the mean by dividing the sum by the element count
    loss = temp[0] / temp[1]
    return loss


class ModelNew(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks using a fused Triton kernel.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Triton implementation of hinge loss = mean(max(0, 1 - predictions * targets))
        return triton_hinge_loss(predictions, targets)