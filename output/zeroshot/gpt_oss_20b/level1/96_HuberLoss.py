import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def smooth_l1_kernel(
    x_ptr,          # Pointer to predictions
    y_ptr,          # Pointer to targets
    out_ptr,        # Pointer to output losses
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load predictions and targets
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    diff = x - y
    abs_diff = tl.abs(diff)

    # Smooth L1: 0.5 * diff^2 if |diff| < 1 else |diff| - 0.5
    loss = tl.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)

    tl.store(out_ptr + offsets, loss, mask=mask)


def triton_smooth_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert pred.is_cuda and target.is_cuda, "Tensors must be on CUDA."
    pred = pred.contiguous()
    target = target.contiguous()

    n_elements = pred.numel()
    out = torch.empty_like(pred, dtype=pred.dtype)

    BLOCK_SIZE = 128  # Tunable

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    smooth_l1_kernel[grid](pred, target, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks
    using a custom Triton kernel.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute element-wise Smooth L1 loss
        per_elem_loss = triton_smooth_l1(predictions, targets)
        # Return the mean loss (reduction='mean')
        return per_elem_loss.mean()