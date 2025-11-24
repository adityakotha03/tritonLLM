import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hinge_loss_kernel(
    pred_ptr,          # pointer to predictions tensor
    target_ptr,        # pointer to targets tensor
    n,                 # total number of elements
    sum_ptr,           # pointer to a single-element sum tensor
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Load the data
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

    # Compute the per-element hinge loss
    loss = tl.maximum(0.0, 1.0 - pred * target)

    # Sum within the block
    block_sum = tl.sum(loss, axis=0)

    # Atomically add the block sum to the global sum
    tl.atomic_add(sum_ptr, block_sum)


def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean hinge loss using a custom Triton kernel.

    Parameters
    ----------
    predictions : torch.Tensor
        Predicted scores (float32) on GPU.
    targets : torch.Tensor
        Binary targets in {-1, 1} (float32) on GPU.

    Returns
    -------
    torch.Tensor
        Scalar mean hinge loss.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    n = predictions.numel()
    sum_tensor = torch.zeros(1, device=predictions.device, dtype=predictions.dtype)

    BLOCK_SIZE = 256  # Tunable parameter
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    hinge_loss_kernel[grid](
        predictions, targets, n, sum_tensor, BLOCK_SIZE=BLOCK_SIZE
    )

    return sum_tensor / n


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return triton_hinge_loss(predictions, targets)