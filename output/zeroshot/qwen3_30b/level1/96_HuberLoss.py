import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def smooth_l1_kernel(
    predictions_ptr,
    targets_ptr,
    loss_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    REDUCTION_BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID for the data block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load predictions and targets
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)

    # Compute the difference
    diff = predictions - targets

    # Apply Smooth L1 loss: if |diff| < 1, use 0.5 * diff^2, else |diff| - 0.5
    squared_term = 0.5 * diff * diff
    linear_term = tl.abs(diff) - 0.5
    loss = tl.where(tl.abs(diff) < 1.0, squared_term, linear_term)

    # Store the per-element loss
    tl.store(loss_ptr + offsets, loss, mask=mask)

    # Block-wise reduction to compute the mean
    # Each block reduces its segment of the loss
    # We use a reduction kernel to compute the sum in shared memory
    # Then use a final block to reduce across all partial sums

    # Shared memory for reduction
    shared = tl.load(loss_ptr + offsets, mask=mask, other=0.0)  # Reuse the same memory
    # We will compute the sum of this block using reduction
    sum_val = tl.sum(shared, axis=0)

    # Store partial sum in the first element of each block
    # Use program_id(1) to coordinate reduction across blocks
    tid = tl.program_id(1)
    if tid == 0:
        # Only one block will write the partial sum to global memory
        tl.store(loss_ptr + (pid * REDUCTION_BLOCK_SIZE), sum_val, mask=pid < (n_elements + REDUCTION_BLOCK_SIZE - 1) // REDUCTION_BLOCK_SIZE)


def triton_smooth_l1_loss(predictions, targets):
    """
    Compute the Smooth L1 (Huber) Loss using Triton kernel.
    Fuses computation and reduction to avoid extra memory copies.
    """
    assert predictions.is_cuda and targets.is_cuda, "Inputs must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    # Output loss tensor (we'll use it for partial sums in reduction)
    n_elements = predictions.numel()
    loss = torch.empty(n_elements, device=predictions.device, dtype=predictions.dtype)

    # Block size for computation
    BLOCK_SIZE = 1024  # Power of 2, good for A100, allows efficient coalescing and occupancy
    REDUCTION_BLOCK_SIZE = 1024  # Same size for consistency

    # Grid: one block per data block
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch the main kernel to compute loss and perform partial reduction
    smooth_l1_kernel[grid](
        predictions,
        targets,
        loss,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        REDUCTION_BLOCK_SIZE=REDUCTION_BLOCK_SIZE,
    )

    # Now perform final reduction: sum all partial sums and divide by number of elements
    # We need to reduce all partial sums across blocks
    n_partial_sums = triton.cdiv(n_elements, REDUCTION_BLOCK_SIZE)
    if n_partial_sums == 1:
        final_sum = loss[0]
    else:
        # Create a buffer for partial sums
        partial_sums = torch.empty(n_partial_sums, device=predictions.device, dtype=predictions.dtype)
        # Copy partial sums from loss
        partial_sums.copy_(loss[:n_partial_sums])
        # Reduce them with a simple sum
        final_sum = torch.sum(partial_sums)

    # Compute mean loss
    mean_loss = final_sum / n_elements

    return mean_loss


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)