import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def kl_div_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,  # total number of elements
    block_size: tl.constexpr,
    num_elements_per_block: tl.constexpr,
    n_blocks: tl.constexpr,
):
    # Each program handles a block of data
    program_id = tl.program_id(0)
    # Offset for this block
    block_start = program_id * block_size

    # Load predictions and targets for this block
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements

    # Load predictions and targets
    preds = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)

    # Clamp predictions to avoid log(0)
    preds = tl.clamp(preds, min=1e-10, max=1.0)

    # Compute log predictions and log targets
    log_preds = tl.log(preds)
    log_targets = tl.log(targets)

    # Compute the KL term: preds * (log_preds - log_targets)
    kl_term = preds * (log_preds - log_targets)

    # Compute sum over this block
    # We use block-level reduction to avoid memory bandwidth from each thread writing to global memory
    # Initialize shared memory for reduction
    # Block-wise reduction: accumulate partial sums in shared memory
    shared_mem = tl.zeros((block_size,), dtype=tl.float32)
    # Store the partial sum in shared memory
    partial_sum = tl.sum(kl_term, axis=0)
    # Write to shared memory
    tl.store(shared_mem + tl.arange(0, block_size), partial_sum)

    # Synchronize threads in the block
    tl.debug_barrier()

    # Now, only one thread per block writes to global output
    if program_id == 0:
        # Global output is a scalar, so only one block writes
        tl.store(output_ptr, tl.sum(shared_mem, axis=0))


def triton_kl_div(predictions, targets):
    # Ensure inputs are on GPU and contiguous
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA"
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    # Get total number of elements
    n_elements = predictions.numel()

    # Define block size
    block_size = 1024  # Power of 2, suitable for large batch

    # Determine number of blocks
    num_blocks = (n_elements + block_size - 1) // block_size

    # Grid: one block per program
    grid = lambda meta: (num_blocks,)

    # Output: scalar
    output = torch.empty(1, dtype=torch.float32, device=predictions.device)

    # Launch the Triton kernel
    kl_div_kernel[grid](
        predictions,
        targets,
        output,
        n_elements,
        BLOCK_SIZE=block_size,
        num_elements_per_block=block_size,
        n_blocks=num_blocks
    )

    # Return scalar result
    return output


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_kl_div(predictions, targets)