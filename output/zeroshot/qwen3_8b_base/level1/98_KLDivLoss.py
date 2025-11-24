import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def kl_div_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    batch_size,
    seq_len,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for out-of-bounds
    mask = offsets < seq_len

    # Load predictions and targets
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)

    # Compute log(predictions)
    log_predictions = tl.math.log(predictions)

    # Compute KL divergence for each element
    kl = (predictions * (log_predictions - tl.math.log(targets))) * (targets != 0.0)

    # Sum over the last dimension (dim)
    if dim == -1:
        kl = tl.sum(kl, axis=-1)
    else:
        kl = tl.sum(kl, axis=dim)

    # Store the result
    tl.store(output_ptr + pid, kl, mask=pid < batch_size)


def triton_kl_div(predictions: torch.Tensor, targets: torch.Tensor):
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    # Compute batch mean
    batch_size = predictions.size(0)
    seq_len = predictions.size(1) if predictions.dim() == 2 else predictions.size(0)
    dim = -1

    # Output tensor
    output = torch.empty(batch_size, dtype=torch.float32, device=predictions.device)

    # Determine block size
    BLOCK_SIZE = 1024  # Tunable parameter

    # Grid size
    grid = lambda meta: (batch_size,)

    # Launch kernel
    kl_div_kernel[grid](predictions, targets, output, batch_size, seq_len, dim, BLOCK_SIZE=BLOCK_SIZE)
    return torch.mean(output)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_kl_div(predictions, targets)