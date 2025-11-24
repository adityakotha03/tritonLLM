import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def kl_div_kernel(
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

    # Load predictions and targets
    pred = tl.load(predictions_ptr + offsets, mask=mask, other=1.0)
    target = tl.load(targets_ptr + offsets, mask=mask, other=0.0)

    # Compute log(predictions)
    log_pred = tl.log(pred)

    # Compute target * log(pred)
    kl_val = target * log_pred

    # Write result
    tl.store(output_ptr + offsets, -kl_val, mask=mask)  # negative because KL is -sum(target * log(pred / target))

@triton.jit
def reduce_batchmean_kernel(
    input_ptr,
    output_ptr,
    total_elements,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Load block of values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    sum_val = tl.sum(x)

    # Perform block-wide reduction using tl.reduce
    batch_reduced = tl.reduce(sum_val, axis=0)

    # Only first block computes final mean
    if pid == 0:
        total_mean = batch_reduced / batch_size
        tl.store(output_ptr, total_mean)

def triton_kl_div(predictions: torch.Tensor, targets: torch.Tensor):
    assert predictions.is_cuda and targets.is_cuda, "Inputs must be on CUDA"
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    batch_size, seq_len = predictions.shape
    total_elements = batch_size * seq_len

    # Allocate output buffer for per-element KL divergence
    kl_elementwise = torch.empty_like(predictions)

    # Grid for elementwise part
    grid_elementwise = lambda meta: ((total_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    # Launch elementwise KL computation
    BLOCK_SIZE_ELEMENT = 1024
    kl_div_kernel[grid_elementwise](
        predictions, targets, kl_elementwise,
        total_elements,
        BLOCK_SIZE=BLOCK_SIZE_ELEMENT
    )

    # Now reduce across all elements with batchmean: sum / batch_size
    output = torch.empty((), dtype=torch.float32, device=predictions.device)
    grid_reduce = lambda meta: ((total_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    BLOCK_SIZE_REDUCE = 1024
    reduce_batchmean_kernel[grid_reduce](
        kl_elementwise,
        output,
        total_elements,
        batch_size,
        BLOCK_SIZE=BLOCK_SIZE_REDUCE
    )

    return output

class ModelNew(nn.Module):
    """
    Optimized version of KL divergence using custom Triton kernels.
    Computes KL divergence with fused log and multiplication, then reduces with batchmean.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_kl_div(predictions, targets)