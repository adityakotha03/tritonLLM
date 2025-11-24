import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def smooth_l1_loss_kernel(
    pred_ptr,           # pointer to predictions
    target_ptr,         # pointer to targets
    loss_ptr,           # pointer to output loss
    n_elements,         # total number of elements
    beta: tl.constexpr, # threshold for smooth L1 loss
    BLOCK_SIZE: tl.constexpr,
):
    # Compute block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load predictions and targets
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

    # Compute absolute difference
    diff = pred - target
    abs_diff = tl.abs(diff)

    # Compute smooth L1 loss:
    # if |diff| < beta: 0.5 * |diff|^2 / beta
    # otherwise: |diff| - 0.5 * beta
    squared_loss = 0.5 * abs_diff * abs_diff / beta
    linear_loss = abs_diff - 0.5 * beta
    loss = tl.where(abs_diff < beta, squared_loss, linear_loss)

    # Store the loss
    tl.store(loss_ptr + offsets, loss, mask=mask)

def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, beta: float = 1.0):
    assert predictions.is_cuda and targets.is_cuda, "Inputs must be on CUDA"
    assert predictions.shape == targets.shape, "Prediction and target shapes must match"
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    out = torch.empty_like(predictions)

    # Heuristic block size
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    smooth_l1_loss_kernel[grid](
        predictions, targets, out, n_elements, beta=beta, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean of the loss, matching PyTorch's default behavior
    return out.mean()

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)