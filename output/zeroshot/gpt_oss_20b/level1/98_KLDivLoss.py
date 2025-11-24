import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    ],
    key=["n"],
)
@triton.jit
def kl_div_kernel(
    pred_ptr,          # pointer to predictions
    target_ptr,        # pointer to targets
    out_ptr,           # pointer to the output scalar
    n,                 # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Load data
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

    # Compute log predictions and targets
    log_pred = tl.math.log(pred)
    log_target = tl.math.log(target)

    # Element‑wise KL contribution
    kl = target * (log_target - log_pred)

    # Reduce the block’s contribution into a single value
    block_sum = tl.sum(kl * mask, axis=0)

    # Atomically add the block sum to the global result
    tl.atomic_add(out_ptr, block_sum)


def triton_kl_div(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence with reduction='batchmean' using a fused Triton kernel.
    """
    assert pred.is_cuda and target.is_cuda, "Tensors must be on CUDA."
    assert pred.shape == target.shape, "predictions and targets must have the same shape"

    # Ensure contiguous tensors
    pred = pred.contiguous()
    target = target.contiguous()

    # Output scalar
    out = torch.empty((), device=pred.device, dtype=pred.dtype)

    n = pred.numel()
    grid = lambda meta: ( (n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], )

    # Launch the kernel
    kl_div_kernel[grid](pred, target, out, n, BLOCK_SIZE=128)

    # Divide by batch size (as per reduction='batchmean')
    batch_size = pred.shape[0]
    out = out / batch_size
    return out


class ModelNew(nn.Module):
    """
    Optimized KL divergence model using a fused Triton kernel.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # predictions and targets are probability distributions (already softmaxed)
        return triton_kl_div(predictions, targets)