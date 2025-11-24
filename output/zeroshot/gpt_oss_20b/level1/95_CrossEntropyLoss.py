import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------- Triton kernel: compute per‑sample cross entropy loss ----------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["N", "C"],
)
@triton.jit
def cross_entropy_kernel(
    preds_ptr,          # float32[batch, classes]
    targets_ptr,        # int32[batch]
    out_ptr,            # float32[batch]
    batch: tl.constexpr,
    classes: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute CrossEntropyLoss for each sample:
        loss = -log( exp(logit_t) / sum(exp(logits)) )
    """

    # Thread id within the block
    thread_idx = tl.program_id(0)

    # Number of samples processed by this program
    block_start = thread_idx * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < batch

    # Load target indices
    tgt = tl.load(targets_ptr + offsets, mask=mask, other=0)

    # Load logits for the whole row (broadcasted)
    # We will load one row per thread that has a valid offset
    # For memory coalescing, load contiguous rows
    # preds_ptr is contiguous [batch, classes]
    preds = tl.load(
        preds_ptr + offsets[:, None] * classes + tl.arange(0, classes)[None, :],
        mask=mask[:, None] & tl.arange(0, classes)[None, :] < classes,
        other=0.0,
    )

    # Find max for numerical stability
    max_logits = tl.max(preds, axis=1, keepdims=True)

    # Subtract max
    logits_minus = preds - max_logits

    # Exponentiate
    exp_logits = tl.exp(logits_minus)

    # Sum of exp
    sum_exp = tl.sum(exp_logits, axis=1, keepdims=True)

    # LogSumExp
    lse = tl.log(sum_exp) + max_logits.squeeze(-1)

    # Gather the logit corresponding to the target class
    tgt_idx = tgt[:, None]
    logit_t = tl.load(
        preds_ptr + offsets[:, None] * classes + tgt_idx,
        mask=mask[:, None],
        other=0.0,
    ).squeeze(-1)

    # Loss per sample
    loss = lse - logit_t

    # Store
    tl.store(out_ptr + offsets, loss, mask=mask)


# ---------- Python wrapper ----------
def triton_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    predictions: [B, C] float32
    targets:    [B] int64
    Returns: loss per sample [B] float32
    """
    assert predictions.is_cuda and targets.is_cuda
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    batch, classes = predictions.shape
    out = torch.empty(batch, device=predictions.device, dtype=torch.float32)

    grid = lambda meta: ((batch + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    cross_entropy_kernel[grid](
        predictions, targets, out, batch, classes, BLOCK_SIZE=256
    )
    return out


# ---------- Optimized model ----------
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        # Forward pass using custom Triton kernel
        return triton_cross_entropy(predictions, targets)