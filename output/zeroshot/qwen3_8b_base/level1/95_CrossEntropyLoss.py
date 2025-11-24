import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    predictions_ptr,  # Pointer to predictions tensor
    targets_ptr,      # Pointer to targets tensor
    output_ptr,       # Pointer to output tensor
    num_classes,      # Number of classes
    batch_size,       # Batch size
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask to avoid out-of-bounds
    mask = offsets < batch_size

    # Load predictions and targets
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)

    # Convert targets to one-hot encoding
    target_one_hot = tl.zeros((BLOCK_SIZE, num_classes), dtype=tl.float32)
    target_one_hot = tl.where(tl.arange(0, num_classes) == targets[:, None], 1.0, 0.0)

    # Compute log_softmax
    predictions_max = tl.max(predictions, axis=1)
    predictions_exp = tl.exp(predictions - predictions_max[:, None])
    predictions_sum = tl.sum(predictions_exp, axis=1)
    log_softmax = -tl.log(predictions_exp / predictions_sum[:, None])

    # Compute cross-entropy loss
    loss = tl.sum(tl.where(target_one_hot, log_softmax, 0.0), axis=1)

    # Store the result
    tl.store(output_ptr + offsets, loss, mask=mask)


def triton_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Custom Triton kernel for cross-entropy loss computation.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    # Compute output shape
    output_shape = (predictions.size(0),)

    # Prepare output tensor
    output = torch.empty(output_shape, device=predictions.device, dtype=predictions.dtype)

    # Determine block size and grid
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    num_blocks = (predictions.size(0) + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    cross_entropy_kernel[ num_blocks ](predictions, targets, output, predictions.size(1), predictions.size(0), BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_cross_entropy(predictions, targets)