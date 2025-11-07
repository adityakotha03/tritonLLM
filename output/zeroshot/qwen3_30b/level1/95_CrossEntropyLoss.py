import torch
import torch.nn as nn
import triton
import triton.language as tl


BLOCK_SIZE = 256
TILE_SIZE = 256


@triton.jit
def cross_entropy_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    num_classes: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    # Get the block ID
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE
    # Number of elements in this block
    num_elements = min(BLOCK_SIZE, batch_size - block_start)

    # Initialize max_val and sum_exp for each row in the block
    # Use a very small number for -inf
    max_val = tl.full((BLOCK_SIZE,), -1e10, dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Number of tiles
    num_tiles = (num_classes + TILE_SIZE - 1) // TILE_SIZE

    # Loop over tiles
    for tile in range(num_tiles):
        tile_start = tile * TILE_SIZE
        tile_end = min(tile_start + TILE_SIZE, num_classes)

        # Create offsets for the current tile
        # (block_start + i) * num_classes + (tile_start + j)
        row_offsets = (block_start + tl.arange(0, BLOCK_SIZE)[:, None]) * num_classes
        col_offsets = tl.arange(0, TILE_SIZE)[None, :] + tile_start
        global_offsets = row_offsets + col_offsets

        # Create mask for valid elements
        # Valid row: block_start + i < batch_size
        row_mask = (block_start + tl.arange(0, BLOCK_SIZE)) < batch_size
        # Valid col: tile_start + j < num_classes
        col_mask = (tl.arange(0, TILE_SIZE)[None, :] + tile_start) < num_classes
        mask = row_mask[:, None] & col_mask

        # Load the tile
        tile_data = tl.load(
            predictions_ptr + global_offsets,
            mask=mask,
            other=-1e10
        )

        # Compute the max over the class dimension (axis=1)
        tile_max = tl.max(tile_data, axis=1)
        max_val = tl.max(max_val, tile_max)

        # Compute the exponential of the shifted values
        shifted = tile_data - max_val[:, None]
        exp_shifted = tl.exp(shifted)
        # Sum along the class dimension
        tile_sum_exp = tl.sum(exp_shifted, axis=1)
        sum_exp = sum_exp + tile_sum_exp

    # Load the target classes for this block
    target_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    target_classes = tl.load(
        targets_ptr + target_offsets,
        mask=(target_offsets < batch_size),
        other=0
    )

    # Get the prediction at the target class for each row
    target_pred_offsets = (block_start + tl.arange(0, BLOCK_SIZE)) * num_classes + target_classes
    target_pred = tl.load(
        predictions_ptr + target_pred_offsets,
        mask=(target_offsets < batch_size),
        other=0.0
    )

    # Compute the loss for each row
    log_sum_exp = tl.log(sum_exp)
    loss_i = -target_pred + max_val + log_sum_exp

    # Apply mask for invalid rows
    row_mask = (block_start + tl.arange(0, BLOCK_SIZE)) < batch_size
    loss_i = loss_i * row_mask

    # Sum the loss for this block
    total_loss = tl.sum(loss_i)

    # Store the partial sum for this block
    output_ptr = output_ptr + block_id
    tl.store(output_ptr, total_loss, mask=block_id < (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE)


def triton_cross_entropy(predictions, targets):
    # Ensure contiguous
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    batch_size = predictions.size(0)
    num_classes = predictions.size(1)
    num_blocks = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Output tensor: one partial sum per block
    out = torch.empty(num_blocks, device=predictions.device, dtype=torch.float32)

    # Grid: one block per block
    grid = lambda meta: (meta["num_blocks"],)

    # Launch the kernel
    cross_entropy_kernel[grid](
        predictions,
        targets,
        out,
        batch_size,
        num_classes,
        BLOCK_SIZE=BLOCK_SIZE,
        TILE_SIZE=TILE_SIZE
    )

    # Reduce across blocks and compute mean
    total_loss = out.sum()
    mean_loss = total_loss / batch_size
    return mean_loss


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return triton_cross_entropy(predictions, targets)