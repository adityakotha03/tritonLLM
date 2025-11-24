import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool1d_kernel(
    x_ptr,
    y_ptr,
    batch_stride,
    feature_stride,
    seq_len,
    output_seq_len,
    kernel_size,
    stride,
    padding,
    dilation,
    batch_size,
    num_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_f = tl.program_id(1)

    offset_m = pid_b * batch_stride + pid_f * feature_stride
    x_ptr += offset_m
    y_ptr += pid_b * num_features * output_seq_len + pid_f * output_seq_len

    for i in range(0, output_seq_len, BLOCK_SIZE_L):
        block_start = i
        block_end = min(i + BLOCK_SIZE_L, output_seq_len)
        offsets_l = block_start + tl.arange(0, BLOCK_SIZE_L)
        mask_l = offsets_l < output_seq_len

        center_pos = offsets_l * stride - padding
        start_pos = center_pos
        end_pos = center_pos + dilation * (kernel_size - 1) + 1

        input_offsets = start_pos + dilation * tl.arange(0, kernel_size)[:, None]
        valid_mask = (input_offsets >= 0) & (input_offsets < seq_len)

        data = tl.load(x_ptr + input_offsets, mask=valid_mask, other=-float('inf'))
        pooled = tl.max(data, axis=0)

        output_mask = mask_l
        tl.store(y_ptr + offsets_l, pooled, mask=output_mask)


class ModelNew(nn.Module):
    """
    Optimized version of Max Pooling 1D using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input tensor must be on GPU."
        x = x.contiguous()

        batch_size, num_features, seq_len = x.shape

        output_seq_len = (seq_len + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out = torch.empty((batch_size, num_features, output_seq_len), dtype=x.dtype, device=x.device)

        def grid(META):
            return (batch_size, num_features)

        # Heuristic for block size
        BLOCK_SIZE_L = triton.next_power_of_2(output_seq_len)
        BLOCK_SIZE_L = min(max(BLOCK_SIZE_L, 64), 1024)
        BLOCK_SIZE_M = 1

        maxpool1d_kernel[grid](
            x,
            out,
            batch_stride=num_features * seq_len,
            feature_stride=seq_len,
            seq_len=seq_len,
            output_seq_len=output_seq_len,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=batch_size,
            num_features=num_features,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_L=BLOCK_SIZE_L,
        )

        return out