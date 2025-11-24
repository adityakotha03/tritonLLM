import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_channels, d_stride, h_stride, w_stride,
    num_elements,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(0)
    offset_dhw = pid
    d = (offset_dhw // (h_stride // d_stride)) // (w_stride // h_stride)
    h = (offset_dhw - d * (h_stride // d_stride) * (w_stride // h_stride)) // (w_stride // h_stride)
    w = offset_dhw % (w_stride // h_stride)
    base_offset = d * d_stride + h * h_stride + w * w_stride

    mask = offset_dhw < num_elements
    if not mask:
        return

    channel_offsets = tl.arange(0, BLOCK_SIZE_C)
    input_offsets = base_offset + channel_offsets
    mask_c = channel_offsets < n_channels
    x = tl.load(input_ptr + input_offsets, mask=mask_c, other=-float('inf'))
    x_max = tl.max(x, 0)
    x_shifted = x - x_max
    x_exp = tl.exp(x_shifted)
    x_sum = tl.sum(x_exp, 0)
    softmax_output = x_exp / x_sum
    tl.store(output_ptr + input_offsets, softmax_output, mask=mask_c)


def triton_softmax(x, dim):
    assert dim == 1, "Only channel dimension (dim=1) supported in this kernel"
    x = x.contiguous()
    batch, channels, depth, height, width = x.shape
    output = torch.empty_like(x)
    n_elements = batch * depth * height * width
    BLOCK_SIZE_C = triton.next_power_of_2(channels)

    def grid(meta):
        return (n_elements,)

    softmax_kernel[grid](
        x, output,
        channels, depth * height * width * channels, height * width * channels, width * channels,
        n_elements,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    return output


class ModelNew(nn.Module):
    """
    Optimized model with Triton-based Softmax.
    The 3D convolution and max pooling operations are kept as native PyTorch ops
    due to their efficient vendor-tuned implementations (cuDNN), while Softmax
    is replaced with a custom Triton kernel for better fusion and memory access control.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool1 = nn.MaxPool3d(pool_kernel_size)
        self.pool2 = nn.MaxPool3d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = triton_softmax(x, dim=1)
        x = self.pool1(x)
        x = self.pool2(x)
        return x