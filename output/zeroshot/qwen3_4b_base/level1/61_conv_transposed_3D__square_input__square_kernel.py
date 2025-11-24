import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    input_ptr,        # pointer to input tensor (batch, in_channels, d, h, w)
    weight_ptr,       # pointer to weight tensor (out_channels, in_channels, k, k, k)
    bias_ptr,         # pointer to bias tensor (out_channels) or None
    output_ptr,       # pointer to output tensor (batch, out_channels, d_out, h_out, w_out)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    output_padding: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute output dimensions
    d_out = (input_ptr.shape[2] - 1) * stride + padding + output_padding
    h_out = (input_ptr.shape[3] - 1) * stride + padding + output_padding
    w_out = (input_ptr.shape[4] - 1) * stride + padding + output_padding

    # Get current block and thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)

    # Compute the output position for this block
    d_out_idx = tl.program_id(2)
    h_out_idx = tl.program_id(3)
    w_out_idx = tl.program_id(4)

    # Compute the actual output indices
    d_out_offset = d_out_idx * BLOCK_SIZE
    h_out_offset = h_out_idx * BLOCK_SIZE
    w_out_offset = w_out_idx * BLOCK_SIZE

    # Compute the range of input indices this block will access
    offsets_d = tl.arange(0, BLOCK_SIZE)
    offsets_h = tl.arange(0, BLOCK_SIZE)
    offsets_w = tl.arange(0, BLOCK_SIZE)

    # Compute input indices using deconvolution formula
    # For transposed conv: output[i] = sum_k input[i - k*stride + padding] * weight[k]
    # We loop over all input positions that contribute to output position (d_out_idx, h_out_idx, w_out_idx)

    # For each output position in this block, compute the input positions
    # We use a 3D kernel with stride and padding
    # We assume input is (B, C_in, D, H, W) and output is (B, C_out, D_out, H_out, W_out)

    # For a given output position, we compute the input position via:
    # d_in = d_out - (d_out_idx * stride) - padding
    # h_in = h_out - (h_out_idx * stride) - padding
    # w_in = w_out - (w_out_idx * stride) - padding

    # But we need to handle the reverse mapping: for each output location, find input locations
    # We use a loop over the kernel size to compute the weighted sum

    # We will process one output location at a time, and for each output location,
    # we compute the input indices and perform the convolution.

    # However, due to complexity and memory constraints, we use a tiling approach with block size
    # Instead of full 3D deconvolution, we use a 3D kernel loop with shared memory and block-wise computation.

    # This implementation is simplified to support a single block of output and assumes
    # that the kernel is applied with a fixed stride and padding.

    # We instead use a more efficient fused kernel that operates on a 3D block of input
    # and computes the output in a coalesced manner.

    # We now define a new approach: process each output location (d_out, h_out, w_out)
    # and compute the input location (d_in, h_in, w_in) via deconvolution.

    # Since we are limited by register count and shared memory, we will use a 3D loop
    # over kernel size and use masking for boundaries.

    # For each output position (d_out_idx, h_out_idx, w_out_idx), we compute input positions
    # and perform the convolution with the kernel.

    # We use a different strategy: process one output channel at a time, and for each
    # output channel, compute the weighted sum over input channels and spatial positions.

    # We assume that the kernel is applied in a way that input positions are mapped via:
    # d_in = d_out - (d_out_idx * stride) - padding
    # h_in = h_out - (h_out_idx * stride) - padding
    # w_in = w_out - (w_out_idx * stride) - padding

    # But we need to ensure that we stay within bounds.

    # We define the input indices for this output location
    d_in = tl.arange(0, kernel_size)
    h_in = tl.arange(0, kernel_size)
    w_in = tl.arange(0, kernel_size)

    # Compute the actual input position for this kernel element
    # We need to compute the input indices for each kernel element
    # For deconvolution, we have: output[d_out, h_out, w_out] = sum_{k} input[d_in, h_in, w_in] * weight[k]

    # We will compute the output for one output position (d_out_idx, h_out_idx, w_out_idx)
    # and one output channel (channel_idx)

    # Compute the input indices for the current output position
    d_in_offset = d_out_idx * stride - padding
    h_in_offset = h_out_idx * stride - padding
    w_in_offset = w_out_idx * stride - padding

    # Compute the input indices for the kernel
    d_in_idx = d_in + d_in_offset
    h_in_idx = h_in + h_in_offset
    w_in_idx = w_in + w_in_offset

    # Create masks to ensure we stay within input bounds
    mask_d = (d_in_idx >= 0) & (d_in_idx < input_ptr.shape[2])
    mask_h = (h_in_idx >= 0) & (h_in_idx < input_ptr.shape[3])
    mask_w = (w_in_idx >= 0) & (w_in_idx < input_ptr.shape[4])

    # Create a 3D mask for all kernel positions
    mask = mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]

    # Load input values
    input_vals = tl.load(
        input_ptr + batch_idx * input_ptr.strides[0] +
        channel_idx * input_ptr.strides[1] +
        d_in_idx * input_ptr.strides[2] +
        h_in_idx * input_ptr.strides[3] +
        w_in_idx * input_ptr.strides[4],
        mask=mask,
        other=0.0
    )

    # Load weights
    weight_vals = tl.load(
        weight_ptr + channel_idx * weight_ptr.strides[1] +
        d_in_idx * weight_ptr.strides[2] +
        h_in_idx * weight_ptr.strides[3] +
        w_in_idx * weight_ptr.strides[4],
        mask=mask,
        other=0.0
    )

    # Compute the dot product over the kernel
    output_val = tl.sum(input_vals * weight_vals, axis=(0, 1, 2))

    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + channel_idx * bias_ptr.strides[0], mask=(channel_idx < out_channels), other=0.0)
        output_val = output_val + bias_val

    # Store output
    output_offset = batch_idx * output_ptr.strides[0] + \
                    channel_idx * output_ptr.strides[1] + \
                    d_out_idx * output_ptr.strides[2] + \
                    h_out_idx * output_ptr.strides[3] + \
                    w_out_idx * output_ptr.strides[4]

    tl.store(output_ptr + output_offset, output_val, mask=(d_out_idx < d_out) & (h_out_idx < h_out) & (w_out_idx < w_out))


def triton_conv_transpose3d(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    bias_tensor: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    groups: int = 1,
    kernel_size: int = 3,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Performs a transposed 3D convolution using a custom Triton kernel.
    """
    assert input_tensor.is_cuda and weight_tensor.is_cuda, "Inputs must be on CUDA."
    input_tensor = input_tensor.contiguous()
    weight_tensor = weight_tensor.contiguous()

    # Prepare output tensor
    batch_size, in_channels, d_in, h_in, w_in = input_tensor.shape
    out_channels, _, k, k, k = weight_tensor.shape
    d_out = (d_in - 1) * stride + padding + output_padding
    h_out = (h_in - 1) * stride + padding + output_padding
    w_out = (w_in - 1) * stride + padding + output_padding

    output_shape = (batch_size, out_channels, d_out, h_out, w_out)
    output_tensor = torch.empty(output_shape, device=input_tensor.device, dtype=input_tensor.dtype)

    # Define grid
    grid = lambda meta: (
        (batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (in_channels + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (d_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (h_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (w_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    # Launch kernel
    conv_transpose3d_kernel[
        grid,
        (block_size, block_size, block_size, block_size, block_size)
    ](
        input_tensor.data_ptr(),
        weight_tensor.data_ptr(),
        bias_tensor.data_ptr() if bias_tensor is not None else None,
        output_tensor.data_ptr(),
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        BLOCK_SIZE=block_size,
    )

    return output_tensor


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Define weight and bias tensors (will be initialized during forward)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using the custom Triton kernel.
        """
        return triton_conv_transpose3d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            kernel_size=self.kernel_size
        )