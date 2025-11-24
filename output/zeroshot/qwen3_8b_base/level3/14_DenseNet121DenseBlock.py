import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C_in, H, W)
    kernel_size,  # (kH, kW)
    stride,  # (sH, sW)
    padding,  # (pH, pW)
    dilation,  # (dH, dW)
    out_channels,  # Number of output channels
    in_channels,  # Number of input channels
    height,  # Height of input
    width,  # Width of input
    BLOCK_SIZE: tl.constexpr,
):
    # Get the thread index
    pid = tl.program_id(0)
    # Get the block index along the channel dimension
    block_idx = pid // (out_channels // 16)
    # Get the output channel index
    out_ch = (pid % (out_channels // 16)) * 16
    # Compute the output position
    out_h = (block_idx // (width // BLOCK_SIZE)) * BLOCK_SIZE
    out_w = (block_idx % (width // BLOCK_SIZE)) * BLOCK_SIZE
    # Compute the input position
    in_h = out_h * stride[0] - padding[0]
    in_w = out_w * stride[1] - padding[1]
    # Compute the range of input indices
    h_offsets = tl.arange(0, kernel_size[0])
    w_offsets = tl.arange(0, kernel_size[1])
    # Compute the input indices
    in_h_offsets = in_h + h_offsets
    in_w_offsets = in_w + w_offsets
    # Compute the output indices
    out_h_offsets = out_h + h_offsets
    out_w_offsets = out_w + w_offsets
    # Initialize the output
    out = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    # Load weights
    weight = tl.load(weight_ptr + out_ch * in_channels * kernel_size[0] * kernel_size[1] + h_offsets * kernel_size[1] + w_offsets, mask=tl.arange(0, kernel_size[0]) < kernel_size[0], other=0.0)
    # Compute the convolution
    for i in range(kernel_size[0]):
        for j in range(kernel_size[1]):
            in_h = in_h_offsets[i]
            in_w = in_w_offsets[j]
            if in_h < 0 or in_h >= height or in_w < 0 or in_w >= width:
                continue
            input_val = tl.load(input_ptr + in_h * width + in_w, mask=tl.arange(0, in_channels) < in_channels, other=0.0)
            out += input_val * weight[i * kernel_size[1] + j]
    # Store the output
    tl.store(output_ptr + out_h_offsets * width + out_w_offsets, out, mask=tl.arange(0, BLOCK_SIZE) < BLOCK_SIZE)


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: tuple, padding: tuple, dilation: tuple):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Prepare output tensor
    out_channels = weight.shape[0]
    out_h = (input.shape[2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    out_w = (input.shape[3] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    out = torch.empty((input.shape[0], out_channels, out_h, out_w), dtype=input.dtype, device=input.device)

    # Number of elements in the tensor
    n_elements = out.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, out, input.shape, (3, 3), stride, padding, dilation, out_channels, input.shape[1], input.shape[2], input.shape[3], BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        :param num_layers: The number of layers in the dense block
        :param num_input_features: The number of input feature maps
        :param growth_rate: The growth rate for the dense block (new features added per layer)
        """
        super(ModelNew, self).__init__()
        layers = []
        for i in range(num_layers):
            in_channels = num_input_features + i * growth_rate
            out_channels = growth_rate
            # Create a single layer with BatchNorm, ReLU, Conv2D, and Dropout.
            # Replace Conv2D with custom Triton kernel
            layers.append(self._make_layer(in_channels, out_channels))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_channels: int, out_channels: int):
        """
        Creates a single layer with BatchNorm, ReLU, Conv2D, and Dropout.
        """
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Replace Conv2D with custom Triton kernel
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )
    
    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Concatenated output tensor with shape (batch_size, num_output_features, height, width)
        """
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)  # Concatenate along channel axis
        return x