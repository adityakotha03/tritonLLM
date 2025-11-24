import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool3d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    input_depth,
    input_height,
    input_width,
    output_depth,
    output_height,
    output_width,
    kernel_size_d,
    kernel_size_h,
    kernel_size_w,
    stride_d,
    stride_h,
    stride_w,
    padding_d,
    padding_h,
    padding_w,
    dilation_d,
    dilation_h,
    dilation_w,
    BD: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
):
    # Program IDs correspond to output spatial locations
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_od = tl.program_id(2)
    pid_oh = tl.program_id(3)
    pid_ow = tl.program_id(4)

    # Compute starting input indices for this output location
    start_d = pid_od * stride_d - padding_d
    start_h = pid_oh * stride_h - padding_h
    start_w = pid_ow * stride_w - padding_w

    # Initialize max value
    max_val = -float('inf')

    # Iterate over kernel
    for kd in range(0, kernel_size_d):
        for kh in range(0, kernel_size_h):
            for kw in range(0, kernel_size_w):
                # Compute input position with dilation
                d = start_d + kd * dilation_d
                h = start_h + kh * dilation_h
                w = start_w + kw * dilation_w

                # Check bounds
                d_valid = (d >= 0) and (d < input_depth)
                h_valid = (h >= 0) and (h < input_height)
                w_valid = (w >= 0) and (w < input_width)

                # Compute input offset if valid
                in_offset = (
                    pid_b * channels * input_depth * input_height * input_width +
                    pid_c * input_depth * input_height * input_width +
                    d * input_height * input_width +
                    h * input_width +
                    w
                )

                # Load input value if in bounds
                val = tl.load(input_ptr + in_offset, mask=d_valid and h_valid and w_valid, other=-float('inf'))

                # Update max
                max_val = tl.maximum(max_val, val)

    # Store output
    out_offset = (
        pid_b * channels * output_depth * output_height * output_width +
        pid_c * output_depth * output_height * output_width +
        pid_od * output_height * output_width +
        pid_oh * output_width +
        pid_ow
    )
    tl.store(output_ptr + out_offset, max_val)


class ModelNew(nn.Module):
    """
    Optimized version of MaxPool3d using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

        # Expand kernel, stride, padding, dilation to 3D
        if isinstance(kernel_size, int):
            self.kernel_size_d = self.kernel_size_h = self.kernel_size_w = kernel_size
        else:
            self.kernel_size_d, self.kernel_size_h, self.kernel_size_w = kernel_size

        if isinstance(self.stride, int):
            self.stride_d = self.stride_h = self.stride_w = self.stride
        else:
            self.stride_d, self.stride_h, self.stride_w = self.stride

        if isinstance(padding, int):
            self.padding_d = self.padding_h = self.padding_w = padding
        else:
            self.padding_d, self.padding_h, self.padding_w = padding

        if isinstance(dilation, int):
            self.dilation_d = self.dilation_h = self.dilation_w = dilation
        else:
            self.dilation_d, self.dilation_h, self.dilation_w = dilation

        # Triton block dimensions
        self.BD = 1
        self.BH = 1
        self.BW = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input tensor must be on GPU."

        batch_size, channels, dim1, dim2, dim3 = x.shape
        input_depth, input_height, input_width = dim1, dim2, dim3

        # Compute output dimensions
        def _compute_output_length(input_length, kernel_size, stride, padding, dilation, ceil_mode):
            length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
            if ceil_mode:
                if (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) % stride != 0:
                    length += 1
            return length

        output_depth = _compute_output_length(input_depth, self.kernel_size_d, self.stride_d, self.padding_d, self.dilation_d, self.ceil_mode)
        output_height = _compute_output_length(input_height, self.kernel_size_h, self.stride_h, self.padding_h, self.dilation_h, self.ceil_mode)
        output_width = _compute_output_length(input_width, self.kernel_size_w, self.stride_w, self.padding_w, self.dilation_w, self.ceil_mode)

        # Allocate output
        output = torch.empty((batch_size, channels, output_depth, output_height, output_width), device=x.device, dtype=x.dtype)

        # Grid configuration: one block per output spatial location
        grid = (batch_size, channels, output_depth, output_height, output_width)

        # Launch kernel
        maxpool3d_kernel[grid](
            x,
            output,
            batch_size,
            channels,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width,
            self.kernel_size_d,
            self.kernel_size_h,
            self.kernel_size_w,
            self.stride_d,
            self.stride_h,
            self.stride_w,
            self.padding_d,
            self.padding_h,
            self.padding_w,
            self.dilation_d,
            self.dilation_h,
            self.dilation_w,
            BD=self.BD,
            BH=self.BH,
            BW=self.BW,
        )

        return output