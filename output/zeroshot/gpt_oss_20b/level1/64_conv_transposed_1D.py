import torch
import torch.nn as nn
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
#                           Triton kernel for ConvTranspose1d                 #
# --------------------------------------------------------------------------- #
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_OUT": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE_OUT": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE_OUT": 512}, num_warps=16),
    ],
    key=["n_out"]
)
@triton.jit
def conv_transpose1d_kernel(
    out_ptr,          # output buffer
    in_ptr,           # input buffer
    weight_ptr,       # weight buffer
    bias_ptr,         # bias buffer (may be None)
    n_batch,          # batch size
    n_in,             # number of input channels
    n_out,            # number of output channels
    in_len,           # input length
    out_len,          # output length
    kernel_size,      # kernel size
    stride,           # stride
    padding,          # padding
    dilation,         # dilation
    BLOCK_SIZE_OUT: tl.constexpr,
):
    """
    Each program processes a contiguous block of output positions along
    the output length dimension. All threads within the block share the
    same batch and output channel indices (broadcasted).
    """
    # Determine the start of the current block in the output length dimension
    out_idx_start = tl.program_id(0) * BLOCK_SIZE_OUT
    offsets_out = out_idx_start + tl.arange(0, BLOCK_SIZE_OUT)
    mask = offsets_out < out_len

    # Load the output channel index for this program
    # (program_id(1) iterates over output channels)
    out_c = tl.program_id(1)
    # Broadcast batch index (program_id(2))
    batch = tl.program_id(2)

    # Preload weight for this output channel and all input channels
    # Weight shape: (out_channels, in_channels, kernel_size)
    # We store it in a contiguous buffer so we can load row-major
    weight_offsets = (
        out_c * n_in * kernel_size
        + tl.arange(0, n_in * kernel_size)
    )
    # Load entire weight slice into registers
    weight_vals = tl.load(weight_ptr + weight_offsets, mask=tl.arange(0, n_in * kernel_size) < n_in * kernel_size)

    # Compute the corresponding input indices for each output position
    # l_in = l_out * stride - padding + k * dilation
    out_positions = offsets_out

    # Prepare a register to accumulate sums for each output position
    acc = tl.zeros((BLOCK_SIZE_OUT,), dtype=tl.float32)

    # Iterate over kernel positions
    for k in range(kernel_size):
        # Compute input index for this kernel offset
        in_positions = out_positions * stride - padding + k * dilation
        # Mask to avoid out of bounds
        valid_in = in_positions >= 0
        valid_in &= in_positions < in_len
        in_mask = valid_in & mask

        # Load input slice for all input channels
        # Input shape: (batch, in_channels, in_len)
        # Compute base pointer for current batch
        base_in = in_ptr + batch * n_in * in_len
        # Compute offsets for each channel
        # We'll load a contiguous chunk of length BLOCK_SIZE_OUT for each channel
        for ic in range(n_in):
            # Compute offset for this channel and position
            in_offset = (
                ic * in_len
                + in_positions
            )
            # Load input values (float32)
            in_vals = tl.load(
                base_in + in_offset,
                mask=in_mask,
                other=0.0,
            )
            # Compute weight index for (out_c, ic, k)
            weight_idx = ic * kernel_size + k
            weight_val = weight_vals[weight_idx]
            # Accumulate
            acc += in_vals * weight_val

    # Add bias if provided
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_c)
        acc += bias_val

    # Store results
    tl.store(out_ptr + out_c * n_batch * out_len + batch * out_len + offsets_out, acc, mask=mask)


# --------------------------------------------------------------------------- #
#                           ModelNew with Triton kernel                      #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        assert groups == 1, "Only groups=1 is supported in this Triton implementation."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias_flag = bias

        # Weight shape: (out_channels, in_channels, kernel_size)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, device="cuda")
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, device="cuda"))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch_size, in_channels, in_length)
        Returns: (batch_size, out_channels, out_length)
        """
        batch_size, _, in_len = x.shape

        # Compute output length as defined by PyTorch's ConvTranspose1d
        out_len = (
            (in_len - 1) * self.stride
            - 2 * self.padding
            + self.kernel_size
            + self.output_padding
        )

        # Prepare output tensor
        out = torch.empty(
            (batch_size, self.out_channels, out_len),
            dtype=x.dtype,
            device=x.device,
        )

        # Launch Triton kernel
        grid = (
            ( (out_len + 127) // 128, ),          # 1st dim: out length
            (self.out_channels, ),                # 2nd dim: output channels
            (batch_size, ),                       # 3rd dim: batch size
        )
        conv_transpose1d_kernel[grid](
            out_ptr=out,
            in_ptr=x,
            weight_ptr=self.weight,
            bias_ptr=self.bias if self.bias_flag else None,
            n_batch=batch_size,
            n_in=self.in_channels,
            n_out=self.out_channels,
            in_len=in_len,
            out_len=out_len,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=1,   # default dilation=1
            BLOCK_SIZE_OUT=128,
        )

        return out