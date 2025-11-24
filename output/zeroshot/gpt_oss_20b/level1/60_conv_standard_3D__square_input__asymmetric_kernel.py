import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------------------------------------------------------
# Triton kernel for a 3‑D convolution (stride = 1, padding = 0, dilation = 1)
# ------------------------------------------------------------------
@triton.jit
def conv3d_kernel(
    inp_ptr: tl.tensor,          # Input tensor (B, C_in, H_in, W_in, D_in)
    weight_ptr: tl.tensor,       # Weight tensor (C_out, C_in, kH, kW, kD)
    bias_ptr: tl.tensor,         # Bias tensor (C_out,)
    out_ptr: tl.tensor,          # Output tensor (B, C_out, H_out, W_out, D_out)

    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H_in: tl.constexpr, W_in: tl.constexpr, D_in: tl.constexpr,
    H_out: tl.constexpr, W_out: tl.constexpr, D_out: tl.constexpr,
    kH: tl.constexpr, kW: tl.constexpr, kD: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes a 3‑D convolution with the following restrictions:

        * stride   = 1
        * padding  = 0
        * dilation = 1
        * groups   = 1

    The kernel is written so that each program processes `BLOCK_SIZE` output
    elements.  All arithmetic is performed in float32 while the tensors
    themselves are stored as float16 in order to exploit the Tensor Cores
    of the A100.
    """

    num_out_elements = B * C_out * H_out * W_out * D_out
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_out_elements
    if not mask.any():
        return

    # ------------------------------------------------------------------
    # Convert the linear offset into 5‑D indices
    # ------------------------------------------------------------------
    out_idx = offsets
    b = out_idx // (C_out * H_out * W_out * D_out)
    tmp = out_idx % (C_out * H_out * W_out * D_out)
    co = tmp // (H_out * W_out * D_out)
    tmp = tmp % (H_out * W_out * D_out)
    h = tmp // (W_out * D_out)
    tmp = tmp % (W_out * D_out)
    w = tmp // D_out
    d = tmp % D_out

    # ------------------------------------------------------------------
    # Compute the convolution sum
    # ------------------------------------------------------------------
    acc = tl.zeros([1], dtype=tl.float32)

    for ic in range(C_in):
        for kh in range(kH):
            for kw in range(kW):
                for kd in range(kD):
                    h_in = h + kh          # stride = 1, no padding
                    w_in = w + kw
                    d_in = d + kd

                    # Index of the input element
                    inp_off = (
                        b * (C_in * H_in * W_in * D_in)
                        + ic * (H_in * W_in * D_in)
                        + h_in * (W_in * D_in)
                        + w_in * D_in
                        + d_in
                    )
                    inp_val = tl.load(inp_ptr + inp_off, mask=mask, other=0.0)

                    # Index of the weight element
                    weight_off = (
                        co * (C_in * kH * kW * kD)
                        + ic * (kH * kW * kD)
                        + kh * (kW * kD)
                        + kw * kD
                        + kd
                    )
                    weight_val = tl.load(weight_ptr + weight_off, mask=mask, other=0.0)

                    acc += inp_val.to(tl.float32) * weight_val.to(tl.float32)

    # ------------------------------------------------------------------
    # Add bias if requested
    # ------------------------------------------------------------------
    if has_bias:
        bias_val = tl.load(bias_ptr + co, mask=mask, other=0.0)
        acc += bias_val.to(tl.float32)

    # Store result (back to float16)
    out_ptr[offsets] = acc.to(tl.float16)


# ------------------------------------------------------------------
# Wrapper that builds the grid and launches the kernel
# ------------------------------------------------------------------
def conv3d_triton(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Performs a 3‑D convolution using the custom Triton kernel.

    The function supports only:
        * stride = 1
        * padding = 0
        * dilation = 1
        * groups = 1

    Parameters
    ----------
    inp   : torch.Tensor
        Input tensor of shape (B, C_in, H_in, W_in, D_in) in float16 or float32.
    weight: torch.Tensor
        Weight tensor of shape (C_out, C_in, kH, kW, kD).
    bias  : torch.Tensor | None
        Optional bias tensor of shape (C_out,).

    Returns
    -------
    torch.Tensor
        The convolution result in the same dtype as the input.
    """
    if not inp.is_cuda:
        raise ValueError("Input tensor must be on CUDA.")
    if not weight.is_cuda:
        raise ValueError("Weight tensor must be on CUDA.")
    if bias is not None and not bias.is_cuda:
        raise ValueError("Bias tensor must be on CUDA.")

    # Use float16 for all data to leverage Tensor Cores
    inp_fp16 = inp.to(torch.float16)
    weight_fp16 = weight.to(torch.float16)
    bias_fp16 = bias.to(torch.float16) if bias is not None else torch.empty(
        0, dtype=torch.float16, device=inp.device
    )

    B, C_in, H_in, W_in, D_in = inp_fp16.shape
    C_out, _, kH, kW, kD = weight_fp16.shape

    # Output dimensions (stride=1, padding=0, dilation=1)
    H_out = H_in - kH + 1
    W_out = W_in - kW + 1
    D_out = D_in - kD + 1

    out_fp16 = torch.empty(
        (B, C_out, H_out, W_out, D_out),
        dtype=torch.float16,
        device=inp.device,
    )

    num_out_elements = B * C_out * H_out * W_out * D_out
    BLOCK_SIZE = 256  # Tunable; power‑of‑two works best

    grid = lambda meta: ((num_out_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    conv3d_kernel[grid](
        inp_fp16,
        weight_fp16,
        bias_fp16,
        out_fp16,
        B, C_in, C_out,
        H_in, W_in, D_in,
        H_out, W_out, D_out,
        kH, kW, kD,
        has_bias= (bias is not None),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Cast result back to the input dtype for consistency
    return out_fp16.to(inp.dtype)


# ------------------------------------------------------------------
# PyTorch model that uses the custom Triton convolution
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    3‑D convolution implemented with a custom Triton kernel.

    Only stride=1, padding=0, dilation=1 and groups=1 are supported.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        if stride != 1 or padding != 0 or dilation != 1:
            raise NotImplementedError(
                "Only stride=1, padding=0, dilation=1 are supported."
            )
        if groups != 1:
            raise NotImplementedError("Only groups=1 is supported.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias

        weight_shape = (out_channels, in_channels, *kernel_size)
        self.weight = nn.Parameter(torch.randn(weight_shape, device="cuda"))
        if bias:
            self.bias_param = nn.Parameter(torch.randn(out_channels, device="cuda"))
        else:
            self.bias_param = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv3d_triton(x, self.weight, self.bias_param)