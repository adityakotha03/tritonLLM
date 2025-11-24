import torch
import torch.nn as nn
import triton
import triton.language as tl

# -------------------------------------------------------------
# Triton kernel for 3‑D transposed convolution (deconvolution)
# -------------------------------------------------------------
@triton.jit
def conv_transpose3d_kernel(
    x_ptr,          # input tensor (B, Cin, D_in, H_in, W_in)
    w_ptr,          # weight tensor (Cin, Cout, KD, KH, KW)
    bias_ptr,       # bias tensor (Cout) or dummy
    out_ptr,        # output tensor (B, Cout, D_out, H_out, W_out)
    batch_size, cin, cout,
    depth_in, height_in, width_in,
    depth_out, height_out, width_out,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    groups, bias_flag,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each thread processes one element of the output tensor.
    """
    # global linear index of the element this thread will write
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # mask to avoid out‑of‑bounds
    total_elems = batch_size * cout * depth_out * height_out * width_out
    mask = idx < total_elems
    if tl.all(~mask):
        return

    # unpack the linear index into 5‑D coordinates
    idx = tl.where(mask, idx, 0)
    out_batch = idx // (cout * depth_out * height_out * width_out)
    rem = idx % (cout * depth_out * height_out * width_out)
    out_c = rem // (depth_out * height_out * width_out)
    rem2 = rem % (depth_out * height_out * width_out)
    out_d = rem2 // (height_out * width_out)
    rem3 = rem2 % (height_out * width_out)
    out_h = rem3 // width_out
    out_w = rem3 % width_out

    # broadcast bias if present
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    if bias_flag:
        bias_val = tl.load(bias_ptr + out_c, mask=mask, other=0.0)
        acc += bias_val

    # compute the group of the current output channel
    out_c_per_group = cout // groups
    group = out_c // out_c_per_group
    cin_per_group = cin // groups
    cin_start = group * cin_per_group
    cin_end   = cin_start + cin_per_group

    # main accumulation loop
    for kd in range(KD):
        for kh in range(KH):
            for kw in range(KW):
                # position of the corresponding input element
                in_d = out_d * stride_d - pad_d + kd
                in_h = out_h * stride_h - pad_h + kh
                in_w = out_w * stride_w - pad_w + kw

                # mask for input bounds
                valid = (in_d >= 0) & (in_d < depth_in) & \
                        (in_h >= 0) & (in_h < height_in) & \
                        (in_w >= 0) & (in_w < width_in)

                if tl.any(valid):
                    for c in range(cin_start, cin_end):
                        # weight offset
                        w_offset = ((c * cout + out_c) * KD * KH * KW +
                                    kd * KH * KW + kh * KW + kw)
                        w_val = tl.load(w_ptr + w_offset)

                        # input offset
                        x_offset = (((out_batch * cin + c) * depth_in + in_d) *
                                    height_in + in_h) * width_in + in_w
                        x_val = tl.load(x_ptr + x_offset, mask=valid, other=0.0)

                        acc += w_val * x_val

    # store result
    tl.store(out_ptr + idx, acc, mask=mask)

# -------------------------------------------------------------
# Helper wrapper that launches the kernel
# -------------------------------------------------------------
def triton_conv_transpose3d(x, weight, bias,
                            stride, padding, output_padding, groups):
    """
    Wrapper that computes a 3‑D transposed convolution using a Triton kernel.
    """
    # parameters
    batch, cin, d_in, h_in, w_in = x.shape
    _, cout, kd, kh, kw = weight.shape
    sd, sh, sw = stride
    pd, ph, pw = padding
    od, oh, ow = output_padding

    # compute output dimensions
    d_out = (d_in - 1) * sd - 2 * pd + kd + od
    h_out = (h_in - 1) * sh - 2 * ph + kh + oh
    w_out = (w_in - 1) * sw - 2 * pw + kw + ow

    # allocate output tensor
    out = torch.empty((batch, cout, d_out, h_out, w_out),
                      dtype=x.dtype, device=x.device)

    # prepare bias pointer
    if bias is not None:
        bias_ptr = bias.data_ptr()
        bias_flag = 1
    else:
        # dummy bias tensor
        bias_dummy = torch.empty((1,), device=x.device)
        bias_ptr = bias_dummy.data_ptr()
        bias_flag = 0

    # compute total number of output elements
    total = batch * cout * d_out * h_out * w_out

    # autotune parameters
    BLOCK_SIZE = 256

    # launch grid
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)

    conv_transpose3d_kernel[grid](
        x, weight, bias_ptr, out,
        batch, cin, cout,
        d_in, h_in, w_in,
        d_out, h_out, w_out,
        sd, sh, sw,
        pd, ph, pw,
        groups, bias_flag,
        KD=kd, KH=kh, KW=kw,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

# -------------------------------------------------------------
# Optimised model using the custom Triton kernel
# -------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution using a custom Triton kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias

        weight_shape = (in_channels, out_channels, *kernel_size)
        self.weight = nn.Parameter(torch.randn(weight_shape, device="cuda"))
        if bias:
            self.bias_param = nn.Parameter(torch.randn(out_channels, device="cuda"))
        else:
            self.bias_param = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x,
            self.weight,
            self.bias_param,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
        )