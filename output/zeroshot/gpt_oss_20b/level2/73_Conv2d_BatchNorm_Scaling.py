import torch
import torch.nn as nn
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
#                     Triton kernel for fused Conv2d + BN + Scale             #
# --------------------------------------------------------------------------- #
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_X': 64, 'BLOCK_SIZE_Y': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_X': 128, 'BLOCK_SIZE_Y': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_X': 256, 'BLOCK_SIZE_Y': 16}, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def conv2d_fused_kernel(
    # pointers to tensors
    input_ptr,          # (B, C, H, W)
    weight_ptr,         # (O, C, KH, KW)
    bias_ptr,           # (O,)
    running_mean_ptr,   # (O,)
    running_var_ptr,    # (O,)
    gamma_ptr,          # (O,)
    beta_ptr,           # (O,)
    # output tensor
    output_ptr,         # (B, O, H_out, W_out)
    # sizes
    B, C, H, W, O, KH, KW, H_out, W_out, stride, padding, eps,
    scaling_factor,
    # metadata
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    """
    Each program instance processes a block of output elements.
    Program 0 handles a tile of size BLOCK_SIZE_X * BLOCK_SIZE_Y output channels.
    """

    # program ids
    prog_x = tl.program_id(0)  # output channel block
    prog_y = tl.program_id(1)  # spatial tile block

    # offsets for output channels
    oc_start = prog_x * BLOCK_SIZE_X
    oc_end   = oc_start + BLOCK_SIZE_X

    # offsets for spatial positions
    # we tile height and width together in a 1D stride
    spatial_start = prog_y * BLOCK_SIZE_Y
    spatial_end   = spatial_start + BLOCK_SIZE_Y

    # loop over batch
    for b in range(B):
        # iterate over output channels in the block
        for oc in range(oc_start, min(oc_end, O)):
            # iterate over spatial positions in the block
            for pos in range(spatial_start, min(spatial_end, H_out * W_out)):
                h_out = pos // W_out
                w_out = pos % W_out

                # compute input region top-left corner
                h_in = h_out * stride - padding
                w_in = w_out * stride - padding

                # accumulate conv result
                acc = tl.zeros([1], dtype=tl.float32)

                # iterate over input channels
                for ic in range(C):
                    # iterate over kernel height and width
                    for kh in range(KH):
                        for kw in range(KW):
                            h_cur = h_in + kh
                            w_cur = w_in + kw

                            # bounds check
                            if (h_cur >= 0) & (h_cur < H) & (w_cur >= 0) & (w_cur < W):
                                inp_idx = ((b * C + ic) * H + h_cur) * W + w_cur
                                inp = tl.load(input_ptr + inp_idx, mask=True, other=0.0)

                                wgt_idx = ((oc * C + ic) * KH + kh) * KW + kw
                                wgt = tl.load(weight_ptr + wgt_idx, mask=True, other=0.0)

                                acc += inp * wgt

                # add bias
                bias = tl.load(bias_ptr + oc, mask=True, other=0.0)
                acc += bias

                # batch norm
                mean = tl.load(running_mean_ptr + oc, mask=True, other=0.0)
                var  = tl.load(running_var_ptr + oc, mask=True, other=0.0)
                gamma = tl.load(gamma_ptr + oc, mask=True, other=0.0)
                beta  = tl.load(beta_ptr + oc, mask=True, other=0.0)

                inv_std = tl.rsqrt(var + eps)
                bn = (acc - mean) * inv_std * gamma + beta

                # scaling
                out_val = bn * scaling_factor

                # store result
                out_idx = ((b * O + oc) * H_out + h_out) * W_out + w_out
                tl.store(output_ptr + out_idx, out_val, mask=True)


# --------------------------------------------------------------------------- #
#                      Helper function to launch the kernel                  #
# --------------------------------------------------------------------------- #
def fused_conv2d_bn_scale(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    scaling_factor: float,
    stride: int = 1,
    padding: int = 0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    input: (B, C, H, W)
    weight: (O, C, KH, KW)
    bias: (O,)
    running_mean: (O,)
    running_var: (O,)
    gamma: (O,)
    beta: (O,)
    scaling_factor: scalar
    """
    assert input.is_cuda and weight.is_cuda
    B, C, H, W = input.shape
    O, _, KH, KW = weight.shape

    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1

    out = torch.empty((B, O, H_out, W_out), dtype=input.dtype, device=input.device)

    grid = lambda meta: (
        ( (O + meta["BLOCK_SIZE_X"] - 1) // meta["BLOCK_SIZE_X"],
          (H_out * W_out + meta["BLOCK_SIZE_Y"] - 1) // meta["BLOCK_SIZE_Y"] ),
    )

    conv2d_fused_kernel[grid](
        input, weight, bias,
        running_mean, running_var,
        gamma, beta,
        out,
        B, C, H, W, O, KH, KW, H_out, W_out,
        stride, padding, eps,
        scaling_factor,
        BLOCK_SIZE_X=meta["BLOCK_SIZE_X"],
        BLOCK_SIZE_Y=meta["BLOCK_SIZE_Y"],
    )

    return out


# --------------------------------------------------------------------------- #
#                               Optimized Model                              #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Optimized model that fuses Conv2d, BatchNorm2d and a scalar scaling
    into a single Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract parameters
        weight = self.conv.weight
        bias   = self.conv.bias
        mean   = self.bn.running_mean
        var    = self.bn.running_var
        gamma  = self.bn.weight
        beta   = self.bn.bias
        eps    = self.bn.eps

        # Call fused Triton kernel
        return fused_conv2d_bn_scale(
            x,
            weight,
            bias,
            mean,
            var,
            gamma,
            beta,
            self.scaling_factor,
            stride=self.conv.stride[0],
            padding=self.conv.padding[0],
            eps=eps,
        )