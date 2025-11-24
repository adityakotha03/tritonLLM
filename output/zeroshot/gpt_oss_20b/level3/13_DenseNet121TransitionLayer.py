import torch
import torch.nn as nn
import triton
import triton.language as tl

# ---------------- Triton kernel ----------------
@triton.jit
def fused_bn_relu_conv_pool_kernel(
    inp_ptr,          # [batch, in_ch, H, W]
    weight_ptr,       # [out_ch, in_ch, 1, 1]
    bias_ptr,         # [out_ch]
    gamma_ptr,        # [out_ch]
    beta_ptr,         # [out_ch]
    mean_ptr,         # [out_ch]
    var_ptr,          # [out_ch]
    out_ptr,          # [batch, out_ch, H/2, W/2]
    batch, in_ch, out_ch,
    H, W,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused: BN -> ReLU -> 1x1 Conv -> AvgPool2d
    """
    # program index
    prog_idx = tl.program_id(0)

    # compute total number of output elements
    num_out = batch * out_ch * (H // 2) * (W // 2)

    stride = BLOCK_SIZE
    start = prog_idx * stride
    end = start + stride

    for idx in range(start, min(end, num_out)):
        # decode idx to batch, oc, y, x
        b = idx // (out_ch * (H // 2) * (W // 2))
        rem = idx % (out_ch * (H // 2) * (W // 2))
        oc = rem // ((H // 2) * (W // 2))
        rem2 = rem % ((H // 2) * (W // 2))
        y = rem2 // (W // 2)
        x = rem2 % (W // 2)

        # compute 1x1 conv output at (b, oc, y*2, x*2) and (y*2+1,x*2+1)
        # we average over 2x2 block after conv
        acc = tl.zeros([1], dtype=tl.float32)

        # loop over input channels
        for ic in range(0, in_ch, 8):
            ic_end = tl.min(ic + 8, in_ch)
            ic_range = ic_end - ic

            # load input for four spatial locations
            inp_off = (b * in_ch * H * W
                       + ic * H * W
                       + (y*2) * W + (x*2))
            inp_offsets = tl.arange(0, ic_range)

            inp_00 = tl.load(inp_ptr + inp_off + inp_offsets, mask=inp_offsets < ic_range, other=0.0)
            inp_01 = tl.load(inp_ptr + inp_off + W + inp_offsets, mask=inp_offsets < ic_range, other=0.0)
            inp_10 = tl.load(inp_ptr + inp_off + 2*W + inp_offsets, mask=inp_offsets < ic_range, other=0.0)
            inp_11 = tl.load(inp_ptr + inp_off + 3*W + inp_offsets, mask=inp_offsets < ic_range, other=0.0)

            # load weights
            w_off = (oc * in_ch + ic) * 1
            w = tl.load(weight_ptr + w_off + inp_offsets, mask=inp_offsets < ic_range, other=0.0)

            # accumulate
            acc += tl.sum(inp_00 * w)
            acc += tl.sum(inp_01 * w)
            acc += tl.sum(inp_10 * w)
            acc += tl.sum(inp_11 * w)

        # Bias
        acc += tl.load(bias_ptr + oc)

        # BatchNorm
        mean = tl.load(mean_ptr + oc)
        var = tl.load(var_ptr + oc)
        gamma = tl.load(gamma_ptr + oc)
        beta = tl.load(beta_ptr + oc)

        bn = (acc - mean) * tl.rsqrt(var + eps)
        bn = bn * gamma + beta

        # ReLU
        relu = tl.maximum(bn, 0.0)

        # store output
        out_off = (b * out_ch * (H//2) * (W//2)
                   + oc * (H//2) * (W//2)
                   + y * (W//2) + x)
        tl.store(out_ptr + out_off, relu, mask=True)

# ---------------- Triton wrapper ----------------
def fused_bn_relu_conv_pool(inp: torch.Tensor,
                            weight: torch.Tensor,
                            bias: torch.Tensor,
                            gamma: torch.Tensor,
                            beta: torch.Tensor,
                            mean: torch.Tensor,
                            var: torch.Tensor) -> torch.Tensor:
    """
    inp: [B, C_in, H, W]
    weight: [C_out, C_in, 1, 1]
    bias, gamma, beta, mean, var: [C_out]
    """
    B, C_in, H, W = inp.shape
    C_out = weight.shape[0]
    H_out = H // 2
    W_out = W // 2

    out = torch.empty((B, C_out, H_out, W_out), device=inp.device, dtype=inp.dtype)

    BLOCK_SIZE = 256
    grid = lambda meta: ( (B * C_out * H_out * W_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    fused_bn_relu_conv_pool_kernel[grid](
        inp_ptr=inp,
        weight_ptr=weight,
        bias_ptr=bias,
        gamma_ptr=gamma,
        beta_ptr=beta,
        mean_ptr=mean,
        var_ptr=var,
        out_ptr=out,
        batch=B,
        in_ch=C_in,
        out_ch=C_out,
        H=H,
        W=W,
        eps=1e-5,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

# ---------------- Optimized Model ----------------
class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features, affine=False, track_running_stats=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=True)
        # The gamma, beta, mean, var are managed by BatchNorm module
        # We will expose them to the Triton kernel
        # No AvgPool layer; fused into kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()

        # BatchNorm forward (use running stats)
        bn_out = self.bn(x)

        # Pull weights and params to CPU tensors for Triton
        weight = self.conv.weight
        bias = self.conv.bias

        gamma = self.bn.weight
        beta = self.bn.bias
        mean = self.bn.running_mean
        var = self.bn.running_var

        # Use Triton fused kernel
        out = fused_bn_relu_conv_pool(
            inp=bn_out,
            weight=weight,
            bias=bias,
            gamma=gamma,
            beta=beta,
            mean=mean,
            var=var
        )
        return out