import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: 3D convolution + Mish + Tanh (fused)
# ---------------------------------------------------------------------------

@triton.jit
def conv3d_mish_tanh_kernel(
    in_ptr,          # input tensor ptr (float16)
    w_ptr,           # weight tensor ptr (float16)
    out_ptr,         # output tensor ptr (float16)
    # tensor shapes
    batch: tl.constexpr,
    in_ch: tl.constexpr,
    out_ch: tl.constexpr,
    D_in: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    K_d: tl.constexpr,
    K_h: tl.constexpr,
    K_w: tl.constexpr,
    D_out: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program processes a contiguous block of output elements.
    The mapping is:
        index = block_start + thread_idx
        (b, oc, od, oh, ow) = unravel_index(index)
    """

    # --------------------------------------------------
    # Helper to convert linear index to multi‑dim
    # --------------------------------------------------
    def unravel(index):
        ow = index % W_out
        index //= W_out
        oh = index % H_out
        index //= H_out
        od = index % D_out
        index //= D_out
        oc = index % out_ch
        index //= out_ch
        b  = index
        return b, oc, od, oh, ow

    # --------------------------------------------------
    # Compute linear offset for the output element
    # --------------------------------------------------
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch * out_ch * D_out * H_out * W_out)

    # Compute output indices
    b, oc, od, oh, ow = unravel(offsets)

    # --------------------------------------------------
    # Compute the convolution sum for each element
    # --------------------------------------------------
    conv_val = tl.zeros([BLOCK_SIZE], dtype=tl.float16)

    # Input and weight strides (in contiguous layout)
    # Input shape: (batch, in_ch, D_in, H_in, W_in)
    # Weight shape: (out_ch, in_ch, K_d, K_h, K_w)
    stride_w_ow = 1
    stride_w_oh = K_w
    stride_w_od = K_w * K_h
    stride_w_ow_in_ch = stride_w_od * K_d
    stride_w_in_ch = stride_w_ow_in_ch * out_ch

    stride_in_ow = 1
    stride_in_oh = W_in
    stride_in_od = W_in * H_in
    stride_in_in_ch = stride_in_od * D_in
    stride_in_batch = stride_in_in_ch * in_ch

    for kd in range(K_d):
        for kh in range(K_h):
            for kw in range(K_w):
                # compute input start positions
                id = od * stride + kd - padding
                ih = oh * stride + kh - padding
                iw = ow * stride + kw - padding

                # load mask for in‑bounds
                in_mask = (id >= 0) & (id < D_in) & (ih >= 0) & (ih < H_in) & (iw >= 0) & (iw < W_in)

                # compute linear offset for weight and input
                w_off = (
                    oc * stride_w_in_ch
                    + tl.arange(0, BLOCK_SIZE) // (D_out * H_out * W_out) * stride_w_ow_in_ch
                    + kd * stride_w_od
                    + kh * stride_w_ow
                    + kw
                )
                in_off = (
                    b * stride_in_batch
                    + tl.arange(0, BLOCK_SIZE) // (D_out * H_out * W_out) * stride_in_in_ch
                    + id * stride_in_od
                    + ih * stride_in_oh
                    + iw
                )

                w_vals = tl.load(w_ptr + w_off, mask=mask, other=0.0)
                in_vals = tl.load(in_ptr + in_off, mask=in_mask & mask, other=0.0)
                conv_val += w_vals * in_vals

    # --------------------------------------------------
    # Apply Mish: x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x))
    # --------------------------------------------------
    x = conv_val
    softplus = tl.log(tl.exp(x) + 1.0)
    mish = x * tl.tanh(softplus)

    # --------------------------------------------------
    # Final Tanh
    # --------------------------------------------------
    out_val = tl.tanh(mish)

    # --------------------------------------------------
    # Store results
    # --------------------------------------------------
    tl.store(out_ptr + offsets, out_val, mask=mask)


# ---------------------------------------------------------------------------
# Triton wrapper function
# ---------------------------------------------------------------------------

def conv3d_mish_tanh(in_tensor, weight_tensor):
    """
    in_tensor:   (B, C_in, D_in, H_in, W_in)  fp16
    weight_tensor: (C_out, C_in, K_d, K_h, K_w) fp16
    returns: output tensor (B, C_out, D_out, H_out, W_out) fp16
    """
    assert in_tensor.is_cuda and weight_tensor.is_cuda
    B, C_in, D_in, H_in, W_in = in_tensor.shape
    C_out, _, K_d, K_h, K_w = weight_tensor.shape

    # compute output dimensions
    D_out = (D_in + 2 * 0 - K_d) // 1 + 1
    H_out = (H_in + 2 * 0 - K_h) // 1 + 1
    W_out = (W_in + 2 * 0 - K_w) // 1 + 1

    out_tensor = torch.empty(
        (B, C_out, D_out, H_out, W_out),
        dtype=in_tensor.dtype,
        device=in_tensor.device,
    )

    BLOCK_SIZE = 256

    grid = lambda meta: ((B * C_out * D_out * H_out * W_out + meta["BLOCK_SIZE"] - 1)
                         // meta["BLOCK_SIZE"],)

    conv3d_mish_tanh_kernel[grid](
        in_tensor,
        weight_tensor,
        out_tensor,
        B, C_in, C_out,
        D_in, H_in, W_in,
        K_d, K_h, K_w,
        D_out, H_out, W_out,
        1, 0, BLOCK_SIZE,
    )

    return out_tensor


# ---------------------------------------------------------------------------
# Optimized model definition
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    """
    Optimized 3D Conv + Mish + Tanh using Triton fused kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size,
                kernel_size,
                dtype=torch.float16,
                device="cuda",
            )
        )
        nn.init.kaiming_normal_(self.conv_weight, mode="fan_in", nonlinearity="relu")

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Ensure input is fp16
        x = x.to(dtype=torch.float16)
        out = conv3d_mish_tanh(x, self.conv_weight)
        return out