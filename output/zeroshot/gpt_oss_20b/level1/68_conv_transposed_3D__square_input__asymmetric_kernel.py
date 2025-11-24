import torch
import torch.nn as nn
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel for a transposed 3D convolution (naïve implementation)
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["N", "C_in", "C_out", "D_in", "W_in", "H_in", "K_d", "K_w", "K_h"],
)
@triton.jit
def conv_transpose3d_kernel(
    X_ptr,          # [B, C_in, D_in, W_in, H_in]
    W_ptr,          # [C_in, C_out, K_d, K_w, K_h]
    OUT_ptr,        # [B, C_out, D_out, W_out, H_out]
    B, C_in, C_out,
    D_in, W_in, H_in,
    D_out, W_out, H_out,
    K_d, K_w, K_h,
    stride_d, stride_w, stride_h,
    padding_d, padding_w, padding_h,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of output voxels
    pid = tl.program_id(0)
    stride = BLOCK_SIZE
    start = pid * stride
    end = start + stride

    # We will loop over all batch elements, output channels and
    # spatial positions inside the block.
    for idx in range(start, end):
        if idx >= B * C_out * D_out * W_out * H_out:
            break

        # Decode 5‑dimensional index from the linear index
        tmp = idx
        h_out = tmp % H_out
        tmp //= H_out
        w_out = tmp % W_out
        tmp //= W_out
        d_out = tmp % D_out
        tmp //= D_out
        c_out = tmp % C_out
        b = tmp // C_out

        # Compute the region of the input that contributes to this output
        d_start = d_out * stride_d - padding_d
        w_start = w_out * stride_w - padding_w
        h_start = h_out * stride_h - padding_h

        acc = tl.zeros([1], dtype=tl.float32)

        # Iterate over kernel and input channels
        for c_in in range(C_in):
            for kd in range(K_d):
                d_in = d_start + kd
                if d_in < 0 or d_in >= D_in:
                    continue
                for kw in range(K_w):
                    w_in = w_start + kw
                    if w_in < 0 or w_in >= W_in:
                        continue
                    for kh in range(K_h):
                        h_in = h_start + kh
                        if h_in < 0 or h_in >= H_in:
                            continue

                        # Compute flat indices
                        x_off = (((b * C_in + c_in) * D_in + d_in) * W_in + w_in) * H_in + h_in
                        w_off = (((c_in * C_out + c_out) * K_d + kd) * K_w + kw) * K_h + kh

                        x_val = tl.load(X_ptr + x_off)
                        w_val = tl.load(W_ptr + w_off)

                        acc = acc + x_val * w_val

        # Store the result
        out_off = (((b * C_out + c_out) * D_out + d_out) * W_out + w_out) * H_out + h_out
        tl.store(OUT_ptr + out_off, acc, mask=True)


# ----------------------------------------------------------------------
# Helper wrapper that launches the kernel
# ----------------------------------------------------------------------
def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple,
    padding: tuple,
    dilation: tuple = (1, 1, 1),
    groups: int = 1,
):
    """
    A minimal wrapper around the Triton kernel.
    Only supports groups=1, dilation=1.
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors."
    assert groups == 1, "Only groups=1 is supported."
    assert dilation == (1, 1, 1), "Only dilation=1 is supported."

    B, C_in, D_in, W_in, H_in = x.shape
    C_in_w, C_out, K_d, K_w, K_h = weight.shape
    assert C_in == C_in_w, "Input channel mismatch."

    stride_d, stride_w, stride_h = stride
    padding_d, padding_w, padding_h = padding

    # Compute output dimensions
    D_out = (D_in - 1) * stride_d - 2 * padding_d + K_d
    W_out = (W_in - 1) * stride_w - 2 * padding_w + K_w
    H_out = (H_in - 1) * stride_h - 2 * padding_h + K_h

    out = torch.empty((B, C_out, D_out, W_out, H_out), device=x.device, dtype=x.dtype)

    # Prepare grid
    num_elements = B * C_out * D_out * W_out * H_out
    grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

    conv_transpose3d_kernel[grid](
        x,
        weight,
        out,
        B, C_in, C_out,
        D_in, W_in, H_in,
        D_out, W_out, H_out,
        K_d, K_w, K_h,
        stride_d, stride_w, stride_h,
        padding_d, padding_w, padding_h,
        BLOCK_SIZE=128,
    )
    return out


# ----------------------------------------------------------------------
# Optimized model that uses the Triton kernel
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with a square input and an asymmetric kernel
    using a custom Triton kernel for improved performance.
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
        self.bias = bias

        # Weight and bias parameters
        self.weight = nn.Parameter(
            torch.randn(
                out_channels,
                in_channels // groups,
                *kernel_size,
                device="cuda",
                dtype=torch.float32,
            )
        )
        if bias:
            self.bias_param = nn.Parameter(
                torch.zeros(out_channels, device="cuda", dtype=torch.float32)
            )
        else:
            self.bias_param = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = triton_conv_transpose3d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )
        if self.bias_param is not None:
            out = out + self.bias_param.view(1, -1, 1, 1, 1)
        return out