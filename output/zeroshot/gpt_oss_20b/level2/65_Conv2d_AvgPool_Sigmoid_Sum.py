import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------- Triton Conv2d Kernel -------------------
@triton.jit
def conv2d_fused_kernel(
    input_ptr,    # [batch, in_c, in_h, in_w]
    weight_ptr,   # [out_c, in_c, k, k]
    bias_ptr,     # [out_c]
    output_ptr,   # [batch, out_c, out_h, out_w]
    batch,
    in_c,
    out_c,
    in_h,
    in_w,
    out_h,
    out_w,
    stride: tl.constexpr,
    padding: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per output element
    # program_id(0) -> (batch * out_c + out_c_idx)
    # program_id(1) -> (out_h * out_w + out_hw_idx)
    bc = tl.program_id(0)
    hw = tl.program_id(1)

    batch_idx = bc // out_c
    out_c_idx = bc % out_c

    out_y = hw // out_w
    out_x = hw % out_w

    # Compute the starting coordinates in the input
    in_y_start = out_y * stride - padding
    in_x_start = out_x * stride - padding

    acc = tl.zeros([1], dtype=tl.float32)

    for ic in range(in_c):
        for ky in range(K):
            in_y = in_y_start + ky
            # mask to avoid out-of-bounds read from input
            in_y_valid = (in_y >= 0) & (in_y < in_h)
            for kx in range(K):
                in_x = in_x_start + kx
                in_x_valid = (in_x >= 0) & (in_x < in_w)

                # Load input element if valid
                in_idx = (batch_idx * in_c * in_h * in_w) + \
                         (ic * in_h * in_w) + \
                         (in_y * in_w) + \
                         in_x
                in_val = tl.load(input_ptr + in_idx,
                                 mask=in_y_valid & in_x_valid,
                                 other=0.0)

                # Load weight element
                w_idx = (out_c_idx * in_c * K * K) + \
                        (ic * K * K) + \
                        (ky * K) + \
                        kx
                w_val = tl.load(weight_ptr + w_idx)

                acc += in_val * w_val

    # Add bias
    bias_val = tl.load(bias_ptr + out_c_idx)
    acc += bias_val

    # Store result
    out_idx = (batch_idx * out_c * out_h * out_w) + \
              (out_c_idx * out_h * out_w) + \
              (out_y * out_w) + \
              out_x
    tl.store(output_ptr + out_idx, acc, mask=True)


# ------------------- Triton Wrapper -------------------
def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                  stride: int = 1, padding: int = 0) -> torch.Tensor:
    """
    x: [B, C_in, H_in, W_in]
    weight: [C_out, C_in, k, k]
    bias: [C_out]
    """
    B, C_in, H_in, W_in = x.shape
    C_out, _, k, _ = weight.shape
    H_out = (H_in + 2 * padding - k) // stride + 1
    W_out = (W_in + 2 * padding - k) // stride + 1

    out = torch.empty((B, C_out, H_out, W_out), dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 128  # Not used in this simple implementation

    grid = lambda meta: ( (B * C_out), (H_out * W_out) )
    conv2d_fused_kernel[grid](
        x,
        weight,
        bias,
        out,
        B, C_in, C_out, H_in, W_in, H_out, W_out,
        stride=stride,
        padding=padding,
        K=k,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ------------------- New Model -------------------
class ModelNew(nn.Module):
    """
    Custom model using a Triton-accelerated Conv2d kernel,
    followed by average pooling, sigmoid activation, and sum reduction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

        # Initialize weights and bias for Conv2d
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size, device='cuda')
        )
        self.bias = nn.Parameter(
            torch.randn(out_channels, device='cuda')
        )

        # PyTorch pooling (kept as is to simplify fusion)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Triton-conv
        x = triton_conv2d(x, self.weight, self.bias, stride=1, padding=0)
        # Average pooling
        x = self.avg_pool(x)
        # Sigmoid activation
        x = torch.sigmoid(x)
        # Sum over spatial dimensions
        x = torch.sum(x, dim=[1, 2, 3])
        return x