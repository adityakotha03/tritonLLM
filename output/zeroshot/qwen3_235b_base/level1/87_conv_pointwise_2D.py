import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1x1_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    batch_size, in_channels, out_channels, height, width,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci,
    stride_outb, stride_outc, stride_outh, stride_outw,
    has_bias: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # 2D block ID for output channel and input spatial location
    pid = tl.program_id(0)
    batch = pid // (out_channels * height * width)
    out_c = (pid // (height * width)) % out_channels
    h = (pid // width) % height
    w = pid % width

    # Compute offsets for output and input
    offset_out = batch * stride_outb + out_c * stride_outc + h * stride_outh + w * stride_outw
    offset_x = batch * stride_xb + h * stride_xh + w * stride_xw

    # Pointers to input and output
    x_ptrs = x_ptr + offset_x + tl.arange(0, BLOCK_SIZE_K) * stride_xc
    w_ptrs = w_ptr + out_c * stride_wco + tl.arange(0, BLOCK_SIZE_K) * stride_wci
    out_ptr += offset_out

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Number of blocks along in_channels
    K = in_channels
    num_blocks = tl.cdiv(K, BLOCK_SIZE_K)

    for k in range(0, num_blocks):
        # Load input (C, ) and weights (out_c, C)
        mask_k = (k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) < K
        x = tl.load(x_ptrs + k * BLOCK_SIZE_K * stride_xc, mask=mask_k, other=0.0)
        w = tl.load(w_ptrs + k * BLOCK_SIZE_K * stride_wci, mask=mask_k, other=0.0)

        # Matmul in blocks
        acc += tl.dot(w[None, :], x[:, None])

    # Add bias if present
    if has_bias:
        b = tl.load(bias_ptr + out_c)
        acc = acc + b

    # Cast to output precision
    acc = acc.to(tl.float16)

    # Store result
    tl.store(out_ptr, acc)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = bias

        # Initialize weight and optional bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Weight initialization
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, in_channels, height, width)
        batch_size, _, height, width = x.shape

        # Reshape input to view each spatial location independently
        x_reshaped = x.view(batch_size, self.in_channels, -1)  # (B, C_in, H*W)
        x_reshaped = x_reshaped.transpose(1, 2)  # (B, H*W, C_in)
        x_reshaped = x_reshaped.reshape(-1, self.in_channels)  # (B*H*W, C_in)

        # Output buffer
        out = torch.empty(
            batch_size * height * width, self.out_channels,
            device=x.device, dtype=torch.float16
        )

        # Weight is (out_channels, in_channels)
        w = self.weight.to(torch.float16)

        # Bias
        b = self.bias.to(torch.float16) if self.bias is not None else None

        # Launch kernel
        def grid(meta):
            return (batch_size * height * width,)

        conv1x1_kernel[grid](
            x_reshaped, w, b, out,
            batch_size, self.in_channels, self.out_channels, height, width,
            x_reshaped.stride(0), x_reshaped.stride(1), 0, 0,
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1), 0, 0,
            self.use_bias,
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=16,
            BLOCK_SIZE_K=64,
        )

        # Reshape back to (B, C_out, H, W)
        out = out.reshape(batch_size, height, width, self.out_channels)
        out = out.transpose(2, 3).transpose(1, 2)

        return out