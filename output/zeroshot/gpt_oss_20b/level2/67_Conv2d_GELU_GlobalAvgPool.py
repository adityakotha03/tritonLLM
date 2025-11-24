import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# ------------------------------------------------------------------
#   Triton kernel: Conv2d (kernel 3x3, stride=1, padding=0) + GELU
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["B", "Cout", "Cin", "H_out", "W_out"],
)
@triton.jit
def conv_gelu_kernel(
    in_ptr,
    w_ptr,
    out_ptr,
    B,
    Cin,
    Cout,
    H_in,
    W_in,
    H_out,
    W_out,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # each program processes a block of output elements
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    end = start + BLOCK_SIZE

    # total number of output elements per batch
    out_per_batch = Cout * H_out * W_out
    total_out = B * out_per_batch

    for out_idx in range(start, min(end, total_out)):
        # compute (b, c_out, h_out, w_out) from flattened index
        b = out_idx // out_per_batch
        rem = out_idx % out_per_batch
        c_out = rem // (H_out * W_out)
        rem2 = rem % (H_out * W_out)
        h_out = rem2 // W_out
        w_out = rem2 % W_out

        # Accumulator for convolution result
        acc = 0.0

        for c_in in range(Cin):
            # base pointer for this input channel
            in_base = (b * Cin + c_in) * H_in * W_in

            # compute top-left corner of kernel in input
            h_in = h_out * stride - padding
            w_in = w_out * stride - padding

            # load 3x3 patch
            for kh in range(3):
                ih = h_in + kh
                if ih < 0 or ih >= H_in:
                    continue
                for kw in range(3):
                    iw = w_in + kw
                    if iw < 0 or iw >= W_in:
                        continue
                    in_offset = in_base + ih * W_in + iw
                    w_offset = (c_out * Cin + c_in) * 9 + kh * 3 + kw
                    inp = tl.load(in_ptr + in_offset)
                    weight = tl.load(w_ptr + w_offset)
                    acc += inp * weight

        # GELU (exact formula)
        # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        x = acc
        x3 = x * x * x
        gelu = 0.5 * x * (1.0 + tl.tanh(sqrt_2_over_pi * (x + 0.044715 * x3)))

        # store output
        out_offset = (b * Cout + c_out) * H_out * W_out + h_out * W_out + w_out
        tl.store(out_ptr + out_offset, gelu)


# ------------------------------------------------------------------
#   Triton kernel: Global average pooling (mean over H_out x W_out)
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["B", "Cout", "H_out", "W_out"],
)
@triton.jit
def global_avg_pool_kernel(
    in_ptr,
    out_ptr,
    B,
    Cout,
    H_out,
    W_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    end = start + BLOCK_SIZE

    # total number of channels per batch
    out_per_batch = Cout
    total_out = B * out_per_batch

    for idx in range(start, min(end, total_out)):
        b = idx // Cout
        c = idx % Cout
        sum_val = 0.0
        for h in range(H_out):
            for w in range(W_out):
                in_offset = (b * Cout + c) * H_out * W_out + h * W_out + w
                sum_val += tl.load(in_ptr + in_offset)
        avg = sum_val / (H_out * W_out)
        out_offset = (b * Cout) + c
        tl.store(out_ptr + out_offset, avg)


# ------------------------------------------------------------------
#   Model with custom Triton kernels
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized model: Conv2d (3x3) + GELU + Global Average Pooling
    implemented with Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        assert kernel_size == 3, "Only kernel size 3 is supported."
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        self.conv.weight.requires_grad = True  # keep as trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Cin, H_in, W_in = x.shape
        Cout = self.conv.out_channels
        kernel = self.conv.weight  # shape (Cout, Cin, 3, 3)

        # Output of conv+gelu
        H_out = H_in - 2  # stride=1, padding=0
        W_out = W_in - 2

        conv_gelu_out = torch.empty(
            (B, Cout, H_out, W_out),
            dtype=x.dtype,
            device=x.device,
        )

        grid = lambda meta: (
            (B * Cout * H_out * W_out + meta["BLOCK_SIZE"] - 1)
            // meta["BLOCK_SIZE"],
        )
        conv_gelu_kernel[grid](
            x,
            kernel,
            conv_gelu_out,
            B,
            Cin,
            Cout,
            H_in,
            W_in,
            H_out,
            W_out,
            stride=1,
            padding=0,
            BLOCK_SIZE=meta["BLOCK_SIZE"],
        )

        # Global average pooling
        out = torch.empty((B, Cout), dtype=x.dtype, device=x.device)
        grid2 = lambda meta: ((B * Cout + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
        global_avg_pool_kernel[grid2](
            conv_gelu_out,
            out,
            B,
            Cout,
            H_out,
            W_out,
            BLOCK_SIZE=meta["BLOCK_SIZE"],
        )

        return out