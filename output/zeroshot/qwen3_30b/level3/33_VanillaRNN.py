import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def concat_kernel(
    x_ptr,
    h_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    input_size: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * (input_size + hidden_size))

    # Load x and h
    x_offsets = (offsets // (input_size + hidden_size)) * input_size + (offsets % (input_size + hidden_size))
    h_offsets = (offsets // (input_size + hidden_size)) * hidden_size + (offsets % (input_size + hidden_size))

    x_mask = offsets < batch_size * input_size
    h_mask = (offsets >= batch_size * input_size) & (offsets < batch_size * (input_size + hidden_size))

    x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
    h_vals = tl.load(h_ptr + h_offsets, mask=h_mask, other=0.0)

    # Write concatenated values
    tl.store(out_ptr + offsets, x_vals, mask=x_mask)
    tl.store(out_ptr + offsets, h_vals, mask=h_mask)


@triton.jit
def linear_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    input_size: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Initialize offsets and mask
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    row_offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Loop over columns of weight matrix (unroll over input_size with tiling)
    for col in range(0, hidden_size, BLOCK_SIZE):
        col_start = col
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = (row_offsets[:, None] < batch_size) & (col_offsets[None, :] < input_size)

        # Load x (B x M) and w (M x N)
        x_vals = tl.load(x_ptr + row_offsets[:, None] * input_size + col_offsets[None, :], mask=mask, other=0.0)
        w_vals = tl.load(w_ptr + col_offsets[None, :] * hidden_size + col_offsets[:, None], mask=mask, other=0.0)

        # Compute dot product (B x N)
        acc = tl.dot(x_vals, w_vals)
        acc = acc + tl.load(b_ptr + col_offsets, mask=col_offsets < hidden_size, other=0.0)

        # Store result (B x N)
        out_offsets = row_offsets[:, None] * hidden_size + col_offsets[None, :]
        tl.store(out_ptr + out_offsets, acc, mask=mask)


@triton.jit
def tanh_kernel(
    x_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.tanh(x)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def linear_tanh_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    input_size: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Fusion: Linear + Tanh in one kernel
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    row_offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Loop over output dimensions (hidden_size) with tiling
    for col in range(0, hidden_size, BLOCK_SIZE):
        col_start = col
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = (row_offsets[:, None] < batch_size) & (col_offsets[None, :] < hidden_size)

        # Load input and weights
        x_vals = tl.load(x_ptr + row_offsets[:, None] * input_size + col_offsets[None, :], mask=mask, other=0.0)
        w_vals = tl.load(w_ptr + col_offsets[None, :] * hidden_size + col_offsets[:, None], mask=mask, other=0.0)

        # Compute linear (B x N)
        acc = tl.dot(x_vals, w_vals)
        bias = tl.load(b_ptr + col_offsets, mask=col_offsets < hidden_size, other=0.0)
        acc += bias

        # Apply tanh
        out = tl.tanh(acc)

        # Store result
        out_offsets = row_offsets[:, None] * hidden_size + col_offsets[None, :]
        tl.store(out_ptr + out_offsets, out, mask=mask)


@triton.jit
def linear_kernel_h2o(
    h_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    output_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute output layer: h2o
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    row_offsets = block_start + tl.arange(0, BLOCK_SIZE)

    for col in range(0, output_size, BLOCK_SIZE):
        col_start = col
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = (row_offsets[:, None] < batch_size) & (col_offsets[None, :] < output_size)

        h_vals = tl.load(h_ptr + row_offsets[:, None] * hidden_size + col_offsets[None, :], mask=mask, other=0.0)
        w_vals = tl.load(w_ptr + col_offsets[None, :] * output_size + col_offsets[:, None], mask=mask, other=0.0)

        acc = tl.dot(h_vals, w_vals)
        bias = tl.load(b_ptr + col_offsets, mask=col_offsets < output_size, other=0.0)
        acc += bias

        out_offsets = row_offsets[:, None] * output_size + col_offsets[None, :]
        tl.store(out_ptr + out_offsets, acc, mask=mask)


def triton_concat(x: torch.Tensor, h: torch.Tensor):
    batch_size, input_size = x.shape
    _, hidden_size = h.shape
    out = torch.empty(batch_size, input_size + hidden_size, device=x.device, dtype=x.dtype)
    n_elements = batch_size * (input_size + hidden_size)
    BLOCK_SIZE = 256

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    concat_kernel[grid](x, h, out, batch_size, input_size, hidden_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    batch_size, input_size = x.shape
    _, hidden_size = w.shape
    out = torch.empty(batch_size, hidden_size, device=x.device, dtype=x.dtype)
    n_elements = batch_size * hidden_size
    BLOCK_SIZE = 256

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    linear_kernel[grid](x, w, b, out, batch_size, input_size, hidden_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_linear_h2o(h: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    batch_size, hidden_size = h.shape
    _, output_size = w.shape
    out = torch.empty(batch_size, output_size, device=h.device, dtype=h.dtype)
    n_elements = batch_size * output_size
    BLOCK_SIZE = 256

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    linear_kernel_h2o[grid](h, w, b, out, batch_size, hidden_size, output_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_tanh(x: torch.Tensor):
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 256

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    tanh_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_linear_tanh(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    batch_size, input_size = x.shape
    _, hidden_size = w.shape
    out = torch.empty(batch_size, hidden_size, device=x.device, dtype=x.dtype)
    n_elements = batch_size * hidden_size
    BLOCK_SIZE = 256

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    linear_tanh_kernel[grid](x, w, b, out, batch_size, input_size, hidden_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.register_buffer('hidden', torch.randn(1, hidden_size))  # Avoid repeated allocation

        # Weights and biases
        self.i2h_weight = nn.Parameter(torch.randn(input_size + hidden_size, hidden_size))
        self.i2h_bias = nn.Parameter(torch.randn(hidden_size))
        self.h2o_weight = nn.Parameter(torch.randn(hidden_size, output_size))
        self.h2o_bias = nn.Parameter(torch.randn(output_size))

    def forward(self, x: torch.Tensor, initial_hidden=None) -> torch.Tensor:
        batch_size = x.shape[0]

        if initial_hidden is not None:
            self.hidden.copy_(initial_hidden)

        # Ensure tensors are contiguous and on the same device
        x = x.contiguous()
        self.hidden = self.hidden.to(x.device)
        self.hidden = self.hidden.expand(batch_size, -1).contiguous()

        # Concatenate x and hidden
        combined = triton_concat(x, self.hidden)

        # Linear + Tanh fused
        self.hidden = triton_linear_tanh(combined, self.i2h_weight, self.i2h_bias)

        # Output layer
        output = triton_linear_h2o(self.hidden, self.h2o_weight, self.h2o_bias)

        return output