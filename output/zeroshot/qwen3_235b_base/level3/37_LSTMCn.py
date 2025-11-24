import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, seq_len, in_features, out_features,
    stride_xb, stride_xs, stride_xi,
    stride_wi, stride_wo,
    stride_ob, stride_os, stride_oo,
    has_bias: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = pid // (tl.cdiv(seq_len, BLOCK_SIZE_M))
    seq_id = pid % (tl.cdiv(seq_len, BLOCK_SIZE_M))

    offset_m = batch_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = tl.arange(0, BLOCK_SIZE_N)
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Compute pointers for this block
    x_ptrs = x_ptr + (offset_m[:, None] // seq_len) * stride_xb + \
             (offset_m[:, None] % seq_len) * stride_xs + offset_k[None, :] * stride_xi
    w_ptrs = w_ptr + offset_k[:, None] * stride_wi + offset_n[None, :] * stride_wo
    b_ptrs = b_ptr + offset_n if has_bias else None
    out_ptrs = out_ptr + (offset_m[:, None] // seq_len) * stride_ob + \
               (offset_m[:, None] % seq_len) * stride_os + offset_n[None, :] * stride_oo

    # Initialize output
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Load input and weights and compute matrix multiplication
    for k in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
        x_mask = (offset_m < batch_size * seq_len)[:, None] & (k * BLOCK_SIZE_K + offset_k < in_features)[None, :]
        w_mask = (k * BLOCK_SIZE_K + offset_k < in_features)[:, None] & (offset_n < out_features)[None, :]
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        accumulator += tl.dot(x, w)
        x_ptrs += BLOCK_SIZE_K * stride_xi
        w_ptrs += BLOCK_SIZE_K * stride_wi

    # Add bias
    if has_bias:
        b = tl.load(b_ptrs, mask=offset_n < out_features, other=0.0)
        accumulator += b[None, :]

    # Cast to input precision and write back
    out = accumulator.to(x_ptr.dtype.element_ty)
    out_mask = (offset_m < batch_size * seq_len)[:, None] & (offset_n < out_features)[None, :]
    tl.store(out_ptrs, out, mask=out_mask)


def triton_fused_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda and weight.is_cuda
    if bias is not None:
        assert bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous() if bias is not None else None

    batch_size, seq_len, in_features = x.shape
    out_features = weight.shape[0]
    out = torch.empty((batch_size, seq_len, out_features), device=x.device, dtype=x.dtype)

    # 1D launch kernel where each block gets its own sequence and batch element
    def grid(meta):
        return (triton.cdiv(seq_len, meta['BLOCK_SIZE_M']) * batch_size,)

    fused_linear_kernel[grid](
        x, weight, bias, out,
        batch_size, seq_len, in_features, out_features,
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(1), weight.stride(0),
        out.stride(0), out.stride(1), out.stride(2),
        has_bias=bias is not None,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
    )
    return out


@triton.jit
def sigmoid_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Sigmoid(x) = 1 / (1 + exp(-x))
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def tanh_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.tanh(x)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_sigmoid(x: torch.Tensor):
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    sigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out


def triton_tanh(x: torch.Tensor):
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    tanh_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out


class LSTMCellTriton(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        self._flat_weights = None

    def flatten_weights(self):
        self._flat_weights = nn.Parameter(torch.cat([
            self.weight_ih.data,
            self.weight_hh.data,
            self.bias_ih.data,
            self.bias_hh.data
        ]))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        if self._flat_weights is None:
            self.flatten_weights()

        # Split the flat weights
        w_ih = self._flat_weights[:4*self.hidden_size*self.input_size].view(4*self.hidden_size, self.input_size)
        w_hh = self._flat_weights[4*self.hidden_size*self.input_size:4*self.hidden_size*(self.input_size+self.hidden_size)].view(4*self.hidden_size, self.hidden_size)
        b_ih = self._flat_weights[4*self.hidden_size*(self.input_size+self.hidden_size):4*self.hidden_size*(self.input_size+self.hidden_size)+4*self.hidden_size]
        b_hh = self._flat_weights[4*self.hidden_size*(self.input_size+self.hidden_size)+4*self.hidden_size:]

        # Compute gates: input, forget, candidate, output
        gates_x = triton_fused_linear(x.unsqueeze(1), w_ih, b_ih).squeeze(1)
        gates_h = triton_fused_linear(h_prev, w_hh, b_hh)
        gates = gates_x + gates_h

        i, f, g, o = gates.chunk(4, dim=-1)

        # Apply activations with Triton kernels
        i = triton_sigmoid(i)
        f = triton_sigmoid(f)
        g = triton_tanh(g)
        o = triton_sigmoid(o)

        # Update cell and hidden state
        c = f * c_prev + i * g
        h = o * triton_tanh(c)

        return h, c


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout

        # Replace nn.LSTM with custom LSTM layers using Triton-optimized cells
        self.lstm_cells = nn.ModuleList([
            LSTMCellTriton(input_size if layer == 0 else hidden_size, hidden_size)
            for layer in range(num_layers)
        ])
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0, c0):
        batch_size, seq_len, _ = x.shape

        # Process each time step
        h = h0
        c = c0

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_new = []
            c_new = []
            for layer in range(self.num_layers):
                h_l = h[layer]
                c_l = c[layer]
                x_input = h_new[-1] if layer > 0 else x_t
                h_out, c_out = self.lstm_cells[layer](x_input, h_l, c_l)
                if self.dropout_layer and layer < self.num_layers - 1:
                    h_out = self.dropout_layer(h_out)
                h_new.append(h_out)
                c_new.append(c_out)
            h = torch.stack(h_new)
            c = torch.stack(c_new)

        # Decode the hidden state of the last time step
        out = self.fc(h[-1])  # Use final layer's hidden state

        return c[-1]