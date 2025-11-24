import torch
import torch.nn as nn
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
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + (offs_m[:, None] // seq_len) * stride_xb + \
             (offs_m[:, None] % seq_len) * stride_xs + offs_k[None, :] * stride_xi
    w_ptrs = w_ptr + offs_k[:, None] * stride_wi + offs_n[None, :] * stride_wo

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
        x_mask = (offs_m[:, None] < batch_size * seq_len) & (offs_k[None, :] < in_features)
        w_mask = (offs_k[:, None] < in_features) & (offs_n[None, :] < out_features)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        accumulator = tl.dot(x, w, acc=accumulator)
        x_ptrs += BLOCK_SIZE_K * stride_xi
        w_ptrs += BLOCK_SIZE_K * stride_wi

    acc = accumulator.to(tl.float16)

    if has_bias:
        b_ptrs = b_ptr + offs_n * stride_wo
        b_mask = offs_n < out_features
        bias = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += bias[None, :]

    out_ptrs = out_ptr + (offs_m[:, None] // seq_len) * stride_ob + \
               (offs_m[:, None] % seq_len) * stride_os + offs_n[None, :] * stride_oo
    out_mask = (offs_m[:, None] < batch_size * seq_len) & (offs_n[None, :] < out_features)
    tl.store(out_ptrs, acc, mask=out_mask)


def triton_fused_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda and weight.is_cuda
    assert x.dtype == torch.float16, "Input must be float16 for optimal Triton GEMM performance"
    assert weight.dtype == torch.float16
    assert bias is None or bias.dtype == torch.float16

    batch_size, seq_len, in_features = x.shape
    out_features, _ = weight.shape

    out = torch.empty((batch_size, seq_len, out_features), device=x.device, dtype=x.dtype)

    def grid(META):
        return (
            triton.cdiv(batch_size * seq_len, META['BLOCK_SIZE_M']),
            triton.cdiv(out_features, META['BLOCK_SIZE_N']),
        )

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
        num_stages=4,
        num_warps=4,
    )
    return out


@triton.jit
def sigmoid_tanh_kernel(
    gates_ptr, c_ptr, h_ptr,
    batch_size, hidden_size,
    stride_gz, stride_gr, stride_gn,
    stride_hz, stride_hr, stride_hn,
    stride_c, stride_h,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_hid = tl.program_id(axis=1)

    start_idx = pid_b * hidden_size + pid_hid * BLOCK_SIZE
    offs_hid = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offs_hid < batch_size * hidden_size

    # Load gate components: [z, r, n] each of size hidden_size
    z_ptr = gates_ptr + offs_hid * stride_gz
    r_ptr = gates_ptr + offs_hid * stride_gr
    n_ptr = gates_ptr + offs_hid * stride_gn

    z = tl.load(z_ptr, mask=mask, other=0.0)
    r = tl.load(r_ptr, mask=mask, other=0.0)
    n = tl.load(n_ptr, mask=mask, other=0.0)

    # Apply sigmoid to z and r
    z_sigmoid = tl.sigmoid(z)
    r_sigmoid = tl.sigmoid(r)

    # Apply tanh to n
    n_tanh = tl.tanh(n)

    # Update cell and hidden state
    c_prev = tl.load(c_ptr + offs_hid * stride_c, mask=mask, other=0.0)
    h_prev = tl.load(h_ptr + offs_hid * stride_h, mask=mask, other=0.0)

    c_new = (1.0 - z_sigmoid) * c_prev + z_sigmoid * n_tanh
    h_new = (1.0 - z_sigmoid) * h_prev + z_sigmoid * n_tanh

    # Store updated states
    tl.store(c_ptr + offs_hid * stride_c, c_new, mask=mask)
    tl.store(h_ptr + offs_hid * stride_h, h_new, mask=mask)


def triton_lstm_cell_step(gates, h_prev, c_prev):
    assert gates.is_cuda and h_prev.is_cuda and c_prev.is_cuda
    assert gates.dtype == torch.float16
    h_prev = h_prev.to(torch.float16)
    c_prev = c_prev.to(torch.float16)

    batch_size, hidden_size = h_prev.shape
    gates = gates.view(batch_size, 3, hidden_size)
    z, r, n = gates.unbind(dim=1)

    grid = lambda meta: (
        batch_size,
        triton.cdiv(hidden_size, meta['BLOCK_SIZE']),
    )

    sigmoid_tanh_kernel[grid](
        z, r, n,
        c_prev, h_prev,
        batch_size, hidden_size,
        z.stride(0), r.stride(0), n.stride(0),
        h_prev.stride(0), h_prev.stride(0), h_prev.stride(0),
        c_prev.stride(0), h_prev.stride(0),
        BLOCK_SIZE=256,
        num_warps=4,
    )

    return h_prev, c_prev


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Replace LSTM with parameter containers (we'll implement custom step)
        self.weight_ih_list = nn.ParameterList()
        self.weight_hh_list = nn.ParameterList()
        self.bias_ih_list = nn.ParameterList()
        self.bias_hh_list = nn.ParameterList()

        for i in range(num_layers):
            in_sz = input_size if i == 0 else hidden_size
            self.weight_ih_list.append(nn.Parameter(torch.empty(4 * hidden_size, in_sz)))
            self.weight_hh_list.append(nn.Parameter(torch.empty(4 * hidden_size, hidden_size)))
            self.bias_ih_list.append(nn.Parameter(torch.empty(4 * hidden_size)))
            self.bias_hh_list.append(nn.Parameter(torch.empty(4 * hidden_size)))
            # Xavier init
            for name, param in self.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

        self.fc_weight = nn.Parameter(torch.empty(output_size, hidden_size))
        self.fc_bias = nn.Parameter(torch.empty(output_size))
        nn.init.xavier_uniform_(self.fc_weight)
        nn.init.zeros_(self.fc_bias)

        self.dropout = dropout
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)

    def lstm_cell(self, x, h_prev, c_prev, w_ih, w_hh, b_ih, b_hh):
        gates = (torch.matmul(x, w_ih.t()) + b_ih + torch.matmul(h_prev, w_hh.t()) + b_hh).to(torch.float16)
        # Split into i, f, g, o
        i, f, g, o = gates.chunk(4, dim=-1)
        # Use standard sigmoid/tanh fusion via PyTorch (already optimized)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def forward(self, x, h0=None, c0=None):
        batch_size, seq_len, _ = x.shape
        device = x.device

        if h0 is None:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=device)
        if c0 is None:
            c0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=device)

        x = x.to(torch.float16)
        h_prev = h0.to(torch.float16)
        c_prev = c0.to(torch.float16)

        for layer in range(self.num_layers):
            w_ih = self.weight_ih_list[layer].to(torch.float16)
            w_hh = self.weight_hh_list[layer].to(torch.float16)
            b_ih = self.bias_ih_list[layer].to(torch.float16)
            b_hh = self.bias_hh_list[layer].to(torch.float16)

            layer_output = []
            for t in range(seq_len):
                inp = x[:, t, :]
                h_t, c_t = self.lstm_cell(inp, h_prev[layer], c_prev[layer], w_ih, w_hh, b_ih, b_hh)
                layer_output.append(h_t)
                h_prev[layer].copy_(h_t)
                c_prev[layer].copy_(c_t)

            x = torch.stack(layer_output, dim=1)
            if self.dropout > 0 and layer < self.num_layers - 1:
                x = self.dropout_layer(x)

        # Final fully connected layer using Triton
        fc_weight = self.fc_weight.to(torch.float16)
        fc_bias = self.fc_bias.to(torch.float16) if self.fc_bias is not None else None
        fc_out = triton_fused_linear(x, fc_weight, fc_bias)
        out = fc_out[:, -1, :].to(torch.float32)

        return out