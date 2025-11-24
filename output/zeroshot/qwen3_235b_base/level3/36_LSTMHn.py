import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_linear_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
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
    num_blocks_m = (batch_size * seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (out_features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_blocks_k = (in_features + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    num_blocks_k = tl.cdiv(in_features, BLOCK_SIZE_K)

    block_m = pid // num_blocks_n
    block_n = pid % num_blocks_n

    offs_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + offs_m[:, None] // seq_len * stride_xb + (offs_m[:, None] % seq_len) * stride_xs + offs_k[None, :] * stride_xi
    weight_ptrs = weight_ptr + offs_k[:, None] * stride_wi + offs_n[None, :] * stride_wo
    output_ptrs = output_ptr + offs_m[:, None] * stride_ob + offs_n[None, :] * stride_oo

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, num_blocks_k):
        mask_k = (k * BLOCK_SIZE_K + offs_k) < in_features
        x = tl.load(x_ptrs, mask=mask_k[None, :], other=0.0)
        w = tl.load(weight_ptrs, mask=mask_k[:, None], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_SIZE_K * stride_xi
        weight_ptrs += BLOCK_SIZE_K * stride_wi

    output = acc.to(tl.float16)

    mask_m = (offs_m < batch_size * seq_len)[:, None]
    mask_n = (offs_n < out_features)[None, :]
    mask = mask_m & mask_n

    if has_bias:
        bias_ptrs = bias_ptr + offs_n * stride_wo
        bias = tl.load(bias_ptrs)[None, :]
        output += bias

    tl.store(output_ptrs, output, mask=mask)


def triton_fused_linear(x, weight, bias=None):
    assert x.is_cuda and weight.is_cuda
    if bias is not None:
        assert bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, seq_len, in_features = x.shape
    out_features = weight.shape[0]

    output = torch.empty((batch_size, seq_len, out_features), device=x.device, dtype=torch.float16)

    def grid(meta):
        return ((batch_size * seq_len + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M'] *
                (out_features + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N'],)

    fused_linear_kernel[grid](
        x, weight, bias, output,
        batch_size, seq_len, in_features, out_features,
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(1), weight.stride(0),
        output.stride(0), output.stride(1), output.stride(2),
        has_bias=(bias is not None),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32
    )
    return output


@triton.jit
def lstm_cell_kernel(
    x_ptr, h_ptr, c_ptr,
    w_ih_ptr, w_hh_ptr, b_ih_ptr, b_hh_ptr,
    h_new_ptr, c_new_ptr,
    batch_size, input_size, hidden_size,
    stride_xb, stride_xi,
    stride_hb, stride_hh,
    stride_cb, stride_ch,
    stride_wihh, stride_wiho,
    stride_whhh, stride_whho,
    stride_bih, stride_bhh,
    stride_hnb, stride_hnh,
    stride_cnb, stride_cnh,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    num_block_h = (hidden_size + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    if pid_b >= batch_size or pid_h >= num_block_h:
        return

    h_start = pid_h * BLOCK_SIZE_H
    h_offs = h_start + tl.arange(0, BLOCK_SIZE_H)
    mask_h = h_offs < hidden_size

    x_ptrs = x_ptr + pid_b * stride_xb + h_offs * stride_xi
    h_ptrs = h_ptr + pid_b * stride_hb + h_offs * stride_hh
    c_ptrs = c_ptr + pid_b * stride_cb + h_offs * stride_ch

    x = tl.load(x_ptrs, mask=mask_h, other=0.0)
    h = tl.load(h_ptrs, mask=mask_h, other=0.0)
    c = tl.load(c_ptrs, mask=mask_h, other=0.0)

    w_ih_ptrs = w_ih_ptr + h_offs[None, :] * stride_wiho + tl.arange(0, 4)[:, None] * stride_wihh
    w_hh_ptrs = w_hh_ptr + h_offs[None, :] * stride_whho + tl.arange(0, 4)[:, None] * stride_whhh

    b_ih_ptrs = b_ih_ptr + tl.arange(0, 4) * stride_bih + h_offs * stride_bhh
    b_hh_ptrs = b_hh_ptr + tl.arange(0, 4) * stride_bhh + h_offs * stride_bhh

    i = tl.dot(w_ih_ptrs, x) + tl.dot(w_hh_ptrs, h) + tl.load(b_ih_ptrs, mask=mask_h[None, :], other=0.0) + tl.load(b_hh_ptrs, mask=mask_h[None, :], other=0.0)

    i0, i1, i2, i3 = i[0], i[1], i[2], i[3]

    f = tl.sigmoid(i0)
    i_gate = tl.sigmoid(i1)
    o = tl.sigmoid(i2)
    g = tl.tanh(i3)

    c_new = f * c + i_gate * g
    h_new = o * tl.tanh(c_new)

    tl.store(h_new_ptr + pid_b * stride_hnb + h_offs * stride_hnh, h_new, mask=mask_h)
    tl.store(c_new_ptr + pid_b * stride_cnb + h_offs * stride_cnh, c_new, mask=mask_h)


def triton_lstm_cell(x, h, c, w_ih, w_hh, b_ih, b_hh):
    assert all(t.is_cuda for t in [x, h, c, w_ih, w_hh])
    if b_ih is not None:
        assert b_ih.is_cuda and b_hh.is_cuda
    x = x.contiguous()
    h = h.contiguous()
    c = c.contiguous()
    w_ih = w_ih.contiguous()
    w_hh = w_hh.contiguous()
    if b_ih is not None:
        b_ih = b_ih.contiguous()
        b_hh = b_hh.contiguous()

    batch_size, input_size = x.shape
    hidden_size = h.shape[1]

    h_new = torch.empty_like(h)
    c_new = torch.empty_like(c)

    grid = (batch_size, triton.cdiv(hidden_size, 64))

    lstm_cell_kernel[grid](
        x, h, c, w_ih, w_hh, b_ih, b_hh,
        h_new, c_new,
        batch_size, input_size, hidden_size,
        x.stride(0), x.stride(1),
        h.stride(0), h.stride(1),
        c.stride(0), c.stride(1),
        w_ih.stride(0), w_ih.stride(1),
        w_hh.stride(0), w_hh.stride(1),
        b_ih.stride(0), b_hh.stride(0),
        h_new.stride(0), h_new.stride(1),
        c_new.stride(0), c_new.stride(1),
        BLOCK_SIZE_H=64,
        BLOCK_SIZE_I=32
    )
    return h_new, c_new


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout

    def forward(self, x, h0, c0):
        batch_size, seq_len, _ = x.shape

        h = h0
        c = c0

        for layer in range(self.num_layers):
            w_ih = self.lstm._parameters[f'weight_ih_l{layer}']
            w_hh = self.lstm._parameters[f'weight_hh_l{layer}']
            b_ih = self.lstm._parameters.get(f'bias_ih_l{layer}', None)
            b_hh = self.lstm._parameters.get(f'bias_hh_l{layer}', None)

            h_layer = []
            c_layer = []

            for t in range(seq_len):
                xt = x[:, t, :]
                h_t, c_t = triton_lstm_cell(xt, h[layer], c[layer], w_ih, w_hh, b_ih, b_hh)
                h_layer.append(h_t)
                c_layer.append(c_t)

            h_layer = torch.stack(h_layer, dim=1)
            c_layer = torch.stack(c_layer, dim=1)

            if self.dropout > 0 and layer < self.num_layers - 1:
                h_layer = F.dropout(h_layer, p=self.dropout, training=self.training)

            h[layer] = h_layer[:, -1, :]
            c[layer] = c_layer[:, -1, :]
            x = h_layer

        out = triton_fused_linear(x, self.fc.weight, self.fc.bias)
        return h