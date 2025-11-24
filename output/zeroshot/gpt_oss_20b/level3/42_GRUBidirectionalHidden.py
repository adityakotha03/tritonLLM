import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernels for the GRU cell
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 256}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_fused_gates(
    a_ptr,  # (batch, 3*hidden)
    b_ptr,  # (3*hidden, hidden)
    out_ptr,  # (batch, 3*hidden)
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    col_start = pid * BLOCK_SIZE_N

    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)

    # Load B block once into registers
    b_matrix = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_M], dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_N):
        a_block = tl.load(a_ptr + row_offsets[:, None] * K + (k + tl.arange(0, BLOCK_SIZE_N))[None, :], mask=row_offsets[:, None] < M, other=0.0)
        b_block = tl.load(b_ptr + (k + tl.arange(0, BLOCK_SIZE_N)) * N + col_offsets[None, :], mask=col_offsets[None, :] < N, other=0.0)
        b_matrix += tl.dot(a_block, b_block)

    # Write out the results
    for i in range(BLOCK_SIZE_M):
        if row_offsets[i] < M:
            tl.store(out_ptr + row_offsets[i] * N + col_offsets, b_matrix[i, :], mask=col_offsets < N)


@triton.jit
def _gru_fusion_kernel(
    input_ptr,
    hidden_ptr,
    output_ptr,
    gate_weight_ptr,
    rec_weight_ptr,
    bias_ptr,
    seq_len,
    batch_size,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a chunk of the batch
    batch_start = tl.program_id(0) * BLOCK_SIZE
    batch_offsets = batch_start + tl.arange(0, BLOCK_SIZE)

    mask = batch_offsets < batch_size

    for t in range(seq_len):
        # Load input slice
        inp = tl.load(input_ptr + t * batch_size * hidden_size + batch_offsets[:, None] * hidden_size, mask=mask[:, None], other=0.0)

        # Linear part for gates: input * W
        inp_gates = tl.zeros([3 * hidden_size], dtype=tl.float32)
        _matmul_fused_gates(inp, gate_weight_ptr, inp_gates, 1, 3 * hidden_size, hidden_size, BLOCK_SIZE_M=1, BLOCK_SIZE_N=hidden_size)

        # Recurrent part: hidden * R
        hid = tl.load(hidden_ptr + batch_offsets[:, None] * hidden_size, mask=mask[:, None], other=0.0)
        hid_gates = tl.zeros([3 * hidden_size], dtype=tl.float32)
        _matmul_fused_gates(hid, rec_weight_ptr, hid_gates, 1, 3 * hidden_size, hidden_size, BLOCK_SIZE_M=1, BLOCK_SIZE_N=hidden_size)

        # Combine gates
        gates = inp_gates + hid_gates + tl.load(bias_ptr, mask=mask, other=0.0)

        # Apply activations
        r = tl.sigmoid(gates[0:hidden_size])
        z = tl.sigmoid(gates[hidden_size:2*hidden_size])
        n = tl.tanh(gates[2*hidden_size:3*hidden_size] + r * hid)

        # New hidden state
        new_h = z * hid + (1 - z) * n
        tl.store(hidden_ptr + batch_offsets[:, None] * hidden_size, new_h, mask=mask[:, None])

    # Store final hidden state
    tl.store(output_ptr + batch_offsets[:, None] * hidden_size, hidden_ptr + batch_offsets[:, None] * hidden_size, mask=mask[:, None])


# ----------------------------------------------------------------------
# Custom GRU implementation using Triton
# ----------------------------------------------------------------------
class TritonGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, batch_first, bidirectional):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        # For simplicity, we only implement the forward direction
        self.weight_ih = nn.Parameter(torch.Tensor(num_layers, hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(num_layers, hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(num_layers, hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(num_layers, hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in [self.weight_ih, self.weight_hh]:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_ih)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_ih, -bound, bound)
            nn.init.uniform_(self.bias_hh, -bound, bound)

    def forward(self, input, h_0=None):
        # input: (seq_len, batch, input_size) or (batch, seq_len, input_size)
        if self.batch_first:
            input = input.transpose(0, 1)  # now (seq_len, batch, input_size)

        seq_len, batch, _ = input.shape
        num_directions = 1  # forward only
        device = input.device

        if h_0 is None:
            h_0 = torch.zeros(self.num_layers * num_directions, batch, self.hidden_size, device=device)

        # Output hidden states for each layer
        hidden = h_0.view(self.num_layers, num_directions, batch, self.hidden_size)

        for layer in range(self.num_layers):
            # Prepare weights and biases
            weight_ih = self.weight_ih[layer].reshape(self.hidden_size, self.input_size)
            weight_hh = self.weight_hh[layer].reshape(self.hidden_size, self.hidden_size)
            bias = torch.cat([self.bias_ih[layer], self.bias_hh[layer]]) if self.bias else None

            # Transpose for Triton kernel (expects (batch, hidden))
            weight_ih_t = weight_ih.t()
            weight_hh_t = weight_hh.t()

            # Allocate hidden state buffer
            h = hidden[layer, 0].contiguous()

            # Launch Triton kernel
            BLOCK_SIZE = 256
            grid = lambda meta: (batch + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE']
            _gru_fusion_kernel[grid](
                input,
                h,
                h,
                weight_ih_t,
                weight_hh_t,
                bias,
                seq_len,
                batch,
                self.hidden_size,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            hidden[layer, 0] = h

        # Return final hidden state
        hn = hidden.view(self.num_layers * num_directions, batch, self.hidden_size)
        return None, hn


# ----------------------------------------------------------------------
# New model using TritonGRU
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super().__init__()
        self.gru = TritonGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            bidirectional=True,
        )

    def forward(self, x, h0):
        output, h_n = self.gru(x, h0)
        return h_n