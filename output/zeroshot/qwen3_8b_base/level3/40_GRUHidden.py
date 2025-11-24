import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.num_directions = 1  # GRU is unidirectional

        # Define Triton kernels for GRU operations
        self._register_triton_kernels()

    def _register_triton_kernels(self):
        # Define Triton kernels for GRU operations
        self._gru_cell = self._make_gru_cell_kernel()
        self._gru_update = self._make_gru_update_kernel()
        self._gru_reset = self._make_gru_reset_kernel()
        self._gru_hidden = self._make_gru_hidden_kernel()

    def _make_gru_cell_kernel(self):
        @triton.jit
        def gru_cell_kernel(
            x_ptr,  # Pointer to input
            h_prev_ptr,  # Pointer to previous hidden state
            W_ih_ptr,  # Pointer to input-hidden weights
            W_hh_ptr,  # Pointer to hidden-hidden weights
            b_ih_ptr,  # Pointer to input bias
            b_hh_ptr,  # Pointer to hidden bias
            output_ptr,  # Pointer to output
            seq_len,  # Sequence length
            batch_size,  # Batch size
            hidden_size,  # Hidden size
            num_layers,  # Number of layers
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            offset = pid * BLOCK_SIZE
            mask = offset + tl.arange(0, BLOCK_SIZE) < seq_len * batch_size * hidden_size
            indices = offset + tl.arange(0, BLOCK_SIZE)
            idx = indices % hidden_size
            batch_idx = (indices // hidden_size) % batch_size
            seq_idx = (indices // (batch_size * hidden_size)) % seq_len
            layer_idx = (indices // (seq_len * batch_size * hidden_size)) % num_layers

            x = tl.load(x_ptr + indices, mask=mask, other=0.0)
            h_prev = tl.load(h_prev_ptr + indices, mask=mask, other=0.0)

            # Compute update gate
            update = self._gru_update(x, h_prev, W_ih_ptr, W_hh_ptr, b_ih_ptr, b_hh_ptr, seq_len, batch_size, hidden_size, num_layers, BLOCK_SIZE)
            # Compute reset gate
            reset = self._gru_reset(x, h_prev, W_ih_ptr, W_hh_ptr, b_ih_ptr, b_hh_ptr, seq_len, batch_size, hidden_size, num_layers, BLOCK_SIZE)
            # Compute hidden state
            hidden = self._gru_hidden(x, h_prev, W_ih_ptr, W_hh_ptr, b_ih_ptr, b_hh_ptr, seq_len, batch_size, hidden_size, num_layers, BLOCK_SIZE)

            output = update * hidden + (1 - update) * h_prev
            tl.store(output_ptr + indices, output, mask=mask)

        return gru_cell_kernel

    def _make_gru_update_kernel(self):
        @triton.jit
        def gru_update_kernel(
            x_ptr,  # Pointer to input
            h_prev_ptr,  # Pointer to previous hidden state
            W_ih_ptr,  # Pointer to input-hidden weights
            W_hh_ptr,  # Pointer to hidden-hidden weights
            b_ih_ptr,  # Pointer to input bias
            b_hh_ptr,  # Pointer to hidden bias
            seq_len,  # Sequence length
            batch_size,  # Batch size
            hidden_size,  # Hidden size
            num_layers,  # Number of layers
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            offset = pid * BLOCK_SIZE
            mask = offset + tl.arange(0, BLOCK_SIZE) < seq_len * batch_size * hidden_size
            indices = offset + tl.arange(0, BLOCK_SIZE)
            idx = indices % hidden_size
            batch_idx = (indices // hidden_size) % batch_size
            seq_idx = (indices // (batch_size * hidden_size)) % seq_len
            layer_idx = (indices // (seq_len * batch_size * hidden_size)) % num_layers

            x = tl.load(x_ptr + indices, mask=mask, other=0.0)
            h_prev = tl.load(h_prev_ptr + indices, mask=mask, other=0.0)

            # Compute update gate
            W_ih = tl.load(W_ih_ptr + indices, mask=mask, other=0.0)
            W_hh = tl.load(W_hh_ptr + indices, mask=mask, other=0.0)
            b_ih = tl.load(b_ih_ptr + indices, mask=mask, other=0.0)
            b_hh = tl.load(b_hh_ptr + indices, mask=mask, other=0.0)

            update = x * W_ih + h_prev * W_hh + b_ih + b_hh
            update = tl.sigmoid(update)
            tl.store(output_ptr + indices, update, mask=mask)

        return gru_update_kernel

    def _make_gru_reset_kernel(self):
        @triton.jit
        def gru_reset_kernel(
            x_ptr,  # Pointer to input
            h_prev_ptr,  # Pointer to previous hidden state
            W_ih_ptr,  # Pointer to input-hidden weights
            W_hh_ptr,  # Pointer to hidden-hidden weights
            b_ih_ptr,  # Pointer to input bias
            b_hh_ptr,  # Pointer to hidden bias
            seq_len,  # Sequence length
            batch_size,  # Batch size
            hidden_size,  # Hidden size
            num_layers,  # Number of layers
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            offset = pid * BLOCK_SIZE
            mask = offset + tl.arange(0, BLOCK_SIZE) < seq_len * batch_size * hidden_size
            indices = offset + tl.arange(0, BLOCK_SIZE)
            idx = indices % hidden_size
            batch_idx = (indices // hidden_size) % batch_size
            seq_idx = (indices // (batch_size * hidden_size)) % seq_len
            layer_idx = (indices // (seq_len * batch_size * hidden_size)) % num_layers

            x = tl.load(x_ptr + indices, mask=mask, other=0.0)
            h_prev = tl.load(h_prev_ptr + indices, mask=mask, other=0.0)

            # Compute reset gate
            W_ih = tl.load(W_ih_ptr + indices, mask=mask, other=0.0)
            W_hh = tl.load(W_hh_ptr + indices, mask=mask, other=0.0)
            b_ih = tl.load(b_ih_ptr + indices, mask=mask, other=0.0)
            b_hh = tl.load(b_hh_ptr + indices, mask=mask, other=0.0)

            reset = x * W_ih + h_prev * W_hh + b_ih + b_hh
            reset = tl.sigmoid(reset)
            tl.store(output_ptr + indices, reset, mask=mask)

        return gru_reset_kernel

    def _make_gru_hidden_kernel(self):
        @triton.jit
        def gru_hidden_kernel(
            x_ptr,  # Pointer to input
            h_prev_ptr,  # Pointer to previous hidden state
            W_ih_ptr,  # Pointer to input-hidden weights
            W_hh_ptr,  # Pointer to hidden-hidden weights
            b_ih_ptr,  # Pointer to input bias
            b_hh_ptr,  # Pointer to hidden bias
            seq_len,  # Sequence length
            batch_size,  # Batch size
            hidden_size,  # Hidden size
            num_layers,  # Number of layers
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            offset = pid * BLOCK_SIZE
            mask = offset + tl.arange(0, BLOCK_SIZE) < seq_len * batch_size * hidden_size
            indices = offset + tl.arange(0, BLOCK_SIZE)
            idx = indices % hidden_size
            batch_idx = (indices // hidden_size) % batch_size
            seq_idx = (indices // (batch_size * hidden_size)) % seq_len
            layer_idx = (indices // (seq_len * batch_size * hidden_size)) % num_layers

            x = tl.load(x_ptr + indices, mask=mask, other=0.0)
            h_prev = tl.load(h_prev_ptr + indices, mask=mask, other=0.0)

            # Compute hidden state
            W_ih = tl.load(W_ih_ptr + indices, mask=mask, other=0.0)
            W_hh = tl.load(W_hh_ptr + indices, mask=mask, other=0.0)
            b_ih = tl.load(b_ih_ptr + indices, mask=mask, other=0.0)
            b_hh = tl.load(b_hh_ptr + indices, mask=mask, other=0.0)

            hidden = x * W_ih + h_prev * W_hh + b_ih + b_hh
            hidden = tl.tanh(hidden)
            tl.store(output_ptr + indices, hidden, mask=mask)

        return gru_hidden_kernel

    def forward(self, x, h0):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, input_size = x.size()
        num_directions = 1
        total_hidden_size = num_directions * self.hidden_size
        output = torch.zeros(seq_len, batch_size, total_hidden_size, device=x.device)
        h_n = torch.zeros(num_directions * self.num_layers, batch_size, self.hidden_size, device=x.device)

        # Initialize hidden states
        if h0 is None:
            h0 = torch.zeros(num_directions * self.num_layers, batch_size, self.hidden_size, device=x.device)

        for layer in range(self.num_layers):
            for seq_idx in range(seq_len):
                x_t = x[seq_idx]
                h_prev = h0[layer]
                # Compute update gate
                update = self._gru_update(x_t, h_prev, self.W_ih, self.W_hh, self.b_ih, self.b_hh, seq_len, batch_size, self.hidden_size, self.num_layers, 128)
                # Compute reset gate
                reset = self._gru_reset(x_t, h_prev, self.W_ih, self.W_hh, self.b_ih, self.b_hh, seq_len, batch_size, self.hidden_size, self.num_layers, 128)
                # Compute hidden state
                hidden = self._gru_hidden(x_t, h_prev, self.W_ih, self.W_hh, self.b_ih, self.b_hh, seq_len, batch_size, self.hidden_size, self.num_layers, 128)
                # Compute final output
                h_t = update * hidden + (1 - update) * h_prev
                output[seq_idx] = h_t
                h0[layer] = h_t

        if self.batch_first:
            output = output.transpose(0, 1)
        return output, h0