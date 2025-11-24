import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_sigmoid_tanh_kernel(
    x_ptr,  # pointer to input (reset gate)
    y_ptr,  # pointer to input (candidate activation)
    h_prev_ptr,  # pointer to previous hidden state
    out_ptr,  # pointer to output hidden state
    n_elements,  # total number of elements
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block handles a slice across batch and hidden dimensions
    pid_b = tl.program_id(0)  # batch dimension
    pid_l = tl.program_id(1)  # layer dimension

    # Compute offsets
    batch_offset = pid_b * hidden_size
    layer_offset = pid_l * hidden_size
    offset = layer_offset + batch_offset

    # Stride for accessing gates (each gate has size hidden_size)
    stride = hidden_size

    # Pointers to reset gate (x), candidate (y), and previous hidden (h_prev)
    x_ptrs = x_ptr + offset + tl.arange(0, BLOCK_SIZE)
    y_ptrs = y_ptr + offset + tl.arange(0, BLOCK_SIZE)
    h_prev_ptrs = h_prev_ptr + offset + tl.arange(0, BLOCK_SIZE)
    out_ptrs = out_ptr + offset + tl.arange(0, BLOCK_SIZE)

    # Load reset gate (apply sigmoid)
    r = x_ptrs
    r_data = tl.load(r, mask=tl.arange(0, BLOCK_SIZE) < hidden_size, other=0.0)
    r_sig = tl.sigmoid(r_data)

    # Load candidate and previous hidden
    h_prev = tl.load(h_prev_ptrs, mask=tl.arange(0, BLOCK_SIZE) < hidden_size, other=0.0)
    y_data = tl.load(y_ptrs, mask=tl.arange(0, BLOCK_SIZE) < hidden_size, other=0.0)

    # Compute candidate activation: tanh(W_c * x + U_c * (r * h_prev))
    candidate = tl.tanh(y_data)

    # Compute new hidden: (1 - z) * h_prev + z * candidate (here z is update gate, but we fuse sigmoid/tanh)
    # In standard GRU: h_t = (1 - z) * h_prev + z * tanh(candidate + r * h_prev)
    # But here we assume z (update gate) is fused in the same way — this kernel fuses sigmoid (for r) and tanh (for candidate)
    # However, in full GRU we need z. So this kernel assumes r and candidate are precomputed, and we're doing the final step.
    # Let's instead assume we are fusing the entire GRU cell forward pass per time step and layer.

    # Actually, we'll refactor: this kernel will compute one time step of GRU for all layers and batch
    # But for now, let's focus on fusing sigmoid + elementwise mul + tanh + hidden update


# Instead, we implement a fused GRU step kernel that processes one time step across all layers and batch
@triton.jit
def gru_step_kernel(
    # Inputs: concatenated weights (3 * hidden_size, input_size + hidden_size) per layer
    weight_ih_ptr,
    weight_hh_ptr,
    bias_ih_ptr,
    bias_hh_ptr,
    x_ptr,  # input at time t: (batch_size, input_size)
    h_ptr,  # previous hidden: (num_layers, batch_size, hidden_size)
    out_h_ptr,  # output hidden: same shape as h_ptr
    # Metadata
    input_size: tl.constexpr,
    hidden_size: tl.constexpr,
    batch_size: tl.constexpr,
    num_layers: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_HIDDEN: tl.constexpr,
):
    # We launch grid of (batch_blocks, layer_blocks, hidden_blocks)
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)

    # Batch block start
    batch_start = pid_b * BLOCK_SIZE_BATCH
    batch_end = tl.minimum(batch_start + BLOCK_SIZE_BATCH, batch_size)
    batch_offs = tl.arange(0, BLOCK_SIZE_BATCH)

    # Hidden block
    hid_start = 0
    hid_end = hidden_size
    hid_offs = tl.arange(0, BLOCK_SIZE_HIDDEN)

    # Layer offset
    layer_offset = pid_l * (3 * hidden_size) * (input_size + hidden_size)

    # Pointers to weights
    w_ih_ptrs = weight_ih_ptr + layer_offset + (tl.arange(0, 3 * BLOCK_SIZE_HIDDEN)[:, None] * (input_size + hidden_size) + tl.arange(0, input_size + hidden_size)[None, :])
    w_hh_ptrs = weight_hh_ptr + layer_offset + (tl.arange(0, 3 * BLOCK_SIZE_HIDDEN)[:, None] * (input_size + hidden_size) + tl.arange(0, input_size + hidden_size)[None, :])

    # Bias pointers
    b_ih_ptrs = bias_ih_ptr + pid_l * 3 * hidden_size + tl.arange(0, 3 * BLOCK_SIZE_HIDDEN)
    b_hh_ptrs = bias_hh_ptr + pid_l * 3 * hidden_size + tl.arange(0, 3 * BLOCK_SIZE_HIDDEN)

    # Input pointer
    x_ptrs = x_ptr + batch_start * input_size + tl.arange(0, input_size)[None, :]
    h_ptrs = h_ptr + pid_l * batch_size * hidden_size + batch_start * hidden_size + tl.arange(0, BLOCK_SIZE_BATCH)[:, None] * hidden_size + tl.arange(0, hidden_size)[None, :]

    # Output pointer
    out_h_ptrs = out_h_ptr + pid_l * batch_size * hidden_size + batch_start * hidden_size + tl.arange(0, BLOCK_SIZE_BATCH)[:, None] * hidden_size + tl.arange(0, hidden_size)[None, :]

    # We process one time step per kernel launch
    # For each layer, we compute: W_ih @ x + b_ih + W_hh @ h + b_hh -> split into r, z, n
    # Then: n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
    # Then: h_new = (1 - z) * h + z * n

    # However, due to complexity of full GRU kernel, we instead use Triton's matmul for fused GEMM + activation

    # Instead, we leverage existing optimized kernels and focus on fusing the GRU cell's inner operations per layer

    # Given the complexity, we instead replace the inner GRU cell with a fused kernel per time step and layer
    # But full implementation of multi-layer GRU with sequence in Triton is very complex
    # So we focus on fusing the pointwise operations in the GRU: sigmoid, tanh, and hidden update

    # We assume the linear projections are done via efficient torch.matmul (which uses cuBLAS)
    # Then we fuse the elementwise GRU update: r, z, n gates -> h_new

    # So this kernel fuses the nonlinear update part of GRU cell


# Due to the complexity of implementing full GRU with sequence loop and layer loop in Triton,
# and since the main bottleneck in GRU is often the sequential recurrence and matmuls,
# we instead focus on fusing the elementwise operations within the GRU cell.

@triton.jit
def fused_gru_cell_kernel(
    # Inputs: pre-activated gates: [r, z, n] each of size (batch_size, hidden_size)
    rz_ptr,  # shape: (batch_size, 3 * hidden_size), concatenated
    h_prev_ptr,  # shape: (batch_size, hidden_size)
    out_ptr,  # output hidden state
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size

    # Pointers to r, z, n (each of size hidden_size)
    r_ptr = rz_ptr + offsets
    z_ptr = rz_ptr + hidden_size + offsets
    n_ptr = rz_ptr + 2 * hidden_size + offsets
    h_prev_ptr = h_prev_ptr + offsets

    # Load
    r = tl.load(r_ptr, mask=mask, other=0.0)
    z = tl.load(z_ptr, mask=mask, other=0.0)
    n = tl.load(n_ptr, mask=mask, other=0.0)
    h_prev = tl.load(h_prev_ptr, mask=mask, other=0.0)

    # Apply sigmoid to r and z
    r_sig = tl.sigmoid(r)
    z_sig = tl.sigmoid(z)

    # Compute candidate: tanh(n + r_sig * h_prev)
    n_tanh = tl.tanh(n + r_sig * h_prev)

    # Compute new hidden: (1 - z_sig) * h_prev + z_sig * n_tanh
    h_new = (1.0 - z_sig) * h_prev + z_sig * n_tanh

    # Store
    tl.store(out_ptr + offsets, h_new, mask=mask)


def fused_gru_cell(rz: torch.Tensor, h_prev: torch.Tensor):
    """
    Fused GRU cell that computes:
        r = sigmoid(r_lin)
        z = sigmoid(z_lin)
        n = tanh(n_lin + r * h_prev)
        h_new = (1 - z) * h_prev + z * n
    Input rz: (batch_size, 3 * hidden_size)
    h_prev: (batch_size, hidden_size)
    Output: h_new (batch_size, hidden_size)
    """
    batch_size, total_hidden = rz.shape
    hidden_size = total_hidden // 3
    assert total_hidden == 3 * hidden_size

    h_new = torch.empty_like(h_prev)

    def grid(meta): return ((hidden_size + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], batch_size)

    fused_gru_cell_kernel[grid](
        rz, h_prev, h_new,
        batch_size=batch_size,
        hidden_size=hidden_size,
        BLOCK_SIZE=1024,
    )
    return h_new


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        # Replace nn.GRU with manual weights and fused kernel
        self.weight_ih_list = nn.ParameterList()
        self.weight_hh_list = nn.ParameterList()
        self.bias_ih_list = nn.ParameterList() if bias else None
        self.bias_hh_list = nn.ParameterList() if bias else None

        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            weight_ih = nn.Parameter(torch.empty(3 * hidden_size, in_size))
            weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
            nn.init.xavier_uniform_(weight_ih)
            nn.init.xavier_uniform_(weight_hh)
            self.weight_ih_list.append(weight_ih)
            self.weight_hh_list.append(weight_hh)

            if bias:
                bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
                bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
                nn.init.zeros_(bias_ih)
                nn.init.zeros_(bias_hh)
                self.bias_ih_list.append(bias_ih)
                self.bias_hh_list.append(bias_hh)

    def gru_cell(self, x, h_prev, weight_ih, weight_hh, bias_ih=None, bias_hh=None):
        # x: (batch_size, input_size)
        # h_prev: (batch_size, hidden_size)
        # Compute W_ih @ x + b_ih
        rzx = F.linear(x, weight_ih, bias_ih)
        rzh = F.linear(h_prev, weight_hh, bias_hh)
        rz = rzx + rzh  # (batch_size, 3 * hidden_size)
        h_new = fused_gru_cell(rz, h_prev)
        return h_new

    def forward(self, x, h0):
        # x: (seq_len, batch_size, input_size) if not batch_first
        if self.batch_first:
            x = x.transpose(0, 1)  # to (seq_len, batch_size, input_size)

        seq_len, batch_size, _ = x.shape
        h = h0

        for t in range(seq_len):
            x_t = x[t]  # (batch_size, input_size)
            h_t = []
            for layer in range(self.num_layers):
                h_prev = h[layer]  # (batch_size, hidden_size)
                weight_ih = self.weight_ih_list[layer]
                weight_hh = self.weight_hh_list[layer]
                bias_ih = self.bias_ih_list[layer] if self.bias else None
                bias_hh = self.bias_hh_list[layer] if self.bias else None
                h_layer = self.gru_cell(x_t, h_prev, weight_ih, weight_hh, bias_ih, bias_hh)
                h_t.append(h_layer)
                x_t = h_layer  # output of layer t becomes input to next layer
            h = torch.stack(h_t)

        # Return final hidden state
        return h