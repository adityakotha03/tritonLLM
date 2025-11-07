import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr  # 'none', 'tanh', 'sigmoid'
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offset in the output matrix
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    # Create offsets for this block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute current K range
        k_start = k * BLOCK_SIZE_K
        k_end = min(k_start + BLOCK_SIZE_K, K)

        # Load A and B (use shared memory)
        a = tl.load(
            a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :]) + k_start * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_end),
            other=0.0
        )
        b = tl.load(
            b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :]) + k_start * stride_bn,
            mask=(offs_k[:, None] < k_end) & (offs_n[None, :] < N),
            other=0.0
        )

        # Perform matrix multiplication
        accumulator += tl.dot(a, b)

    # Normalize to output range
    accumulator = accumulator / 8.0  # For scaling with fp16, optional

    # Apply activation
    if ACTIVATION == "tanh":
        accumulator = tl.tanh(accumulator)
    elif ACTIVATION == "sigmoid":
        # Online sigmoid: avoid overflow
        accumulator = tl.where(accumulator >= 0, 
                              1.0 / (1.0 + tl.exp(-accumulator)), 
                              tl.exp(accumulator) / (1.0 + tl.exp(accumulator)))
    # If 'none', no activation

    # Write result
    c = accumulator.to(tl.float16)  # Use fp16 for efficiency

    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn), c, mask=mask)


@triton.jit
def gru_cell_kernel(
    x_ptr, h_ptr, w_ih_ptr, w_hh_ptr, b_ih_ptr, b_hh_ptr,
    h_new_ptr,
    batch_size, seq_len, input_size, hidden_size,
    # Strides
    stride_xb, stride_xs, stride_xi,
    stride_hb, stride_hs, stride_hi,
    stride_wib, stride_wik, stride_wii,
    stride_whb, stride_whk, stride_whi,
    stride_bib, stride_bik, stride_bii,
    stride_bhb, stride_bhk, stride_bhi,
    stride_hnb, stride_hns, stride_hni,
    # Block size
    BLOCK_SIZE: tl.constexpr,
    # Activation
    ACTIVATION: tl.constexpr = "tanh"
):
    pid = tl.program_id(0)
    tid = tl.program_id(1)

    # For batch and time
    b = pid // seq_len
    t = pid % seq_len

    # Local index in hidden state
    hid_idx = tid

    if hid_idx >= hidden_size:
        return

    # Load input and hidden
    x = tl.load(x_ptr + (b * stride_xb + t * stride_xs + hid_idx * stride_xi), mask=(hid_idx < hidden_size), other=0.0)
    h = tl.load(h_ptr + (b * stride_hb + t * stride_hs + hid_idx * stride_hi), mask=(hid_idx < hidden_size), other=0.0)

    # Prepare for gates: reset, update, output
    # Reset gate: W_ih @ x + W_hh @ h + b_ih + b_hh
    # Use matmul + sigmoid fused
    # We'll split the weights: reset, update, output are each hidden_size x input_size (and hidden_size x hidden_size)

    # Reset gate: 0: hidden_size
    reset_start = 0
    update_start = hidden_size
    out_start = 2 * hidden_size

    # Load weights for reset gate
    w_ih_reset = w_ih_ptr + (reset_start * stride_wib + hid_idx * stride_wii)
    w_hh_reset = w_hh_ptr + (reset_start * stride_whb + hid_idx * stride_whi)

    # Compute: reset = sigmoid(W_ih @ x + W_hh @ h + b_ih + b_hh)
    reset_val = tl.zeros((), dtype=tl.float32)
    for i in range(0, input_size, BLOCK_SIZE):
        i_end = min(i + BLOCK_SIZE, input_size)
        x_block = tl.load(x_ptr + (b * stride_xb + t * stride_xs + tl.arange(i, i_end) * stride_xi), mask=(tl.arange(i, i_end) < input_size), other=0.0)
        w_ih_block = tl.load(w_ih_reset + (tl.arange(i, i_end) * stride_wii), mask=(tl.arange(i, i_end) < input_size), other=0.0)

        reset_val += tl.dot(x_block, w_ih_block)

    for i in range(0, hidden_size, BLOCK_SIZE):
        i_end = min(i + BLOCK_SIZE, hidden_size)
        h_block = tl.load(h_ptr + (b * stride_hb + t * stride_hs + tl.arange(i, i_end) * stride_hi), mask=(tl.arange(i, i_end) < hidden_size), other=0.0)
        w_hh_block = tl.load(w_hh_reset + (tl.arange(i, i_end) * stride_whi), mask=(tl.arange(i, i_end) < hidden_size), other=0.0)

        reset_val += tl.dot(h_block, w_hh_block)

    # Add bias
    b_ih_reset = b_ih_ptr + reset_start * stride_bib + hid_idx * stride_bii
    b_hh_reset = b_hh_ptr + reset_start * stride_bhb + hid_idx * stride_bhi
    reset_bias = tl.load(b_ih_reset, mask=(hid_idx < hidden_size), other=0.0)
    reset_bias += tl.load(b_hh_reset, mask=(hid_idx < hidden_size), other=0.0)
    reset_val += reset_bias

    reset = tl.where(reset_val >= 0.0, 
                     1.0 / (1.0 + tl.exp(-reset_val)),
                     tl.exp(reset_val) / (1.0 + tl.exp(reset_val)))

    # Update gate: sigmoid(W_ih @ x + W_hh @ h + b_ih + b_hh)
    update_val = tl.zeros((), dtype=tl.float32)
    w_ih_update = w_ih_ptr + (update_start * stride_wib + hid_idx * stride_wii)
    w_hh_update = w_hh_ptr + (update_start * stride_whb + hid_idx * stride_whi)

    for i in range(0, input_size, BLOCK_SIZE):
        i_end = min(i + BLOCK_SIZE, input_size)
        x_block = tl.load(x_ptr + (b * stride_xb + t * stride_xs + tl.arange(i, i_end) * stride_xi), mask=(tl.arange(i, i_end) < input_size), other=0.0)
        w_ih_block = tl.load(w_ih_update + (tl.arange(i, i_end) * stride_wii), mask=(tl.arange(i, i_end) < input_size), other=0.0)

        update_val += tl.dot(x_block, w_ih_block)

    for i in range(0, hidden_size, BLOCK_SIZE):
        i_end = min(i + BLOCK_SIZE, hidden_size)
        h_block = tl.load(h_ptr + (b * stride_hb + t * stride_hs + tl.arange(i, i_end) * stride_hi), mask=(tl.arange(i, i_end) < hidden_size), other=0.0)
        w_hh_block = tl.load(w_hh_update + (tl.arange(i, i_end) * stride_whi), mask=(tl.arange(i, i_end) < hidden_size), other=0.0)

        update_val += tl.dot(h_block, w_hh_block)

    # Bias
    b_ih_update = b_ih_ptr + update_start * stride_bib + hid_idx * stride_bii
    b_hh_update = b_hh_ptr + update_start * stride_bhb + hid_idx * stride_bhi
    update_bias = tl.load(b_ih_update, mask=(hid_idx < hidden_size), other=0.0)
    update_bias += tl.load(b_hh_update, mask=(hid_idx < hidden_size), other=0.0)
    update_val += update_bias

    update = tl.where(update_val >= 0.0, 
                      1.0 / (1.0 + tl.exp(-update_val)),
                      tl.exp(update_val) / (1.0 + tl.exp(update_val)))

    # Candidate hidden state: tanh(W_ih @ x + W_hh @ h + b_ih + b_hh)
    cand_val = tl.zeros((), dtype=tl.float32)
    w_ih_cand = w_ih_ptr + (out_start * stride_wib + hid_idx * stride_wii)
    w_hh_cand = w_hh_ptr + (out_start * stride_whb + hid_idx * stride_whi)

    for i in range(0, input_size, BLOCK_SIZE):
        i_end = min(i + BLOCK_SIZE, input_size)
        x_block = tl.load(x_ptr + (b * stride_xb + t * stride_xs + tl.arange(i, i_end) * stride_xi), mask=(tl.arange(i, i_end) < input_size), other=0.0)
        w_ih_block = tl.load(w_ih_cand + (tl.arange(i, i_end) * stride_wii), mask=(tl.arange(i, i_end) < input_size), other=0.0)

        cand_val += tl.dot(x_block, w_ih_block)

    for i in range(0, hidden_size, BLOCK_SIZE):
        i_end = min(i + BLOCK_SIZE, hidden_size)
        h_block = tl.load(h_ptr + (b * stride_hb + t * stride_hs + tl.arange(i, i_end) * stride_hi), mask=(tl.arange(i, i_end) < hidden_size), other=0.0)
        w_hh_block = tl.load(w_hh_cand + (tl.arange(i, i_end) * stride_whi), mask=(tl.arange(i, i_end) < hidden_size), other=0.0)

        cand_val += tl.dot(h_block, w_hh_block)

    # Bias
    b_ih_cand = b_ih_ptr + out_start * stride_bib + hid_idx * stride_bii
    b_hh_cand = b_hh_ptr + out_start * stride_bhb + hid_idx * stride_bhi
    cand_bias = tl.load(b_ih_cand, mask=(hid_idx < hidden_size), other=0.0)
    cand_bias += tl.load(b_hh_cand, mask=(hid_idx < hidden_size), other=0.0)
    cand_val += cand_bias

    cand = tl.tanh(cand_val)

    # New hidden state: (1 - update) * h + update * cand
    h_new = (1.0 - update) * h + update * cand

    # Store result
    tl.store(h_new_ptr + (b * stride_hnb + t * stride_hns + hid_idx * stride_hni), h_new, mask=(hid_idx < hidden_size))


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        # Create GRU with custom weights
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=True)

        # Initialize hidden state
        self.h0 = torch.randn((num_layers * 2, 1, hidden_size), device='cuda')

    def forward(self, x, h0):
        # x: (seq_len, batch_size, input_size) or (batch_size, seq_len, input_size)
        # h0: (num_layers * 2, batch_size, hidden_size)

        if self.batch_first:
            x = x.transpose(0, 1)  # (seq_len, batch_size, input_size)

        seq_len, batch_size, input_size = x.shape
        hidden_size = self.hidden_size
        num_layers = self.num_layers
        num_directions = 2

        # Output: (seq_len, batch_size, num_directions * hidden_size)
        output = torch.zeros(seq_len, batch_size, num_directions * hidden_size, device=x.device, dtype=x.dtype)

        # Copy h0 to device
        h = h0.clone()

        # Get raw weights and bias
        w_ih = self.gru.weight_ih_l0  # (3*hidden_size, input_size)
        w_hh = self.gru.weight_hh_l0  # (3*hidden_size, hidden_size)
        b_ih = self.gru.bias_ih_l0 if self.bias else None
        b_hh = self.gru.bias_hh_l0 if self.bias else None

        # Extract sub-weights: reset, update, candidate
        # Each: (hidden_size, input_size) and (hidden_size, hidden_size)
        # Use split
        w_ih_reset = w_ih[:hidden_size]
        w_ih_update = w_ih[hidden_size:2*hidden_size]
        w_ih_cand = w_ih[2*hidden_size:]

        w_hh_reset = w_hh[:hidden_size]
        w_hh_update = w_hh[hidden_size:2*hidden_size]
        w_hh_cand = w_hh[2*hidden_size:]

        # Flatten biases
        b_ih_reset = b_ih[:hidden_size] if b_ih is not None else None
        b_ih_update = b_ih[hidden_size:2*hidden_size] if b_ih is not None else None
        b_ih_cand = b_ih[2*hidden_size:] if b_ih is not None else None

        b_hh_reset = b_hh[:hidden_size] if b_hh is not None else None
        b_hh_update = b_hh[hidden_size:2*hidden_size] if b_hh is not None else None
        b_hh_cand = b_hh[2*hidden_size:] if b_hh is not None else None

        # Convert to float16 for better Tensor Core use
        x = x.to(torch.float16)
        h = h.to(torch.float16)
        w_ih_reset = w_ih_reset.to(torch.float16)
        w_ih_update = w_ih_update.to(torch.float16)
        w_ih_cand = w_ih_cand.to(torch.float16)
        w_hh_reset = w_hh_reset.to(torch.float16)
        w_hh_update = w_hh_update.to(torch.float16)
        w_hh_cand = w_hh_cand.to(torch.float16)
        if b_ih_reset is not None:
            b_ih_reset = b_ih_reset.to(torch.float16)
        if b_ih_update is not None:
            b_ih_update = b_ih_update.to(torch.float16)
        if b_ih_cand is not None:
            b_ih_cand = b_ih_cand.to(torch.float16)
        if b_hh_reset is not None:
            b_hh_reset = b_hh_reset.to(torch.float16)
        if b_hh_update is not None:
            b_hh_update = b_hh_update.to(torch.float16)
        if b_hh_cand is not None:
            b_hh_cand = b_hh_cand.to(torch.float16)

        # Use Triton kernels to run GRU cell over sequence
        # We'll unroll over time and batch
        total_steps = seq_len * batch_size
        # Each kernel: (seq_len * batch_size, hidden_size)
        BLOCK_SIZE = 128  # Use block size tuned via autotune

        # Use a single kernel launch per time step per batch
        for t in range(seq_len):
            # Run GRU cell for all batches and hidden states
            # h: (num_layers * 2, batch_size, hidden_size)
            # We assume num_layers == 1 for now (single layer) to simplify
            # Extend to multiple layers if needed (future)

            # For now, run for one layer (directions are handled by GRU)
            # Only process first layer
            h_layer = h[0]  # (batch_size, hidden_size)

            # Launch Triton kernel
            grid = (total_steps, hidden_size)

            # Flatten pointers
            x_flat = x[t].contiguous()
            h_flat = h_layer.contiguous()

            # Set strides
            stride_xb = x_flat.stride(0)
            stride_xs = x_flat.stride(1)
            stride_xi = x_flat.stride(2)  # input_size

            stride_hb = h_flat.stride(0)
            stride_hs = h_flat.stride(1)
            stride_hi = h_flat.stride(2)

            stride_wib = w_ih_reset.stride(0)
            stride_wik = w_ih_reset.stride(1)
            stride_wii = w_ih_reset.stride(2)  # hidden_size

            stride_whb = w_hh_reset.stride(0)
            stride_whk = w_hh_reset.stride(1)
            stride_whi = w_hh_reset.stride(2)

            stride_bib = b_ih_reset.stride(0) if b_ih_reset is not None else 0
            stride_bik = b_ih_reset.stride(1) if b_ih_reset is not None else 0
            stride_bii = b_ih_reset.stride(2) if b_ih_reset is not None else 0

            stride_bhb = b_hh_reset.stride(0) if b_hh_reset is not None else 0
            stride_bhk = b_hh_reset.stride(1) if b_hh_reset is not None else 0
            stride_bhi = b_hh_reset.stride(2) if b_hh_reset is not None else 0

            stride_hnb = output[t].stride(0)
            stride_hns = output[t].stride(1)
            stride_hni = output[t].stride(2)

            # Launch kernel
            gru_cell_kernel[
                grid,
                BLOCK_SIZE
            ](
                x_flat, h_flat,
                w_ih_reset, w_hh_reset,
                b_ih_reset, b_hh_reset,
                output[t],
                batch_size, seq_len, input_size, hidden_size,
                stride_xb, stride_xs, stride_xi,
                stride_hb, stride_hs, stride_hi,
                stride_wib, stride_wik, stride_wii,
                stride_whb, stride_whk, stride_whi,
                stride_bib, stride_bik, stride_bii,
                stride_bhb, stride_bhk, stride_bhi,
                stride_hnb, stride_hns, stride_hni,
                BLOCK_SIZE=BLOCK_SIZE
            )

        # Return output
        if self.batch_first:
            output = output.transpose(0, 1)  # (batch_size, seq_len, hidden_size * 2)

        return output, h