import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_add_tanh_kernel(
    i2h_weight_ptr, i2h_bias_ptr,
    h2o_weight_ptr, h2o_bias_ptr,
    x_ptr, h_ptr,
    out_h_ptr, out_o_ptr,
    input_size: tl.constexpr, hidden_size: tl.constexpr, output_size: tl.constexpr,
    batch_size: tl.constexpr,
    stride_xb, stride_hb,
    stride_i2h_w, stride_i2h_k,
    stride_h2o_w, stride_h2o_k,
    stride_oh_b, stride_oo_b,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_HIDDEN: tl.constexpr,
    BLOCK_SIZE_OUTPUT: tl.constexpr,
):
    # Batch block
    batch_pid = tl.program_id(0)
    offset_batch = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    batch_mask = offset_batch < batch_size

    # Hidden block
    hidden_pid = tl.program_id(1)
    offset_hidden = hidden_pid * BLOCK_SIZE_HIDDEN + tl.arange(0, BLOCK_SIZE_HIDDEN)
    hidden_mask = offset_hidden < hidden_size

    # Output block for h2o
    output_pid = tl.program_id(2)
    offset_output = output_pid * BLOCK_SIZE_OUTPUT + tl.arange(0, BLOCK_SIZE_OUTPUT)
    output_mask = offset_output < output_size

    # Pointers to current batch slice
    x_batch_ptr = x_ptr + offset_batch[:, None] * stride_xb
    h_batch_ptr = h_ptr + offset_batch[:, None] * stride_hb

    # Load input and hidden
    x = tl.load(x_batch_ptr, mask=batch_mask[:, None], other=0.0)
    h = tl.load(h_batch_ptr, mask=batch_mask[:, None], other=0.0)

    # Concatenate x and h -> shape (batch, input_size + hidden_size)
    total_input_size = input_size + hidden_size
    xh = tl.zeros((BLOCK_SIZE_BATCH, total_input_size), dtype=tl.float32)
    xh = tl.store(xh, x, mask=batch_mask[:, None])
    xh = tl.store(xh[:, input_size:], h, mask=batch_mask[:, None])

    # Matmul: (batch, total_input_size) @ (total_input_size, hidden_size) -> (batch, hidden_size)
    acc_h = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_HIDDEN), dtype=tl.float32)
    for k_block in range(0, total_input_size, BLOCK_SIZE_HIDDEN):
        offset_k = k_block + tl.arange(0, BLOCK_SIZE_HIDDEN)
        k_mask = offset_k < total_input_size
        xh_block = tl.load(xh[:, offset_k], mask=batch_mask[:, None] & k_mask[None, :], other=0.0)
        w1_block = tl.load(
            i2h_weight_ptr + offset_k[None, :] * stride_i2h_k + offset_hidden[:, None] * stride_i2h_w,
            mask=k_mask[None, :] & hidden_mask[:, None], other=0.0
        )
        acc_h += tl.dot(xh_block, w1_block)
    
    # Add bias and tanh
    bias_h = tl.load(i2h_bias_ptr + offset_hidden, mask=hidden_mask, other=0.0)
    acc_h += bias_h[None, :]
    new_hidden = tl.tanh(acc_h)

    # Store new hidden state
    h_out_ptr = out_h_ptr + offset_batch[:, None] * stride_oh_b + offset_hidden[None, :] * 1
    tl.store(h_out_ptr, new_hidden, mask=batch_mask[:, None] & hidden_mask[None, :])

    # Matmul for h2o: (batch, hidden_size) @ (hidden_size, output_size) -> (batch, output_size)
    acc_o = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_OUTPUT), dtype=tl.float32)
    for k_block in range(0, hidden_size, BLOCK_SIZE_OUTPUT):
        offset_k = k_block + tl.arange(0, BLOCK_SIZE_OUTPUT)
        k_mask = offset_k < hidden_size
        h_block = tl.load(new_hidden[:, offset_k], mask=batch_mask[:, None] & k_mask[None, :], other=0.0)
        w2_block = tl.load(
            h2o_weight_ptr + offset_k[None, :] * stride_h2o_k + offset_output[:, None] * stride_h2o_w,
            mask=k_mask[None, :] & output_mask[:, None], other=0.0
        )
        acc_o += tl.dot(h_block, w2_block)

    # Add output bias
    bias_o = tl.load(h2o_bias_ptr + offset_output, mask=output_mask, other=0.0)
    output = acc_o + bias_o[None, :]

    # Store output
    o_out_ptr = out_o_ptr + offset_batch[:, None] * stride_oo_b + offset_output[None, :] * 1
    tl.store(o_out_ptr, output, mask=batch_mask[:, None] & output_mask[None, :])


def fused_i2h_h2o_tanh(
    x: torch.Tensor,
    h: torch.Tensor,
    i2h_weight: torch.Tensor,
    i2h_bias: torch.Tensor,
    h2o_weight: torch.Tensor,
    h2o_bias: torch.Tensor,
):
    batch_size, _ = x.shape
    hidden_size = i2h_bias.shape[0]
    output_size = h2o_bias.shape[0]
    input_size = x.shape[1] - hidden_size  # x includes [input; hidden]

    # Output tensors
    new_hidden = torch.empty((batch_size, hidden_size), device=x.device, dtype=x.dtype)
    output = torch.empty((batch_size, output_size), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(batch_size, meta['BLOCK_SIZE_BATCH']),
            triton.cdiv(hidden_size, meta['BLOCK_SIZE_HIDDEN']),
            triton.cdiv(output_size, meta['BLOCK_SIZE_OUTPUT']),
        )

    matmul_add_tanh_kernel[grid](
        i2h_weight, i2h_bias,
        h2o_weight, h2o_bias,
        x, h,
        new_hidden, output,
        input_size, hidden_size, output_size, batch_size,
        x.stride(0), h.stride(0),
        i2h_weight.stride(0), i2h_weight.stride(1),
        h2o_weight.stride(0), h2o_weight.stride(1),
        new_hidden.stride(0), output.stride(0),
        BLOCK_SIZE_BATCH=64,
        BLOCK_SIZE_HIDDEN=64,
        BLOCK_SIZE_OUTPUT=64,
    )
    return new_hidden, output


class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, _ = x.size()
        hidden = h0.to(x.device)
        outputs = []

        # Extract weights and biases once outside the loop
        i2h_weight = self.i2h.weight.t().contiguous()  # (hidden_size, input_size + hidden_size)
        i2h_bias = self.i2h.bias.contiguous()
        h2o_weight = self.h2o.weight.t().contiguous()  # (output_size, hidden_size)
        h2o_bias = self.h2o.bias.contiguous()

        for t in range(seq_len):
            # Concatenate input and hidden state
            combined = torch.cat((x[t], hidden), dim=1)  # (batch_size, input_size + hidden_size)
            # Fused i2h (linear + tanh) and h2o (linear)
            hidden, output = fused_i2h_h2o_tanh(
                combined, hidden,
                i2h_weight, i2h_bias,
                h2o_weight, h2o_bias
            )
            outputs.append(output)

        return torch.stack(outputs, dim=0)