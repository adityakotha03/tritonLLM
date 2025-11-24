import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_mish_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, hidden_size, input_size,
    scale_factor,
    clamp_min, clamp_max,
    X_stride, W_stride, O_stride,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    x_ptrs = x_ptr + offs_m[:, None] * X_stride + offs_k[None, :] 
    w_ptrs = weight_ptr + offs_k[:, None] * W_stride + offs_n[None, :]

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, input_size, BLOCK_SIZE_K):
        x_mask = (offs_m[:, None] < batch_size) & (offs_k[None, :] < input_size)
        w_mask = (offs_k[:, None] < input_size) & (offs_n[None, :] < hidden_size)
        
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        accumulator += tl.dot(x, w)
        
        x_ptrs += BLOCK_SIZE_K
        w_ptrs += BLOCK_SIZE_K * W_stride

    c = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    bias_ptrs = bias_ptr + offs_n
    bias_mask = offs_n < hidden_size
    bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
    accumulator = accumulator + bias[None, :]

    accumulator = accumulator * scale_factor
    accumulator = accumulator + accumulator

    accumulator = tl.maximum(accumulator, clamp_min)
    accumulator = tl.minimum(accumulator, clamp_max)

    row_max = tl.max(accumulator, axis=1)
    row_max = tl.where(row_max == float('-inf'), 0.0, row_max)
    exp_x = tl.exp(accumulator - row_max[:, None])
    row_sum = tl.sum(exp_x, axis=1)
    logsumexp = row_max + tl.log(row_sum)

    logsumexp_repeated = logsumexp[:, None].broadcast_to((BLOCK_SIZE_M, BLOCK_SIZE_N))

    mish = logsumexp_repeated * tl.tanh(tl.log(1.0 + tl.exp(logsumexp_repeated)))

    out_ptrs = out_ptr + offs_m[:, None] * O_stride + offs_n[None, :]
    out_mask = (offs_m[:, None] < batch_size) & (offs_n[None, :] < hidden_size)
    tl.store(out_ptrs, mish, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        assert x.is_cuda and self.linear.weight.is_cuda and self.linear.bias.is_cuda

        batch_size, input_size = x.shape
        hidden_size = self.linear.out_features

        x = x.contiguous()
        weight = self.linear.weight.contiguous()
        bias = self.linear.bias.contiguous()

        out = torch.empty((batch_size, hidden_size), device=x.device, dtype=x.dtype)

        def grid(META):
            return (
                triton.cdiv(batch_size, META['BLOCK_SIZE_M']),
                triton.cdiv(hidden_size, META['BLOCK_SIZE_N']),
            )

        matmul_mish_kernel[grid](
            x_ptr=x,
            weight_ptr=weight,
            bias_ptr=bias,
            out_ptr=out,
            batch_size=batch_size,
            hidden_size=hidden_size,
            input_size=input_size,
            scale_factor=self.scale_factor,
            clamp_min=self.clamp_min,
            clamp_max=self.clamp_max,
            X_stride=x.stride(0),
            W_stride=weight.stride(0),
            O_stride=out.stride(0),
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
        )

        return out