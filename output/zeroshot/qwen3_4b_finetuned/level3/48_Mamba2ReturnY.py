import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_exp_mean_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl_math.exp(tmp0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp2, 0))
    tmp5 = 64.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr0 + x0, tmp6, xmask)


@triton.jit
def triton_poi_fused_exp_mean_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl_math.exp(tmp0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp2, 0))
    tmp5 = 16.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr0 + x0, tmp6, xmask)


@triton.jit
def triton_poi_fused_add_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128 % 64
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_mul_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 860480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = xindex // 1024
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tmp3 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + x2, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4 * tmp6
    tl.store(out_ptr0 + x2, tmp5, xmask)


@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8
    x1 = xindex // 8
    tmp0 = tl.load(in_ptr0 + (x2 + 218464), xmask)
    tmp1 = tl.load(in_ptr0 + (218464 + x0 + 8 * x1), xmask)
    tmp3 = tl.load(in_ptr0 + (129264 + x0 + 8 * x1), xmask)
    tmp6 = tl.load(in_ptr0 + (86800 + x0 + 8 * x1), xmask)
    tmp9 = tl.load(in_ptr0 + (54592 + x0 + 8 * x1), xmask)
    tmp12 = tl.load(in_ptr0 + (30968 + x0 + 8 * x1), xmask)
    tmp15 = tl.load(in_ptr0 + (20104 + x0 + 8 * x1), xmask)
    tmp18 = tl.load(in_ptr0 + (10240 + x0 + 8 * x1), xmask)
    tmp21 = tl.load(in_ptr0 + (184 + x0 + 8 * x1), xmask)
    tmp2 = tl_math.exp(tmp0)
    tmp4 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp6 * tmp5
    tmp8 = tmp9 * tmp7
    tmp10 = tmp12 * tmp8
    tmp11 = tmp15 * tmp10
    tmp13 = tmp18 * tmp11
    tmp14 = tmp21 * tmp13
    tmp16 = tl_math.exp(tmp14)
    tmp17 = tmp16 * tmp0
    tmp19 = tmp7 * tmp2
    tmp20 = tmp19 * tmp1
    tmp22 = tmp20 + tmp17
    tmp23 = tmp8 * tmp3
    tmp24 = tmp23 * tmp2
    tmp25 = tmp10 * tmp24
    tmp26 = tmp25 + tmp22
    tmp27 = tmp11 * tmp6
    tmp28 = tmp27 * tmp2
    tmp29 = tmp28 * tmp1
    tmp30 = tmp29 + tmp26
    tmp31 = tmp13 * tmp9
    tmp32 = tmp31 * tmp2
    tmp33 = tmp32 * tmp1
    tmp34 = tmp33 + tmp30
    tmp35 = tmp16 * tmp12
    tmp36 = tmp35 * tmp2
    tmp37 = tmp36 * tmp1
    tmp38 = tmp37 + tmp34
    tmp39 = tmp17 * tmp7
    tmp40 = tmp39 * tmp2
    tmp41 = tmp40 * tmp1
    tmp42 = tmp41 + tmp38
    tl.store(out_ptr0 + x2, tmp42, xmask)


@triton.jit
def triton_poi_fused_add_mul_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4,
    in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 8
    x2 = xindex // 128
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + x2, xmask)
    tmp10 = tl.load(in_ptr3 + x3, xmask)
    tmp14 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + x3, xmask)
    tmp23 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + x3, xmask)
    tmp1 = tl_math.exp(tmp0)
    tmp2 = tl_math.exp(tmp3)
    tmp4 = tmp1 * tmp2
    tmp5 = tmp4 * tmp6
    tmp7 = tl_math.exp(tmp5)
    tmp8 = tmp7 * tmp10
    tmp9 = tmp8 * tmp1
    tmp11 = tl_math.exp(tmp9)
    tmp12 = tmp11 * tmp14
    tmp13 = tmp12 * tmp1
    tmp15 = tmp12 * tmp18
    tmp16 = tmp15 * tmp1
    tmp17 = tmp13 + tmp16
    tmp19 = tmp15 * tmp23
    tmp20 = tmp19 * tmp1
    tmp21 = tmp20 * tmp1
    tmp22 = tmp17 + tmp21
    tmp24 = tmp17 * tmp27
    tmp25 = tmp24 * tmp1
    tmp26 = tmp25 * tmp1
    tmp28 = tmp26 + tmp22
    tl.store(out_ptr0 + x3, tmp28, xmask)


@triton.jit
def triton_poi_fused_add_mul_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4,
    in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 8
    x2 = xindex // 128
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + x2, xmask)
    tmp10 = tl.load(in_ptr3 + x3, xmask)
    tmp14 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + x3, xmask)
    tmp23 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + x3, xmask)
    tmp32 = tl.load(in_ptr8 + x0, xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr9 + x3, xmask)
    tmp1 = tl_math.exp(tmp0)
    tmp2 = tl_math.exp(tmp3)
    tmp4 = tmp1 * tmp2
    tmp5 = tmp4 * tmp6
    tmp7 = tl_math.exp(tmp5)
    tmp8 = tmp7 * tmp10
    tmp9 = tmp8 * tmp1
    tmp11 = tl_math.exp(tmp9)
    tmp12 = tmp11 * tmp14
    tmp13 = tmp12 * tmp1
    tmp15 = tmp12 * tmp18
    tmp16 = tmp15 * tmp1
    tmp17 = tmp13 + tmp16
    tmp19 = tmp15 * tmp23
    tmp20 = tmp19 * tmp1
    tmp21 = tmp20 * tmp1
    tmp22 = tmp17 + tmp21
    tmp24 = tmp17 * tmp27
    tmp25 = tmp24 * tmp1
    tmp26 = tmp25 * tmp1
    tmp28 = tmp26 + tmp22
    tmp29 = tmp11 * tmp32
    tmp30 = tmp29 * tmp1
    tmp31 = tmp30 * tmp1
    tmp33 = tmp31 + tmp28
    tmp34 = tmp29 * tmp36
    tmp35 = tmp34 * tmp1
    tmp36 = tmp35 * tmp1
    tmp37 = tmp33 + tmp36
    tl.store(out_ptr0 + x3, tmp37, xmask)


@triton.jit
def triton_poi_fused_add_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp3 = tl.load(in_ptr2 + x0, xmask)
    tmp5 = tl.load(in_ptr3 + x0, xmask)
    tmp7 = tl.load(in_ptr4 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + x0, tmp8, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13) = args
    args.clear()
    assert_size_stride(primals_1, (2048, 128, 8), (1024, 8, 1))
    assert_size_stride(primals_2, (2048, 128, 8, 16), (16384, 1024, 128, 1))
    assert_size_stride(primals_3, (2048, 128, 8, 16), (16384, 1024, 128, 1))
    assert_size_stride(primals_4, (64, 128, 8, 64), (65536, 512, 64, 1))
    assert_size_stride(primals_5, (64, 64), (64, 1))
    assert_size_stride(primals_6, (64, 64), (64, 1))
    assert_size_stride(primals_7, (64, 64), (64, 1))
    assert_size_stride(primals_8, (64, 64), (64, 1))
    assert_size_stride(primals_9, (64, 64), (64, 1))
    assert_size_stride(primals_10, (64, 64), (64, 1))
    assert_size_stride(primals_11, (64, 64), (64, 1))
    assert_size_stride(primals_12, (64, 64), (64, 1))
    assert_size_stride(primals_13, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_exp_mean_0[grid(64)](primals_5, buf0, 64, XBLOCK=
            64, num_warps=1, num_stages=1)
        buf1 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        triton_poi_fused_exp_mean_0[grid(16)](primals_6, buf1, 16, XBLOCK=
            16, num_warps=1, num_stages=1)
        buf2 = empty_strided_cuda((2048, 128, 8, 64), (65536, 512, 64, 1),
            torch.float32)
        extern_kernels.bmm(primals_1, primals_4, out=buf2)
        del primals_1
        del primals_4
        buf3 = empty_strided_cuda((2048, 128, 8, 16), (16384, 1024, 128, 1),
            torch.float32)
        extern_kernels.bmm(buf2, primals_2, out=buf3)
        del primals_2
        buf4 = empty_strided_cuda((2048, 128, 8), (1024, 8, 1), torch.float32)
        extern_kernels.bmm(primals_1, primals_3, out=buf4)
        del primals_1
        del primals_3
        buf5 = empty_strided_cuda((64, 8, 64), (512, 64, 1), torch.float32)
        extern_kernels.bmm(buf3, buf4, out=buf5)
        del buf4
        buf6 = empty_strided_cuda((2048, 128, 8), (1024, 8, 1), torch.float32)
        extern_kernels.bmm(primals_8, primals_5, out=buf6)
        del primals_5
        buf7 = empty_strided_cuda((2048, 128, 1), (128, 1, 1024), torch.float32)
        extern_kernels.bmm(primals_7, primals_9, out=buf7)
        del primals_7
        buf8 = empty_strided_cuda((2048, 128, 8, 1), (1024, 8, 1, 1024),
            torch.float32)
        extern_kernels.bmm(primals_8, primals_6, out=buf8)
        del primals_6
        buf9 = empty_strided_cuda((2048, 128, 8, 1), (1024, 8, 1, 1024),
            torch.float32)
        extern_kernels.bmm(primals_9, primals_8, out=buf9)
        del primals_8
        buf10 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        extern_kernels.mm(primals_10, primals_11, out=buf10)
        del primals_10
        buf11 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        extern_kernels.mm(primals_11, primals_12, out=buf11)
        del primals_11
        buf12 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        extern_kernels.mm(primals_12, primals_13, out=buf12)
        del primals_13
        buf13 = empty_strided_cuda((2048, 128, 1), (128, 1, 1024), torch.
            float32)
        extern_kernels.bmm(buf10, buf11, out=buf13)
        del buf10
        del buf11
        buf14 = empty_strided_cuda((2048, 128, 8), (1024, 8, 1), torch.float32)
        extern_kernels.bmm(buf12, buf13, out=buf14)
        del buf12
        del buf13
        buf15 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
        extern_kernels.mm(primals_9, primals_12, out=buf15)
        del primals_12
        buf16 = empty_strided_cuda((2048, 128, 1), (128, 1, 1024), torch.
            float32)
        extern_kernels.bmm(buf15, primals_10, out=buf16)
        del buf15
        del primals_10
        buf17 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        extern_kernels.mm(primals_12, primals_10, out=buf17)
        del primals_10
        buf18 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        extern_kernels.mm(primals_13, primals_12, out=buf18)
        del primals_13
        buf19 = empty_strided_cuda((2048, 128, 8, 1), (1024, 8, 1, 1024),
            torch.float32)
        extern_kernels.bmm(primals_9, primals_11, out=buf19)
        del primals_11
        buf20 = empty_strided_cuda((2048, 128, 16), (2048, 16, 1), torch.
            float32)
        triton_poi_fused_add_2[grid(196608)](buf20, buf14, 196608, XBLOCK=
            1024, num_warps=4, num_stages=1)
        buf21 = empty_strided_cuda((64, 8, 16), (128, 16, 1), torch.float32)
        triton_poi_fused_mul_3[grid(860480)](buf20, buf19, primals_10,
            buf21, 860480, XBLOCK=512, num_warps=4, num_stages=1)
        del buf19
        buf22 = empty_strided_cuda((64, 8, 64), (512, 64, 1), torch.float32)
        triton_poi_fused_4[grid(32768)](buf21, buf22, 32768, XBLOCK=256,
            num_warps=4, num_stages=1)
        del buf21
        buf23 = empty_strided_cuda((2048, 128, 16), (2048, 16, 1), torch.
            float32)
        triton_poi_fused_add_mul_5[grid(16384)](buf22, buf14, primals_9,
            primals_10, primals_11, primals_12, primals_13, primals_5, buf23,
            16384, XBLOCK=128, num_warps=4, num_stages=1)
        del buf22
        del primals_5
        buf24 = empty_strided_cuda((2048, 128, 16), (2048, 16, 1), torch.
            float32)
        triton_poi_fused_add_mul_6[grid(16384)](buf23, buf14, primals_9,
            primals_10, primals_11, primals_12, primals_13, primals_5,
            primals_10, primals_9, buf24, 16384, XBLOCK=256, num_warps=4,
            num_stages=1)
        del buf23
        del primals_13
        buf25 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        triton_poi_fused_exp_mean_1[grid(16)](primals_10, buf25, 16, XBLOCK
            =16, num_warps=1, num_stages=1)
        del primals_10
        buf26 = empty_strided_cuda((2048, 128, 8), (1024, 8, 1), torch.float32)
        extern_kernels.bmm(primals_11, buf18, out=buf26)
        del primals_11
        buf27 = empty_strided_cuda((2048, 128, 8), (1024, 8, 1), torch.float32)
        triton_poi_fused_add_7[grid(2048)](buf14, primals_12, primals_9,
            buf26, buf25, buf27, 2048, XBLOCK=256, num_warps=4, num_stages=1)
        del buf14
        del primals_12
        del primals_9
        del buf25
        del buf26
    return (buf24, primals_12, primals_13, buf0, buf1, primals_11, primals_8,
        primals_6, primals_9, primals_11, primals_12, primals_13, buf2, buf3,
        buf6, buf7, buf8, buf9, buf14, buf16, buf17, buf18, buf20, buf21,
        buf24, buf27)


class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len
        =64):
        """
        Mamba Structured State Space model implementation for benchmarking.
        
        :param batch_size: Size of the batch
        :param seq_length: Length of the input sequence
        :param n_heads: Number of attention heads
        :param d_head: Dimension of each head
        :param d_state: Dimension of the state space
        :param block_len: Length of each block for chunked computation
        """
        super(ModelNew, self).__init__()
        
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        
        # Initialize parameters
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        
    def segsum(self, x):
        """Naive segment sum calculation."""
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum
    
    def forward(self, input_0):
        primals_5 = self.A
        primals_6 = self.B
        primals_7 = self.C
        primals_8 = self.A
        primals_9 = self.B
        primals_10 = self.C
        primals_11 = self.A
        primals_12 = self.B
        primals_13 = self.C
        primals_4 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8, primals_9,
            primals_10, primals_11, primals_12, primals_13])
        return output[0]
