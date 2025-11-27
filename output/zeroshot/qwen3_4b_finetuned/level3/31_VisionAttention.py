import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_clone_0(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 2
    y1 = yindex // 2
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 2 * x2 + 256 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 128 * y3), tmp2, xmask & ymask)


@triton.jit
def triton_poi_fused_mul_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.1538461538461537
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr0 + x0, tmp3, xmask)


@triton.jit
def triton_poi_fused_mul_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.1538461538461537
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr0 + x0, tmp3, xmask)


@triton.jit
def triton_poi_fused_mul_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.1538461538461537
    tmp3 = tmp1 * tmp2
    tl.store(out_ptr0 + x0, tmp3, xmask)


@triton.jit
def triton_poi_fused__softmax_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 4 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4 * x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4 * x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4 * x1), xmask, eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + x2, tmp9, xmask)


@triton.jit
def triton_poi_fused__softmax_5(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 4 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4 * x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4 * x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4 * x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_clone_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 2
    y1 = yindex // 2
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 2 * x2 + 256 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 128 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_add_native_layer_norm_7(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + 4 * x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (1 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (2 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4 * x0), xmask, eviction_policy='evict_last'
        )
    tmp12 = tl.load(in_ptr1 + (3 + 4 * x0), xmask, eviction_policy='evict_last'
        )
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tmp17 = tmp2 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tmp5 - tmp16
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 + tmp20
    tmp22 = tmp9 - tmp16
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 + tmp23
    tmp25 = tmp13 - tmp16
    tmp26 = tmp25 * tmp25
    tmp27 = tmp24 + tmp26
    tmp28 = tmp27 / tmp15
    tl.store(out_ptr0 + x0, tmp16, xmask)
    tl.store(out_ptr1 + x0, tmp28, xmask)


@triton.jit
def triton_poi_fused_add_native_layer_norm_8(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x2, xmask)
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + x2, tmp13, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9) = args
    args.clear()
    assert_size_stride(primals_1, (2, 128, 128, 128), (2097152, 16384, 128,
        1))
    assert_size_stride(primals_2, (4, 128, 1), (128, 1, 1))
    assert_size_stride(primals_3, (4,), (1,))
    assert_size_stride(primals_4, (4, 128, 128), (16384, 128, 1))
    assert_size_stride(primals_5, (4, 128, 128), (16384, 128, 1))
    assert_size_stride(primals_6, (4,), (1,))
    assert_size_stride(primals_7, (4,), (1,))
    assert_size_stride(primals_8, (4,), (1,))
    assert_size_stride(primals_9, (128,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, 128, 128, 128), (2097152, 16384, 128,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(512, 128)](primals_1, primals_2, buf0,
            512, 128, XBLOCK=32, YBLOCK=32, num_warps=4, num_stages=1)
        del primals_2
        buf1 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
        triton_poi_fused_mul_1[grid(512)](primals_3, buf1, 512, XBLOCK=256,
            num_warps=4, num_stages=1)
        del primals_3
        buf2 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf0, (512, 128), (128, 1), 0),
            reinterpret_tensor(buf1, (128, 128), (1, 128), 0), out=buf2)
        buf3 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
        triton_poi_fused_mul_2[grid(512)](primals_4, buf3, 512, XBLOCK=256,
            num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf2, (512, 128, 1), (128, 1,
            0), 0), reinterpret_tensor(buf3, (512, 1, 128), (128, 1, 1), 0),
            out=buf4)
        buf5 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
        triton_poi_fused_mul_3[grid(512)](primals_5, buf5, 512, XBLOCK=256,
            num_warps=4, num_stages=1)
        buf6 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf4, (512, 1, 128), (128, 
            128, 1), 0), reinterpret_tensor(buf5, (512, 128, 1), (128, 1, 
            128), 0), out=buf6)
        buf7 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
        triton_poi_fused__softmax_4[grid(2048)](buf6, buf7, 2048, XBLOCK=
            128, num_warps=4, num_stages=1)
        buf8 = reinterpret_tensor(buf6, (512, 128), (128, 1), 0)
        del buf6
        triton_poi_fused__softmax_5[grid(2048)](buf7, buf8, 2048, XBLOCK=
            128, num_warps=4, num_stages=1)
        del buf7
        buf9 = empty_strided_cuda((2, 128, 128, 128), (2097152, 16384, 128,
            1), torch.float32)
        triton_poi_fused_clone_6[grid(512, 128)](primals_1, buf9, 512, 128,
            XBLOCK=32, YBLOCK=32, num_warps=4, num_stages=1)
        buf10 = reinterpret_tensor(buf1, (512, 128), (128, 1), 0)
        del buf1
        extern_kernels.mm(reinterpret_tensor(buf8, (512, 128), (128, 1), 0),
            reinterpret_tensor(buf9, (128, 512), (512, 1), 0), out=buf10)
        buf11 = empty_strided_cuda((2, 128, 128, 1), (16384, 128, 1, 1),
            torch.float32)
        buf12 = empty_strided_cuda((2, 128, 1), (128, 1, 1), torch.float32)
        buf13 = empty_strided_cuda((2, 128, 1), (128, 1, 128), torch.float32)
        triton_poi_fused_add_native_layer_norm_7[grid(256)](buf10, primals_1,
            buf12, buf13, 256, XBLOCK=256, num_warps=4, num_stages=1)
        buf14 = reinterpret_tensor(buf5, (2, 128, 128), (16384, 128, 1), 0)
        del buf5
        triton_poi_fused_add_native_layer_norm_8[grid(1024)](buf10,
            primals_1, buf12, buf13, primals_6, primals_7, buf14, 1024,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf12
        del buf13
        del primals_7
    return buf14, primals_1, primals_6, primals_8, reinterpret_tensor(buf0,
        (512, 128), (128, 1), 0), buf2, reinterpret_tensor(buf3, (128, 512),
        (1, 128), 0), reinterpret_tensor(buf4, (512, 128, 1), (128, 1, 128),
        0), reinterpret_tensor(buf8, (512, 128), (128, 1), 0
        ), reinterpret_tensor(buf9, (128, 512), (512, 1), 0
        ), buf10, primals_9, primals_5, primals_4


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Attention Block using Multihead Self-Attention.
        :param embed_dim: Embedding dimension (the number of channels)
        :param num_heads: Number of attention heads
        """
        super(ModelNew, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input_0):
        primals_2 = self.attn.in_proj_weight
        primals_3 = self.attn.in_proj_bias
        primals_4 = self.attn.out_proj.weight
        primals_6 = self.attn.out_proj.bias
        primals_7 = self.attn.out_proj.bias
        primals_8 = self.norm.weight
        primals_9 = self.norm.bias
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8, primals_9])
        return output[0]
