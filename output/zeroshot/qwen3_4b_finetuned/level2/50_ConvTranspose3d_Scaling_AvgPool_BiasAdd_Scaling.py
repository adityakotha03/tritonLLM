import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_mul_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 144 % 4
    x0 = xindex % 144
    x2 = xindex // 576
    x3 = xindex
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (16 * x0 + x2), xmask, eviction_policy='evict_last'
        )
    tmp4 = tl.load(in_out_ptr0 + (144 + x3), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (4 + x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (16 * x0 + 144 + x2), xmask,
        eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (8 + x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (16 * x0 + 288 + x2), xmask,
        eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (12 + x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (16 * x0 + 432 + x2), xmask,
        eviction_policy='evict_last')
    tmp3 = tmp0 * tmp1
    tmp6 = tmp3 + tmp2
    tmp8 = tmp4 * tmp5
    tmp9 = tmp7 + tmp8
    tmp11 = tmp6 + tmp9
    tmp13 = tmp10 * tmp15
    tmp14 = tmp12 + tmp13
    tmp16 = tmp11 + tmp14
    tmp18 = tmp17 + tmp16
    tmp19 = 0.5
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20 + tmp17
    tl.store(in_out_ptr0 + x3, tmp21, xmask)


@triton.jit
def triton_poi_fused_avg_pool3d_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 27216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 32 % 24
    x0 = xindex % 32
    x2 = xindex // 768
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 128 * x1 + 2048 * x2), xmask)
    tmp1 = tl.load(in_ptr0 + (32 + 2 * x0 + 128 * x1 + 2048 * x2), xmask)
    tmp3 = tl.load(in_ptr0 + (64 + 2 * x0 + 128 * x1 + 2048 * x2), xmask)
    tmp5 = tl.load(in_ptr0 + (96 + 2 * x0 + 128 * x1 + 2048 * x2), xmask)
    tmp10 = tl.load(in_ptr0 + (128 * x1 + 2048 * x2), xmask)
    tmp11 = tl.load(in_ptr0 + (32 + 128 * x1 + 2048 * x2), xmask)
    tmp13 = tl.load(in_ptr0 + (64 + 128 * x1 + 2048 * x2), xmask)
    tmp15 = tl.load(in_ptr0 + (96 + 128 * x1 + 2048 * x2), xmask)
    tmp19 = tl.load(in_ptr0 + (128 * x1 + 128 * x0 + 2048 * x2), xmask)
    tmp20 = tl.load(in_ptr0 + (160 + 128 * x1 + 128 * x0 + 2048 * x2), xmask)
    tmp22 = tl.load(in_ptr0 + (192 + 128 * x1 + 128 * x0 + 2048 * x2), xmask)
    tmp24 = tl.load(in_ptr0 + (224 + 128 * x1 + 128 * x0 + 2048 * x2), xmask)
    tmp29 = tl.load(in_ptr0 + (192 + 128 * x1 + 256 * x0 + 2048 * x2), xmask)
    tmp30 = tl.load(in_ptr0 + (224 + 128 * x1 + 256 * x0 + 2048 * x2), xmask)
    tmp32 = tl.load(in_ptr0 + (256 + 128 * x1 + 256 * x0 + 2048 * x2), xmask)
    tmp34 = tl.load(in_ptr0 + (288 + 128 * x1 + 256 * x0 + 2048 * x2), xmask)
    tmp39 = tl.load(in_ptr0 + (256 + 128 * x1 + 256 * x0 + 128 * x0 + 2048 *
        x2), xmask)
    tmp40 = tl.load(in_ptr0 + (288 + 128 * x1 + 256 * x0 + 128 * x0 + 2048 *
        x2), xmask)
    tmp42 = tl.load(in_ptr0 + (320 + 128 * x1 + 256 * x0 + 128 * x0 + 2048 *
        x2), xmask)
    tmp44 = tl.load(in_ptr0 + (352 + 128 * x1 + 256 * x0 + 128 * x0 + 2048 *
        x2), xmask)
    tmp49 = tl.load(in_ptr0 + (320 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 2048 * x2), xmask)
    tmp50 = tl.load(in_ptr0 + (352 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 2048 * x2), xmask)
    tmp52 = tl.load(in_ptr0 + (384 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 2048 * x2), xmask)
    tmp54 = tl.load(in_ptr0 + (416 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 2048 * x2), xmask)
    tmp59 = tl.load(in_ptr0 + (384 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 2048 * x2), xmask)
    tmp60 = tl.load(in_ptr0 + (416 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 2048 * x2), xmask)
    tmp62 = tl.load(in_ptr0 + (448 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 2048 * x2), xmask)
    tmp64 = tl.load(in_ptr0 + (480 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 2048 * x2), xmask)
    tmp69 = tl.load(in_ptr0 + (448 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp70 = tl.load(in_ptr0 + (480 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp72 = tl.load(in_ptr0 + (512 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp74 = tl.load(in_ptr0 + (544 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp77 = tl.load(in_ptr0 + (512 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp78 = tl.load(in_ptr0 + (544 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp80 = tl.load(in_ptr0 + (576 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp82 = tl.load(in_ptr0 + (608 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp87 = tl.load(in_ptr0 + (576 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp88 = tl.load(in_ptr0 + (608 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp90 = tl.load(in_ptr0 + (640 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp92 = tl.load(in_ptr0 + (672 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp95 = tl.load(in_ptr0 + (640 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 *
        x2), xmask)
    tmp96 = tl.load(in_ptr0 + (672 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 *
        x2), xmask)
    tmp98 = tl.load(in_ptr0 + (704 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 *
        x2), xmask)
    tmp100 = tl.load(in_ptr0 + (736 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 *
        x2), xmask)
    tmp105 = tl.load(in_ptr0 + (704 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 2048 * x2), xmask)
    tmp106 = tl.load(in_ptr0 + (736 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 2048 * x2), xmask)
    tmp108 = tl.load(in_ptr0 + (768 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 2048 * x2), xmask)
    tmp110 = tl.load(in_ptr0 + (800 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 2048 * x2), xmask)
    tmp115 = tl.load(in_ptr0 + (768 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 2048 * x2), xmask)
    tmp116 = tl.load(in_ptr0 + (800 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 2048 * x2), xmask)
    tmp118 = tl.load(in_ptr0 + (832 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 2048 * x2), xmask)
    tmp120 = tl.load(in_ptr0 + (864 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 2048 * x2), xmask)
    tmp123 = tl.load(in_ptr0 + (832 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp124 = tl.load(in_ptr0 + (864 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp126 = tl.load(in_ptr0 + (896 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp128 = tl.load(in_ptr0 + (928 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp133 = tl.load(in_ptr0 + (896 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp134 = tl.load(in_ptr0 + (928 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp136 = tl.load(in_ptr0 + (960 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 *
        x2), xmask)
    tmp138 = tl.load(in_ptr0 + (992 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 *
        x2), xmask)
    tmp141 = tl.load(in_ptr0 + (960 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 2048 * x2), xmask)
    tmp142 = tl.load(in_ptr0 + (992 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 2048 * x2), xmask)
    tmp144 = tl.load(in_ptr0 + (1024 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 2048 * x2), xmask)
    tmp146 = tl.load(in_ptr0 + (1056 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 2048 * x2), xmask)
    tmp149 = tl.load(in_ptr0 + (1024 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp150 = tl.load(in_ptr0 + (1056 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp152 = tl.load(in_ptr0 + (1088 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp154 = tl.load(in_ptr0 + (1120 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp157 = tl.load(in_ptr0 + (1088 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp158 = tl.load(in_ptr0 + (1120 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp160 = tl.load(in_ptr0 + (1152 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp162 = tl.load(in_ptr0 + (1184 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp165 = tl.load(in_ptr0 + (1152 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp166 = tl.load(in_ptr0 + (1184 + 128 * x1 + 256 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 *
        x0 + 128 * x0 + 128 * x0 + 128 * x0 + 128 * x0 + 2048 * x2), xmask)
    tmp168 =