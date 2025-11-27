import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_per_fused__native_batch_norm_legit_0(in_out_ptr0, in_ptr0,
    in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 32 * x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + r1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 32, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 32.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + x0, tmp23, xmask)
    tl.store(out_ptr1 + x0, tmp12, xmask)
    tl.store(out_ptr0 + x0, tmp18, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 32 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 32 * x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 32 * x1), xmask, eviction_policy='evict_last'
        )
    tmp6 = tl.load(in_ptr0 + (3 + 32 * x1), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (4 + 32 * x1), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (5 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (6 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (7 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr0 + (8 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp20 = tl.load(in_ptr0 + (9 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr0 + (10 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (11 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_ptr0 + (12 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp28 = tl.load(in_ptr0 + (13 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp31 = tl.load(in_ptr0 + (14 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (15 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_ptr0 + (16 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp37 = tl.load(in_ptr0 + (17 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (18 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr0 + (19 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp43 = tl.load(in_ptr0 + (20 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (21 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp47 = tl.load(in_ptr0 + (22 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_ptr0 + (23 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (24 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp53 = tl.load(in_ptr0 + (25 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp55 = tl.load(in_ptr0 + (26 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (27 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_ptr0 + (28 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp61 = tl.load(in_ptr0 + (29 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (30 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp65 = tl.load(in_ptr0 + (31 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = triton_helpers.maximum(tmp2, tmp1)
    tmp5 = triton_helpers.maximum(tmp4, tmp3)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp8 = tmp0 - tmp7
    tmp10 = triton_helpers.exp(tmp9)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp8 - tmp16
    tmp21 = triton_helpers.exp(tmp20)
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tmp29 = triton_helpers.maximum(tmp28, tmp27)
    tmp30 = tmp17 - tmp29
    tmp32 = triton_helpers.exp(tmp31)
    tmp34 = triton_helpers.maximum(tmp33, tmp32)
    tmp36 = triton_helpers.maximum(tmp35, tmp34)
    tmp38 = triton_helpers.maximum(tmp37, tmp36)
    tmp40 = triton_helpers.maximum(tmp39, tmp38)
    tmp42 = triton_helpers.exp(tmp41)
    tmp44 = triton_helpers.maximum(tmp43, tmp42)
    tmp46 = triton_helpers.maximum(tmp45, tmp44)
    tmp48 = triton_helpers.maximum(tmp47, tmp46)
    tmp50 = triton_helpers.maximum(tmp49, tmp48)
    tmp52 = triton_helpers.exp(tmp51)
    tmp54 = triton_helpers.maximum(tmp53, tmp52)
    tmp56 = triton_helpers.maximum(tmp55, tmp54)
    tmp58 = triton_helpers.maximum(tmp57, tmp56)
    tmp60 = triton_helpers.exp(tmp59)
    tmp62 = triton_helpers.maximum(tmp61, tmp60)
    tmp64 = triton_helpers.maximum(tmp63, tmp62)
    tmp66 = triton_helpers.maximum(tmp65, tmp64)
    tmp67 = tmp30 - tmp66
    tmp68 = triton_helpers.exp(tmp67)
    tl.store(out_ptr0 + x2, tmp68, xmask)


@triton.jit
def triton_poi_fused__softmax_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 32 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 32 * x1), xmask, eviction_policy='evict_last'
        )
    tmp4 = tl.load(in_ptr0 + (2 + 32 * x1), xmask, eviction_policy='evict_last'
        )
    tmp6 = tl.load(in_ptr0 + (3 + 32 * x1), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (4 + 32 * x1), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (5 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (6 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (7 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr0 + (8 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp20 = tl.load(in_ptr0 + (9 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr0 + (10 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (11 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_ptr0 + (12 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp28 = tl.load(in_ptr0 + (13 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp31 = tl.load(in_ptr0 + (14 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (15 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_ptr0 + (16 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp37 = tl.load(in_ptr0 + (17 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (18 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr0 + (19 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp43 = tl.load(in_ptr0 + (20 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (21 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp47 = tl.load(in_ptr0 + (22 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_ptr0 + (23 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (24 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp53 = tl.load(in_ptr0 + (25 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp55 = tl.load(in_ptr0 + (26 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (27 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_ptr0 + (28 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp61 = tl.load(in_ptr0 + (29 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (30 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp65 = tl.load(in_ptr0 + (31 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tmp10 = tmp9 + tmp11
    tmp12 = tmp10 + tmp13
    tmp14 = tmp12 + tmp15
    tmp16 = tmp8 / tmp14
    tmp17 = tmp16 / tmp16
    tmp19 = tmp20 + tmp22
    tmp21 = tmp19 + tmp24
    tmp23 = tmp21 + tmp26
    tmp25 = tmp23 + tmp28
    tmp27 = tmp17 / tmp25
    tmp28 = tmp27 / tmp27
    tmp30 = tmp31 + tmp33
    tmp32 = tmp30 + tmp35
    tmp34 = tmp32 + tmp37
    tmp36 = tmp34 + tmp39
    tmp38 = tmp28 / tmp36
    tmp39 = tmp38 / tmp38
    tmp40 = tmp41 + tmp43
    tmp42 = tmp40 + tmp45
    tmp44 = tmp42 + tmp47
    tmp46 = tmp44 + tmp49
    tmp48 = tmp39 / tmp46
    tmp49 = tmp48 / tmp48
    tmp50 = tmp51 + tmp53
    tmp52 = tmp50 + tmp55
    tmp54 = tmp52 + tmp57
    tmp56 = tmp54 + tmp59
    tmp58 = tmp49 / tmp56
    tmp59 = tmp58 / tmp58
    tmp60 = tmp61 + tmp63
    tmp62 = tmp60 + tmp65
    tmp64 = tmp62 / tmp64
    tmp65 = tmp64 / tmp64
    tl.store(out_ptr0 + x2, tmp65, xmask)


@triton.jit
def triton_poi_fused_clone_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_sub_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.load(in_ptr0 + x2, xmask)
    tmp4 = tmp3 - tmp2
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_7(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_8(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused_div_9(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_div_10(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_div_11(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_linalg_vector_norm_12(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp2, 0))
    tmp5 = libdevice.sqrt(tmp4)
    tl.store(out_ptr0 + x0, tmp5, xmask)


@triton.jit
def triton_poi_fused_div_13(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_div_14(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_linalg_vector_norm_15(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp2, 0))
    tmp5 = libdevice.sqrt(tmp4)
    tl.store(out_ptr0 + x0, tmp5, xmask)


@triton.jit
def triton_poi_fused_div_16(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_linalg_vector_norm_17(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp2, 0))
    tmp5 = libdevice.sqrt(tmp4)
    tl.store(out_ptr0 + x0, tmp5, xmask)


@triton.jit
def triton_poi_fused_div_18(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_div_19(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_linalg_vector_norm_20(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp2, 0))
    tmp5 = libdevice.sqrt(tmp4)
    tl.store(out_ptr0 + x0, tmp5, xmask)


@triton.jit
def triton_poi_fused_div_21(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_linalg_vector_norm_22(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp2, 0))
    tmp5 = libdevice.sqrt(tmp4)
    tl.store(out_ptr0 + x0, tmp5, xmask)


@triton.jit
def triton_poi_fused_div_23(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_linalg_vector_norm_24(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp2, 0))
    tmp5 = libdevice.sqrt(tmp4)
    tl.store(out_ptr0 + x0, tmp5, xmask)


@triton.jit
def triton_poi_fused_div_25(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_linalg_vector_norm_26(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp2, 0))
    tmp5 = libdevice.sqrt(tmp4)
    tl.store(out_ptr0 + x0, tmp5, xmask)


