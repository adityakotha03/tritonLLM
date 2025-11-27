import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__unsafe_index_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 2048
    x0 = xindex % 2048
    x2 = xindex
    tmp0 = x1
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 32, tl.int64)
    tmp9 = tl.load(in_ptr0 + (2048 + x0), tmp6 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp10 = tmp0 < tmp3
    tl.full([1], 48, tl.int64)
    tmp13 = tl.load(in_ptr0 + (4096 + x0), tmp10 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp14 = tmp0 >= tmp3
    tl.full([1], 64, tl.int64)
    tmp17 = tl.load(in_ptr0 + (6144 + x0), tmp14 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp18 = tmp0 < tmp3
    tl.full([1], 80, tl.int64)
    tmp21 = tl.load(in_ptr0 + (8192 + x0), tmp18 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp22 = tmp0 >= tmp3
    tl.full([1], 96, tl.int64)
    tmp25 = tl.load(in_ptr0 + (10240 + x0), tmp22 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp26 = tmp0 < tmp3
    tl.full([1], 112, tl.int64)
    tmp29 = tl.load(in_ptr0 + (12288 + x0), tmp26 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp30 = tmp0 >= tmp3
    tl.full([1], 128, tl.int64)
    tmp33 = tl.load(in_ptr0 + (14336 + x0), tmp30 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp34 = tmp0 < tmp3
    tl.full([1], 144, tl.int64)
    tmp37 = tl.load(in_ptr0 + (16384 + x0), tmp34 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp38 = tmp0 >= tmp3
    tl.full([1], 160, tl.int64)
    tmp41 = tl.load(in_ptr0 + (18432 + x0), tmp38 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp42 = tmp0 < tmp3
    tl.full([1], 176, tl.int64)
    tmp45 = tl.load(in_ptr0 + (20480 + x0), tmp42 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp46 = tmp0 >= tmp3
    tl.full([1], 192, tl.int64)
    tmp49 = tl.load(in_ptr0 + (22528 + x0), tmp46 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp50 = tl.load(in_ptr1 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp51 = tmp5 + tmp50
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp4, tmp51, tmp52)
    tmp54 = tl.load(in_ptr1 + (2048 + x0), tmp6 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp55 = tmp9 + tmp54
    tmp56 = tl.full(tmp55.shape, 0.0, tmp55.dtype)
    tmp57 = tl.where(tmp6, tmp55, tmp56)
    tmp58 = tl.where(tmp4, tmp53, tmp57)
    tmp59 = tl.load(in_ptr1 + (4096 + x0), tmp10 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp60 = tmp13 + tmp59
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp10, tmp60, tmp61)
    tmp63 = tl.where(tmp6, tmp58, tmp62)
    tmp64 = tl.load(in_ptr1 + (6144 + x0), tmp14 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp65 = tmp17 + tmp64
    tmp66 = tl.full(tmp65.shape, 0.0, tmp65.dtype)
    tmp67 = tl.where(tmp14, tmp65, tmp66)
    tmp68 = tl.where(tmp10, tmp63, tmp67)
    tmp69 = tl.load(in_ptr1 + (8192 + x0), tmp18 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp70 = tmp21 + tmp69
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp18, tmp70, tmp71)
    tmp73 = tl.where(tmp14, tmp68, tmp72)
    tmp74 = tl.load(in_ptr1 + (10240 + x0), tmp22 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp75 = tmp25 + tmp74
    tmp76 = tl.full(tmp75.shape, 0.0, tmp75.dtype)
    tmp77 = tl.where(tmp22, tmp75, tmp76)
    tmp78 = tl.where(tmp18, tmp73, tmp77)
    tmp79 = tl.load(in_ptr1 + (12288 + x0), tmp26 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp80 = tmp29 + tmp79
    tmp81 = tl.full(tmp80.shape, 0.0, tmp80.dtype)
    tmp82 = tl.where(tmp26, tmp80, tmp81)
    tmp83 = tl.where(tmp22, tmp78, tmp82)
    tmp84 = tl.load(in_ptr1 + (14336 + x0), tmp30 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp85 = tmp33 + tmp84
    tmp86 = tl.full(tmp85.shape, 0.0, tmp85.dtype)
    tmp87 = tl.where(tmp30, tmp85, tmp86)
    tmp88 = tl.where(tmp26, tmp83, tmp87)
    tmp89 = tl.load(in_ptr1 + (16384 + x0), tmp34 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp90 = tmp37 + tmp89
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp34, tmp90, tmp91)
    tmp93 = tl.where(tmp30, tmp88, tmp92)
    tmp94 = tl.load(in_ptr1 + (18432 + x0), tmp38 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp95 = tmp41 + tmp94
    tmp96 = tl.full(tmp95.shape, 0.0, tmp95.dtype)
    tmp97 = tl.where(tmp38, tmp95, tmp96)
    tmp98 = tl.where(tmp34, tmp93, tmp97)
    tmp99 = tl.load(in_ptr1 + (20480 + x0), tmp42 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp100 = tmp45 + tmp99
    tmp101 = tl.full(tmp100.shape, 0.0, tmp100.dtype)
    tmp102 = tl.where(tmp42, tmp100, tmp101)
    tmp103 = tl.where(tmp38, tmp98, tmp102)
    tmp104 = tl.load(in_ptr1 + (22528 + x0), tmp46 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp105 = tmp49 + tmp104
    tmp106 = tl.full(tmp105.shape, 0.0, tmp105.dtype)
    tmp107 = tl.where(tmp46, tmp105, tmp106)
    tmp108 = tl.where(tmp42, tmp103, tmp107)
    tl.store(out_ptr0 + x2, tmp108, xmask)


@triton.jit
def triton_poi_fused__unsafe_index_1(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 2048
    x0 = xindex % 2048
    x2 = xindex
    tmp0 = x1
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2048 + x0), tmp4 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 32, tl.int64)
    tmp9 = tl.load(in_ptr0 + (4096 + x0), tmp6 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp10 = tmp0 < tmp3
    tl.full([1], 48, tl.int64)
    tmp13 = tl.load(in_ptr0 + (6144 + x0), tmp10 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp14 = tmp0 >= tmp3
    tl.full([1], 64, tl.int64)
    tmp17 = tl.load(in_ptr0 + (8192 + x0), tmp14 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp18 = tmp0 < tmp3
    tl.full([1], 80, tl.int64)
    tmp21 = tl.load(in_ptr0 + (10240 + x0), tmp18 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp22 = tmp0 >= tmp3
    tl.full([1], 96, tl.int64)
    tmp25 = tl.load(in_ptr0 + (12288 + x0), tmp22 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp26 = tmp0 < tmp3
    tl.full([1], 112, tl.int64)
    tmp29 = tl.load(in_ptr0 + (14336 + x0), tmp26 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp30 = tmp0 >= tmp3
    tl.full([1], 128, tl.int64)
    tmp33 = tl.load(in_ptr0 + (16384 + x0), tmp30 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp34 = tmp0 < tmp3
    tl.full([1], 144, tl.int64)
    tmp37 = tl.load(in_ptr0 + (18432 + x0), tmp34 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp38 = tmp0 >= tmp3
    tl.full([1], 160, tl.int64)
    tmp41 = tl.load(in_ptr0 + (20480 + x0), tmp38 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp42 = tmp0 < tmp3
    tl.full([1], 176, tl.int64)
    tmp45 = tl.load(in_ptr0 + (22528 + x0), tmp42 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp46 = tmp0 >= tmp3
    tl.full([1], 192, tl.int64)
    tmp49 = tl.load(in_ptr0 + (24576 + x0), tmp46 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp50 = tl.load(in_ptr1 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp51 = tmp5 + tmp50
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp4, tmp51, tmp52)
    tmp54 = tl.load(in_ptr1 + (2048 + x0), tmp6 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp55 = tmp9 + tmp54
    tmp56 = tl.full(tmp55.shape, 0.0, tmp55.dtype)
    tmp57 = tl.where(tmp6, tmp55, tmp56)
    tmp58 = tl.where(tmp4, tmp53, tmp57)
    tmp59 = tl.load(in_ptr1 + (4096 + x0), tmp10 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp60 = tmp13 + tmp59
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp10, tmp60, tmp61)
    tmp63 = tl.where(tmp6, tmp58, tmp62)
    tmp64 = tl.load(in_ptr1 + (6144 + x0), tmp14 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp65 = tmp17 + tmp64
    tmp66 = tl.full(tmp65.shape, 0.0, tmp65.dtype)
    tmp67 = tl.where(tmp14, tmp65, tmp66)
    tmp68 = tl.where(tmp10, tmp63, tmp67)
    tmp69 = tl.load(in_ptr1 + (8192 + x0), tmp18 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp70 = tmp21 + tmp69
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp18, tmp70, tmp71)
    tmp73 = tl.where(tmp14, tmp68, tmp72)
    tmp74 = tl.load(in_ptr1 + (10240 + x0), tmp22 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp75 = tmp25 + tmp74
    tmp76 = tl.full(tmp75.shape, 0.0, tmp75.dtype)
    tmp77 = tl.where(tmp22, tmp75, tmp76)
    tmp78 = tl.where(tmp18, tmp73, tmp77)
    tmp79 = tl.load(in_ptr1 + (12288 + x0), tmp26 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp80 = tmp29 + tmp79
    tmp81 = tl.full(tmp80.shape, 0.0, tmp80.dtype)
    tmp82 = tl.where(tmp26, tmp80, tmp81)
    tmp83 = tl.where(tmp22, tmp78, tmp82)
    tmp84 = tl.load(in_ptr1 + (14336 + x0), tmp30 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp85 = tmp33 + tmp84
    tmp86 = tl.full(tmp85.shape, 0.0, tmp85.dtype)
    tmp87 = tl.where(tmp30, tmp85, tmp86)
    tmp88 = tl.where(tmp26, tmp83, tmp87)
    tmp89 = tl.load(in_ptr1 + (16384 + x0), tmp34 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp90 = tmp37 + tmp89
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp34, tmp90, tmp91)
    tmp93 = tl.where(tmp30, tmp88, tmp92)
    tmp94 = tl.load(in_ptr1 + (18432 + x0), tmp38 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp95 = tmp41 + tmp94
    tmp96 = tl.full(tmp95.shape, 0.0, tmp95.dtype)
    tmp97 = tl.where(tmp38, tmp95, tmp96)
    tmp98 = tl.where(tmp34, tmp93, tmp97)
    tmp99 = tl.load(in_ptr1 + (20480 + x0), tmp42 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp100 = tmp45 + tmp99
    tmp101 = tl.full(tmp100.shape, 0.0, tmp100.dtype)
    tmp102 = tl.where(tmp42, tmp100, tmp101)
    tmp103 = tl.where(tmp38, tmp98, tmp102)
    tmp104 = tl.load(in_ptr1 + (22528 + x0), tmp46 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp105 = tmp49 + tmp104
    tmp106 = tl.full(tmp105.shape, 0.0, tmp105.dtype)
    tmp107 = tl.where(tmp46, tmp105, tmp106)
    tmp108 = tl.where(tmp42, tmp103, tmp107)
    tl.store(out_ptr0 + x2, tmp108, xmask)


@triton.jit
def triton_poi_fused__unsafe_index_2(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 2048
    x0 = xindex % 2048
    x2 = xindex
    tmp0 = x1
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (4096 + x0), tmp4 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 32, tl.int64)
    tmp9 = tl.load(in_ptr0 + (6144 + x0), tmp6 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp10 = tmp0 < tmp3
    tl.full([1], 48, tl.int64)
    tmp13 = tl.load(in_ptr0 + (8192 + x0), tmp10 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp14 = tmp0 >= tmp3
    tl.full([1], 64, tl.int64)
    tmp17 = tl.load(in_ptr0 + (10240 + x0), tmp14 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp18 = tmp0 < tmp3
    tl.full([1], 80, tl.int64)
    tmp21 = tl.load(in_ptr0 + (12288 + x0), tmp18 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp22 = tmp0 >= tmp3
    tl.full([1], 96, tl.int64)
    tmp25 = tl.load(in_ptr0 + (14336 + x0), tmp22 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp26 = tmp0 < tmp3
    tl.full([1], 112, tl.int64)
    tmp29 = tl.load(in_ptr0 + (16384 + x0), tmp26 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp30 = tmp0 >= tmp3
    tl.full([1], 128, tl.int64)
    tmp33 = tl.load(in_ptr0 + (18432 + x0), tmp30 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp34 = tmp0 < tmp3
    tl.full([1], 144, tl.int64)
    tmp37 = tl.load(in_ptr0 + (20480 + x0), tmp34 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp38 = tmp0 >= tmp3
    tl.full([1], 160, tl.int64)
    tmp41 = tl.load(in_ptr0 + (22528 + x0), tmp38 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp42 = tmp0 < tmp3
    tl.full([1], 176, tl.int64)
    tmp45 = tl.load(in_ptr0 + (24576 + x0), tmp42 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp46 = tmp0 >= tmp3
    tl.full([1], 192, tl.int64)
    tmp49 = tl.load(in_ptr0 + (26624 + x0), tmp46 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp50 = tl.load(in_ptr1 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp51 = tmp5 + tmp50
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp4, tmp51, tmp52)
    tmp54 = tl.load(in_ptr1 + (2048 + x0), tmp6 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp55 = tmp9 + tmp54
    tmp56 = tl.full(tmp55.shape, 0.0, tmp55.dtype)
    tmp57 = tl.where(tmp6, tmp55, tmp56)
    tmp58 = tl.where(tmp4, tmp53, tmp57)
    tmp59 = tl.load(in_ptr1 + (4096 + x0), tmp10 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp60 = tmp13 + tmp59
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp10, tmp60, tmp61)
    tmp63 = tl.where(tmp6, tmp58, tmp62)
    tmp64 = tl.load(in_ptr1 + (6144 + x0), tmp14 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp65 = tmp17 + tmp64
    tmp66 = tl.full(tmp65.shape, 0.0, tmp65.dtype)
    tmp67 = tl.where(tmp14, tmp65, tmp66)
    tmp68 = tl.where(tmp10, tmp63, tmp67)
    tmp69 = tl.load(in_ptr1 + (8192 + x0), tmp18 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp70 = tmp21 + tmp69
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp18, tmp70, tmp71)
    tmp73 = tl.where(tmp14, tmp68, tmp72)
    tmp74 = tl.load(in_ptr1 + (10240 + x0), tmp22 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp75 = tmp25 + tmp74
    tmp76 = tl.full(tmp75.shape, 0.0, tmp75.dtype)
    tmp77 = tl.where(tmp22, tmp75, tmp76)
    tmp78 = tl.where(tmp18, tmp73, tmp77)
    tmp79 = tl.load(in_ptr1 + (12288 + x0), tmp26 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp80 = tmp29 + tmp79
    tmp81 = tl.full(tmp80.shape, 0.0, tmp80.dtype)
    tmp82 = tl.where(tmp26, tmp80, tmp81)
    tmp83 = tl.where(tmp22, tmp78, tmp82)
    tmp84 = tl.load(in_ptr1 + (14336 + x0), tmp30 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp85 = tmp33 + tmp84
    tmp86 = tl.full(tmp85.shape, 0.0, tmp85.dtype)
    tmp87 = tl.where(tmp30, tmp85, tmp86)
    tmp88 = tl.where(tmp26, tmp83, tmp87)
    tmp89 = tl.load(in_ptr1 + (16384 + x0), tmp34 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp90 = tmp37 + tmp89
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp34, tmp90, tmp91)
    tmp93 = tl.where(tmp30, tmp88, tmp92)
    tmp94 = tl.load(in_ptr1 + (18432 + x0), tmp38 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp95 = tmp41 + tmp94
    tmp96 = tl.full(tmp95.shape, 0.0, tmp95.dtype)
    tmp97 = tl.where(tmp38, tmp95, tmp96)
    tmp98 = tl.where(tmp34, tmp93, tmp97)
    tmp99 = tl.load(in_ptr1 + (20480 + x0), tmp42 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp100 = tmp45 + tmp99
    tmp101 = tl.full(tmp100.shape, 0.0, tmp100.dtype)
    tmp102 = tl.where(tmp42, tmp100, tmp101)
    tmp103 = tl.where(tmp38, tmp98, tmp102)
    tmp104 = tl.load(in_ptr1 + (22528 + x0), tmp46 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp105 = tmp49 + tmp104
    tmp106 = tl.full(tmp105.shape, 0.0, tmp105.dtype)
    tmp107 = tl.where(tmp46, tmp105, tmp106)
    tmp108 = tl.where(tmp42, tmp103, tmp107)
    tl.store(out_ptr0 + x2, tmp108, xmask)


@triton.jit
def triton_poi_fused__unsafe_index_3(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 2048
    x0 = xindex % 2048
    x2 = xindex
    tmp0 = x1
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (6144 + x0), tmp4 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 32, tl.int64)
    tmp9 = tl.load(in_ptr0 + (8192 + x0), tmp6 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp10 = tmp0 < tmp3
    tl.full([1], 48, tl.int64)
    tmp13 = tl.load(in_ptr0 + (10240 + x0), tmp10 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp14 = tmp0 >= tmp3
    tl.full([1], 64, tl.int64)
    tmp17 = tl.load(in_ptr0 + (12288 + x0), tmp14 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp18 = tmp0 < tmp3
    tl.full([1], 80, tl.int64)
    tmp21 = tl.load(in_ptr0 + (14336 + x0), tmp18 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp22 = tmp0 >= tmp3
    tl.full([1], 96, tl.int64)
    tmp25 = tl.load(in_ptr0 + (16384 + x0), tmp22 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp26 = tmp0 < tmp3
    tl.full([1], 112, tl.int64)
    tmp29 = tl.load(in_ptr0 + (18432 + x0), tmp26 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp30 = tmp0 >= tmp3
    tl.full([1], 128, tl.int64)
    tmp33 = tl.load(in_ptr0 + (20480 + x0), tmp30 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp34 = tmp0 < tmp3
    tl.full([1], 144, tl.int64)
    tmp37 = tl.load(in_ptr0 + (22528 + x0), tmp34 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp38 = tmp0 >= tmp3
    tl.full([1], 160, tl.int64)
    tmp41 = tl.load(in_ptr0 + (24576 + x0), tmp38 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp42 = tmp0 < tmp3
    tl.full([1], 176, tl.int64)
    tmp45 = tl.load(in_ptr0 + (26624 + x0), tmp42 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp46 = tmp0 >= tmp3
    tl.full([1], 192, tl.int64)
    tmp49 = tl.load(in_ptr0 + (28672 + x0), tmp46 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp50 = tl.load(in_ptr1 + x0, tmp4 & xmask, eviction_policy='