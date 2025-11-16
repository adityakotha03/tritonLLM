import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_0(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 13824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 9
    x0 = xindex % 9
    x1 = xindex // 9 % 2
    x3 = xindex // 144
    x4 = xindex % 144
    tmp0 = tl.load(in_ptr0 + (x2 + 64 * x0 + 576 * x1 + 1152 * x3), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < 2
    tmp4 = tmp2 & tmp3
    tl.store(out_ptr0 + (x4 + 144 * x3), tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_1(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 3456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + x2, xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_2(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 27648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 9
    x0 = xindex % 9
    x1 = xindex // 9 % 2
    x3 = xindex // 144
    x4 = xindex % 144
    tmp0 = tl.load(in_ptr0 + (x2 + 64 * x0 + 576 * x1 + 1152 * x3), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < 2
    tmp4 = tmp2 & tmp3
    tl.store(out_ptr0 + (x4 + 144 * x3), tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_3(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 27648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 9
    x0 = xindex % 9
    x1 = xindex // 9 % 2
    x3 = xindex // 144
    x4 = xindex % 144
    tmp0 = tl.load(in_ptr0 + (x2 + 64 * x0 + 576 * x1 + 1152 * x3), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < 2
    tmp4 = tmp2 & tmp3
    tl.store(out_ptr0 + (x4 + 144 * x3), tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_4(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 32
    x3 = xindex
    x4 = xindex % 32
    tmp0 = tl.load(in_ptr0 + x2, xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < 2
    tmp4 = tmp2 & tmp3
    tl.store(out_ptr0 + x4, tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_5(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 10752
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 32
    x3 = xindex
    x4 = xindex % 32
    tmp0 = tl.load(in_ptr0 + x2, xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < 2
    tmp4 = tmp2 & tmp3
    tl.store(out_ptr0 + x4, tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_6(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex % 589824
    x1 = xindex // 9
    x2 = xindex % 9
    x5 = xindex // 144
    x6 = xindex % 144
    x3 = xindex // 13824
    x7 = xindex // 589824
    tmp0 = tl.load(in_ptr0 + (x1 + 13824 * x4 + 1382400 * x5 + 13824000 * x7
        + 138240000 * x3), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < 2
    tmp4 = tmp2 & tmp3
    tl.store(out_ptr0 + (x2 + 9 * x4 + 81 * x5 + 729 * x7 + 6561 * x3),
        tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_7(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 870400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex % 870400
    x1 = xindex // 16
    x2 = xindex % 16
    x5 = xindex // 1152
    x6 = xindex % 1152
    x3 = xindex // 55296
    x7 = xindex // 870400
    tmp0 = tl.load(in_ptr0 + (x1 + 55296 * x4 + 4423680 * x5 + 44236800 * x7
        + 442368000 * x3), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < 2
    tmp4 = tmp2 & tmp3
    tl.store(out_ptr0 + (x2 + 16 * x4 + 256 * x5 + 4096 * x7 + 65536 * x3),
        tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_8(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 297856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex % 297856
    x1 = xindex // 32
    x2 = xindex % 32
    x5 = xindex // 952
    x6 = xindex % 952
    x3 = xindex // 30464
    x7 = xindex // 297856
    tmp0 = tl.load(in_ptr0 + (x1 + 30464 * x4 + 2944512 * x5 + 29445120 * x7
        + 294451200 * x3), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < 2
    tmp4 = tmp2 & tmp3
    tl.store(out_ptr0 + (x2 + 32 * x4 + 1024 * x5 + 32768 * x7 + 1048576 *
        x3), tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_9(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 148928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex % 148928
    x1 = xindex // 32
    x2 = xindex % 32
    x5 = xindex // 464
    x6 = xindex % 464
    x3 = xindex // 14928
    x7 = xindex // 148928
    tmp0 = tl.load(in_ptr0 + (x1 + 14928 * x4 + 1443168 * x5 + 14431680 *
        x7 + 144316800 * x3), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < 2
    tmp4 = tmp2 & tmp3
    tl.store(out_ptr0 + (x2 + 32 * x4 + 1024 * x5 + 32768 * x7 + 1048576 *
        x3), tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_10(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 266240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex % 266240
    x1 = xindex // 32
    x2 = xindex % 32
    x5 = xindex // 836
    x6 = xindex % 836
    x3 = xindex // 26656
    x7 = xindex // 266240
    tmp0 = tl.load(in_ptr0 + (x1 + 26656 * x4 + 2281856 * x5 + 22818560 *
        x7 + 228185600 * x3), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < 2
    tmp4 = tmp2 & tmp3
    tl.store(out_ptr0 + (x2 + 32 * x4 + 1024 * x5 + 32768 * x7 + 1048576 *
        x3), tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_11(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 287232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex % 287232
    x1 = xindex // 32
    x2 = xindex % 32
    x5 = xindex // 898
    x6 = xindex % 898
    x3 = xindex // 28736
    x7 = xindex // 287232
    tmp0 = tl.load(in_ptr0 + (x1 + 28736 * x4 + 2535360 * x5 + 25353600 *
        x7 + 253536000 * x3), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < 2
    tmp4 = tmp2 & tmp3
    tl.store(out_ptr0 + (x2 + 32 * x4 + 1024 * x5 + 32768 * x7 + 1048576 *
        x3), tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_12(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 504064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex % 504064
    x1 = xindex // 32
    x2 = xindex % 32
    x5 = xindex // 15752
    x6 = xindex % 15752
    x3 = xindex // 504064
    x7 = xindex // 504064
    tmp0 = tl.load(in_ptr0 + (x1 + 504064 * x4 + 77650240 * x5 + 776502400 *
        x7 + 7765024000 * x3), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < 2
    tmp4 = tmp2 & tmp3
    tl.store(out_ptr0 + (x2 + 32 * x4 + 1024 * x5 + 32768 * x7 + 1048576 *
        x3), tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_13(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 510336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex % 510336
    x1 = xindex // 32
    x2 = xindex % 32
    x5 = xindex // 15951
    x6 = xindex % 15951
    x3 = xindex // 510432
    x7 = xindex // 510336
    tmp0 = tl.load(in_ptr0 + (x1 + 510432 * x4 + 78504960 * x5 + 785049600
        * x7 + 7850496000 * x3), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < 2
    tmp4 = tmp2 & tmp3
    tl.store(out_ptr0 + (x2 + 32 * x4 + 1024 * x5 + 32768 * x7 + 1048576 *
        x3), tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_14(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 732864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex % 732864
    x1 = xindex // 32
    x2 = xindex % 32
    x5 = xindex // 22802
    x6 = xindex % 22802
    x3 = xindex // 732864
    x7 = xindex // 732864
    tmp0 = tl.load(in_ptr0 + (x1 + 732864 * x4 + 11742240 * x5 + 117422400
        * x7 + 1174224000 * x3), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < 2
    tmp4 = tmp2 & tmp3
    tl.store(out_ptr0 + (x2 + 32 * x4 + 1024 * x5 + 32768 * x7 + 1048576 *
        x3), tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_15(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 544352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex % 544352
    x1 = xindex // 32
    x2 = xindex % 32
    x5 = xindex // 17011
    x6 = xindex % 17011
    x3 = xindex // 544352
    x7 = xindex // 544352
    tmp0 = tl.load(in_ptr0 + (x1 + 544352 * x4 + 8711760 * x5 + 87117600 *
        x7 + 871176000 * x3), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < 2
    tmp4 = tmp2 & tmp3
    tl.store(out_ptr0 + (x2 + 32 * x4 + 1024 * x5 + 32768 * x7 + 1048576 *
        x3), tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_16(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1097376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex % 1097376
    x1 = xindex // 32
    x2 = xindex % 32
    x5 = xindex // 34218
    x6 = xindex % 34218
    x3 = xindex // 1097376
    x7 = xindex // 1097376
    tmp0 = tl.load(in_ptr0 + (x1 + 1097376 * x4 + 34303680 * x5 + 343036800
        * x7 + 3430368000 * x3), xmask, eviction_policy='evict_last',
        other=0.0)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < 2
    tmp4 = tmp2 & tmp3
    tl.store(out_ptr0 + (x2 + 32 * x4 + 1024 * x5 + 32768 * x7 + 1048576 *
        x3), tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_17(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2194752
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex % 2194752
    x1 = xindex // 32
    x2 = xindex % 32
    x5 = xindex // 67328
    x6 = xindex % 67328
    x3 = xindex // 2194752
    x7 = xindex // 2194752
    tmp0 = tl.load(in_ptr0 + (x1 + 2194752 * x4 + 69793024 * x5 + 697930240
        * x7 + 6979302400 * x3), xmask, eviction_policy='evict_last',
        other=0.0)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < 2
    tmp4 = tmp2 & tmp3
    tl.store(out_ptr0 + (x2 + 32 * x4 + 1024 * x5 + 32768 * x7 + 1048576 *
        x3), tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_18(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 266240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex % 266240
    x1 = xindex // 32
    x2 = xindex % 32
    x5 = xindex // 836
    x6 = xindex % 836
    x3 = xindex // 26656
    x7 = xindex // 266240
    tmp0 = tl.load(in_ptr0 + (x1 + 26656 * x4 + 2281856 * x5 + 22818560 *
        x7 + 228185600 * x3), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < 2
    tmp4 = tmp2 & tmp3
    tl.store(out_ptr0 + (x2 + 32 * x4 + 1024 * x5 + 32768 * x7 + 1048576 *
        x3), tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_19(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 287232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex % 287232
    x1 = xindex // 32
    x2 = xindex % 32
    x5 = xindex // 898
    x6 = xindex % 898
    x3 = xindex // 28736
    x7 = xindex // 287232
    tmp0 = tl.load(in_ptr0 + (x1 + 28736 * x4 + 2535360 * x5 + 25353600 *
        x7 + 253536000 * x3), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < 2
    tmp4 = tmp2 & tmp3
    tl.store(out_ptr0 + (x2 + 32 * x4 + 1024 * x5 + 32768 * x7 + 1048576 *
        x3), tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_20(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 504064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex % 504064
    x1 = xindex // 32
    x2 = xindex % 32
    x5 = xindex // 15752
    x6 = xindex % 15752
    x3 = xindex // 504064
    x7 = xindex // 504064
    tmp0 = tl.load(in_ptr0 + (x1 + 504064 * x4 + 77650240 * x5 + 776502400
        * x7 + 7765024000 * x3), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < 2
    tmp4 = tmp2 & tmp3
    tl.store(out_ptr0 + (x2 + 32 * x4 + 1024 * x5 + 32768 * x7 + 1048576 *
        x3), tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_21(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 510336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex % 510336
    x1 = xindex // 32
    x2 = xindex % 32
    x5 = xindex // 15951
    x6 = xindex % 15951
    x3 = xindex // 510432
    x7 = xindex // 510336
    tmp0 = tl.load(in_ptr0 + (x1 + 510432 * x4 + 78504960 * x5 + 785049600
        * x7 + 7850496000 * x3), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < 2
    tmp4 = tmp2 & tmp3
    tl.store(out_ptr0 + (x2 + 32 * x4 + 1024 * x5 + 32768 * x7 + 1048576 *
        x3), tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_22(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 732864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex % 732864
    x1 = xindex // 32
    x2 = xindex % 32
    x5 = xindex // 22802
    x6 = xindex % 22802
    x3 = xindex // 732864
    x7 = xindex // 732864
    tmp0 = tl.load(in_ptr0 + (x1 + 732864 * x4 + 11742240 * x5 + 117422400
        * x7 + 1174224000 * x3), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < 2
    tmp4 = tmp2 & tmp3
    tl.store(out_ptr0 + (x2 + 32 * x4 + 1024 * x5 + 32768 * x7 + 1048576 *
        x3), tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_max_pool2d_with_indices_23(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 544352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex % 544352
    x1 = xindex // 32
    x2 = xindex % 32
    x5 = xindex // 17011
    x6 = xindex % 17011
    x3 = xindex // 544352
    x7 = xindex // 544352
    tmp0 = tl.load(in_ptr0 + (x1 + 544352 * x4 + 8711760 * x5 + 87117600 *
        x7 + 871176000 * x3), xmask, eviction_policy='evict_last', other=0.0)
    tmp1