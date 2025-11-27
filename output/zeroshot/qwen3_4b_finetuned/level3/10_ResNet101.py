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
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 192
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 49 * y3), xmask & ymask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 147 * y1), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 192
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 49 * y3), xmask & ymask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 147 * y1), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 256
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 16
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask & ymask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 16 * x2 + 144 * y1), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 256
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 16
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask & ymask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 16 * x2 + 144 * y1), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 16
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask, eviction_policy='evict_last'
        )
    tl.store(out_ptr0 + (y0 + 16 * x2 + 144 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_relu_5(in_out_ptr0,
    in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1024
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (r2 + 64 * x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 64.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tl.store(in_out_ptr0 + (r2 + 64 * x3), tmp2, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + x3, tmp23, xmask)
    tl.store(out_ptr0 + x3, tmp12, xmask)
    tl.store(out_ptr1 + x3, tmp18, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_6(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 16 % 16
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_7(in_out_ptr0,
    in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 16 % 32
    x2 = xindex // 512
    x4 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(out_ptr1 + (x4 + 512 * x2), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(out_ptr2 + (x4 + 512 * x2), xmask, eviction_policy=
        'evict_last')
    tmp8 = tl.load(out_ptr3 + (x4 + 512 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6 * tmp8
    tmp9 = tl.broadcast_to(tmp7, [XBLOCK])
    tl.where(xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 0)
    tmp16 = tl.full([1], 512, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 0)
    tmp25 = 512.0
    tmp26 = tmp24 / tmp25
    tmp27 = 1e-05
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tl.store(in_out_ptr0 + x3, tmp7, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + x3, tmp29, xmask)
    tl.store(out_ptr0 + x3, tmp18, xmask)
    tl.store(out_ptr2 + x3, tmp24, xmask)
    tl.store(out_ptr3 + x3, tmp28, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_8(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 16 % 32
    x2 = xindex // 512
    x4 = xindex % 512
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_9(in_out_ptr0,
    in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 16 % 64
    x2 = xindex // 1024
    x4 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(out_ptr1 + (x4 + 1024 * x2), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(out_ptr2 + (x4 + 1024 * x2), xmask, eviction_policy=
        'evict_last')
    tmp8 = tl.load(out_ptr3 + (x4 + 1024 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6 * tmp8
    tmp9 = tl.broadcast_to(tmp7, [XBLOCK])
    tl.where(xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 0)
    tmp16 = tl.full([1], 1024, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 0)
    tmp25 = 1024.0
    tmp26 = tmp24 / tmp25
    tmp27 = 1e-05
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tl.store(in_out_ptr0 + x3, tmp7, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + x3, tmp29, xmask)
    tl.store(out_ptr0 + x3, tmp18, xmask)
    tl.store(out_ptr2 + x3, tmp24, xmask)
    tl.store(out_ptr3 + x3, tmp28, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_10(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 16 % 64
    x2 = xindex // 1024
    x4 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_11(in_out_ptr0,
    in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 16 % 128
    x2 = xindex // 2048
    x4 = xindex % 2048
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(out_ptr1 + (x4 + 2048 * x2), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(out_ptr2 + (x4 + 2048 * x2), xmask, eviction_policy=
        'evict_last')
    tmp8 = tl.load(out_ptr3 + (x4 + 2048 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6 * tmp8
    tmp9 = tl.broadcast_to(tmp7, [XBLOCK])
    tl.where(xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 0)
    tmp16 = tl.full([1], 2048, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 0)
    tmp25 = 2048.0
    tmp26 = tmp24 / tmp25
    tmp27 = 1e-05
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tl.store(in_out_ptr0 + x3, tmp7, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + x3, tmp29, xmask)
    tl.store(out_ptr0 + x3, tmp18, xmask)
    tl.store(out_ptr2 + x3, tmp24, xmask)
    tl.store(out_ptr3 + x3, tmp28, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_12(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 16 % 128
    x2 = xindex // 2048
    x4 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused__to_copy_13(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__unsafe_index_add_mul_14(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10,
    in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17,
    in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24,
    in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31,
    in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38,
    in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45,
    in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52,
    in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59,
    in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66,
    in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73,
    in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80,
    in_ptr81, in_ptr82, in_ptr83, in_ptr84, in_ptr85, in_ptr86, in_ptr87,
    in_ptr88, in_ptr89, in_ptr90, in_ptr91, in_ptr92, in_ptr93, in_ptr94,
    in_ptr95, in_ptr96, in_ptr97, in_ptr98, in_ptr99, in_ptr100, in_ptr101,
    in_ptr102, in_ptr103, in_ptr104, in_ptr105, in_ptr106, in_ptr107,
    in_ptr108, in_ptr109, in_ptr110, in_ptr111, in_ptr112, in_ptr113,
    in_ptr114, in_ptr115, in_ptr116, in_ptr117, in_ptr118, in_ptr119,
    in_ptr120, in_ptr121, in_ptr122, in_ptr123, in_ptr124, in_ptr125,
    in_ptr126, in_ptr127, in_ptr128, in_ptr129, in_ptr130, in_ptr131,
    in_ptr132, in_ptr133, in_ptr134, in_ptr135, in_ptr136, in_ptr137,
    in_ptr138, in_ptr139, in_ptr140, in_ptr141, in_ptr142, in_ptr143,
    in_ptr144, in_ptr145, in_ptr146, in_ptr147, in_ptr148, in_ptr149,
    in_ptr150, in_ptr151, in_ptr152, in_ptr153, in_ptr154, in_ptr155,
    in_ptr156, in_ptr157, in_ptr158, in_ptr159, in_ptr160, in_ptr161,
    in_ptr162, in_ptr163, in_ptr164, in_ptr165, in_ptr166, in_ptr167,
    in_ptr168, in_ptr169, in_ptr170, in_ptr171, in_ptr172, in_ptr173,
    in_ptr174, in_ptr175, in_ptr176, in_ptr177, in_ptr178, in_ptr179,
    in_ptr180, in_ptr181, in_ptr182, in_ptr183, in_ptr184, in_ptr185,
    in_ptr186, in_ptr187, in_ptr188, in_ptr189, in_ptr190, in_ptr191,
    in_ptr192, in_ptr193, in_ptr194, in_ptr195, in_ptr196, in_ptr197,
    in_ptr198, in_ptr199, in_ptr200, in_ptr201, in_ptr202, in_ptr203,
    in_ptr204, in_ptr205, in_ptr206, in_ptr207, in_ptr208, in_ptr209,
    in_ptr210, in_ptr211, in_ptr212, in_ptr213, in_ptr214, in_ptr215,
    in_ptr216, in_ptr217, in_ptr218, in_ptr219, in_ptr220, in_ptr221,
    in_ptr222, in_ptr223, in_ptr224, in_ptr225, in_ptr226, in_ptr227,
    in_ptr228, in_ptr229, in_ptr230, in_ptr231, in_ptr232, in_ptr233,
    in_ptr234, in_ptr235, in_ptr236, in_ptr237, in_ptr238, in_ptr239,
    in_ptr240, in_ptr241, in_ptr242, in_ptr243, in_ptr244, in_ptr245,
    in_ptr246, in_ptr247, in_ptr248, in_ptr249, in_ptr250, in_ptr251,
    in_ptr252, in_ptr253, in_ptr254, in_ptr255, in_ptr256, in_ptr257,
    in_ptr258, in_ptr259, in_ptr260, in_ptr261, in_ptr262, in_ptr263,
    in_ptr264, in_ptr265, in_ptr266, in_ptr267, in_ptr268, in_ptr269,
    in_ptr270, in_ptr271, in_ptr272, in_ptr273, in_ptr274, in_ptr275,
    in_ptr276, in_ptr277, in_ptr278, in_ptr279, in_ptr280, in_ptr281,
    in_ptr282, in_ptr283, in_ptr284, in_ptr285, in_ptr286, in_ptr287,
    in_ptr288, in_ptr289, in_ptr290, in_ptr291, in_ptr292, in_ptr293,
    in_ptr294, in_ptr295, in_ptr296, in_ptr297, in_ptr298, in_ptr299,
    in_ptr300, in_ptr301, in_ptr302, in_ptr303, in_ptr304, in_ptr305,
    in_ptr306, in_ptr307, in_ptr308, in_ptr309, in_ptr310, in_ptr311,
    in_ptr312, in_ptr313, in_ptr314, in_ptr315, in_ptr316, in_ptr317,
    in_ptr318, in_ptr319, in_ptr320, in_ptr321, in_ptr322, in_ptr323,
    in_ptr324, in_ptr325, in_ptr326, in_ptr327, in_ptr328, in_ptr329,
    in_ptr330, in_ptr331, in_ptr332, in_ptr333, in_ptr334, in_ptr335,
    in_ptr336, in_ptr337, in_ptr338, in_ptr339, in_ptr340, in_ptr341,
    in_ptr342, in_ptr343, in_ptr344, in_ptr345, in_ptr346, in_ptr347,
    in_ptr348, in_ptr349, in_ptr350, in_ptr351, in_ptr352, in_ptr353,
    in_ptr354, in_ptr355, in_ptr356, in_ptr357, in_ptr358, in_ptr359,
    in_ptr360, in_ptr361, in_ptr362, in_ptr363, in_ptr364, in_ptr365,
    in_ptr366, in_ptr367, in_ptr368, in_ptr369, in_ptr370, in_ptr371,
    in_ptr372, in_ptr373, in_ptr374, in_ptr375, in_ptr376, in_ptr377,
    in_ptr378, in_ptr379, in_ptr380, in_ptr381, in_ptr382, in_ptr383,
    in_ptr384, in_ptr385, in_ptr386, in_ptr387, in_ptr388, in_ptr389,
    in_ptr390, in_ptr391, in_ptr392, in_ptr393, in_ptr394, in_ptr395,
    in_ptr396, in_ptr397, in_ptr398, in_ptr399, in_ptr400, in_ptr401,
    in_ptr402, in_ptr403, in_ptr404, in_ptr405, in_ptr406, in_ptr407,
    in_ptr408, in_ptr409, in_ptr410, in_ptr411, in_ptr412, in_ptr413,
    in_ptr414, in_ptr415, in_ptr416, in_ptr417, in_ptr418, in_ptr419,
    in_ptr420, in_ptr421, in_ptr422, in_ptr423, in_ptr424, in_ptr425,
    in_ptr426, in_ptr427, in_ptr428, in_ptr429, in_ptr430, in_ptr431,
    in_ptr432, in_ptr433, in_ptr434, in_ptr435, in_ptr436, in_ptr437,
    in_ptr438, in_ptr439, in_ptr440, in_ptr441, in_ptr442, in_ptr443,
    in_ptr444, in_ptr445, in_ptr446, in_ptr447, in_ptr448, in_ptr449,
    in_ptr450, in_ptr451, in_ptr452, in_ptr453, in_ptr454, in_ptr455,
    in_ptr456, in_ptr457, in_ptr458, in_ptr459, in_ptr460, in_ptr461,
    in_ptr462, in_ptr463, in_ptr464, in_ptr465, in_ptr466, in_ptr467,
    in_ptr468, in_ptr469, in_ptr470, in_ptr471, in_ptr472, in_ptr473,
    in_ptr474, in_ptr475, in_ptr476, in_ptr477, in_ptr478, in_ptr479,
   