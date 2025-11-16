import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x0, tmp1, xmask)


@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 128 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (128 + 128 * x1), xmask, eviction_policy=
        'evict_last')
    tmp4 = tl.load(in_ptr0 + (256 + 128 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (512 + 128 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 3.0
    tmp8 = tmp5 / tmp6
    tmp9 = tmp0 - tmp8
    tmp10 = 0.0078125
    tmp11 = tmp9 * tmp10
    tl.store(out_ptr0 + x2, tmp11, xmask)


@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x0, tmp1, xmask)


@triton.jit
def triton_poi_fused__to_copy_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 512 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (512 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp4 = tl.load(in_ptr0 + (1024 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (2048 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 4.0
    tmp8 = tmp5 / tmp6
    tmp9 = tmp0 - tmp8
    tmp10 = 0.0078125
    tmp11 = tmp9 * tmp10
    tl.store(out_ptr0 + x2, tmp11, xmask)


@triton.jit
def triton_poi_fused__to_copy_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x0, tmp1, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10) = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_4, (64,), (1,))
    assert_size_stride(primals_5, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_6, (64,), (1,))
    assert_size_stride(primals_7, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_8, (64,), (1,))
    assert_size_stride(primals_9, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_10, (128,), (1,))
    assert_size_stride(primals_11, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_12, (128,), (1,))
    assert_size_stride(primals_13, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_14, (128,), (1,))
    assert_size_stride(primals_15, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_16, (128,), (1,))
    assert_size_stride(primals_17, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_18, (256,), (1,))
    assert_size_stride(primals_19, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_20, (256,), (1,))
    assert_size_stride(primals_21, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_22, (256,), (1,))
    assert_size_stride(primals_23, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_24, (256,), (1,))
    assert_size_stride(primals_25, (10, 256), (256, 1))
    assert_size_stride(primals_26, (10,), (1,))
    assert_size_stride(primals_27, (8, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_28, (8,), (1,))
    assert_size_stride(primals_29, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_30, (8,), (1,))
    assert_size_stride(primals_31, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_32, (8,), (1,))
    assert_size_stride(primals_33, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_34, (8,), (1,))
    assert_size_stride(primals_35, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_36, (8,), (1,))
    assert_size_stride(primals_37, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_38, (8,), (1,))
    assert_size_stride(primals_39, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_40, (8,), (1,))
    assert_size_stride(primals_41, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_42, (8,), (1,))
    assert_size_stride(primals_43, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_44, (8,), (1,))
    assert_size_stride(primals_45, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_46, (8,), (1,))
    assert_size_stride(primals_47, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_48, (8,), (1,))
    assert_size_stride(primals_49, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_50, (8,), (1,))
    assert_size_stride(primals_51, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_52, (8,), (1,))
    assert_size_stride(primals_53, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_54, (8,), (1,))
    assert_size_stride(primals_55, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_56, (8,), (1,))
    assert_size_stride(primals_57, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_58, (8,), (1,))
    assert_size_stride(primals_59, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_60, (8,), (1,))
    assert_size_stride(primals_61, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_62, (8,), (1,))
    assert_size_stride(primals_63, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_64, (8,), (1,))
    assert_size_stride(primals_65, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_66, (8,), (1,))
    assert_size_stride(primals_67, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_68, (8,), (1,))
    assert_size_stride(primals_69, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_70, (8,), (1,))
    assert_size_stride(primals_71, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_72, (8,), (1,))
    assert_size_stride(primals_73, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_74, (8,), (1,))
    assert_size_stride(primals_75, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_76, (8,), (1,))
    assert_size_stride(primals_77, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_78, (8,), (1,))
    assert_size_stride(primals_79, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_80, (8,), (1,))
    assert_size_stride(primals_81, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_82, (8,), (1,))
    assert_size_stride(primals_83, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_84, (8,), (1,))
    assert_size_stride(primals_85, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_86, (8,), (1,))
    assert_size_stride(primals_87, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_88, (8,), (1,))
    assert_size_stride(primals_89, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_90, (8,), (1,))
    assert_size_stride(primals_91, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_92, (8,), (1,))
    assert_size_stride(primals_93, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_94, (8,), (1,))
    assert_size_stride(primals_95, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_96, (8,), (1,))
    assert_size_stride(primals_97, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_98, (8,), (1,))
    assert_size_stride(primals_99, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_100, (8,), (1,))
    assert_size_stride(primals_101, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_102, (8,), (1,))
    assert_size_stride(primals_103, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_104, (8,), (1,))
    assert_size_stride(primals_105, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_106, (8,), (1,))
    assert_size_stride(primals_107, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_108, (8,), (1,))
    assert_size_stride(primals_109, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_110, (8,), (1,))
    assert_size_stride(primals_111, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_112, (8,), (1,))
    assert_size_stride(primals_113, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_114, (8,), (1,))
    assert_size_stride(primals_115, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_116, (8,), (1,))
    assert_size_stride(primals_117, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_118, (8,), (1,))
    assert_size_stride(primals_119, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_120, (8,), (1,))
    assert_size_stride(primals_121, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_122, (8,), (1,))
    assert_size_stride(primals_123, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_124, (8,), (1,))
    assert_size_stride(primals_125, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_126, (8,), (1,))
    assert_size_stride(primals_127, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_128, (8,), (1,))
    assert_size_stride(primals_129, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_130, (8,), (1,))
    assert_size_stride(primals_131, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_132, (8,), (1,))
    assert_size_stride(primals_133, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_134, (8,), (1,))
    assert_size_stride(primals_135, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_136, (8,), (1,))
    assert_size_stride(primals_137, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_138, (8,), (1,))
    assert_size_stride(primals_139, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_140, (8,), (1,))
    assert_size_stride(primals_141, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_142, (8,), (1,))
    assert_size_stride(primals_143, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_144, (8,), (1,))
    assert_size_stride(primals_145, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_146, (8,), (1,))
    assert_size_stride(primals_147, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_148, (8,), (1,))
    assert_size_stride(primals_149, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_150, (8,), (1,))
    assert_size_stride(primals_151, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_152, (8,), (1,))
    assert_size_stride(primals_153, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_154, (8,), (1,))
    assert_size_stride(primals_155, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_156, (8,), (1,))
    assert_size_stride(primals_157, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_158, (8,), (1,))
    assert_size_stride(primals_159, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_160, (8,), (1,))
    assert_size_stride(primals_161, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_162, (8,), (1,))
    assert_size_stride(primals_163, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_164, (8,), (1,))
    assert_size_stride(primals_165, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_166, (8,), (1,))
    assert_size_stride(primals_167, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_168, (8,), (1,))
    assert_size_stride(primals_169, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_170, (8,), (1,))
    assert_size_stride(primals_171, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_172, (8,), (1,))
    assert_size_stride(primals_173, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_174, (8,), (1,))
    assert_size_stride(primals_175, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_176, (8,), (1,))
    assert_size_stride(primals_177, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_178, (8,), (1,))
    assert_size_stride(primals_179, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_180, (8,), (1,))
    assert_size_stride(primals_181, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_182, (8,), (1,))
    assert_size_stride(primals_183, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_184, (8,), (1,))
    assert_size_stride(primals_185, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_186, (8,), (1,))
    assert_size_stride(primals_187, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_188, (8,), (1,))
    assert_size_stride(primals_189, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_190, (8,), (1,))
    assert_size_stride(primals_191, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_192, (8,), (1,))
    assert_size_stride(primals_193, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_194, (8,), (1,))
    assert_size_stride(primals_195, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_196, (8,), (1,))
    assert_size_stride(primals_197, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_198, (8,), (1,))
    assert_size_stride(primals_199, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_200, (8,), (1,))
    assert_size_stride(primals_201, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_202, (8,), (1,))
    assert_size_stride(primals_203, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_204, (8,), (1,))
    assert_size_stride(primals_205, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_206, (8,), (1,))
    assert_size_stride(primals_207, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_208, (8,), (1,))
    assert_size_stride(primals_209, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_210, (8,), (1,))
    assert_size_stride(primals_211, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_212, (8,), (1,))
    assert_size_stride(primals_213, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_214, (8,), (1,))
    assert_size_stride(primals_215, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_216, (8,), (1,))
    assert_size_stride(primals_217, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_218, (8,), (1,))
    assert_size_stride(primals_219, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_220, (8,), (1,))
    assert_size_stride(primals_221, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_222, (8,), (1,))
    assert_size_stride(primals_223, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_224, (8,), (1,))
    assert_size_stride(primals_225, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_226, (8,), (1,))
    assert_size_stride(primals_227, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_228, (8,), (1,))
    assert_size_stride(primals_229, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_230, (8,), (1,))
    assert_size_stride(primals_231, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_232, (8,), (1,))
    assert_size_stride(primals_233, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_234, (8,), (1,))
    assert_size_stride(primals_235, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_236, (8,), (1,))
    assert_size_stride(primals_237, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_238, (8,), (1,))
    assert_size_stride(primals_239, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_240, (8,), (1,))
    assert_size_stride(primals_241, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_242, (8,), (1,))
    assert_size_stride(primals_243, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_244, (8,), (1,))
    assert_size_stride(primals_245, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_246, (8,), (1,))
    assert_size_stride(primals_247, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_248, (8,), (1,))
    assert_size_stride(primals_249, (8, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_250, (8,), (1,))
    assert_size_stride(primals_251, (8, 8, 3