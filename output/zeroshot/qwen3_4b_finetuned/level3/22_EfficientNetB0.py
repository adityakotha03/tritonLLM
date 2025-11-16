import torch
from torch._inductor.select_algorithm import extern_kernels
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
def triton_poi_fused_convolution_relu_threshold_backward_0(in_out_ptr0,
    in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_1(in_out_ptr0,
    in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_2(in_out_ptr0,
    in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_3(in_out_ptr0,
    in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_4(in_out_ptr0,
    in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_5(in_out_ptr0,
    in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_6(in_out_ptr0,
    in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_7(in_out_ptr0,
    in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 15360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_8(in_out_ptr0,
    in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 25600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_9(in_out_ptr0,
    in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 51200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_10(in_out_ptr0,
    in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 102400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_11(in_out_ptr0,
    in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 153600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_12(in_out_ptr0,
    in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 256000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_13(in_out_ptr0,
    in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 512000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_14(in_out_ptr0,
    in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr0 + x3, tmp6, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13, primals_14, primals_15, primals_16, primals_17,
        primals_18, primals_19, primals_20, primals_21, primals_22,
        primals_23, primals_24, primals_25, primals_26, primals_27,
        primals_28, primals_29, primals_30, primals_31, primals_32,
        primals_33, primals_34, primals_35, primals_36, primals_37,
        primals_38, primals_39, primals_40, primals_41, primals_42,
        primals_43, primals_44, primals_45, primals_46, primals_47,
        primals_48, primals_49, primals_50, primals_51, primals_52,
        primals_53, primals_54, primals_55, primals_56, primals_57,
        primals_58, primals_59, primals_60, primals_61, primals_62,
        primals_63, primals_64, primals_65, primals_66, primals_67,
        primals_68, primals_69, primals_70, primals_71, primals_72,
        primals_73, primals_74, primals_75, primals_76, primals_77,
        primals_78, primals_79, primals_80, primals_81, primals_82,
        primals_83, primals_84, primals_85, primals_86, primals_87,
        primals_88, primals_89, primals_90, primals_91, primals_92,
        primals_93, primals_94, primals_95, primals_96, primals_97,
        primals_98, primals_99, primals_100, primals_101, primals_102,
        primals_103, primals_104, primals_105, primals_106, primals_107,
        primals_108, primals_109, primals_110, primals_111, primals_112,
        primals_113, primals_114, primals_115, primals_116, primals_117,
        primals_118, primals_119, primals_120, primals_121, primals_122,
        primals_123, primals_124, primals_125, primals_126, primals_127,
        primals_128, primals_129, primals_130, primals_131, primals_132,
        primals_133, primals_134, primals_135, primals_136, primals_137,
        primals_138, primals_139, primals_140, primals_141, primals_142,
        primals_143, primals_144, primals_145, primals_146, primals_147,
        primals_148, primals_149, primals_150, primals_151, primals_152,
        primals_153, primals_154, primals_155, primals_156, primals_157,
        primals_158, primals_159, primals_160, primals_161, primals_162,
        primals_163, primals_164, primals_165, primals_166, primals_167,
        primals_168, primals_169, primals_170, primals_171, primals_172,
        primals_173, primals_174, primals_175, primals_176, primals_177,
        primals_178, primals_179, primals_180, primals_181, primals_182,
        primals_183, primals_184, primals_185, primals_186, primals_187,
        primals_188, primals_189, primals_190, primals_191, primals_192,
        primals_193, primals_194, primals_195, primals_196, primals_197,
        primals_198, primals_199, primals_200, primals_201, primals_202,
        primals_203, primals_204, primals_205, primals_206, primals_207,
        primals_208, primals_209, primals_210, primals_211, primals_212,
        primals_213, primals_214, primals_215, primals_216, primals_217,
        primals_218, primals_219, primals_220, primals_221, primals_222,
        primals_223, primals_224, primals_225, primals_226, primals_227,
        primals_228, primals_229, primals_230, primals_231, primals_232,
        primals_233, primals_234, primals_235, primals_236, primals_237,
        primals_238, primals_239, primals_240, primals_241, primals_242,
        primals_243, primals_244, primals_245, primals_246, primals_247,
        primals_248, primals_249, primals_250, primals_251, primals_252,
        primals_253, primals_254, primals_255, primals_256, primals_257,
        primals_258, primals_259, primals_260, primals_261, primals_262,
        primals_263, primals_264, primals_265, primals_266, primals_267,
        primals_268, primals_269, primals_270, primals_271, primals_272,
        primals_273, primals_274, primals_275, primals_276, primals_277,
        primals_278, primals_279, primals_280, primals_281, primals_282,
        primals_283, primals_284, primals_285, primals_286, primals_287,
        primals_288, primals_289, primals_290, primals_291, primals_292,
        primals_293, primals_294, primals_295, primals_296, primals_297,
        primals_298, primals_299, primals_300, primals_301, primals_302,
        primals_303, primals_304, primals_305, primals_306, primals_307,
        primals_308, primals_309, primals_310, primals_311, primals_312,
        primals_313, primals_314, primals_315, primals_316, primals_317,
        primals_318, primals_319, primals_320, primals_321, primals_322,
        primals_323, primals_324, primals_325, primals_326, primals_327,
        primals_328, primals_329, primals_330, primals_331, primals_332,
        primals_333, primals_334, primals_335, primals_336, primals_337,
        primals_338, primals_339, primals_340, primals_341, primals_342,
        primals_343, primals_344, primals_345, primals_346, primals_347,
        primals_348, primals_349, primals_350, primals_351, primals_352,
        primals_353, primals_354, primals_355, primals_356, primals_357,
        primals_358, primals_359, primals_360, primals_361, primals_362,
        primals_363, primals_364, primals_365, primals_366, primals_367,
        primals_368, primals_369, primals_370, primals_371, primals_372,
        primals_373, primals_374, primals_375, primals_376, primals_377,
        primals_378, primals_379, primals_380, primals_381, primals_382,
        primals_383, primals_384, primals_385, primals_386, primals_387,
        primals_388, primals_389, primals_390, primals_391, primals_392,
        primals_393, primals_394, primals_395, primals_396, primals_397,
        primals_398, primals_399, primals_400, primals_401, primals_402,
        primals_403, primals_404, primals_405, primals_406, primals_407,
        primals_408, primals_409, primals_410, primals_411, primals_412,
        primals_413, primals_414, primals_415, primals_416, primals_417,
        primals_418, primals_419, primals_420, primals_421, primals_422,
        primals_423, primals_424, primals_425, primals_426, primals_427,
        primals_428, primals_429, primals_430, primals_431, primals_432,
        primals_433, primals_434, primals_435, primals_436, primals_437,
        primals_438, primals_439, primals_440, primals_441, primals_442,
        primals_443, primals_444, primals_445, primals_446, primals_447,
        primals_448, primals_449, primals_450, primals_451, primals_452,
        primals_453, primals_454, primals_455, primals_456, primals_457,
        primals_458, primals_459, primals_460, primals_461, primals_462,
        primals_463, primals_464, primals_465, primals_466, primals_467,
        primals_468, primals_469, primals_470, primals_471, primals_472,
        primals_473, primals_474, primals_475, primals_476, primals_477,
        primals_478, primals_479, primals_480, primals_481, primals_482,
        primals_483, primals_484, primals_485, primals_486, primals_487,
        primals_488, primals_489, primals_490, primals_491, primals_492,
        primals_493, primals_494, primals_495, primals_496, primals_497,
        primals_498, primals_499, primals_500, primals_501, primals_502,
        primals_503, primals_504, primals_505, primals_506, primals_507,
        primals_508, primals_509, primals_510, primals_511, primals_512,
        primals_513, primals_514, primals_515, primals_516, primals_517,
        primals_518, primals_519, primals_520, primals_521, primals_522,
        primals_523, primals_524, primals_525, primals_526, primals_527,
        primals_528, primals_529, primals_530, primals_531, primals_532,
        primals_533, primals_534, primals_535, primals_536, primals_537,
        primals_538, primals_539, primals_540, primals_541, primals_542,
        primals_543, primals_544, primals_545, primals_546, primals_547,
        primals_548, primals_549, primals_550, primals_551, primals_552,
        primals_553, primals_554, primals_555, primals_556, primals_557,
        primals_558, primals_559, primals_560, primals_561, primals_562,
        primals_563, primals_564, primals_565, primals_566, primals_567,
        primals_568, primals_569, primals_570, primals_571, primals_572,
        primals_573, primals_574, primals_575, primals_576, primals_577,
        primals_578, primals_579, primals_580, primals_581, primals_582,
        primals_583, primals_584, primals_585, primals_586, primals_587,
        primals_588, primals_589, primals_590, primals_591, primals_592,
        primals_593, primals_594, primals_595, primals_596, primals_597,
        primals_598, primals_599, primals_600, primals_601, primals_602,
        primals_603, primals_604, primals_605, primals_606, primals_607,
        primals_608, primals_609, primals_610, primals_611, primals_612,
        primals_613, primals_6