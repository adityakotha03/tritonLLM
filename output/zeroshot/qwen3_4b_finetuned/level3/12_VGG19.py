import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_relu_0(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 24176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 224
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp3 <= tmp2
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr0 + (224 + x2), tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_1(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 19600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 49
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp3 <= tmp2
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr0 + (49 + x2), tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_2(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 16800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 36
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp3 <= tmp2
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr0 + (36 + x2), tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_3(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 12250
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 25
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp3 <= tmp2
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr0 + (25 + x2), tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_4(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 8400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 16
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp3 <= tmp2
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr0 + (16 + x2), tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_5(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 4900
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp3 <= tmp2
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr0 + (4 + x2), tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_6(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 4900
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp3 <= tmp2
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr0 + (4 + x2), tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_7(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = tmp4 <= tmp3
    tl.store(out_ptr0 + x0, tmp3, xmask)
    tl.store(out_ptr0 + (1000 + x0), tmp5, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_8(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = tmp4 <= tmp3
    tl.store(out_ptr0 + x0, tmp3, xmask)
    tl.store(out_ptr0 + (1000 + x0), tmp5, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_9(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = tmp4 <= tmp3
    tl.store(out_ptr0 + x0, tmp3, xmask)
    tl.store(out_ptr0 + (1000 + x0), tmp5, xmask)


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
        primals_613, primals_614, primals_615, primals_616, primals_617,
        primals_618, primals_619, primals_620, primals_621, primals_622,
        primals_623, primals_624, primals_625, primals_626, primals_627,
        primals_628, primals_629, primals_630, primals_631, primals_632,
        primals_633, primals_634, primals_635, primals_636, primals_637,
        primals_638, primals_639, primals_640, primals_641, primals_642,
        primals_643, primals_644, primals_645, primals_646, primals_647,
        primals_648, primals_649, primals_650, primals_651, primals_652,
        primals_653, primals_654, primals_655, primals_656, primals_657,
        primals_658, primals_659, primals_660, primals_661, primals_662,
        primals_663, primals_664, primals_665, primals_666, primals_667,
        primals_668, primals_669, primals_670, primals_671, primals_672,
        primals_673, primals_674, primals_675, primals_676, primals_677,
        primals_678, primals_679, primals_680, primals_681, primals_682,
        primals_683, primals_684, primals_685, primals_686, primals_687,
        primals_688, primals_689, primals_690, primals_691, primals_692,
        primals_693, primals_694, primals_695, primals_696, primals_697,
        primals_698, primals_699, primals_700, primals_701, primals_702,
        primals_703, primals_704, primals_705, primals_706, primals_707,
        primals_708, primals_709, primals_710, primals_711, primals_712,
        primals_713, primals_714, primals_715, primals_716, primals_717,
        primals_718, primals_719, primals_720, primals_721, primals_722,
        primals_723, primals_724, primals_725, primals_726, primals_727,
        primals_728, primals_729, primals_730, primals_731, primals_732,
        primals_733, primals_734, primals_735, primals_736, primals_737,
        primals_738, primals_739, primals_740, primals_741, primals_742,
        primals_743, primals_744, primals_745, primals_746, primals_747,
        primals_748, primals_749, primals_750, primals_751, primals_752,
        primals_753, primals_754, primals_755, primals_756, primals_757,
        primals_758, primals_759, primals_760, primals_761, primals_762,
        primals_763, primals_764, primals_765, primals_766, primals_767,
        primals_768, primals_769, primals_770, primals_771, primals_772,
        primals_773, primals_774, primals_775, primals_776, primals_777,
        primals_778, primals_779, primals_780, primals_781, primals_782,
        primals_783, primals_784, primals_785, primals_786, primals_787,
        primals_788, primals_789, primals_790, primals_791, primals_792,
        primals_793, primals_794, primals_795, primals_796, primals_797,
        primals_798, primals_799, primals_800, primals_801, primals_802,
        primals_803, primals_804, primals_805, primals_806, primals_807,
        primals_808, primals_809, primals_810, primals_811, primals_812,
        primals_813, primals_814, primals_815, primals_816, primals_817,
        primals_818, primals_819, primals_820, primals_821, primals_822,
        primals_823, primals_824, primals_825, primals_826, primals_827,
        primals_828, primals_829, primals_8