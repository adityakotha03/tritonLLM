import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex // 1024 % 32
    tmp0 = tl.load(in_out_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, None)


@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x2 = xindex // 1024
    x1 = xindex // 32 % 32
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x2 + 32 * x1), xmask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + x3, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK:
    tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 8
    xnumel = 1600
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 5120 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1600 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_add_mean_sub_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex // 1024
    x2 = xindex % 1024
    x4 = xindex
    x1 = xindex // 32 % 32
    tmp0 = tl.load(in_out_ptr0 + x4, xmask)
    tmp1 = tl.load(in_ptr0 + x3, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (1024 + x2), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_out_ptr0 + (2048 + x2), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_out_ptr0 + (3072 + x2), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr1 + (x1 + 32 * x3), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr2 + (x1 + 32 * x3), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 / 1024.0
    tmp12 = tmp9 - tmp10
    tmp13 = tmp11 - tmp10
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14 * 0.5
    tmp16 = tmp2 + tmp15
    tl.store(in_out_ptr0 + x4, tmp16, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (32, 16, 3, 3, 3), (432, 27, 9, 3, 1))
    assert_size_stride(primals_2, (32,), (1,))
    assert_size_stride(primals_3, (16, 16, 16, 32, 32), (16777216, 1048576,
        65536, 2048, 64))
    assert_size_stride(primals_4, (32,), (1,))
    assert_size_stride(primals_5, (32,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(2, 
            2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (16, 32, 17, 33, 33), (3574336, 109216, 6400,
            196, 1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(3574336)](buf1, primals_2, 3574336
            , XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((16, 32, 17, 33, 33), (3574336, 109216,
            6400, 196, 1), torch.float32)
        triton_poi_fused__to_copy_1[grid(20480)](primals_1, buf2, 20480,
            XBLOCK=256, num_warps=4, num_stages=1)
        del primals_1
        buf3 = empty_strided_cuda((8, 1600), (1600, 1), torch.float32)
        extern_kernels.mm(buf1, buf2, out=buf3)
        del buf2
        buf5 = empty_strided_cuda((16, 32, 17, 33, 33), (3574336, 109216,
            6400, 196, 1), torch.float32)
        extern_kernels.convolution(buf3, primals_3, stride=(2, 2, 2),
            padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=1, bias=None)
        buf4 = empty_strided_cuda((16, 32, 17, 33, 33), (3574336, 109216,
            6400, 196, 1), torch.float32)
        triton_poi_fused__to_copy_2[grid(8, 1600)](buf5, buf4, 8, 1600,
            XBLOCK=64, YBLOCK=4, num_warps=1, num_stages=1)
        del buf5
        triton_poi_fused_add_mean_sub_3[grid(20480)](buf4, primals_4, primals_5,
            buf1, 20480, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_4
        del buf1
    return buf4, primals_3, primals_5


class ModelNew(nn.Module):
    """
    A 3D convolutional transpose layer followed by Batch Normalization and subtraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_3 = self.batch_norm.weight
        primals_4 = self.batch_norm.bias
        primals_5 = self.batch_norm.running_mean
        primals_6 = self.batch_norm.running_var
        primals_7 = self.batch_norm.weight
        primals_8 = self.batch_norm.bias
        primals_9 = self.batch_norm.weight
        primals_10 = self.batch_norm.bias
        primals_11 = self.batch_norm.weight
        primals_12 = self.batch_norm.bias
        primals_13 = self.batch_norm.running_mean
        primals_14 = self.batch_norm.running_var
        primals_15 = self.batch_norm.weight
        primals_16 = self.batch_norm.bias
        primals_17 = self.batch_norm.weight
        primals_18 = self.batch_norm.bias
        primals_19 = self.batch_norm.running_mean
        primals_20 = self.batch_norm.running_var
        primals_21 = self.batch_norm.weight
        primals_22 = self.batch_norm.bias
        primals_23 = self.batch_norm.running_mean
        primals_24 = self.batch_norm.running_var
        primals_25 = self.batch_norm.weight
        primals_26 = self.batch_norm.bias
        primals_27 = self.batch_norm.running_mean
        primals_28 = self.batch_norm.running_var
        primals_29 = self.batch_norm.weight
        primals_30 = self.batch_norm.bias
        primals_31 = self.batch_norm.running_mean
        primals_32 = self.batch_norm.running_var
        primals_33 = self.batch_norm.weight
        primals_34 = self.batch_norm.bias
        primals_35 = self.batch_norm.running_mean
        primals_36 = self.batch_norm.running_var
        primals_37 = self.batch_norm.weight
        primals_38 = self.batch_norm.bias
        primals_39 = self.batch_norm.running_mean
        primals_40 = self.batch_norm.running_var
        primals_41 = self.batch_norm.weight
        primals_42 = self.batch_norm.bias
        primals_43 = self.batch_norm.running_mean
        primals_44 = self.batch_norm.running_var
        primals_45 = self.batch_norm.weight
        primals_46 = self.batch_norm.bias
        primals_47 = self.batch_norm.running_mean
        primals_48 = self.batch_norm.running_var
        primals_49 = self.batch_norm.weight
        primals_50 = self.batch_norm.bias
        primals_51 = self.batch_norm.running_mean
        primals_52 = self.batch_norm.running_var
        primals_53 = self.batch_norm.weight
        primals_54 = self.batch_norm.bias
        primals_55 = self.batch_norm.running_mean
        primals_56 = self.batch_norm.running_var
        primals_57 = self.batch_norm.weight
        primals_58 = self.batch_norm.bias
        primals_59 = self.batch_norm.running_mean
        primals_60 = self.batch_norm.running_var
        primals_61 = self.batch_norm.weight
        primals_62 = self.batch_norm.bias
        primals_63 = self.batch_norm.running_mean
        primals_64 = self.batch_norm.running_var
        primals_65 = self.batch_norm.weight
        primals_66 = self.batch_norm.bias
        primals_67 = self.batch_norm.running_mean
        primals_68 = self.batch_norm.running_var
        primals_69 = self.batch_norm.weight
        primals_70 = self.batch_norm.bias
        primals_71 = self.batch_norm.running_mean
        primals_72 = self.batch_norm.running_var
        primals_73 = self.batch_norm.weight
        primals_74 = self.batch_norm.bias
        primals_75 = self.batch_norm.running_mean
        primals_76 = self.batch_norm.running_var
        primals_77 = self.batch_norm.weight
        primals_78 = self.batch_norm.bias
        primals_79 = self.batch_norm.running_mean
        primals_80 = self.batch_norm.running_var
        primals_81 = self.batch_norm.weight
        primals_82 = self.batch_norm.bias
        primals_83 = self.batch_norm.running_mean
        primals_84 = self.batch_norm.running_var
        primals_85 = self.batch_norm.weight
        primals_86 = self.batch_norm.bias
        primals_87 = self.batch_norm.running_mean
        primals_88 = self.batch_norm.running_var
        primals_89 = self.batch_norm.weight
        primals_90 = self.batch_norm.bias
        primals_91 = self.batch_norm.running_mean
        primals_92 = self.batch_norm.running_var
        primals_93 = self.batch_norm.weight
        primals_94 = self.batch_norm.bias
        primals_95 = self.batch_norm.running_mean
        primals_96 = self.batch_norm.running_var
        primals_97 = self.batch_norm.weight
        primals_98 = self.batch_norm.bias
        primals_99 = self.batch_norm.running_mean
        primals_100 = self.batch_norm.running_var
        primals_101 = self.batch_norm.weight
        primals_102 = self.batch_norm.bias
        primals_103 = self.batch_norm.running_mean
        primals_104 = self.batch_norm.running_var
        primals_105 = self.batch_norm.weight
        primals_106 = self.batch_norm.bias
        primals_107 = self.batch_norm.running_mean
        primals_108 = self.batch_norm.running_var
        primals_109 = self.batch_norm.weight
        primals_110 = self.batch_norm.bias
        primals_111 = self.batch_norm.running_mean
        primals_112 = self.batch_norm.running_var
        primals_113 = self.batch_norm.weight
        primals_114 = self.batch_norm.bias
        primals_115 = self.batch_norm.running_mean
        primals_116 = self.batch_norm.running_var
        primals_117 = self.batch_norm.weight
        primals_118 = self.batch_norm.bias
        primals_119 = self.batch_norm.running_mean
        primals_120 = self.batch_norm.running_var
        primals_121 = self.batch_norm.weight
        primals_122 = self.batch_norm.bias
        primals_123 = self.batch_norm.running_mean
        primals_124 = self.batch_norm.running_var
        primals_125 = self.batch_norm.weight
        primals_126 = self.batch_norm.bias
        primals_127 = self.batch_norm.running_mean
        primals_128 = self.batch_norm.running_var
        primals_129 = self.batch_norm.weight
        primals_130 = self.batch_norm.bias
        primals_131 = self.batch_norm.running_mean
        primals_132 = self.batch_norm.running_var
        primals_133 = self.batch_norm.weight
        primals_134 = self.batch_norm.bias
        primals_135 = self.batch_norm.running_mean
        primals_136 = self.batch_norm.running_var
        primals_137 = self.batch_norm.weight
        primals_138 = self.batch_norm.bias
        primals_139 = self.batch_norm.running_mean
        primals_140 = self.batch_norm.running_var
        primals_141 = self.batch_norm.weight
        primals_142 = self.batch_norm.bias
        primals_143 = self.batch_norm.running_mean
        primals_144 = self.batch_norm.running_var
        primals_145 = self.batch_norm.weight
        primals_146 = self.batch_norm.bias
        primals_147 = self.batch_norm.running_mean
        primals_148 = self.batch_norm.running_var
        primals_149 = self.batch_norm.weight
        primals_150 = self.batch_norm.bias
        primals_151 = self.batch_norm.running_mean
        primals_152 = self.batch_norm.running_var
        primals_153 = self.batch_norm.weight
        primals_154 = self.batch_norm.bias
        primals_155 = self.batch_norm.running_mean
        primals_156 = self.batch_norm.running_var
        primals_157 = self.batch_norm.weight
        primals_158 = self.batch_norm.bias
        primals_159 = self.batch_norm.running_mean
        primals_160 = self.batch_norm.running_var
        primals_161 = self.batch_norm.weight
        primals_162 = self.batch_norm.bias
        primals_163 = self.batch_norm.running_mean
        primals_164 = self.batch_norm.running_var
        primals_165 = self.batch_norm.weight
        primals_166 = self.batch_norm.bias
        primals_167 = self.batch_norm.running_mean
        primals_168 = self.batch_norm.running_var
        primals_169 = self.batch_norm.weight
        primals_170 = self.batch_norm.bias
        primals_171 = self.batch_norm.running_mean
        primals_172 = self.batch_norm.running_var
        primals_173 = self.batch_norm.weight
        primals_174 = self.batch_norm.bias
        primals_175 = self.batch_norm.running_mean
        primals_176 = self.batch_norm.running_var
        primals_177 = self.batch_norm.weight
        primals_178 = self.batch_norm.bias
        primals_179 = self.batch_norm.running_mean
        primals_180 = self.batch_norm.running_var
        primals_181 = self.batch_norm.weight
        primals_182 = self.batch_norm.bias
        primals_183 = self.batch_norm.running_mean
        primals_184 = self.batch_norm.running_var
        primals_185 = self.batch_norm.weight
        primals_186 = self.batch_norm.bias
        primals_187 = self.batch_norm.running_mean
        primals_188 = self.batch_norm.running_var
        primals_189 = self.batch_norm.weight
        primals_190 = self.batch_norm.bias
        primals_191 = self.batch_norm.running_mean
        primals_192 = self.batch_norm.running_var
        primals_193 = self.batch_norm.weight
        primals_194 = self.batch_norm.bias
        primals_195 = self.batch_norm.running_mean
        primals_196 = self.batch_norm.running_var
        primals_197 = self.batch_norm.weight
        primals_198 = self.batch_norm.bias
        primals_199 = self.batch_norm.running_mean
        primals_200 = self.batch_norm.running_var
        primals_201 = self.batch_norm.weight
        primals_202 = self.batch_norm.bias
        primals_203 = self.batch_norm.running_mean
        primals_204 = self.batch_norm.running_var
        primals_205 = self.batch_norm.weight
        primals_206 = self.batch_norm.bias
        primals_207 = self.batch_norm.running_mean
        primals_208 = self.batch_norm.running_var
        primals_209 = self.batch_norm.weight
        primals_210 = self.batch_norm.bias
        primals_211 = self.batch_norm.running_mean
        primals_212 = self.batch_norm.running_var
        primals_213 = self.batch_norm.weight
        primals_214 = self.batch_norm.bias
        primals_215 = self.batch_norm.running_mean
        primals_216 = self.batch_norm.running_var
        primals_217 = self.batch_norm.weight
        primals_218 = self.batch_norm.bias
        primals_219 = self.batch_norm.running_mean
        primals_220 = self.batch_norm.running_var
        primals_221 = self.batch_norm.weight
        primals_222 = self.batch_norm.bias
        primals_223 = self.batch_norm.running_mean
        primals_224 = self.batch_norm.running_var
        primals_225 = self.batch_norm.weight
        primals_226 = self.batch_norm.bias
        primals_227 = self.batch_norm.running_mean
        primals_228 = self.batch_norm.running_var
        primals_229 = self.batch_norm.weight
        primals_230 = self.batch_norm.bias
        primals_231 = self.batch_norm.running_mean
        primals_232 = self.batch_norm.running_var
        primals_233 = self.batch_norm.weight
        primals_234 = self.batch_norm.bias
        primals_235 = self.batch_norm.running_mean
        primals_236 = self.batch_norm.running_var
        primals_237 = self.batch_norm.weight
        primals_238 = self.batch_norm.bias
        primals_239 = self.batch_norm.running_mean
        primals_240 = self.batch_norm.running_var
        primals_241 = self.batch_norm.weight
        primals_242 = self.batch_norm.bias
        primals_243 = self.batch_norm.running_mean
        primals_244 = self.batch_norm.running_var
        primals_245 = self.batch_norm.weight
        primals_246 = self.batch_norm.bias
        primals_247 = self.batch_norm.running_mean
        primals_248 = self.batch_norm.running_var
        primals_249 = self.batch_norm.weight
        primals_250 = self.batch_norm.bias
        primals_251 = self.batch_norm.running_mean
        primals_252 = self.batch_norm.running_var
        primals_253 = self.batch_norm.weight
        primals_254 = self.batch_norm.bias
        primals_255 = self.batch_norm.running_mean
        primals_256 = self.batch_norm.running_var
        primals_257 = self.batch_norm.weight
        primals_258 = self.batch_norm.bias
        primals_259 = self.batch_norm.running_mean
        primals_260 = self.batch_norm.running_var
        primals_261 = self.batch_norm.weight
        primals_262 = self.batch_norm.bias
        primals_263 = self.batch_norm.running_mean
        primals_264 = self.batch_norm.running_var
        primals_265 = self.batch_norm.weight
        primals_266 = self.batch_norm.bias
        primals_267 = self.batch_norm.running_mean
        primals_268 = self.batch_norm.running_var
        primals_269 = self.batch_norm.weight
        primals_270 = self.batch_norm.bias
        primals_271 = self.batch_norm.running_mean
        primals_272 = self.batch_norm.running_var
        primals_273 = self.batch_norm.weight
        primals_274 = self.batch_norm.bias
        primals_275 = self.batch_norm.running_mean
        primals_276 = self.batch_norm.running_var
        primals_277 = self.batch_norm.weight
        primals_278 = self.batch_norm.bias
        primals_279 = self.batch_norm.running_mean
        primals_280 = self.batch_norm.running_var
        primals_281 = self.batch_norm.weight
        primals_282 = self.batch_norm.bias
        primals_283 = self.batch_norm.running_mean
        primals_284 = self.batch_norm.running_var
        primals_285 = self.batch_norm.weight
        primals_286 = self.batch_norm.bias
        primals_287 = self.batch_norm.running_mean
        primals_288 = self.batch_norm.running_var
        primals_289 = self.batch_norm.weight
        primals_290 = self.batch_norm.bias
        primals_291 = self.batch_norm.running_mean
        primals_292 = self.batch_norm.running_var
        primals_293 = self.batch_norm.weight
        primals_294 = self.batch_norm.bias
        primals_295 = self.batch_norm.running_mean
        primals_296 = self.batch_norm.running_var
        primals_297 = self.batch_norm.weight
        primals_298 = self.batch_norm.bias
        primals_299 = self.batch_norm.running_mean
        primals_300 = self.batch_norm.running_var
        primals_301 = self.batch_norm.weight
        primals_302 = self.batch_norm.bias
        primals_303 = self.batch_norm.running_mean
        primals_304 = self.batch_norm.running_var
        primals_305 = self.batch_norm.weight
        primals_306 = self.batch_norm.bias
        primals_307 = self.batch_norm.running_mean
        primals_308 = self.batch_norm.running_var
        primals_309 = self.batch_norm.weight
        primals_310 = self.batch_norm.bias
        primals_311 = self.batch_norm.running_mean
        primals_312 = self.batch_norm.running_var
        primals_313 = self.batch_norm.weight
        primals_314 = self.batch_norm.bias
        primals_315 = self.batch_norm.running_mean
        primals_316 = self.batch_norm.running_var
        primals_317 = self.batch_norm.weight
        primals_318 = self.batch_norm.bias
        primals_319 = self.batch_norm.running_mean
        primals_320 = self.batch_norm.running_var
        primals_321 = self.batch_norm.weight
        primals_322 = self.batch_norm.bias
        primals_323 = self.batch_norm.running_mean
        primals_324 = self.batch_norm.running_var
        primals_325 = self.batch_norm.weight
        primals_326 = self.batch_norm.bias
        primals_327 = self.batch_norm.running_mean
        primals_328 = self.batch_norm.running_var
        primals_329 = self.batch_norm.weight
        primals_330 = self.batch_norm.bias
        primals_331 = self.batch_norm.running_mean
        primals_332 = self.batch_norm.running_var
        primals_333 = self.batch_norm.weight
        primals_334 = self.batch_norm.bias
        primals_335 = self.batch_norm.running_mean
        primals_336 = self.batch_norm.running_var
        primals_337 = self.batch_norm.weight
        primals_338 = self.batch_norm.bias
        primals_339 = self.batch_norm.running_mean
        primals_340 = self.batch_norm.running_var
        primals_341 = self.batch_norm.weight
        primals_342 = self.batch_norm.bias
        primals_343 = self.batch_norm.running_mean
        primals_344 = self.batch_norm.running_var
        primals_345 = self.batch_norm.weight
        primals_346 = self.batch_norm.bias
        primals_347 = self.batch_norm.running_mean
        primals_348 = self.batch_norm.running_var
        primals_349 = self.batch_norm.weight
        primals_350 = self.batch_norm.bias
        primals_351 = self.batch_norm.running_mean
        primals_352 = self.batch_norm.running_var
        primals_353 = self.batch_norm.weight
        primals_354 = self.batch_norm.bias
        primals_355 = self.batch_norm.running_mean
        primals_356 = self.batch_norm.running_var
        primals_357 = self.batch_norm.weight
        primals_358 = self.batch_norm.bias
        primals_359 = self.batch_norm.running_mean
        primals_360 = self.batch_norm.running_var
        primals_361 = self.batch_norm.weight
        primals_362 = self.batch_norm.bias
        primals_363 = self.batch_norm.running_mean
        primals_364 = self.batch_norm.running_var
        primals_365 = self.batch_norm.weight
        primals_366 = self.batch_norm.bias
        primals_367 = self.batch_norm.running_mean
        primals_368 = self.batch_norm.running_var
        primals_369 = self.batch_norm.weight
        primals_370 = self.batch_norm.bias
        primals_371 = self.batch_norm.running_mean
        primals_372 = self.batch_norm.running_var
        primals_373 = self.batch_norm.weight
        primals_374 = self.batch_norm.bias
        primals_375 = self.batch_norm.running_mean
        primals_376 = self.batch_norm.running_var
        primals_377 = self.batch_norm.weight
        primals_378 = self.batch_norm.bias
        primals_379 = self.batch_norm.running_mean
        primals_380 = self.batch_norm.running_var
        primals_381 = self.batch_norm.weight
        primals_382 = self.batch_norm.bias
        primals_383 = self.batch_norm.running_mean
        primals_384 = self.batch_norm.running_var
        primals_385 = self.batch_norm.weight
        primals_386 = self.batch_norm.bias
        primals_387 = self.batch_norm.running_mean
        primals_388 = self.batch_norm.running_var
        primals_389 = self.batch_norm.weight
        primals_390 = self.batch_norm.bias
        primals_391 = self.batch_norm.running_mean
        primals_392 = self.batch_norm.running_var
        primals_393 = self.batch_norm.weight
        primals_394 = self.batch_norm.bias
        primals_395 = self.batch_norm.running_mean
        primals_396 = self.batch_norm.running_var
        primals_397 = self.batch_norm.weight
        primals_398 = self.batch_norm.bias
        primals_399 = self.batch_norm.running_mean
        primals_400 = self.batch_norm.running_var
        primals_401 = self.batch_norm.weight
        primals_402 = self.batch_norm.bias
        primals_403 = self.batch_norm.running_mean
        primals_404 = self.batch_norm.running_var
        primals_405 = self.batch_norm.weight
        primals_406 = self.batch_norm.bias
        primals_407 = self.batch_norm.running_mean
        primals_408 = self.batch_norm.running_var
        primals_409 = self.batch_norm.weight
        primals_410 = self.batch_norm.bias
        primals_411 = self.batch_norm.running_mean
        primals_412 = self.batch_norm.running_var
        primals_413 = self.batch_norm.weight
        primals_414 = self.batch_norm.bias
        primals_415 = self.batch_norm.running_mean
        primals_416 = self.batch_norm.running_var
        primals_417 = self.batch_norm.weight
        primals_418 = self.batch_norm.bias
        primals_419 = self.batch_norm.running_mean
        primals_420 = self.batch_norm.running_var
        primals_421 = self.batch_norm.weight
        primals_422 = self.batch_norm.bias
        primals_423 = self.batch_norm.running_mean
        primals_424 = self.batch_norm.running_var
        primals_425 = self.batch_norm.weight
        primals_426 = self.batch_norm.bias
        primals_427 = self.batch_norm.running_mean
        primals_428 = self.batch_norm.running_var
        primals_429 = self.batch_norm.weight
        primals_430 = self.batch_norm.bias
        primals_431 = self.batch_norm.running_mean
        primals_432 = self.batch_norm.running_var
        primals_433 = self.batch_norm.weight
        primals_434 = self.batch_norm.bias
        primals_435 = self.batch_norm.running_mean
        primals_436 = self.batch_norm.running_var
        primals_437 = self.batch_norm.weight
        primals_438 = self.batch_norm.bias
        primals_439 = self.batch_norm.running_mean
        primals_440 = self.batch_norm.running_var
        primals_441 = self.batch_norm.weight
        primals_442 = self.batch_norm.bias
        primals_443 = self.batch_norm.running_mean
        primals_444 = self.batch_norm.running_var
       