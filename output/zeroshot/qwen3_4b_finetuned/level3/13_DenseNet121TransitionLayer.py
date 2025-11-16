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
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__unsafe_index_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 256
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = triton_helpers.maximum(tmp2, 0.0)
    tmp6 = tmp4 + tmp3
    tmp8 = tmp6 / tmp5
    tmp9 = tmp8 * tmp7
    tl.store(out_ptr0 + x2, tmp9, xmask)


@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0 > 0.0
    tl.store(out_ptr0 + x0, tmp1, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_2, (32,), (1,))
    assert_size_stride(primals_3, (128, 32, 256, 256), (2097152, 65536, 256,
        1))
    assert_size_stride(primals_4, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_5, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 32, 1, 1), (32, 1, 32, 32), torch.float32
            )
        get_raw_stream(0)
        triton_poi_fused__to_copy_0[grid(2592)](primals_2, buf0, 2592,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf1 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (128, 32, 256, 256), (2097152, 65536, 256, 1
            ))
        buf2 = empty_strided_cuda((128, 32, 256, 256), (2097152, 65536, 256,
            1), torch.float32)
        triton_poi_fused__unsafe_index_1[grid(8192)](buf1, buf0, primals_1,
            primals_4, primals_5, buf2, 8192, XBLOCK=128, num_warps=4,
            num_stages=1)
        del buf0
        del buf1
        del primals_1
        del primals_4
        del primals_5
        buf3 = empty_strided_cuda((128, 32, 128, 128), (524288, 16384, 128,
            1), torch.bool)
        triton_poi_fused_2[grid(8192)](buf2, buf3, 8192, XBLOCK=128,
            num_warps=4, num_stages=1)
        del buf2
    return reinterpret_tensor(buf3, (128, 64, 128, 128), (1048576, 16384, 128,
        1), 0), primals_3


class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        :param num_input_features: The number of input feature maps
        :param num_output_features: The number of output feature maps
        """
        super(ModelNew, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1,
                bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, input_0):
        primals_1 = self.transition[0].weight
        primals_2 = self.transition[0].running_mean
        primals_3 = self.transition[0].running_var
        primals_4 = self.transition[0].num_batches_tracked
        primals_5 = self.transition[1].weight
        primals_6 = self.transition[3].weight
        primals_7 = self.transition[3].bias
        primals_8 = self.transition[2].weight
        primals_9 = self.transition[2].bias
        primals_10 = self.transition[1].bias
        primals_11 = self.transition[3].stride
        primals_12 = self.transition[3].padding
        primals_13 = self.transition[3].dilation
        primals_14 = self.transition[3].transposed
        primals_15 = self.transition[3].output_padding
        primals_16 = self.transition[3].groups
        primals_17 = self.transition[3].padding_mode
        primals_18 = self.transition[3].kernel_size
        primals_19 = self.transition[3].stride_2
        primals_20 = self.transition[3].stride_3
        primals_21 = self.transition[3].stride_4
        primals_22 = self.transition[3].stride_5
        primals_23 = self.transition[3].stride_6
        primals_24 = self.transition[3].stride_7
        primals_25 = self.transition[3].stride_8
        primals_26 = self.transition[3].stride_9
        primals_27 = self.transition[3].stride_10
        primals_28 = self.transition[3].stride_11
        primals_29 = self.transition[3].stride_12
        primals_30 = self.transition[3].stride_13
        primals_31 = self.transition[3].stride_14
        primals_32 = self.transition[3].stride_15
        primals_33 = self.transition[3].stride_16
        primals_34 = self.transition[3].stride_17
        primals_35 = self.transition[3].stride_18
        primals_36 = self.transition[3].stride_19
        primals_37 = self.transition[3].stride_20
        primals_38 = self.transition[3].stride_21
        primals_39 = self.transition[3].stride_22
        primals_40 = self.transition[3].stride_23
        primals_41 = self.transition[3].stride_24
        primals_42 = self.transition[3].stride_25
        primals_43 = self.transition[3].stride_26
        primals_44 = self.transition[3].stride_27
        primals_45 = self.transition[3].stride_28
        primals_46 = self.transition[3].stride_29
        primals_47 = self.transition[3].stride_30
        primals_48 = self.transition[3].stride_31
        primals_49 = self.transition[3].stride_32
        primals_50 = self.transition[3].stride_33
        primals_51 = self.transition[3].stride_34
        primals_52 = self.transition[3].stride_35
        primals_53 = self.transition[3].stride_36
        primals_54 = self.transition[3].stride_37
        primals_55 = self.transition[3].stride_38
        primals_56 = self.transition[3].stride_39
        primals_57 = self.transition[3].stride_40
        primals_58 = self.transition[3].stride_41
        primals_59 = self.transition[3].stride_42
        primals_60 = self.transition[3].stride_43
        primals_61 = self.transition[3].stride_44
        primals_62 = self.transition[3].stride_45
        primals_63 = self.transition[3].stride_46
        primals_64 = self.transition[3].stride_47
        primals_65 = self.transition[3].stride_48
        primals_66 = self.transition[3].stride_49
        primals_67 = self.transition[3].stride_50
        primals_68 = self.transition[3].stride_51
        primals_69 = self.transition[3].stride_52
        primals_70 = self.transition[3].stride_53
        primals_71 = self.transition[3].stride_54
        primals_72 = self.transition[3].stride_55
        primals_73 = self.transition[3].stride_56
        primals_74 = self.transition[3].stride_57
        primals_75 = self.transition[3].stride_58
        primals_76 = self.transition[3].stride_59
        primals_77 = self.transition[3].stride_60
        primals_78 = self.transition[3].stride_61
        primals_79 = self.transition[3].stride_62
        primals_80 = self.transition[3].stride_63
        primals_81 = self.transition[3].stride_64
        primals_82 = self.transition[3].stride_65
        primals_83 = self.transition[3].stride_66
        primals_84 = self.transition[3].stride_67
        primals_85 = self.transition[3].stride_68
        primals_86 = self.transition[3].stride_69
        primals_87 = self.transition[3].stride_70
        primals_88 = self.transition[3].stride_71
        primals_89 = self.transition[3].stride_72
        primals_90 = self.transition[3].stride_73
        primals_91 = self.transition[3].stride_74
        primals_92 = self.transition[3].stride_75
        primals_93 = self.transition[3].stride_76
        primals_94 = self.transition[3].stride_77
        primals_95 = self.transition[3].stride_78
        primals_96 = self.transition[3].stride_79
        primals_97 = self.transition[3].stride_80
        primals_98 = self.transition[3].stride_81
        primals_99 = self.transition[3].stride_82
        primals_100 = self.transition[3].stride_83
        primals_101 = self.transition[3].stride_84
        primals_102 = self.transition[3].stride_85
        primals_103 = self.transition[3].stride_86
        primals_104 = self.transition[3].stride_87
        primals_105 = self.transition[3].stride_88
        primals_106 = self.transition[3].stride_89
        primals_107 = self.transition[3].stride_90
        primals_108 = self.transition[3].stride_91
        primals_109 = self.transition[3].stride_92
        primals_110 = self.transition[3].stride_93
        primals_111 = self.transition[3].stride_94
        primals_112 = self.transition[3].stride_95
        primals_113 = self.transition[3].stride_96
        primals_114 = self.transition[3].stride_97
        primals_115 = self.transition[3].stride_98
        primals_116 = self.transition[3].stride_99
        primals_117 = self.transition[3].stride_100
        primals_118 = self.transition[3].stride_101
        primals_119 = self.transition[3].stride_102
        primals_120 = self.transition[3].stride_103
        primals_121 = self.transition[3].stride_104
        primals_122 = self.transition[3].stride_105
        primals_123 = self.transition[3].stride_106
        primals_124 = self.transition[3].stride_107
        primals_125 = self.transition[3].stride_108
        primals_126 = self.transition[3].stride_109
        primals_127 = self.transition[3].stride_110
        primals_128 = self.transition[3].stride_111
        primals_129 = self.transition[3].stride_112
        primals_130 = self.transition[3].stride_113
        primals_131 = self.transition[3].stride_114
        primals_132 = self.transition[3].stride_115
        primals_133 = self.transition[3].stride_116
        primals_134 = self.transition[3].stride_117
        primals_135 = self.transition[3].stride_118
        primals_136 = self.transition[3].stride_119
        primals_137 = self.transition[3].stride_120
        primals_138 = self.transition[3].stride_121
        primals_139 = self.transition[3].stride_122
        primals_140 = self.transition[3].stride_123
        primals_141 = self.transition[3].stride_124
        primals_142 = self.transition[3].stride_125
        primals_143 = self.transition[3].stride_126
        primals_144 = self.transition[3].stride_127
        primals_145 = self.transition[3].stride_128
        primals_146 = self.transition[3].stride_129
        primals_147 = self.transition[3].stride_130
        primals_148 = self.transition[3].stride_131
        primals_149 = self.transition[3].stride_132
        primals_150 = self.transition[3].stride_133
        primals_151 = self.transition[3].stride_134
        primals_152 = self.transition[3].stride_135
        primals_153 = self.transition[3].stride_136
        primals_154 = self.transition[3].stride_137
        primals_155 = self.transition[3].stride_138
        primals_156 = self.transition[3].stride_139
        primals_157 = self.transition[3].stride_140
        primals_158 = self.transition[3].stride_141
        primals_159 = self.transition[3].stride_142
        primals_160 = self.transition[3].stride_143
        primals_161 = self.transition[3].stride_144
        primals_162 = self.transition[3].stride_145
        primals_163 = self.transition[3].stride_146
        primals_164 = self.transition[3].stride_147
        primals_165 = self.transition[3].stride_148
        primals_166 = self.transition[3].stride_149
        primals_167 = self.transition[3].stride_150
        primals_168 = self.transition[3].stride_151
        primals_169 = self.transition[3].stride_152
        primals_170 = self.transition[3].stride_153
        primals_171 = self.transition[3].stride_154
        primals_172 = self.transition[3].stride_155
        primals_173 = self.transition[3].stride_156
        primals_174 = self.transition[3].stride_157
        primals_175 = self.transition[3].stride_158
        primals_176 = self.transition[3].stride_159
        primals_177 = self.transition[3].stride_160
        primals_178 = self.transition[3].stride_161
        primals_179 = self.transition[3].stride_162
        primals_180 = self.transition[3].stride_163
        primals_181 = self.transition[3].stride_164
        primals_182 = self.transition[3].stride_165
        primals_183 = self.transition[3].stride_166
        primals_184 = self.transition[3].stride_167
        primals_185 = self.transition[3].stride_168
        primals_186 = self.transition[3].stride_169
        primals_187 = self.transition[3].stride_170
        primals_188 = self.transition[3].stride_171
        primals_189 = self.transition[3].stride_172
        primals_190 = self.transition[3].stride_173
        primals_191 = self.transition[3].stride_174
        primals_192 = self.transition[3].stride_175
        primals_193 = self.transition[3].stride_176
        primals_194 = self.transition[3].stride_177
        primals_195 = self.transition[3].stride_178
        primals_196 = self.transition[3].stride_179
        primals_197 = self.transition[3].stride_180
        primals_198 = self.transition[3].stride_181
        primals_199 = self.transition[3].stride_182
        primals_200 = self.transition[3].stride_183
        primals_201 = self.transition[3].stride_184
        primals_202 = self.transition[3].stride_185
        primals_203 = self.transition[3].stride_186
        primals_204 = self.transition[3].stride_187
        primals_205 = self.transition[3].stride_188
        primals_206 = self.transition[3].stride_189
        primals_207 = self.transition[3].stride_190
        primals_208 = self.transition[3].stride_191
        primals_209 = self.transition[3].stride_192
        primals_210 = self.transition[3].stride_193
        primals_211 = self.transition[3].stride_194
        primals_212 = self.transition[3].stride_195
        primals_213 = self.transition[3].stride_196
        primals_214 = self.transition[3].stride_197
        primals_215 = self.transition[3].stride_198
        primals_216 = self.transition[3].stride_199
        primals_217 = self.transition[3].stride_200
        primals_218 = self.transition[3].stride_201
        primals_219 = self.transition[3].stride_202
        primals_220 = self.transition[3].stride_203
        primals_221 = self.transition[3].stride_204
        primals_222 = self.transition[3].stride_205
        primals_223 = self.transition[3].stride_206
        primals_224 = self.transition[3].stride_207
        primals_225 = self.transition[3].stride_208
        primals_226 = self.transition[3].stride_209
        primals_227 = self.transition[3].stride_210
        primals_228 = self.transition[3].stride_211
        primals_229 = self.transition[3].stride_212
        primals_230 = self.transition[3].stride_213
        primals_231 = self.transition[3].stride_214
        primals_232 = self.transition[3].stride_215
        primals_233 = self.transition[3].stride_216
        primals_234 = self.transition[3].stride_217
        primals_235 = self.transition[3].stride_218
        primals_236 = self.transition[3].stride_219
        primals_237 = self.transition[3].stride_220
        primals_238 = self.transition[3].stride_221
        primals_239 = self.transition[3].stride_222
        primals_240 = self.transition[3].stride_223
        primals_241 = self.transition[3].stride_224
        primals_242 = self.transition[3].stride_225
        primals_243 = self.transition[3].stride_226
        primals_244 = self.transition[3].stride_227
        primals_245 = self.transition[3].stride_228
        primals_246 = self.transition[3].stride_229
        primals_247 = self.transition[3].stride_230
        primals_248 = self.transition[3].stride_231
        primals_249 = self.transition[3].stride_232
        primals_250 = self.transition[3].stride_233
        primals_251 = self.transition[3].stride_234
        primals_252 = self.transition[3].stride_235
        primals_253 = self.transition[3].stride_236
        primals_254 = self.transition[3].stride_237
        primals_255 = self.transition[3].stride_238
        primals_256 = self.transition[3].stride_239
        primals_257 = self.transition[3].stride_240
        primals_258 = self.transition[3].stride_241
        primals_259 = self.transition[3].stride_242
        primals_260 = self.transition[3].stride_243
        primals_261 = self.transition[3].stride_244
        primals_262 = self.transition[3].stride_245
        primals_263 = self.transition[3].stride_246
        primals_264 = self.transition[3].stride_247
        primals_265 = self.transition[3].stride_248
        primals_266 = self.transition[3].stride_249
        primals_267 = self.transition[3].stride_250
        primals_268 = self.transition[3].stride_251
        primals_269 = self.transition[3].stride_252
        primals_270 = self.transition[3].stride_253
        primals_271 = self.transition[3].stride_254
        primals_272 = self.transition[3].stride_255
        primals_273 = self.transition[3].stride_256
        primals_274 = self.transition[3].stride_257
        primals_275 = self.transition[3].stride_258
        primals_276 = self.transition[3].stride_259
        primals_277 = self.transition[3].stride_260
        primals_278 = self.transition[3].stride_261
        primals_279 = self.transition[3].stride_262
        primals_280 = self.transition[3].stride_263
        primals_281 = self.transition[3].stride_264
        primals_282 = self.transition[3].stride_265
        primals_283 = self.transition[3].stride_266
        primals_284 = self.transition[3].stride_267
        primals_285 = self.transition[3].stride_268
        primals_286 = self.transition[3].stride_269
        primals_287 = self.transition[3].stride_270
        primals_288 = self.transition[3].stride_271
        primals_289 = self.transition[3].stride_272
        primals_290 = self.transition[3].stride_273
        primals_291 = self.transition[3].stride_274
        primals_292 = self.transition[3].stride_275
        primals_293 = self.transition[3].stride_276
        primals_294 = self.transition[3].stride_277
        primals_295 = self.transition[3].stride_278
        primals_296 = self.transition[3].stride_279
        primals_297 = self.transition[3].stride_280
        primals_298 = self.transition[3].stride_281
        primals_299 = self.transition[3].stride_282
        primals_300 = self.transition[3].stride_283
        primals_301 = self.transition[3].stride_284
        primals_302 = self.transition[3].stride_285
        primals_303 = self.transition[3].stride_286
        primals_304 = self.transition[3].stride_287
        primals_305 = self.transition[3].stride_288
        primals_306 = self.transition[3].stride_289
        primals_307 = self.transition[3].stride_290
        primals_308 = self.transition[3].stride_291
        primals_309 = self.transition[3].stride_292
        primals_310 = self.transition[3].stride_293
        primals_311 = self.transition[3].stride_294
        primals_312 = self.transition[3].stride_295
        primals_313 = self.transition[3].stride_296
        primals_314 = self.transition[3].stride_297
        primals_315 = self.transition[3].stride_298
        primals_316 = self.transition[3].stride_299
        primals_317 = self.transition[3].stride_300
        primals_318 = self.transition[3].stride_301
        primals_319 = self.transition[3].stride_302
        primals_320 = self.transition[3].stride_303
        primals_321 = self.transition[3].stride_304
        primals_322 = self.transition[3].stride_305
        primals_323 = self.transition[3].stride_306
        primals_324 = self.transition[3].stride_307
        primals_325 = self.transition[3].stride_308
        primals_326 = self.transition[3].stride_309
        primals_327 = self.transition[3].stride_310
        primals_328 = self.transition[3].stride_311
        primals_329 = self.transition[3].stride_312
        primals_330 = self.transition[3].stride_313
        primals_331 = self.transition[3].stride_314
        primals_332 = self.transition[3].stride_315
        primals_333 = self.transition[3].stride_316
        primals_334 = self.transition[3].stride_317
        primals_335 = self.transition[3].stride_318
        primals_336 = self.transition[3].stride_319
        primals_337 = self.transition[3].stride_320
        primals_338 = self.transition[3].stride_321
        primals_339 = self.transition[3].stride_322
        primals_340 = self.transition[3].stride_323
        primals_341 = self.transition[3].stride_324
        primals_342 = self.transition[3].stride_325
        primals_343 = self.transition[3].stride_326
        primals_344 = self.transition[3].stride_327
        primals_345 = self.transition[3].stride_328
        primals_346 = self.transition[3].stride_329
        primals_347 = self.transition[3].stride_330
        primals_348 = self.transition[3].stride_331
        primals_349 = self.transition[3].stride_332
        primals_350 = self.transition[3].stride_333
        primals_351 = self.transition[3].stride_334
        primals_352 = self.transition[3].stride_335
        primals_353 = self.transition[3].stride_336
        primals_354 = self.transition[3].stride_337
        primals_355 = self.transition[3].stride_338
        primals_356 = self.transition[3].stride_339
        primals_357 = self.transition[3].stride_340
        primals_358 = self.transition[3].stride_341
        primals_359 = self.transition[3].stride_342
        primals_360 = self.transition[3].stride_343
        primals_361 = self.transition[3].stride_344
        prim