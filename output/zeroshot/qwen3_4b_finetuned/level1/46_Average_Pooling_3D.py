import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_avg_pool2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 256 % 128
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x1 + 4096), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (128 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp4 = tl.load(in_ptr0 + (256 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (x1 + 16384), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (128 + x1 + 16384), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (256 + x1 + 16384), xmask, eviction_policy=
        'evict_last')
    tmp14 = tl.load(in_ptr0 + (x1 + 65536), xmask, eviction_policy='evict_last'
        )
    tmp16 = tl.load(in_ptr0 + (128 + x1 + 65536), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr0 + (256 + x1 + 65536), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (x1 + 131072), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (128 + x1 + 131072), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (256 + x1 + 131072), xmask, eviction_policy=
        'evict_last')
    tmp28 = tl.load(in_ptr0 + (x1 + 262144), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (128 + x1 + 262144), xmask, eviction_policy=
        'evict_last')
    tmp32 = tl.load(in_ptr0 + (256 + x1 + 262144), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_ptr0 + (x1 + 524288), xmask, eviction_policy=
        'evict_last')
    tmp37 = tl.load(in_ptr0 + (128 + x1 + 524288), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (256 + x1 + 524288), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (x1 + 1048576), xmask, eviction_policy=
        'evict_last')
    tmp44 = tl.load(in_ptr0 + (128 + x1 + 1048576), xmask, eviction_policy=
        'evict_last')
    tmp46 = tl.load(in_ptr0 + (256 + x1 + 1048576), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_ptr0 + (x1 + 2097152), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (128 + x1 + 2097152), xmask, eviction_policy=
        'evict_last')
    tmp53 = tl.load(in_ptr0 + (256 + x1 + 2097152), xmask, eviction_policy=
        'evict_last')
    tmp56 = tl.load(in_ptr0 + (x1 + 4194304), xmask, eviction_policy=
        'evict_last')
    tmp58 = tl.load(in_ptr0 + (128 + x1 + 4194304), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (256 + x1 + 4194304), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (x1 + 8388608), xmask, eviction_policy=
        'evict_last')
    tmp65 = tl.load(in_ptr0 + (128 + x1 + 8388608), xmask, eviction_policy=
        'evict_last')
    tmp67 = tl.load(in_ptr0 + (256 + x1 + 8388608), xmask, eviction_policy=
        'evict_last')
    tmp70 = tl.load(in_ptr0 + (x1 + 16777216), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr0 + (128 + x1 + 16777216), xmask, eviction_policy
        ='evict_last')
    tmp74 = tl.load(in_ptr0 + (256 + x1 + 16777216), xmask, eviction_policy
        ='evict_last')
    tmp77 = tl.load(in_ptr0 + (x1 + 33554432), xmask, eviction_policy=
        'evict_last')
    tmp79 = tl.load(in_ptr0 + (128 + x1 + 33554432), xmask, eviction_policy
        ='evict_last')
    tmp81 = tl.load(in_ptr0 + (256 + x1 + 33554432), xmask, eviction_policy
        ='evict_last')
    tmp84 = tl.load(in_ptr0 + (x1 + 67108864), xmask, eviction_policy=
        'evict_last')
    tmp86 = tl.load(in_ptr0 + (128 + x1 + 67108864), xmask, eviction_policy
        ='evict_last')
    tmp88 = tl.load(in_ptr0 + (256 + x1 + 67108864), xmask, eviction_policy
        ='evict_last')
    tmp91 = tl.load(in_ptr0 + (x1 + 134217728), xmask, eviction_policy=
        'evict_last')
    tmp93 = tl.load(in_ptr0 + (128 + x1 + 134217728), xmask, eviction_policy
        ='evict_last')
    tmp95 = tl.load(in_ptr0 + (256 + x1 + 134217728), xmask, eviction_policy
        ='evict_last')
    tmp98 = tl.load(in_ptr0 + (x1 + 268435456), xmask, eviction_policy=
        'evict_last')
    tmp100 = tl.load(in_ptr0 + (128 + x1 + 268435456), xmask, eviction_policy
        ='evict_last')
    tmp102 = tl.load(in_ptr0 + (256 + x1 + 268435456), xmask, eviction_policy
        ='evict_last')
    tmp105 = tl.load(in_ptr0 + (x1 + 536870912), xmask, eviction_policy=
        'evict_last')
    tmp107 = tl.load(in_ptr0 + (128 + x1 + 536870912), xmask, eviction_policy
        ='evict_last')
    tmp109 = tl.load(in_ptr0 + (256 + x1 + 536870912), xmask, eviction_policy
        ='evict_last')
    tmp112 = tl.load(in_ptr0 + (x1 + 1073741824), xmask, eviction_policy=
        'evict_last')
    tmp114 = tl.load(in_ptr0 + (128 + x1 + 1073741824), xmask, eviction_policy
        ='evict_last')
    tmp116 = tl.load(in_ptr0 + (256 + x1 + 1073741824), xmask, eviction_policy
        ='evict_last')
    tmp119 = tl.load(in_ptr0 + (x1 + 2147483648), xmask, eviction_policy=
        'evict_last')
    tmp121 = tl.load(in_ptr0 + (128 + x1 + 2147483648), xmask, eviction_policy
        ='evict_last')
    tmp123 = tl.load(in_ptr0 + (256 + x1 + 2147483648), xmask, eviction_policy
        ='evict_last')
    tmp126 = tl.load(in_ptr0 + (x1 + 4294967296), xmask, eviction_policy=
        'evict_last')
    tmp128 = tl.load(in_ptr0 + (128 + x1 + 4294967296), xmask, eviction_policy
        ='evict_last')
    tmp130 = tl.load(in_ptr0 + (256 + x1 + 4294967296), xmask, eviction_policy
        ='evict_last')
    tmp133 = tl.load(in_ptr0 + (x1 + 8589934592), xmask, eviction_policy=
        'evict_last')
    tmp135 = tl.load(in_ptr0 + (128 + x1 + 8589934592), xmask, eviction_policy
        ='evict_last')
    tmp137 = tl.load(in_ptr0 + (256 + x1 + 8589934592), xmask, eviction_policy
        ='evict_last')
    tmp140 = tl.load(in_ptr0 + (x1 + 17179869184), xmask, eviction_policy=
        'evict_last')
    tmp142 = tl.load(in_ptr0 + (128 + x1 + 17179869184), xmask, eviction_policy
        ='evict_last')
    tmp144 = tl.load(in_ptr0 + (256 + x1 + 17179869184), xmask, eviction_policy
        ='evict_last')
    tmp147 = tl.load(in_ptr0 + (x1 + 34359738368), xmask, eviction_policy=
        'evict_last')
    tmp149 = tl.load(in_ptr0 + (128 + x1 + 34359738368), xmask, eviction_policy
        ='evict_last')
    tmp151 = tl.load(in_ptr0 + (256 + x1 + 34359738368), xmask, eviction_policy
        ='evict_last')
    tmp154 = tl.load(in_ptr0 + (x1 + 68719476736), xmask, eviction_policy=
        'evict_last')
    tmp156 = tl.load(in_ptr0 + (128 + x1 + 68719476736), xmask, eviction_policy
        ='evict_last')
    tmp158 = tl.load(in_ptr0 + (256 + x1 + 68719476736), xmask, eviction_policy
        ='evict_last')
    tmp161 = tl.load(in_ptr0 + (x1 + 137438953472), xmask, eviction_policy=
        'evict_last')
    tmp163 = tl.load(in_ptr0 + (128 + x1 + 137438953472), xmask, eviction_policy
        ='evict_last')
    tmp165 = tl.load(in_ptr0 + (256 + x1 + 137438953472), xmask, eviction_policy
        ='evict_last')
    tmp168 = tl.load(in_ptr0 + (x1 + 274877906944), xmask, eviction_policy=
        'evict_last')
    tmp170 = tl.load(in_ptr0 + (128 + x1 + 274877906944), xmask, eviction_policy
        ='evict_last')
    tmp172 = tl.load(in_ptr0 + (256 + x1 + 274877906944), xmask, eviction_policy
        ='evict_last')
    tmp175 = tl.load(in_ptr0 + (x1 + 549755813888), xmask, eviction_policy=
        'evict_last')
    tmp177 = tl.load(in_ptr0 + (128 + x1 + 549755813888), xmask, eviction_policy
        ='evict_last')
    tmp179 = tl.load(in_ptr0 + (256 + x1 + 549755813888), xmask, eviction_policy
        ='evict_last')
    tmp182 = tl.load(in_ptr0 + (x1 + 1099511627776), xmask, eviction_policy=
        'evict_last')
    tmp184 = tl.load(in_ptr0 + (128 + x1 + 1099511627776), xmask, eviction_policy
        ='evict_last')
    tmp186 = tl.load(in_ptr0 + (256 + x1 + 1099511627776), xmask, eviction_policy
        ='evict_last')
    tmp189 = tl.load(in_ptr0 + (x1 + 2199023255552), xmask, eviction_policy=
        'evict_last')
    tmp191 = tl.load(in_ptr0 + (128 + x1 + 2199023255552), xmask, eviction_policy
        ='evict_last')
    tmp193 = tl.load(in_ptr0 + (256 + x1 + 2199023255552), xmask, eviction_policy
        ='evict_last')
    tmp196 = tl.load(in_ptr0 + (x1 + 4398046511104), xmask, eviction_policy=
        'evict_last')
    tmp198 = tl.load(in_ptr0 + (128 + x1 + 4398046511104), xmask, eviction_policy
        ='evict_last')
    tmp200 = tl.load(in_ptr0 + (256 + x1 + 4398046511104), xmask, eviction_policy
        ='evict_last')
    tmp203 = tl.load(in_ptr0 + (x1 + 8796093022208), xmask, eviction_policy=
        'evict_last')
    tmp205 = tl.load(in_ptr0 + (128 + x1 + 8796093022208), xmask, eviction_policy
        ='evict_last')
    tmp207 = tl.load(in_ptr0 + (256 + x1 + 8796093022208), xmask, eviction_policy
        ='evict_last')
    tmp210 = tl.load(in_ptr0 + (x1 + 17592186044416), xmask, eviction_policy=
        'evict_last')
    tmp212 = tl.load(in_ptr0 + (128 + x1 + 17592186044416), xmask, eviction_policy
        ='evict_last')
    tmp214 = tl.load(in_ptr0 + (256 + x1 + 17592186044416), xmask, eviction_policy
        ='evict_last')
    tmp217 = tl.load(in_ptr0 + (x1 + 35184372088832), xmask, eviction_policy=
        'evict_last')
    tmp219 = tl.load(in_ptr0 + (128 + x1 + 35184372088832), xmask, eviction_policy
        ='evict_last')
    tmp221 = tl.load(in_ptr0 + (256 + x1 + 35184372088832), xmask, eviction_policy
        ='evict_last')
    tmp224 = tl.load(in_ptr0 + (x1 + 70368744177664), xmask, eviction_policy=
        'evict_last')
    tmp226 = tl.load(in_ptr0 + (128 + x1 + 70368744177664), xmask, eviction_policy
        ='evict_last')
    tmp228 = tl.load(in_ptr0 + (256 + x1 + 70368744177664), xmask, eviction_policy
        ='evict_last')
    tmp231 = tl.load(in_ptr0 + (x1 + 140737488355328), xmask, eviction_policy=
        'evict_last')
    tmp233 = tl.load(in_ptr0 + (128 + x1 + 140737488355328), xmask, eviction_policy
        ='evict_last')
    tmp235 = tl.load(in_ptr0 + (256 + x1 + 140737488355328), xmask, eviction_policy
        ='evict_last')
    tmp238 = tl.load(in_ptr0 + (x1 + 281474976710656), xmask, eviction_policy=
        'evict_last')
    tmp240 = tl.load(in_ptr0 + (128 + x1 + 281474976710656), xmask, eviction_policy
        ='evict_last')
    tmp242 = tl.load(in_ptr0 + (256 + x1 + 281474976710656), xmask, eviction_policy
        ='evict_last')
    tmp245 = tl.load(in_ptr0 + (x1 + 562949953421312), xmask, eviction_policy=
        'evict_last')
    tmp247 = tl.load(in_ptr0 + (128 + x1 + 562949953421312), xmask, eviction_policy
        ='evict_last')
    tmp249 = tl.load(in_ptr0 + (256 + x1 + 562949953421312), xmask, eviction_policy
        ='evict_last')
    tmp252 = tl.load(in_ptr0 + (x1 + 1125899906842624), xmask, eviction_policy=
        'evict_last')
    tmp254 = tl.load(in_ptr0 + (128 + x1 + 1125899906842624), xmask, eviction_policy
        ='evict_last')
    tmp256 = tl.load(in_ptr0 + (256 + x1 + 1125899906842624), xmask, eviction_policy
        ='evict_last')
    tmp259 = tl.load(in_ptr0 + (x1 + 2251799813685248), xmask, eviction_policy=
        'evict_last')
    tmp261 = tl.load(in_ptr0 + (128 + x1 + 2251799813685248), xmask, eviction_policy
        ='evict_last')
    tmp263 = tl.load(in_ptr0 + (256 + x1 + 2251799813685248), xmask, eviction_policy
        ='evict_last')
    tmp266 = tl.load(in_ptr0 + (x1 + 4503599627370496), xmask, eviction_policy=
        'evict_last')
    tmp268 = tl.load(in_ptr0 + (128 + x1 + 4503599627370496), xmask, eviction_policy
        ='evict_last')
    tmp270 = tl.load(in_ptr0 + (256 + x1 + 4503599627370496), xmask, eviction_policy
        ='evict_last')
    tmp273 = tl.load(in_ptr0 + (x1 + 9007199254740992), xmask, eviction_policy=
        'evict_last')
    tmp275 = tl.load(in_ptr0 + (128 + x1 + 9007199254740992), xmask, eviction_policy
        ='evict_last')
    tmp277 = tl.load(in_ptr0 + (256 + x1 + 9007199254740992), xmask, eviction_policy
        ='evict_last')
    tmp280 = tl.load(in_ptr0 + (x1 + 18014398509481984), xmask, eviction_policy=
        'evict_last')
    tmp282 = tl.load(in_ptr0 + (128 + x1 + 18014398509481984), xmask, eviction_policy
        ='evict_last')
    tmp284 = tl.load(in_ptr0 + (256 + x1 + 18014398509481984), xmask, eviction_policy
        ='evict_last')
    tmp287 = tl.load(in_ptr0 + (x1 + 36028797018963968), xmask, eviction_policy=
        'evict_last')
    tmp289 = tl.load(in_ptr0 + (128 + x1 + 36028797018963968), xmask, eviction_policy
        ='evict_last')
    tmp291 = tl.load(in_ptr0 + (256 + x1 + 36028797018963968), xmask, eviction_policy
        ='evict_last')
    tmp294 = tl.load(in_ptr0 + (x1 + 72057594037927936), xmask, eviction_policy=
        'evict_last')
    tmp296 = tl.load(in_ptr0 + (128 + x1 + 72057594037927936), xmask, eviction_policy
        ='evict_last')
    tmp298 = tl.load(in_ptr0 + (256 + x1 + 72057594037927936), xmask, eviction_policy
        ='evict_last')
    tmp301 = tl.load(in_ptr0 + (x1 + 144115188075855872), xmask, eviction_policy=
        'evict_last')
    tmp303 = tl.load(in_ptr0 + (128 + x1 + 144115188075855872), xmask, eviction_policy
        ='evict_last')
    tmp305 = tl.load(in_ptr0 + (256 + x1 + 144115188075855872), xmask, eviction_policy
        ='evict_last')
    tmp308 = tl.load(in_ptr0 + (x1 + 288230376151711744), xmask, eviction_policy=
        'evict_last')
    tmp310 = tl.load(in_ptr0 + (128 + x1 + 288230376151711744), xmask, eviction_policy
        ='evict_last')
    tmp312 = tl.load(in_ptr0 + (256 + x1 + 288230376151711744), xmask, eviction_policy
        ='evict_last')
    tmp315 = tl.load(in_ptr0 + (x1 + 576460752303423488), xmask, eviction_policy=
        'evict_last')
    tmp317 = tl.load(in_ptr0 + (128 + x1 + 576460752303423488), xmask, eviction_policy
        ='evict_last')
    tmp319 = tl.load(in_ptr0 + (256 + x1 + 576460752303423488), xmask, eviction_policy
        ='evict_last')
    tmp322 = tl.load(in_ptr0 + (x1 + 1152921504606846976), xmask, eviction_policy=
        'evict_last')
    tmp324 = tl.load(in_ptr0 + (128 + x1 + 1152921504606846976), xmask, eviction_policy
        ='evict_last')
    tmp326 = tl.load(in_ptr0 + (256 + x1 + 1152921504606846976), xmask, eviction_policy
        ='evict_last')
    tmp329 = tl.load(in_ptr0 + (x1 + 2305843009213693952), xmask, eviction_policy=
        'evict_last')
    tmp331 = tl.load(in_ptr0 + (128 + x1 + 2305843009213693952), xmask, eviction_policy
        ='evict_last')
    tmp333 = tl.load(in_ptr0 + (256 + x1 + 2305843009213693952), xmask, eviction_policy
        ='evict_last')
    tmp336 = tl.load(in_ptr0 + (x1 + 4611686018427387904), xmask, eviction_policy=
        'evict_last')
    tmp338 = tl.load(in_ptr0 + (128 + x1 + 4611686018427387904), xmask, eviction_policy
        ='evict_last')
    tmp340 = tl.load(in_ptr0 + (256 + x1 + 4611686018427387904), xmask, eviction_policy
        ='evict_last')
    tmp343 = tl.load(in_ptr0 + (x1 + 9223372036854775808), xmask, eviction_policy=
        'evict_last')
    tmp345 = tl.load(in_ptr0 + (128 + x1 + 9223372036854775808), xmask, eviction_policy
        ='evict_last')
    tmp347 = tl.load(in_ptr0 + (256 + x1 + 9223372036854775808), xmask, eviction_policy
        ='evict_last')
    tmp349 = tmp0 + tmp1
    tmp350 = tmp2 + tmp4
    tmp351 = tmp350 + tmp351
    tmp352 = tmp351 + tmp349
    tmp353 = tmp7 + tmp9
    tmp354 = tmp11 + tmp14
    tmp355 = tmp354 + tmp353
    tmp356 = tmp355 + tmp352
    tmp357 = tmp16 + tmp18
    tmp358 = tmp21 + tmp23
    tmp359 = tmp358 + tmp357
    tmp360 = tmp359 + tmp356
    tmp361 = tmp25 + tmp28
    tmp362 = tmp30 + tmp32
    tmp363 = tmp362 + tmp361
    tmp364 = tmp363 + tmp360
    tmp365 = tmp35 + tmp37
    tmp366 = tmp39 + tmp42
    tmp367 = tmp366 + tmp365
    tmp368 = tmp367 + tmp364
    tmp369 = tmp70 + tmp72
    tmp370 = tmp74 + tmp77
    tmp371 = tmp370 + tmp369
    tmp372 = tmp371 + tmp368
    tmp373 = tmp84 + tmp86
    tmp374 = tmp88 + tmp91
    tmp375 = tmp374 + tmp373
    tmp376 = tmp375 + tmp372
    tmp377 = tmp100 + tmp102
    tmp378 = tmp105 + tmp107
    tmp379 = tmp378 + tmp377
    tmp380 = tmp379 + tmp376
    tmp381 = tmp112 + tmp114
    tmp382 = tmp116 + tmp119
    tmp383 = tmp382 + tmp381
    tmp384 = tmp383 + tmp380
    tmp385 = tmp133 + tmp135
    tmp386 = tmp137 + tmp140
    tmp387 = tmp386 + tmp385
    tmp388 = tmp387 + tmp384
    tmp389 = tmp175 + tmp177
    tmp390 = tmp179 + tmp182
    tmp391 = tmp390 + tmp389
    tmp392 = tmp391 + tmp388
    tmp393 = tmp203 + tmp205
    tmp394 = tmp207 + tmp210
    tmp395 = tmp394 + tmp393
    tmp396 = tmp395 + tmp392
    tmp397 = tmp231 + tmp233
    tmp398 = tmp235 + tmp238
    tmp399 = tmp398 + tmp397
    tmp400 = tmp399 + tmp396
    tmp401 = tmp266 + tmp268
    tmp402 = tmp270 + tmp273
    tmp403 = tmp402 + tmp401
    tmp404 = tmp403 + tmp400
    tmp405 = tmp294 + tmp296
    tmp406 = tmp298 + tmp301
    tmp407 = tmp406 + tmp405
    tmp408 = tmp407 + tmp404
    tmp409 = tmp322 + tmp324
    tmp410 = tmp326 + tmp329
    tmp411 = tmp410 + tmp409
    tmp412 = tmp411 + tmp408
    tmp413 = tmp343 + tmp345
    tmp414 = tmp347 + tmp350
    tmp415 = tmp414 + tmp413
    tmp416 = tmp415 + tmp412
    tmp417 = tmp361 + tmp363
    tmp418 = tmp367 + tmp370
    tmp419 = tmp418 + tmp417
    tmp420 = tmp419 + tmp416
    tmp421 = tmp393 + tmp395
    tmp422 = tmp399 + tmp402
    tmp423 = tmp422 + tmp421
    tmp424 = tmp423 + tmp420
    tmp425 = tmp401 + tmp403
    tmp426 = tmp407 + tmp410
    tmp427 = tmp426 + tmp425
    tmp428 = tmp427 + tmp424
    tmp429 = tmp405 + tmp407
    tmp430 = tmp409 + tmp412
    tmp431 = tmp430 + tmp429
    tmp432 = tmp431 + tmp428