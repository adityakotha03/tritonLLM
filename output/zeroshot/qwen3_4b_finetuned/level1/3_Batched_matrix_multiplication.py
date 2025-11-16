import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_per_fused_bmm_0(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK:
    tl.constexpr):
    xnumel = 4096
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x1 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (r2 + 256 * x1), xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0 + 128 * r2), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.where(xmask, tmp1, float('nan'))
    tmp5 = tl.tril(tmp4, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 1))
    tmp7 = tl.load(in_ptr0 + (256 + r2 + 256 * x1), xmask, other=0.0)
    tmp9 = tl.where(xmask, tmp7, float('nan'))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp13 = tl.where(xmask, tmp10, float('nan'))
    tmp14 = tl.tril(tmp13, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 1))
    tmp16 = tl.load(in_ptr0 + (512 + r2 + 256 * x1), xmask, other=0.0)
    tmp18 = tl.where(xmask, tmp16, float('nan'))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp22 = tl.where(xmask, tmp19, float('nan'))
    tmp23 = tl.tril(tmp22, 0)
    tmp25 = tl.sum(tmp23, 1)
    tmp26 = tl.load(in_ptr1 + (128 + x0 + 128 * r2), xmask, eviction_policy
        ='evict_last')
    tmp27 = tl.load(in_ptr1 + (256 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr1 + (384 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr1 + (512 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (768 + r2 + 256 * x1), xmask, other=0.0)
    tmp32 = tl.where(xmask, tmp30, float('nan'))
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
    tmp36 = tl.where(xmask, tmp33, float('nan'))
    tmp37 = tl.tril(tmp36, 0)
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 1))
    tmp39 = tl.load(in_ptr0 + (1024 + r2 + 256 * x1), xmask, other=0.0)
    tmp41 = tl.where(xmask, tmp39, float('nan'))
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK, RBLOCK])
    tmp45 = tl.where(xmask, tmp42, float('nan'))
    tmp46 = tl.tril(tmp45, 0)
    tmp47 = triton_helpers.promote_to_tensor(tl.sum(tmp46, 1))
    tmp48 = tl.load(in_ptr0 + (1280 + r2 + 256 * x1), xmask, other=0.0)
    tmp50 = tl.where(xmask, tmp48, float('nan'))
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, RBLOCK])
    tmp54 = tl.where(xmask, tmp51, float('nan'))
    tmp55 = tl.tril(tmp54, 0)
    tmp56 = triton_helpers.promote_to_tensor(tl.sum(tmp55, 1))
    tmp57 = tl.load(in_ptr1 + (640 + x0 + 128 * r2), xmask, eviction_policy
        ='evict_last')
    tmp58 = tl.load(in_ptr1 + (768 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr1 + (928 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr1 + (1088 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr0 + (1536 + r2 + 256 * x1), xmask, other=0.0)
    tmp63 = tl.where(xmask, tmp61, float('nan'))
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK, RBLOCK])
    tmp67 = tl.where(xmask, tmp64, float('nan'))
    tmp68 = tl.tril(tmp67, 0)
    tmp69 = triton_helpers.promote_to_tensor(tl.sum(tmp68, 1))
    tmp70 = tl.load(in_ptr0 + (1792 + r2 + 256 * x1), xmask, other=0.0)
    tmp72 = tl.where(xmask, tmp70, float('nan'))
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
    tmp76 = tl.where(xmask, tmp73, float('nan'))
    tmp77 = tl.tril(tmp76, 0)
    tmp78 = triton_helpers.promote_to_tensor(tl.sum(tmp77, 1))
    tmp79 = tl.load(in_ptr0 + (2048 + r2 + 256 * x1), xmask, other=0.0)
    tmp81 = tl.where(xmask, tmp79, float('nan'))
    tmp82 = tl.broadcast_to(tmp81, [XBLOCK, RBLOCK])
    tmp85 = tl.where(xmask, tmp82, float('nan'))
    tmp86 = tl.tril(tmp85, 0)
    tmp87 = triton_helpers.promote_to_tensor(tl.sum(tmp86, 1))
    tmp88 = tl.load(in_ptr1 + (1280 + x0 + 128 * r2), xmask, eviction_policy
        ='evict_last')
    tmp89 = tl.load(in_ptr1 + (1408 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp90 = tl.load(in_ptr1 + (1536 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr1 + (1664 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp92 = tl.load(in_ptr0 + (2304 + r2 + 256 * x1), xmask, other=0.0)
    tmp94 = tl.where(xmask, tmp92, float('nan'))
    tmp95 = tl.broadcast_to(tmp94, [XBLOCK, RBLOCK])
    tmp98 = tl.where(xmask, tmp95, float('nan'))
    tmp99 = tl.tril(tmp98, 0)
    tmp100 = triton_helpers.promote_to_tensor(tl.sum(tmp99, 1))
    tmp101 = tl.load(in_ptr0 + (2560 + r2 + 256 * x1), xmask, other=0.0)
    tmp103 = tl.where(xmask, tmp101, float('nan'))
    tmp104 = tl.broadcast_to(tmp103, [XBLOCK, RBLOCK])
    tmp107 = tl.where(xmask, tmp104, float('nan'))
    tmp108 = tl.tril(tmp107, 0)
    tmp109 = triton_helpers.promote_to_tensor(tl.sum(tmp108, 1))
    tmp110 = tl.load(in_ptr1 + (1920 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp111 = tl.load(in_ptr1 + (2048 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp112 = tl.load(in_ptr1 + (2176 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp113 = tl.load(in_ptr1 + (2304 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp114 = tl.load(in_ptr0 + (2816 + r2 + 256 * x1), xmask, other=0.0)
    tmp116 = tl.where(xmask, tmp114, float('nan'))
    tmp117 = tl.broadcast_to(tmp116, [XBLOCK, RBLOCK])
    tmp120 = tl.where(xmask, tmp117, float('nan'))
    tmp121 = tl.tril(tmp120, 0)
    tmp122 = triton_helpers.promote_to_tensor(tl.sum(tmp121, 1))
    tmp123 = tl.load(in_ptr0 + (3072 + r2 + 256 * x1), xmask, other=0.0)
    tmp125 = tl.where(xmask, tmp123, float('nan'))
    tmp126 = tl.broadcast_to(tmp125, [XBLOCK, RBLOCK])
    tmp129 = tl.where(xmask, tmp126, float('nan'))
    tmp130 = tl.tril(tmp129, 0)
    tmp131 = triton_helpers.promote_to_tensor(tl.sum(tmp130, 1))
    tmp132 = tl.load(in_ptr1 + (2240 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp133 = tl.load(in_ptr1 + (2368 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp134 = tl.load(in_ptr1 + (2512 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp135 = tl.load(in_ptr1 + (2672 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp136 = tl.load(in_ptr0 + (3328 + r2 + 256 * x1), xmask, other=0.0)
    tmp138 = tl.where(xmask, tmp136, float('nan'))
    tmp139 = tl.broadcast_to(tmp138, [XBLOCK, RBLOCK])
    tmp142 = tl.where(xmask, tmp139, float('nan'))
    tmp143 = tl.tril(tmp142, 0)
    tmp144 = triton_helpers.promote_to_tensor(tl.sum(tmp143, 1))
    tmp145 = tl.load(in_ptr0 + (3584 + r2 + 256 * x1), xmask, other=0.0)
    tmp147 = tl.where(xmask, tmp145, float('nan'))
    tmp148 = tl.broadcast_to(tmp147, [XBLOCK, RBLOCK])
    tmp151 = tl.where(xmask, tmp148, float('nan'))
    tmp152 = tl.tril(tmp151, 0)
    tmp153 = triton_helpers.promote_to_tensor(tl.sum(tmp152, 1))
    tmp154 = tl.load(in_ptr1 + (2560 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp155 = tl.load(in_ptr1 + (2688 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp156 = tl.load(in_ptr1 + (2832 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp157 = tl.load(in_ptr1 + (2992 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp158 = tl.load(in_ptr0 + (3840 + r2 + 256 * x1), xmask, other=0.0)
    tmp160 = tl.where(xmask, tmp158, float('nan'))
    tmp161 = tl.broadcast_to(tmp160, [XBLOCK, RBLOCK])
    tmp164 = tl.where(xmask, tmp161, float('nan'))
    tmp165 = tl.tril(tmp164, 0)
    tmp166 = triton_helpers.promote_to_tensor(tl.sum(tmp165, 1))
    tmp167 = tl.load(in_ptr0 + (4096 + r2 + 256 * x1), xmask, other=0.0)
    tmp169 = tl.where(xmask, tmp167, float('nan'))
    tmp170 = tl.broadcast_to(tmp169, [XBLOCK, RBLOCK])
    tmp173 = tl.where(xmask, tmp170, float('nan'))
    tmp174 = tl.tril(tmp173, 0)
    tmp175 = triton_helpers.promote_to_tensor(tl.sum(tmp174, 1))
    tmp176 = tl.load(in_ptr1 + (2880 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp177 = tl.load(in_ptr1 + (3008 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp178 = tl.load(in_ptr1 + (3168 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp179 = tl.load(in_ptr1 + (3328 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp180 = tl.load(in_ptr0 + (4352 + r2 + 256 * x1), xmask, other=0.0)
    tmp182 = tl.where(xmask, tmp180, float('nan'))
    tmp183 = tl.broadcast_to(tmp182, [XBLOCK, RBLOCK])
    tmp186 = tl.where(xmask, tmp183, float('nan'))
    tmp187 = tl.tril(tmp186, 0)
    tmp188 = triton_helpers.promote_to_tensor(tl.sum(tmp187, 1))
    tmp189 = tl.load(in_ptr0 + (4608 + r2 + 256 * x1), xmask, other=0.0)
    tmp191 = tl.where(xmask, tmp189, float('nan'))
    tmp192 = tl.broadcast_to(tmp191, [XBLOCK, RBLOCK])
    tmp195 = tl.where(xmask, tmp192, float('nan'))
    tmp196 = tl.tril(tmp195, 0)
    tmp197 = triton_helpers.promote_to_tensor(tl.sum(tmp196, 1))
    tmp198 = tl.load(in_ptr1 + (3200 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp199 = tl.load(in_ptr1 + (3328 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp200 = tl.load(in_ptr1 + (3488 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp201 = tl.load(in_ptr1 + (3648 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp202 = tl.load(in_ptr0 + (4864 + r2 + 256 * x1), xmask, other=0.0)
    tmp204 = tl.where(xmask, tmp202, float('nan'))
    tmp205 = tl.broadcast_to(tmp204, [XBLOCK, RBLOCK])
    tmp208 = tl.where(xmask, tmp205, float('nan'))
    tmp209 = tl.tril(tmp208, 0)
    tmp210 = triton_helpers.promote_to_tensor(tl.sum(tmp209, 1))
    tmp211 = tl.load(in_ptr0 + (5120 + r2 + 256 * x1), xmask, other=0.0)
    tmp213 = tl.where(xmask, tmp211, float('nan'))
    tmp214 = tl.broadcast_to(tmp213, [XBLOCK, RBLOCK])
    tmp217 = tl.where(xmask, tmp214, float('nan'))
    tmp218 = tl.tril(tmp217, 0)
    tmp219 = triton_helpers.promote_to_tensor(tl.sum(tmp218, 1))
    tmp220 = tl.load(in_ptr1 + (3520 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp221 = tl.load(in_ptr1 + (3648 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp222 = tl.load(in_ptr1 + (3824 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp223 = tl.load(in_ptr1 + (3984 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp224 = tl.load(in_ptr0 + (5376 + r2 + 256 * x1), xmask, other=0.0)
    tmp226 = tl.where(xmask, tmp224, float('nan'))
    tmp227 = tl.broadcast_to(tmp226, [XBLOCK, RBLOCK])
    tmp230 = tl.where(xmask, tmp227, float('nan'))
    tmp231 = tl.tril(tmp230, 0)
    tmp232 = triton_helpers.promote_to_tensor(tl.sum(tmp231, 1))
    tmp233 = tl.load(in_ptr0 + (5632 + r2 + 256 * x1), xmask, other=0.0)
    tmp235 = tl.where(xmask, tmp233, float('nan'))
    tmp236 = tl.broadcast_to(tmp235, [XBLOCK, RBLOCK])
    tmp239 = tl.where(xmask, tmp236, float('nan'))
    tmp240 = tl.tril(tmp239, 0)
    tmp241 = triton_helpers.promote_to_tensor(tl.sum(tmp240, 1))
    tmp242 = tl.load(in_ptr1 + (3840 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp243 = tl.load(in_ptr1 + (3968 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp244 = tl.load(in_ptr1 + (4160 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp245 = tl.load(in_ptr1 + (4320 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp246 = tl.load(in_ptr0 + (5888 + r2 + 256 * x1), xmask, other=0.0)
    tmp248 = tl.where(xmask, tmp246, float('nan'))
    tmp249 = tl.broadcast_to(tmp248, [XBLOCK, RBLOCK])
    tmp252 = tl.where(xmask, tmp249, float('nan'))
    tmp253 = tl.tril(tmp252, 0)
    tmp254 = triton_helpers.promote_to_tensor(tl.sum(tmp253, 1))
    tmp255 = tl.load(in_ptr0 + (6144 + r2 + 256 * x1), xmask, other=0.0)
    tmp257 = tl.where(xmask, tmp255, float('nan'))
    tmp258 = tl.broadcast_to(tmp257, [XBLOCK, RBLOCK])
    tmp261 = tl.where(xmask, tmp258, float('nan'))
    tmp262 = tl.tril(tmp261, 0)
    tmp263 = triton_helpers.promote_to_tensor(tl.sum(tmp262, 1))
    tmp264 = tl.load(in_ptr1 + (4160 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp265 = tl.load(in_ptr1 + (4288 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp266 = tl.load(in_ptr1 + (4480 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp267 = tl.load(in_ptr1 + (4640 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp268 = tl.load(in_ptr0 + (6400 + r2 + 256 * x1), xmask, other=0.0)
    tmp270 = tl.where(xmask, tmp268, float('nan'))
    tmp271 = tl.broadcast_to(tmp270, [XBLOCK, RBLOCK])
    tmp274 = tl.where(xmask, tmp271, float('nan'))
    tmp275 = tl.tril(tmp274, 0)
    tmp276 = triton_helpers.promote_to_tensor(tl.sum(tmp275, 1))
    tmp277 = tl.load(in_ptr0 + (6656 + r2 + 256 * x1), xmask, other=0.0)
    tmp279 = tl.where(xmask, tmp277, float('nan'))
    tmp280 = tl.broadcast_to(tmp279, [XBLOCK, RBLOCK])
    tmp283 = tl.where(xmask, tmp280, float('nan'))
    tmp284 = tl.tril(tmp283, 0)
    tmp285 = triton_helpers.promote_to_tensor(tl.sum(tmp284, 1))
    tmp286 = tl.load(in_ptr1 + (4480 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp287 = tl.load(in_ptr1 + (4608 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp288 = tl.load(in_ptr1 + (4800 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp289 = tl.load(in_ptr1 + (4960 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp290 = tl.load(in_ptr0 + (6896 + r2 + 256 * x1), xmask, other=0.0)
    tmp292 = tl.where(xmask, tmp290, float('nan'))
    tmp293 = tl.broadcast_to(tmp292, [XBLOCK, RBLOCK])
    tmp296 = tl.where(xmask, tmp293, float('nan'))
    tmp297 = tl.tril(tmp296, 0)
    tmp298 = triton_helpers.promote_to_tensor(tl.sum(tmp297, 1))
    tmp299 = tl.load(in_ptr0 + (7152 + r2 + 256 * x1), xmask, other=0.0)
    tmp301 = tl.where(xmask, tmp299, float('nan'))
    tmp302 = tl.broadcast_to(tmp301, [XBLOCK, RBLOCK])
    tmp305 = tl.where(xmask, tmp302, float('nan'))
    tmp306 = tl.tril(tmp305, 0)
    tmp307 = triton_helpers.promote_to_tensor(tl.sum(tmp306, 1))
    tmp308 = tl.load(in_ptr1 + (4800 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp309 = tl.load(in_ptr1 + (4928 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp310 = tl.load(in_ptr1 + (5120 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp311 = tl.load(in_ptr1 + (5280 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp312 = tl.load(in_ptr0 + (7424 + r2 + 256 * x1), xmask, other=0.0)
    tmp314 = tl.where(xmask, tmp312, float('nan'))
    tmp315 = tl.broadcast_to(tmp314, [XBLOCK, RBLOCK])
    tmp318 = tl.where(xmask, tmp315, float('nan'))
    tmp319 = tl.tril(tmp318, 0)
    tmp320 = triton_helpers.promote_to_tensor(tl.sum(tmp319, 1))
    tmp321 = tl.load(in_ptr0 + (7680 + r2 + 256 * x1), xmask, other=0.0)
    tmp323 = tl.where(xmask, tmp321, float('nan'))
    tmp324 = tl.broadcast_to(tmp323, [XBLOCK, RBLOCK])
    tmp327 = tl.where(xmask, tmp324, float('nan'))
    tmp328 = tl.tril(tmp327, 0)
    tmp329 = triton_helpers.promote_to_tensor(tl.sum(tmp328, 1))
    tmp330 = tl.load(in_ptr1 + (5120 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp331 = tl.load(in_ptr1 + (5248 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp332 = tl.load(in_ptr1 + (5440 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp333 = tl.load(in_ptr1 + (5600 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp334 = tl.load(in_ptr0 + (7936 + r2 + 256 * x1), xmask, other=0.0)
    tmp336 = tl.where(xmask, tmp334, float('nan'))
    tmp337 = tl.broadcast_to(tmp336, [XBLOCK, RBLOCK])
    tmp340 = tl.where(xmask, tmp337, float('nan'))
    tmp341 = tl.tril(tmp340, 0)
    tmp342 = triton_helpers.promote_to_tensor(tl.sum(tmp341, 1))
    tmp343 = tl.load(in_ptr0 + (8192 + r2 + 256 * x1), xmask, other=0.0)
    tmp345 = tl.where(xmask, tmp343, float('nan'))
    tmp346 = tl.broadcast_to(tmp345, [XBLOCK, RBLOCK])
    tmp349 = tl.where(xmask, tmp346, float('nan'))
    tmp350 = tl.tril(tmp349, 0)
    tmp351 = triton_helpers.promote_to_tensor(tl.sum(tmp350, 1))
    tmp352 = tl.load(in_ptr1 + (5440 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp353 = tl.load(in_ptr1 + (5568 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp354 = tl.load(in_ptr1 + (5760 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp355 = tl.load(in_ptr1 + (5920 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp356 = tl.load(in_ptr0 + (8448 + r2 + 256 * x1), xmask, other=0.0)
    tmp358 = tl.where(xmask, tmp356, float('nan'))
    tmp359 = tl.broadcast_to(tmp358, [XBLOCK, RBLOCK])
    tmp362 = tl.where(xmask, tmp359, float('nan'))
    tmp363 = tl.tril(tmp362, 0)
    tmp364 = triton_helpers.promote_to_tensor(tl.sum(tmp363, 1))
    tmp365 = tl.load(in_ptr0 + (8704 + r2 + 256 * x1), xmask, other=0.0)
    tmp367 = tl.where(xmask, tmp365, float('nan'))
    tmp368 = tl.broadcast_to(tmp367, [XBLOCK, RBLOCK])
    tmp371 = tl.where(xmask, tmp368, float('nan'))
    tmp372 = tl.tril(tmp371, 0)
    tmp373 = triton_helpers.promote_to_tensor(tl.sum(tmp372, 1))
    tmp374 = tl.load(in_ptr1 + (5760 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp375 = tl.load(in_ptr1 + (5888 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp376 = tl.load(in_ptr1 + (6080 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp377 = tl.load(in_ptr1 + (6240 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp378 = tl.load(in_ptr0 + (8992 + r2 + 256 * x1), xmask, other=0.0)
    tmp380 = tl.where(xmask, tmp378, float('nan'))
    tmp381 = tl.broadcast_to(tmp380, [XBLOCK, RBLOCK])
    tmp384 = tl.where(xmask, tmp381, float('nan'))
    tmp385 = tl.tril(tmp384, 0)
    tmp386 = triton_helpers.promote_to_tensor(tl.sum(tmp385, 1))
    tmp387 = tl.load(in_ptr0 + (9248 + r2 + 256 * x1), xmask, other=0.0)
    tmp389 = tl.where(xmask, tmp387, float('nan'))
    tmp390 = tl.broadcast_to(tmp389, [XBLOCK, RBLOCK])
    tmp393 = tl.where(xmask, tmp390, float('nan'))
    tmp394 = tl.tril(tmp393, 0)
    tmp395 = triton_helpers.promote_to_tensor(tl.sum(tmp394, 1))
    tmp396 = tl.load(in_ptr1 + (6080 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp397 = tl.load(in_ptr1 + (6208 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp398 = tl.load(in_ptr1 + (6400 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp399 = tl.load(in_ptr1 + (6560 + x0 + 128 * r2), xmask,
        eviction_policy='evict_last')
    tmp400 = tl.load(in_ptr0 + (9552 + r2 + 256 * x1), xmask, other=0.0)
    tmp402 = tl.where(xmask, tmp400, float('nan'))
    tmp403 = tl.broadcast_to(tmp402, [XBLOCK, RBLOCK])
    tmp406 = tl.where(xmask, tmp403, float('nan'))
    tmp407 = tl.tril(tmp406, 0)
    tmp408 = triton_helpers.promote_to_tensor(tl.sum(tmp