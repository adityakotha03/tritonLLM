import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_group_norm_0(in_ptr0, out_ptr0, out_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp3 = tl.load(in_ptr0 + 8192, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (16384 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (24576 + x0), xmask, eviction_policy='evict_last'
        )
    tmp13 = tl.load(in_ptr0 + (32768 + x0), xmask, eviction_policy='evict_last'
        )
    tmp16 = tl.load(in_ptr0 + (40960 + x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr0 + (49152 + x0), xmask, eviction_policy='evict_last'
        )
    tmp22 = tl.load(in_ptr0 + (57344 + x0), xmask, eviction_policy='evict_last'
        )
    tmp25 = tl.load(in_ptr0 + (65536 + x0), xmask, eviction_policy='evict_last'
        )
    tmp28 = tl.load(in_ptr0 + (73728 + x0), xmask, eviction_policy='evict_last'
        )
    tmp31 = tl.load(in_ptr0 + (81920 + x0), xmask, eviction_policy='evict_last'
        )
    tmp34 = tl.load(in_ptr0 + (90112 + x0), xmask, eviction_policy='evict_last'
        )
    tmp37 = tl.load(in_ptr0 + (98304 + x0), xmask, eviction_policy='evict_last'
        )
    tmp40 = tl.load(in_ptr0 + (106496 + x0), xmask, eviction_policy='evict_last'
        )
    tmp43 = tl.load(in_ptr0 + (114688 + x0), xmask, eviction_policy='evict_last'
        )
    tmp46 = tl.load(in_ptr0 + (122880 + x0), xmask, eviction_policy='evict_last'
        )
    tmp49 = tl.load(in_ptr0 + (131072 + x0), xmask, eviction_policy='evict_last'
        )
    tmp52 = tl.load(in_ptr0 + (139264 + x0), xmask, eviction_policy='evict_last'
        )
    tmp55 = tl.load(in_ptr0 + (147456 + x0), xmask, eviction_policy='evict_last'
        )
    tmp58 = tl.load(in_ptr0 + (155648 + x0), xmask, eviction_policy='evict_last'
        )
    tmp61 = tl.load(in_ptr0 + (163840 + x0), xmask, eviction_policy='evict_last'
        )
    tmp64 = tl.load(in_ptr0 + (172032 + x0), xmask, eviction_policy='evict_last'
        )
    tmp67 = tl.load(in_ptr0 + (180224 + x0), xmask, eviction_policy='evict_last'
        )
    tmp70 = tl.load(in_ptr0 + (188416 + x0), xmask, eviction_policy='evict_last'
        )
    tmp73 = tl.load(in_ptr0 + (196608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp76 = tl.load(in_ptr0 + (204800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp79 = tl.load(in_ptr0 + (213088 + x0), xmask, eviction_policy='evict_last'
        )
    tmp82 = tl.load(in_ptr0 + (221280 + x0), xmask, eviction_policy='evict_last'
        )
    tmp85 = tl.load(in_ptr0 + (229472 + x0), xmask, eviction_policy='evict_last'
        )
    tmp88 = tl.load(in_ptr0 + (237664 + x0), xmask, eviction_policy='evict_last'
        )
    tmp91 = tl.load(in_ptr0 + (245856 + x0), xmask, eviction_policy='evict_last'
        )
    tmp94 = tl.load(in_ptr0 + (254048 + x0), xmask, eviction_policy='evict_last'
        )
    tmp97 = tl.load(in_ptr0 + (262240 + x0), xmask, eviction_policy='evict_last'
        )
    tmp100 = tl.load(in_ptr0 + (270432 + x0), xmask, eviction_policy='evict_last'
        )
    tmp103 = tl.load(in_ptr0 + (278624 + x0), xmask, eviction_policy='evict_last'
        )
    tmp106 = tl.load(in_ptr0 + (286816 + x0), xmask, eviction_policy='evict_last'
        )
    tmp109 = tl.load(in_ptr0 + (295008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp112 = tl.load(in_ptr0 + (303200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp115 = tl.load(in_ptr0 + (311392 + x0), xmask, eviction_policy='evict_last'
        )
    tmp118 = tl.load(in_ptr0 + (319584 + x0), xmask, eviction_policy='evict_last'
        )
    tmp121 = tl.load(in_ptr0 + (327776 + x0), xmask, eviction_policy='evict_last'
        )
    tmp124 = tl.load(in_ptr0 + (335968 + x0), xmask, eviction_policy='evict_last'
        )
    tmp127 = tl.load(in_ptr0 + (344160 + x0), xmask, eviction_policy='evict_last'
        )
    tmp130 = tl.load(in_ptr0 + (352352 + x0), xmask, eviction_policy='evict_last'
        )
    tmp133 = tl.load(in_ptr0 + (360544 + x0), xmask, eviction_policy='evict_last'
        )
    tmp136 = tl.load(in_ptr0 + (368736 + x0), xmask, eviction_policy='evict_last'
        )
    tmp139 = tl.load(in_ptr0 + (376928 + x0), xmask, eviction_policy='evict_last'
        )
    tmp142 = tl.load(in_ptr0 + (385120 + x0), xmask, eviction_policy='evict_last'
        )
    tmp145 = tl.load(in_ptr0 + (393312 + x0), xmask, eviction_policy='evict_last'
        )
    tmp148 = tl.load(in_ptr0 + (401504 + x0), xmask, eviction_policy='evict_last'
        )
    tmp151 = tl.load(in_ptr0 + (409696 + x0), xmask, eviction_policy='evict_last'
        )
    tmp154 = tl.load(in_ptr0 + (417888 + x0), xmask, eviction_policy='evict_last'
        )
    tmp157 = tl.load(in_ptr0 + (426080 + x0), xmask, eviction_policy='evict_last'
        )
    tmp160 = tl.load(in_ptr0 + (434272 + x0), xmask, eviction_policy='evict_last'
        )
    tmp163 = tl.load(in_ptr0 + (442464 + x0), xmask, eviction_policy='evict_last'
        )
    tmp166 = tl.load(in_ptr0 + (450656 + x0), xmask, eviction_policy='evict_last'
        )
    tmp169 = tl.load(in_ptr0 + (458848 + x0), xmask, eviction_policy='evict_last'
        )
    tmp172 = tl.load(in_ptr0 + (467040 + x0), xmask, eviction_policy='evict_last'
        )
    tmp175 = tl.load(in_ptr0 + (475232 + x0), xmask, eviction_policy='evict_last'
        )
    tmp178 = tl.load(in_ptr0 + (483424 + x0), xmask, eviction_policy='evict_last'
        )
    tmp181 = tl.load(in_ptr0 + (491616 + x0), xmask, eviction_policy='evict_last'
        )
    tmp184 = tl.load(in_ptr0 + (499808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp187 = tl.load(in_ptr0 + (508000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp190 = tl.load(in_ptr0 + (516192 + x0), xmask, eviction_policy='evict_last'
        )
    tmp193 = tl.load(in_ptr0 + (524384 + x0), xmask, eviction_policy='evict_last'
        )
    tmp196 = tl.load(in_ptr0 + (532576 + x0), xmask, eviction_policy='evict_last'
        )
    tmp199 = tl.load(in_ptr0 + (540768 + x0), xmask, eviction_policy='evict_last'
        )
    tmp202 = tl.load(in_ptr0 + (548960 + x0), xmask, eviction_policy='evict_last'
        )
    tmp205 = tl.load(in_ptr0 + (557152 + x0), xmask, eviction_policy='evict_last'
        )
    tmp208 = tl.load(in_ptr0 + (565344 + x0), xmask, eviction_policy='evict_last'
        )
    tmp211 = tl.load(in_ptr0 + (573536 + x0), xmask, eviction_policy='evict_last'
        )
    tmp214 = tl.load(in_ptr0 + (581728 + x0), xmask, eviction_policy='evict_last'
        )
    tmp217 = tl.load(in_ptr0 + (589920 + x0), xmask, eviction_policy='evict_last'
        )
    tmp220 = tl.load(in_ptr0 + (598112 + x0), xmask, eviction_policy='evict_last'
        )
    tmp223 = tl.load(in_ptr0 + (606304 + x0), xmask, eviction_policy='evict_last'
        )
    tmp226 = tl.load(in_ptr0 + (614496 + x0), xmask, eviction_policy='evict_last'
        )
    tmp229 = tl.load(in_ptr0 + (622688 + x0), xmask, eviction_policy='evict_last'
        )
    tmp232 = tl.load(in_ptr0 + (630880 + x0), xmask, eviction_policy='evict_last'
        )
    tmp235 = tl.load(in_ptr0 + (639072 + x0), xmask, eviction_policy='evict_last'
        )
    tmp238 = tl.load(in_ptr0 + (647264 + x0), xmask, eviction_policy='evict_last'
        )
    tmp241 = tl.load(in_ptr0 + (655456 + x0), xmask, eviction_policy='evict_last'
        )
    tmp244 = tl.load(in_ptr0 + (663648 + x0), xmask, eviction_policy='evict_last'
        )
    tmp247 = tl.load(in_ptr0 + (671840 + x0), xmask, eviction_policy='evict_last'
        )
    tmp250 = tl.load(in_ptr0 + (679984 + x0), xmask, eviction_policy='evict_last'
        )
    tmp253 = tl.load(in_ptr0 + (688176 + x0), xmask, eviction_policy='evict_last'
        )
    tmp256 = tl.load(in_ptr0 + (696368 + x0), xmask, eviction_policy='evict_last'
        )
    tmp259 = tl.load(in_ptr0 + (704560 + x0), xmask, eviction_policy='evict_last'
        )
    tmp262 = tl.load(in_ptr0 + (712752 + x0), xmask, eviction_policy='evict_last'
        )
    tmp265 = tl.load(in_ptr0 + (720944 + x0), xmask, eviction_policy='evict_last'
        )
    tmp268 = tl.load(in_ptr0 + (729136 + x0), xmask, eviction_policy='evict_last'
        )
    tmp271 = tl.load(in_ptr0 + (737328 + x0), xmask, eviction_policy='evict_last'
        )
    tmp274 = tl.load(in_ptr0 + (745520 + x0), xmask, eviction_policy='evict_last'
        )
    tmp277 = tl.load(in_ptr0 + (753712 + x0), xmask, eviction_policy='evict_last'
        )
    tmp280 = tl.load(in_ptr0 + (761904 + x0), xmask, eviction_policy='evict_last'
        )
    tmp283 = tl.load(in_ptr0 + (770096 + x0), xmask, eviction_policy='evict_last'
        )
    tmp286 = tl.load(in_ptr0 + (778288 + x0), xmask, eviction_policy='evict_last'
        )
    tmp289 = tl.load(in_ptr0 + (786480 + x0), xmask, eviction_policy='evict_last'
        )
    tmp292 = tl.load(in_ptr0 + (794672 + x0), xmask, eviction_policy='evict_last'
        )
    tmp295 = tl.load(in_ptr0 + (802864 + x0), xmask, eviction_policy='evict_last'
        )
    tmp298 = tl.load(in_ptr0 + (811056 + x0), xmask, eviction_policy='evict_last'
        )
    tmp301 = tl.load(in_ptr0 + (819248 + x0), xmask, eviction_policy='evict_last'
        )
    tmp304 = tl.load(in_ptr0 + (827440 + x0), xmask, eviction_policy='evict_last'
        )
    tmp307 = tl.load(in_ptr0 + (835632 + x0), xmask, eviction_policy='evict_last'
        )
    tmp310 = tl.load(in_ptr0 + (843824 + x0), xmask, eviction_policy='evict_last'
        )
    tmp313 = tl.load(in_ptr0 + (852016 + x0), xmask, eviction_policy='evict_last'
        )
    tmp316 = tl.load(in_ptr0 + (860208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp319 = tl.load(in_ptr0 + (868400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp322 = tl.load(in_ptr0 + (876592 + x0), xmask, eviction_policy='evict_last'
        )
    tmp325 = tl.load(in_ptr0 + (884784 + x0), xmask, eviction_policy='evict_last'
        )
    tmp328 = tl.load(in_ptr0 + (892976 + x0), xmask, eviction_policy='evict_last'
        )
    tmp331 = tl.load(in_ptr0 + (901168 + x0), xmask, eviction_policy='evict_last'
        )
    tmp334 = tl.load(in_ptr0 + (909360 + x0), xmask, eviction_policy='evict_last'
        )
    tmp337 = tl.load(in_ptr0 + (917552 + x0), xmask, eviction_policy='evict_last'
        )
    tmp340 = tl.load(in_ptr0 + (925744 + x0), xmask, eviction_policy='evict_last'
        )
    tmp343 = tl.load(in_ptr0 + (933936 + x0), xmask, eviction_policy='evict_last'
        )
    tmp346 = tl.load(in_ptr0 + (942128 + x0), xmask, eviction_policy='evict_last'
        )
    tmp349 = tl.load(in_ptr0 + (950320 + x0), xmask, eviction_policy='evict_last'
        )
    tmp352 = tl.load(in_ptr0 + (958512 + x0), xmask, eviction_policy='evict_last'
        )
    tmp355 = tl.load(in_ptr0 + (966704 + x0), xmask, eviction_policy='evict_last'
        )
    tmp358 = tl.load(in_ptr0 + (974896 + x0), xmask, eviction_policy='evict_last'
        )
    tmp361 = tl.load(in_ptr0 + (983088 + x0), xmask, eviction_policy='evict_last'
        )
    tmp364 = tl.load(in_ptr0 + (991280 + x0), xmask, eviction_policy='evict_last'
        )
    tmp367 = tl.load(in_ptr0 + (999472 + x0), xmask, eviction_policy='evict_last'
        )
    tmp370 = tl.load(in_ptr0 + (1007664 + x0), xmask, eviction_policy='evict_last'
        )
    tmp373 = tl.load(in_ptr0 + (1015856 + x0), xmask, eviction_policy='evict_last'
        )
    tmp376 = tl.load(in_ptr0 + (1024048 + x0), xmask, eviction_policy='evict_last'
        )
    tmp379 = tl.load(in_ptr0 + (1032240 + x0), xmask, eviction_policy='evict_last'
        )
    tmp382 = tl.load(in_ptr0 + (1040432 + x0), xmask, eviction_policy='evict_last'
        )
    tmp385 = tl.load(in_ptr0 + (1048624 + x0), xmask, eviction_policy='evict_last'
        )
    tmp388 = tl.load(in_ptr0 + (1056816 + x0), xmask, eviction_policy='evict_last'
        )
    tmp391 = tl.load(in_ptr0 + (1065008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp394 = tl.load(in_ptr0 + (1073200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp397 = tl.load(in_ptr0 + (1081392 + x0), xmask, eviction_policy='evict_last'
        )
    tmp400 = tl.load(in_ptr0 + (1089584 + x0), xmask, eviction_policy='evict_last'
        )
    tmp403 = tl.load(in_ptr0 + (1097776 + x0), xmask, eviction_policy='evict_last'
        )
    tmp406 = tl.load(in_ptr0 + (1105968 + x0), xmask, eviction_policy='evict_last'
        )
    tmp409 = tl.load(in_ptr0 + (1114160 + x0), xmask, eviction_policy='evict_last'
        )
    tmp412 = tl.load(in_ptr0 + (1122352 + x0), xmask, eviction_policy='evict_last'
        )
    tmp415 = tl.load(in_ptr0 + (1130544 + x0), xmask, eviction_policy='evict_last'
        )
    tmp418 = tl.load(in_ptr0 + (1138736 + x0), xmask, eviction_policy='evict_last'
        )
    tmp421 = tl.load(in_ptr0 + (1146928 + x0), xmask, eviction_policy='evict_last'
        )
    tmp424 = tl.load(in_ptr0 + (1155120 + x0), xmask, eviction_policy='evict_last'
        )
    tmp427 = tl.load(in_ptr0 + (1163312 + x0), xmask, eviction_policy='evict_last'
        )
    tmp430 = tl.load(in_ptr0 + (1171504 + x0), xmask, eviction_policy='evict_last'
        )
    tmp433 = tl.load(in_ptr0 + (1179696 + x0), xmask, eviction_policy='evict_last'
        )
    tmp436 = tl.load(in_ptr0 + (1187888 + x0), xmask, eviction_policy='evict_last'
        )
    tmp439 = tl.load(in_ptr0 + (1196080 + x0), xmask, eviction_policy='evict_last'
        )
    tmp442 = tl.load(in_ptr0 + (1204272 + x0), xmask, eviction_policy='evict_last'
        )
    tmp445 = tl.load(in_ptr0 + (1212464 + x0), xmask, eviction_policy='evict_last'
        )
    tmp448 = tl.load(in_ptr0 + (1220656 + x0), xmask, eviction_policy='evict_last'
        )
    tmp451 = tl.load(in_ptr0 + (1228848 + x0), xmask, eviction_policy='evict_last'
        )
    tmp454 = tl.load(in_ptr0 + (1237040 + x0), xmask, eviction_policy='evict_last'
        )
    tmp457 = tl.load(in_ptr0 + (1245232 + x0), xmask, eviction_policy='evict_last'
        )
    tmp460 = tl.load(in_ptr0 + (1253424 + x0), xmask, eviction_policy='evict_last'
        )
    tmp463 = tl.load(in_ptr0 + (1261616 + x0), xmask, eviction_policy='evict_last'
        )
    tmp466 = tl.load(in_ptr0 + (1269808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp469 = tl.load(in_ptr0 + (1277999 + x0), xmask, eviction_policy='evict_last'
        )
    tmp472 = tl.load(in_ptr0 + (1286192 + x0), xmask, eviction_policy='evict_last'
        )
    tmp475 = tl.load(in_ptr0 + (1294384 + x0), xmask, eviction_policy='evict_last'
        )
    tmp478 = tl.load(in_ptr0 + (1302576 + x0), xmask, eviction_policy='evict_last'
        )
    tmp481 = tl.load(in_ptr0 + (1310768 + x0), xmask, eviction_policy='evict_last'
        )
    tmp484 = tl.load(in_ptr0 + (1318960 + x0), xmask, eviction_policy='evict_last'
        )
    tmp487 = tl.load(in_ptr0 + (1327152 + x0), xmask, eviction_policy='evict_last'
        )
    tmp490 = tl.load(in_ptr0 + (1335344 + x0), xmask, eviction_policy='evict_last'
        )
    tmp493 = tl.load(in_ptr0 + (1343536 + x0), xmask, eviction_policy='evict_last'
        )
    tmp496 = tl.load(in_ptr0 + (1351728 + x0), xmask, eviction_policy='evict_last'
        )
    tmp499 = tl.load(in_ptr0 + (1359920 + x0), xmask, eviction_policy='evict_last'
        )
    tmp502 = tl.load(in_ptr0 + (1368112 + x0), xmask, eviction_policy='evict_last'
        )
    tmp505 = tl.load(in_ptr0 + (1376304 + x0), xmask, eviction_policy='evict_last'
        )
    tmp508 = tl.load(in_ptr0 + (1384496 + x0), xmask, eviction_policy='evict_last'
        )
    tmp511 = tl.load(in_ptr0 + (1392688 + x0), xmask, eviction_policy='evict_last'
        )
    tmp514 = tl.load(in_ptr0 + (1400880 + x0), xmask, eviction_policy='evict_last'
        )
    tmp517 = tl.load(in_ptr0 + (1409072 + x0), xmask, eviction_policy='evict_last'
        )
    tmp520 = tl.load(in_ptr0 + (1417264 + x0), xmask, eviction_policy='evict_last'
        )
    tmp523 = tl.load(in_ptr0 + (1425456 + x0), xmask, eviction_policy='evict_last'
        )
    tmp526 = tl.load(in_ptr0 + (1433648 + x0), xmask, eviction_policy='evict_last'
        )
    tmp529 = tl.load(in_ptr0 + (1441840 + x0), xmask, eviction_policy='evict_last'
        )
    tmp532 = tl.load(in_ptr0 + (1449999 + x0), xmask, eviction_policy='evict_last'
        )
    tmp535 = tl.load(in_ptr0 + (1458192 + x0), xmask, eviction_policy='evict_last'
        )
    tmp538 = tl.load(in_ptr0 + (1466384 + x0), xmask, eviction_policy='evict_last'
        )
    tmp541 = tl.load(in_ptr0 + (1474576 + x0), xmask, eviction_policy='evict_last'
        )
    tmp544 = tl.load(in_ptr0 + (1482768 + x0), xmask, eviction_policy='evict_last'
        )
    tmp547 = tl.load(in_ptr0 + (1490960 + x0), xmask, eviction_policy='evict_last'
        )
    tmp550 = tl.load(in_ptr0 + (1499152 + x0), xmask, eviction_policy='evict_last'
        )
    tmp553 = tl.load(in_ptr0 + (1507344 + x0), xmask, eviction_policy='evict_last'
        )
    tmp556 = tl.load(in_ptr0 + (1515536 + x0), xmask, eviction_policy='evict_last'
        )
    tmp559 = tl.load(in_ptr0 + (1523728 + x0), xmask, eviction_policy='evict_last'
        )
    tmp562 = tl.load(in_ptr0 + (1531920 + x0), xmask, eviction_policy='evict_last'
        )
    tmp565 = tl.load(in_ptr0 + (1540112 + x0), xmask, eviction_policy='evict_last'
        )
    tmp568 = tl.load(in_ptr0 + (1548304 + x0), xmask, eviction_policy='evict_last'
        )
    tmp571 = tl.load(in_ptr0 + (1556496 + x0), xmask, eviction_policy='evict_last'
        )
    tmp574 = tl.load(in_ptr0 + (1564688 + x0), xmask, eviction_policy='evict_last'
        )
    tmp577 = tl.load(in_ptr0 + (1572880 + x0), xmask, eviction_policy='evict_last'
        )
    tmp580 = tl.load(in_ptr0 + (1581072 + x0), xmask, eviction_policy='evict_last'
        )
    tmp583 = tl.load(in_ptr0 + (1589264 + x0), xmask, eviction_policy='evict_last'
        )
    tmp586 = tl.load(in_ptr0 + (1597456 + x0), xmask, eviction_policy='evict_last'
        )
    tmp589 = tl.load(in_ptr0 + (1605648 + x0), xmask, eviction_policy='evict_last'
        )
    tmp592 = tl.load(in_ptr0 + (1613840 + x0), xmask, eviction_policy='evict_last'
        )
    tmp595 = tl.load(in_ptr0 + (1622032 + x0), xmask, eviction_policy='evict_last'
        )
    tmp598 = tl.load(in_ptr0 + (1630224 + x0), xmask, eviction_policy='evict_last'
        )
    tmp601 = tl.load(in_ptr0 + (1638416 + x0), xmask, eviction_policy='evict_last'
        )
    tmp604 = tl.load(in_ptr0 + (1646608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp607 = tl.load(in_ptr0 + (1654800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp610 = tl.load(in_ptr0 + (1662992 + x0), xmask, eviction_policy='evict_last'
        )
    tmp613 = tl.load(in_ptr0 + (1671184 + x0), xmask, eviction_policy='evict_last'
        )
    tmp616 = tl.load(in_ptr0 + (1679376 + x0), xmask, eviction_policy='evict_last'
        )
    tmp619 = tl.load(in_ptr0 + (1687568 + x0), xmask, eviction_policy='evict_last'
        )
    tmp622 = tl.load(in_ptr0 + (1695760 + x0), xmask, eviction_policy='evict_last'
        )
    tmp625 = tl.load(in_ptr0 + (1703952 + x0), xmask, eviction_policy='evict_last'
        )
    tmp628 = tl.load(in_ptr0 + (1712144 + x0), xmask, eviction_policy='evict_last'
        )
    tmp631 = tl.load(in_ptr0 + (1720336 + x0), xmask, eviction_policy='evict_last'
        )
    tmp634 = tl.load(in_ptr0 + (1728528 + x0), xmask, eviction_policy='evict_last'
        )
    tmp637 = tl.load(in_ptr0 + (1736720 + x0), xmask, eviction_policy='evict_last'
        )
    tmp640 = tl.load(in_ptr0 + (1744912 + x0), xmask, eviction_policy='evict_last'
        )
    tmp643 = tl.load(in_ptr0 + (1753104 + x0), xmask, eviction_policy='evict_last'
        )
    tmp646 = tl.load(in_ptr0 + (1761296 + x0), xmask, eviction_policy='evict_last'
        )
    tmp649 = tl.load(in_ptr0 + (1769488 + x0), xmask, eviction_policy='evict_last'
        )
    tmp652 = tl.load(in_ptr0 + (1777680 + x0), xmask, eviction_policy='evict_last'
        )
    tmp655 = tl.load(in_ptr0 + (1785872 + x0), xmask, eviction_policy='evict_last'
        )
    tmp658 = tl.load(in_ptr0 + (1794064 + x0), xmask, eviction_policy='evict_last'
        )
    tmp661 = tl.load(in_ptr0 + (1802256 + x0), xmask, eviction_policy='evict_last'
        )
    tmp664 = tl.load(in_ptr0 + (1810448 + x0), xmask, eviction_policy='evict_last'
        )
    tmp667 = tl.load(in_ptr0 + (1818640 + x0), xmask, eviction_policy='evict_last'
        )
    tmp670 = tl.load(in_ptr0 + (1826832 + x0), xmask, eviction_policy='evict_last'
        )
    tmp673 = tl.load(in_ptr0 + (1835024 + x0), xmask, eviction_policy='evict_last'
        )
    tmp676 = tl.load(in_ptr0 + (1843216 + x0), xmask, eviction_policy='evict_last'
        )
    tmp679 = tl.load(in_ptr0 + (1851408 + x