import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch as th
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_fused_add_mul_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 * tmp2
    tmp4 = 1.0
    tmp5 = tmp3 + tmp4
    tl.store(in_out_ptr0 + x0, tmp5, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (2048, 100, 512), (52428800, 524288, 512))
    assert_size_stride(primals_2, (32, 512), (512, 1))
    assert_size_stride(primals_3, (32,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2048, 512), (512, 1), torch.float32)
        get_raw_stream(0)
        triton_fused_add_mul_0[grid(2048)](buf0, primals_3, 2048, XBLOCK=
            256, num_warps=4, num_stages=1)
        del primals_3
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((2048, 512), (512, 1), torch.float32)
        triton_fused_add_mul_0[grid(2048)](buf2, primals_2, 2048, XBLOCK=
            256, num_warps=4, num_stages=1)
        del primals_2
        buf3 = torch.ops.aten.add.mm(primals_1, buf1, out=buf2)
        buf4 = buf2
        del buf2
        buf5 = torch.ops.aten.add.mm(primals_1, buf3, out=buf4)
    return buf5, primals_1, buf1, buf3, buf4


class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        # The `clusters` weights are the `(w,b)` in the paper
        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size

    def forward(self, input_0):
        primals_2 = self.clusters
        primals_3 = self.clusters2
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]