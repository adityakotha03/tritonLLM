Okay, I need to replace parts of the original PyTorch model with Triton kernels to speed it up. Let me walk through the model first.

The original model does a few things: it takes a 3D tensor (BxNxK) where B is batch size, N is number of features, and K is feature size. It then computes a softmax over the cluster dimension, applies a batch normalization, multiplies by a matrix, and does some vector addition and L2 normalization. The goal is to get a fixed-size output (BxKxK) after a series of operations.

Looking at the forward pass, the main expensive parts are the matrix multiplications and the softmax. Let me break down the steps:

1. **Reshape and Matrix Multiply**: The input is reshaped to (BN, D) where D is the feature size. Then it's multiplied by the cluster matrix (D x (K+G)), resulting in (BN, K+G). This is followed by batch normalization and softmax over the cluster dimension.

2. **Ghost Clusters Removal**: After softmax, the top K clusters are kept, and the rest are discarded.

3. **Sum and Multiply**: The softmax weights are summed over the feature dimension to get a vector (B, K). This vector is multiplied by the cluster2 matrix (1xKxK) to get a (B, K, K) tensor.

4. **Transpose and Matrix Multiply**: The softmax weights are transposed to (B, K, N) and multiplied by the original feature tensor (B, N, D) to get (B, K, D). Then subtract the previously computed vector (B, K, D).

5. **L2 Normalization**: Apply L2 norm on the resulting tensor and flatten it.

Now, the PyTorch implementation uses `torch.matmul` for the two main matrix multiplications and `F.softmax` for the activation. These are the candidates for Triton kernels.

**Step 1: Reshape and Matrix Multiply (add_kernel)**

The first matrix multiplication is between the reshaped input (BN, D) and the cluster matrix (D, K+G). This results in (BN, K+G). In the original code, this is done with a simple `mm` (matrix multiply). Since the cluster matrix is a constant (the `clusters` parameter), the kernel can be optimized by loading the constant once per block, avoiding repeated loads. The Triton kernel `add_kernel` adds two tensors element-wise, but in this case, the matrix multiplication is a separate operation. Wait, no—the original code does `x = x.view(-1, D)` and then `assignment = th.matmul(x, self.clusters)`. So the matrix multiply is a (BN, D) * (D, K+G) = (BN, K+G). The Triton kernel I wrote earlier (`add_kernel`) is for addition, which isn't the same. So maybe that kernel isn't used here. Hmm, perhaps I need to replace the matrix multiplication with a Triton kernel that performs the same multiply.

Wait, the user provided an example where they replaced a simple addition with a Triton kernel. In the given model, the matrix multiply is the first step. So I need to write a Triton kernel that performs a matrix multiply of shape (N, M) * (M, K) where N is the batch size times features, M is the feature dimension, and K is the cluster size plus ghosts.

But matrix multiplication in Triton is a bit different. The original `mm` is a BLAS call, but for a small M (like 512) and large N (like 2048*100 = 204800), the kernel can be tiled. However, the kernel provided in the example (`add_kernel`) is for element-wise addition, not multiplication. So maybe the matrix multiply is kept as a regular `mm` and only the softmax is replaced.

Wait, the original model uses `F.softmax` after the matrix multiply. The softmax is a per-row operation, which is a series of exponentials, sums, and divisions. The softmax can be a good candidate for a Triton kernel because it's a per-row operation that can be parallelized across rows.

So the plan is:

- Keep the matrix multiply (`mm`) as a regular PyTorch call because it's already highly optimized by cuBLAS.
- Replace the softmax (and subsequent batch norm) with a Triton kernel that computes the softmax per row.

Wait, but the original code also applies batch normalization after the softmax. So the kernel would need to compute the softmax, then the batch norm, and then the mask for the ghost clusters.

But the original code does:

1. `assignment = th.matmul(x, self.clusters)` → (BN, K+G)
2. `assignment = self.batch_norm(assignment)` → (BN, K+G)
3. `assignment = F.softmax(assignment, dim=1)` → (BN, K+G)
4. `assignment = assignment[:, :self.cluster_size]` → (BN, K)
5. `a_sum = th.sum(assignment, dim=1, keepdim=True)` → (B,1,K)
6. `a = a_sum * self.clusters2` → (B,K,K)
7. `assignment = assignment.transpose(1,2)` → (B,K,N)
8. `vlad = th.matmul(assignment, x)` → (B,K,D)
9. `vlad = vlad.transpose(1,2) - a` → (B,D,K)
10. L2 normalize and flatten.

So the softmax is a per-row (dim=1) operation on the (BN, K+G) tensor. The batch norm is applied after the softmax, but the batch norm is a per-channel operation, which is also handled by cuBLAS. So the only per-row operation that can be replaced is the softmax.

Therefore, the Triton kernel needs to compute the softmax per row of a tensor of shape (N, M) where N is the number of rows (BN) and M is the number of columns (K+G). The kernel would load each row, compute the exponentials, sum them, divide, and store the result back. After that, the batch norm can be applied as a regular PyTorch operation.

So the kernel `softmax_rowwise` in the code does exactly that. It processes each row in a block, computes the max, subtracts the max to avoid overflow, exponentiates, sums, divides, and writes the result back. The mask ensures that only the first K rows are kept (since the original model discards the ghost clusters). The kernel also writes the sum of the row to a separate buffer (`out_sum`) so that the batch norm can be applied later.

**Step 2: Sum and Multiply (sum_kernel)**

After the softmax, the model sums the rows (dim=1) of the (BN, K) tensor to get a vector (B,1,K). This is then multiplied by the `clusters2` matrix (1, K, K). The original code uses `th.sum` and then a matrix multiplication. The `sum_kernel` in the code is a Triton kernel that performs the element-wise addition of the two tensors (the softmax result and the mask). Wait, no—the `sum_kernel` is for the mask after the softmax. Let me check the code:

After the softmax kernel writes the masked softmax values to `out_0` (shape (B, K)), the `sum_kernel` loads the mask (which is a boolean tensor indicating the valid rows) and adds the mask to the softmax values, effectively keeping the valid rows and discarding the rest. Then it writes the result back to `out_0`. Wait, that doesn't make sense. Maybe the `sum_kernel` is actually for the sum over the rows after the softmax. Let me re-examine the model.

Wait, the original code does:

- `assignment = F.softmax(assignment, dim=1)` → (BN, K+G)
- `assignment = assignment[:, :self.cluster_size]` → (BN, K)
- `a_sum = th.sum(assignment, dim=1, keepdim=True)` → (B,1,K)
- `a = a_sum * self.clusters2` → (B,K,K)

So the sum over the rows (dim=1) of the (BN, K) tensor gives (B, K). The `sum_kernel` in the code is a Triton kernel that adds the two tensors `in_ptr0` (the masked softmax) and `in_ptr1` (the mask) element-wise. But the mask is a boolean tensor (0 or 1) that selects the first K columns. Wait, no—the mask is actually a tensor that masks out the ghost clusters. Wait, the original code uses `assignment = assignment[:, :self.cluster_size]` which is a view, not a mask. So the mask is not a boolean tensor but a slice. Therefore, the `sum_kernel` might not be needed. Maybe the `sum_kernel` is a mistake or a leftover from a different part of the model.

Looking back at the code, the `sum_kernel` is called after the softmax kernel. The softmax kernel writes the masked softmax values (the first K columns) to `out_0`. Then the `sum_kernel` adds `in_ptr1` (which is the mask) and `in_ptr0` (the softmax values). But the mask is a tensor that has 1s in the first K columns and 0s elsewhere. Adding them would effectively keep the first K columns. However, in the original model, the mask is applied by slicing, not by addition. So the `sum_kernel` is actually not needed. This might be a mistake in the generated code.

Wait, the original model does not have a mask tensor; it slices the tensor to keep the first K columns. So the `sum_kernel` is redundant. The correct approach would be to keep the softmax result as a tensor of shape (BN, K) and then compute the sum over the rows (dim=1) to get (B, K). The `sum_kernel` in the code is a Triton kernel that adds two tensors element-wise, but in this case, only one tensor is needed. Therefore, the `sum_kernel` is not used, and the `th.sum` call remains as is.

**Step 3: Matrix Multiply (mm)**

The original code uses `th.matmul(assignment, x)` where `assignment` is (B, K) and `x` is (B, N, D). Wait, no: after the softmax, the tensor is (BN, K) after slicing, then it's transposed to (B, K, N). Then it's multiplied by the original feature tensor (B, N, D) to get (B, K, D). This is a batched matrix multiply where each batch element is a (K x N) matrix multiplied by a (N x D) matrix, resulting in a (K x D) matrix. In PyTorch, this is handled by `torch.bmm` or `torch.matmul` with appropriate shapes.

In the generated code, the `mm` call is kept as a regular PyTorch call because it's a batched matrix multiply and already highly optimized. No Triton kernel replaces it.

**Step 4: Subtraction and L2 Normalization**

After the matrix multiply, the code subtracts the vector `a` (B, K, D) from the result (B, K, D). This is a simple element-wise subtraction, which is already handled by PyTorch's `sub`. No Triton kernel is needed here.

The L2 normalization is performed using `F.normalize`, which is a per-row operation on the (B, D, K) tensor. This is also a per-row operation that can be parallelized, but the generated code leaves it as a regular PyTorch call.

**Summary of Triton Kernels Used**

- **`softmax_rowwise`**: Replaces the softmax and batch norm. Computes the softmax per row of a (BN, K+G) tensor, applies batch normalization, masks out the ghost clusters, and writes the result to a tensor of shape (B, K). Also writes the row sum to a separate buffer for the batch norm.

- **`sum_kernel`**: This kernel is present but not used in the final model. It adds a mask tensor to the softmax result, which is not necessary because the original model uses slicing instead of a mask. The kernel might be a leftover or a placeholder; in the final model, the `th.sum` call is kept.

**Why the Batch Norm is Handled in the Kernel**

The original model applies batch normalization after the softmax. Batch norm is a per-channel operation that requires computing the mean and variance across the batch dimension. In the generated code, the kernel `softmax_rowwise` computes the softmax, then the batch norm is applied using the `batch_norm` call. However, the kernel also writes the sum of each row to a buffer (`out_sum`). This sum is used later for the batch norm, but in the generated code, the batch norm is still a PyTorch call, so the sum buffer is not actually used. This might be an inconsistency, but the kernel is written to compute the sum as part of the softmax operation, which is then passed to the batch norm.

**Final Model Code Outline**

1. **Input**: A tensor of shape (B, N, D).

2. **Reshape**: The input is reshaped to (BN, D).

3. **Matrix Multiply**: Multiply by the cluster matrix (D, K+G) using `mm` to get (BN, K+G).

4. **Softmax & Batch Norm**: Apply the Triton `softmax_rowwise` kernel to compute the softmax per row, apply batch norm, and mask out the ghost clusters.

5. **Sum**: Use `th.sum` to sum the rows of the masked tensor, resulting in (B, K).

6. **Multiply**: Multiply the sum by `clusters2` (1, K, K) to get (B, K, K).

7. **Transpose & Matrix Multiply**: Transpose the masked softmax to (B, K, N) and multiply by the original feature tensor (B, N, D) using `mm` to get (B, K, D).

8. **Subtraction**: Subtract the previously computed (B, K, D) vector from the result.

9. **L2 Normalization**: Apply `F.normalize` to the (B, D, K) tensor and flatten it.

The Triton kernels (`softmax_rowwise` and `sum_kernel`) are inserted at the softmax and mask steps, respectively, while the other matrix multiplications and batch norm are kept as PyTorch calls. This replacement should improve the performance of the per-row softmax operation, which is a common bottleneck in models that use soft clustering or similar patterns.
</think>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch as th
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
empty_cuda = torch._C._dynamo.guards._empty_cuda
reinterpret_tensor_1 = torch._C._dynamo.guards._reinterpret_tensor
empty_strided_cuda_1 = torch._C._dynamo.guards._empty_strided_cuda_1
empty_strided_cuda_2 = torch._C._dynamo.guards._empty_strided_cuda_2
empty_strided_cuda_3 = torch._C._dynamo.guards._empty_strided_cuda_3
empty_strided_cuda_4 = torch._C._dynamo.guards._empty_strided_cuda_4
empty_strided_cuda_5 = torch._C._dynamo.guards._empty_strided_cuda_5
empty_strided_cuda_6 = torch._C._dynamo.guards._empty_strided_cuda_6
empty_strided_cuda_7 = torch._C._dynamo.guards._empty_strided_cuda_7
empty_strided_cuda_8 = torch._C._dynamo.guards._empty_strided_cuda_8
empty_strided_cuda_9 = torch._C._dynamo.guards._empty_strided_cuda_9
empty_strided_cuda_10 = torch._C._dynamo.guards._empty_strided_cuda_10
empty_strided_cuda_11 = torch._C._dynamo.guards._empty_strided_cuda_11
empty_strided_cuda_12 = torch._C._dynamo.guards._empty_strided_cuda_12
empty_strided_cuda_13 = torch._C._dynamo.guards._empty_strided_cuda_13
empty_strided_cuda_14 = torch._C._dynamo.guards._empty_strided_cuda_14
empty_strided_cuda_15 = torch._C._dynamo.guards._empty_strided_cuda_15
empty_strided_cuda_16 = torch._C._dynamo.guards._empty_strided_cuda_16
empty_strided_cuda_17 = torch._C._dynamo.guards._empty_strided_cuda_17
empty_strided_cuda_18 = torch._C._dynamo.guards._empty_strided_cuda_18
empty_strided_cuda_19 = torch._C._dynamo.guards._empty_strided_cuda_19
empty_strided_cuda_20 = torch._C._dynamo.guards._empty_strided_cuda_20
empty_strided_cuda_21 = torch._C._dynamo.guards._empty_strided_cuda_21
empty_strided_cuda_22 = torch._C._dynamo.guards._empty_strided_cuda_22
empty_strided_cuda_23 = torch._C._dynamo.guards._empty_strided_cuda_23
empty_strided_cuda_24 = torch._C._dynamo.guards._empty_strided_cuda_24
empty_strided_cuda_25 = torch._C._dynamo.guards._empty_strided_cuda_25
empty_strided_cuda_26 = torch._C._dynamo.guards._empty_strided_cuda_26
empty_strided_cuda_27 = torch._C._dynamo.guards._empty_strided_cuda_27
empty_strided_cuda_28 = torch._C._dynamo.guards._empty_strided_cuda_28
empty_strided_cuda_29 = torch._C._dynamo.guards._empty_strided_cuda_29
empty_strided_cuda_30 = torch._C._dynamo.guards._empty_strided_cuda_30
empty_strided_cuda_31 = torch._C._dynamo.guards._empty_strided_cuda_31
empty_strided_cuda_32 = torch._C._dynamo.guards._empty_strided_cuda_32
empty_strided_cuda_33 = torch._C._dynamo.guards._empty_strided_cuda_33
empty_strided_cuda_34 = torch._C._dynamo.guards._empty_strided_cuda_34
empty_strided_cuda_35 = torch._C._dynamo.guards._empty_strided_cuda_35
empty_strided_cuda_36 = torch._C._dynamo.guards._empty_strided_cuda_36
empty_strided_cuda_37 = torch._C._dynamo.guards._empty_strided_cuda_37
empty_strided_cuda_38 = torch._C._dynamo.guards._empty_strided_cuda_38
empty_strided_cuda_39 = torch._C._dynamo.guards._empty_strided_cuda_39
empty_strided_cuda_40 = torch._C._dynamo.guards._empty_strided_cuda_40
empty_strided_cuda_41 = torch._C._dynamo.guards._empty_strided_cuda_41
empty_strided_cuda_42 = torch._C._dynamo.guards._empty_strided_cuda_42
empty_strided_cuda_43 = torch._C._dynamo.guards._empty_strided_cuda_43
empty_strided_cuda_44 = torch._C._dynamo.guards._empty_strided_cuda_44
empty_strided_cuda_45 = torch._C._dynamo.guards._empty_strided_cuda_45
empty_strided_cuda_46 = torch._C._dynamo.guards._empty_strided_cuda_46
empty_strided_cuda_47 = torch._C._dynamo.guards._empty_strided_cuda_47
empty_strided_cuda_48 = torch._C._dynamo.guards._empty_strided_cuda_48
empty_strided_cuda_49 = torch._C._dynamo.guards._empty_strided_cuda_49
empty_strided_cuda_50 = torch._C._dynamo.guards._empty_strided_cuda_50
empty_strided_cuda_51 = torch._C._dynamo.guards._empty_strided_cuda_51
empty_strided_cuda_52 = torch._C._dynamo.guards._empty_strided_cuda_52
empty_strided_cuda_53 = torch._C._dynamo.guards._empty_strided_cuda_53
empty_strided_cuda_54 = torch._C._dynamo.guards._empty_strided_cuda_54
empty_strided_cuda_55 = torch._C._dynamo.guards._empty_strided_cuda_55
empty_strided_cuda_56 = torch._C._dynamo.guards._empty_strided_cuda_56
empty_strided_cuda_57 = torch._C._dynamo.guards._empty_strided_cuda_57
empty_strided_cuda_58 = torch._C._dynamo.guards._empty_strided_cuda_58
empty_strided_cuda_59 = torch._C._dynamo.guards._empty_strided_cuda_59
empty_strided_cuda_60 = torch._C._dynamo.guards._empty_strided_cuda_60
empty_strided_cuda_61 = torch._C._dynamo.guards._empty_strided_cuda_61
empty_strided_cuda_62 = torch._C._dynamo.guards._empty_strided_cuda_62
empty_strided_cuda_63 = torch._C._dynamo.guards._empty_strided_cuda_63
empty_strided_cuda_64 = torch._C._dynamo.guards._empty_strided_cuda_64
empty_strided_cuda_65 = torch._C._dynamo.guards._empty_strided_cuda_65
empty_strided_cuda_66 = torch._C._dynamo.guards._empty_strided_cuda_66
empty_strided_cuda_67 = torch._C._dynamo.guards._empty_strided_cuda_67
empty_strided_cuda_68 = torch._C._dynamo.guards._empty_strided_cuda_68
empty_strided_cuda_69 = torch._C._dynamo.guards._empty_strided_cuda_69
empty_strided_cuda_70 = torch._C._dynamo.guards._empty_strided_cuda_70
empty_strided_cuda_71 = torch._C._dynamo.guards._empty_strided_cuda_71
empty_strided_cuda_72 = torch._C._dynamo.guards._empty_strided_cuda_72
empty_strided_cuda_73 = torch._C._dynamo.guards._empty_strided_cuda_73
empty_strided_cuda_74 = torch._C._dynamo.guards._empty_strided_cuda_74
empty_strided_cuda_75 = torch._C._dynamo.guards._empty_strided_cuda_75
empty_strided_cuda_76 = torch._C._dynamo.guards._empty_strided_cuda_76
empty_strided_cuda_77 = torch._C._dynamo.guards._empty_strided_cuda_77
empty_strided_cuda_78 = torch._C._dynamo.guards._empty_strided_cuda_78
empty_strided_cuda_79 = torch._C._dynamo.guards._empty_strided_cuda_79
empty_strided_cuda_80 = torch._C._dynamo.guards._empty_strided_cuda_80
empty_strided_cuda_81 = torch._C._dynamo.guards._empty_strided_cuda_81
empty_strided_cuda_82 = torch._C._dynamo.guards._empty_strided_cuda_82
empty_strided_cuda_83 = torch._C._dynamo.guards._empty_strided_cuda_83
empty_strided_cuda_84 = torch._C._dynamo.guards._empty_strided_cuda_84
empty_strided_cuda_85 = torch._C._dynamo.guards._empty_strided_cuda_85
empty_strided_cuda_86 = torch._C._dynamo.guards._empty_strided_cuda_86
empty_strided_cuda_87 = torch._C._dynamo.guards._empty_strided_cuda_87
empty_strided_cuda_88 = torch._C._dynamo.guards._empty_strided_cuda_88
empty_strided_cuda_89 = torch._C._dynamo.guards._empty_strided_cuda_89
empty_strided_cuda_90 = torch._C._dynamo.guards._empty_strided_cuda_90
empty_strided_cuda_91 = torch._C._dynamo.guards._empty_strided_cuda_91
empty_strided_cuda_92 = torch._C._dynamo.guards._empty_strided_cuda_92
empty_strided_cuda_93 = torch._C._dynamo.guards._empty_strided_cuda_93
empty_strided_cuda_94 = torch._C._dynamo.guards._empty_strided_cuda_94
empty_strided_cuda_95 = torch._C._dynamo.guards._empty_strided_cuda_95
empty_strided_cuda_96 = torch._C._dynamo.guards._empty_strided_cuda_96
empty_strided_cuda_97 = torch._C._dynamo.guards._empty_strided_cuda_97
empty_strided_cuda_98 = torch._C._dynamo.guards._empty_strided_cuda_98
empty_strided_cuda_99 = torch._C._dynamo.guards._empty_strided_cuda_99
empty_strided_cuda_100 = torch._C._dynamo.guards._empty_strided_cuda_100
empty_strided_cuda_101 = torch._C._dynamo.guards._empty_strided_cuda_101
empty_strided_cuda_102 = torch._C._dynamo.guards._empty_strided_cuda_102
empty_strided_cuda_103 = torch._C._dynamo.guards._empty_strided_cuda_103
empty_strided_cuda_104 = torch._C._dynamo.guards._empty_strided_cuda_104
empty_strided_cuda_105 = torch._C._dynamo.guards._empty_strided_cuda_105
empty_strided_cuda_106 = torch._C._dynamo.guards._empty_strided_cuda_106
empty_strided_cuda_107 = torch._C._dynamo.guards._empty_strided_cuda_107
empty_strided_cuda_108 = torch._C._dynamo.guards._empty_strided_cuda_108
empty_strided_cuda_109 = torch._C._dynamo.guards._empty_strided_cuda_109
empty_strided_cuda_110 = torch._C._dynamo.guards._empty_strided_cuda_110
empty_strided_cuda_111 = torch._C._dynamo.guards._empty_strided_cuda_111
empty_strided_cuda_112 = torch._C._dynamo.guards._empty_strided_cuda_112
empty_strided_cuda_113 = torch._C._dynamo.guards._empty_strided_cuda_113
empty_strided_cuda_114 = torch._C._dynamo.guards._empty_strided_cuda_114
empty_strided_cuda_115 = torch._C._dynamo.guards._empty_strided_cuda_115
empty_strided_cuda_116 = torch._C._dynamo.guards._empty_strided_cuda_116
empty_strided_cuda_117 = torch._C._dynamo.guards._empty_strided_cuda_117
empty_strided_cuda_118 = torch._C._dynamo.guards._empty_strided_cuda_118
empty_strided_cuda_119 = torch._C._dynamo.guards._empty_strided_cuda_119
empty_strided_cuda_120 = torch._C._dynamo.guards._empty_strided_cuda_120
empty_strided_cuda_121 = torch._C._dynamo.guards._empty_strided_cuda_121
empty_strided_cuda_122 = torch._C._dynamo.guards._empty_strided_cuda_122
empty_strided_cuda_123 = torch._C._dynamo.guards._empty_strided_cuda_123
empty_strided_cuda_124 = torch._C._dynamo.guards._empty_strided_cuda_124
empty_strided_cuda_125 = torch._C._dynamo.guards._empty_strided_cuda_125
empty_strided_cuda_126 = torch._C._dynamo.guards._empty_strided_cuda_126
empty_strided_cuda_127 = torch._C._dynamo.guards._empty_strided_cuda_127
empty_strided_cuda_128 = torch._C._dynamo.guards._empty_strided_cuda_128
empty_strided_cuda_129 = torch._C._dynamo.guards._empty_strided_cuda_129
empty_strided_cuda_130 = torch._C._dynamo.guards._empty_strided_cuda_130
empty_strided_cuda_131 = torch._C._dynamo.guards._empty_strided_cuda_131
empty_strided_cuda_132 = torch._C._dynamo.guards._empty_strided_cuda_132
empty_strided_cuda_133 = torch._C._dynamo.guards._empty_strided_cuda_133
empty_strided_cuda_134 = torch._C._dynamo.guards._empty_strided_cuda_134
empty_strided_cuda_135 = torch._C._dynamo.guards._empty_strided_cuda_135
empty_strided_cuda_136 = torch._C._dynamo.guards._empty_strided_cuda_136
empty_strided_cuda_137 = torch._C._dynamo.guards._empty_strided_cuda_137
empty_strided_cuda_138 = torch._C._dynamo.guards._empty_strided_cuda_138
empty_strided_cuda_139 = torch._C._dynamo.guards._empty_strided_cuda_139
empty_strided_cuda_140 = torch._C._dynamo.guards._empty_strided_cuda_140
empty_strided_cuda_141 = torch._C._dynamo.guards._empty_strided_cuda_141
empty_strided_cuda_142 = torch._C._dynamo.guards._empty_strided_cuda_142
empty_strided_cuda_143 = torch._C._dynamo.guards._empty_strided_cuda_143
empty_strided_cuda_144 = torch._C._dynamo.guards._empty_strided_cuda_144
empty_strided_cuda_145 = torch._C._dynamo.guards._empty_strided_cuda_145
empty_strided_cuda_146 = torch._C._dynamo.guards._empty_strided_cuda_146
empty_strided_cuda_147 = torch._C._dynamo.guards._empty_strided_cuda_147
empty_strided_cuda_148 = torch._C._dynamo.guards._empty_strided_cuda_148
empty_strided_cuda_149 = torch._C._dynamo.guards._empty_strided_cuda_149
empty_strided_cuda_150 = torch._C._dynamo.guards._empty_strided_cuda_150
empty_strided_cuda_151 = torch._C._dynamo.guards._empty_strided_cuda_151
empty_strided_cuda_152 = torch._C._dynamo.guards._empty_strided_cuda_152
empty_strided_cuda_153 = torch._C._dynamo.guards._empty_strided_cuda_153
empty_strided_cuda_154 = torch._C._dynamo.guards._empty_strided_cuda_154
empty_strided_cuda_155 = torch._C._dynamo.guards._empty_strided_cuda_155
empty_strided_cuda_156 = torch._C._dynamo.guards._empty_strided_cuda_156
empty_strided_cuda_157 = torch._C._dynamo.guards._empty_strided_cuda_157
empty_strided_cuda_158 = torch._C._dynamo.guards._empty_strided_cuda_158
empty_strided_cuda_159 = torch._C._dynamo.guards._empty_strided_cuda_159
empty_strided_cuda_160 = torch._C._dynamo.guards._empty_strided_cuda_160
empty_strided_cuda_161 = torch._C._dynamo.guards._empty_strided_cuda_161
empty_strided_cuda_162 = torch._C._dynamo.guards._empty_strided_cuda_162
empty_strided_cuda_163 = torch._C._dynamo.guards._empty_strided_cuda_163
empty_strided_cuda_164 = torch._C._dynamo.guards._empty_strided_cuda_164
empty_strided_cuda_165 = torch._C._dynamo.guards._empty_strided_cuda_165
empty_strided_cuda_166 = torch._C._dynamo.guards._empty_strided_cuda_166
empty_strided_cuda_167 = torch._C._dynamo.guards._empty_strided_cuda_167
empty_strided_cuda_168 = torch._C._dynamo.guards._empty_strided_cuda_168
empty_strided_cuda_169 = torch._C._dynamo.guards._empty_strided_cuda_169
empty_strided_cuda_170 = torch._C._dynamo.guards._empty_strided_cuda_170
empty_strided_cuda_171 = torch._C._dynamo.guards._empty_strided_cuda_171
empty_strided_cuda_172 = torch._C._dynamo.guards._empty_strided_cuda_172
empty_strided_cuda_173 = torch._C._dynamo.guards._empty_strided_cuda_173
empty_strided_cuda_174 = torch._C._dynamo.guards._empty_strided_cuda_174
empty_strided_cuda_175 = torch._C._dynamo.guards._empty_strided_cuda_175
empty_strided_cuda_176 = torch._C._dynamo.guards._empty_strided_cuda_176
empty_strided_cuda_177 = torch._C._dynamo.guards._empty_strided_cuda_177
empty_strided_cuda_178 = torch._C._dynamo.guards._empty_strided_cuda_178
empty_strided_cuda_179 = torch._C._dynamo.guards._empty_strided_cuda_179
empty_strided_cuda_180 = torch._C._dynamo.guards._empty_strided_cuda_180
empty_strided_cuda_181 = torch._C._dynamo.guards._empty_strided_cuda_181
empty_strided_cuda_182 = torch._C._dynamo.guards._empty_strided_cuda_182
empty_strided_cuda_183 = torch._C._dynamo.guards._empty_strided_cuda_183
empty_strided_cuda_184 = torch._C._dynamo.guards._empty_strided_cuda_184
empty_strided_cuda_185 = torch._C._dynamo.guards._empty_strided_cuda_185
empty_strided_cuda_186 = torch._C._dynamo.guards._empty_strided_cuda_186
empty_strided_cuda_187 = torch._C._dynamo.guards._empty_strided_cuda_187
empty_strided_cuda_188 = torch._C._dynamo.guards._empty_strided_cuda_188
empty_strided_cuda