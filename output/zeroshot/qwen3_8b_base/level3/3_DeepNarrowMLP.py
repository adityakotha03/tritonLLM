import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_relu_kernel(
    a_ptr,  # Pointer to first input (matrix A)
    b_ptr,  # Pointer to second input (matrix B)
    out_ptr,  # Pointer to output
    m,  # Number of rows in A
    n,  # Number of columns in B
    k,  # Number of columns in A / rows in B
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    num_blocks = (m + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_row = pid % num_blocks
    block_col = pid // num_blocks

    # Compute the block's row and column indices
    row_start = block_row * BLOCK_SIZE
    col_start = block_col * BLOCK_SIZE

    # Compute the block's row and column offsets
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)

    # Load matrix A and B
    a = tl.load(a_ptr + row_offsets[:, None] * k + col_offsets[None, :], mask=(row_offsets < m)[:, None] & (col_offsets < k)[None, :], other=0.0)
    b = tl.load(b_ptr + row_offsets[:, None] * n + col_offsets[None, :], mask=(row_offsets < m)[:, None] & (col_offsets < n)[None, :], other=0.0)

    # Compute the matrix multiplication
    c = tl.dot(a, b)

    # Apply ReLU
    c = tl.maximum(c, 0.0)

    # Store the result
    tl.store(out_ptr + row_offsets[:, None] * n + col_offsets[None, :], c, mask=(row_offsets < m)[:, None] & (col_offsets < n)[None, :])


def triton_matmul_relu(a: torch.Tensor, b: torch.Tensor, m, n, k):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    # Prepare output tensor
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)

    # Determine the number of blocks needed
    BLOCK_SIZE = 128  # Tunable parameter for block size
    num_blocks = (m + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    grid = (num_blocks,)
    matmul_relu_kernel[grid](a, b, out, m, n, k, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes

        # Replace the linear + ReLU operations with custom Triton kernels
        self.layers = []
        current_input_size = input_size

        for hidden_size in hidden_layer_sizes:
            self.layers.append(self._create_linear_relu(hidden_size, current_input_size))
            current_input_size = hidden_size

        self.layers.append(nn.Linear(current_input_size, output_size))

        self.network = nn.Sequential(*self.layers)

    def _create_linear_relu(self, out_features, in_features):
        """
        Creates a linear layer followed by a ReLU, implemented with a custom Triton kernel.
        """
        def forward(x):
            # Ensure x is contiguous
            x = x.contiguous()
            # Create output tensor
            out = torch.empty_like(x)
            # Launch the Triton kernel
            triton_matmul_relu(x, self.weight, x.size(0), out_features, in_features)
            return out

        return forward

    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        for layer in self.layers:
            x = layer(x)
        return x