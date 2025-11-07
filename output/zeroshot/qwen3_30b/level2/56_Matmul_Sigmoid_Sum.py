W_tile = tl.load(
    W_ptr + k * hidden_size + j + tl.arange(0, BLOCK_SIZE_K)[:, None] * hidden_size + tl.arange(0, BLOCK_SIZE_J)[None, :],
    mask=tl.arange(0, BLOCK_SIZE_K)[:, None] < (input_size - k)  # This is not correct
)