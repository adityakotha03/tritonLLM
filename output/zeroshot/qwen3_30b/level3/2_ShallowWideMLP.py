for k in range(0, K, BLOCK_SIZE_K):
    # Load tile of A: (BLOCK_SIZE_M, BLOCK_SIZE_K)
    # Load tile of B: (BLOCK_SIZE_K, BLOCK_SIZE_N)
    # Compute partial result: (BLOCK_SIZE_M, BLOCK_SIZE_N)
    # Accumulate in C