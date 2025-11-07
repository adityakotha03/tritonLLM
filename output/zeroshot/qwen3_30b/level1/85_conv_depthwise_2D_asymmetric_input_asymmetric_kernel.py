block_idx = tl.program_id(0)  # 1D index over spatial dimensions
channel = tl.program_id(1)    # channel index
h_out = block_idx // W_out
w_out = block_idx % W_out