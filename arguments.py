class Args:
    # model_args;
    restore_ckpt = "checkpoints/100000_raft-stereo.pth"
    dataset="scared"
    mixed_precision = False
    valid_iters=32
    
    hidden_dims = [128]*3
    corr_implementation = "reg"
    corr_levels = 4
    corr_radius = 4
    n_downsample = 3
    n_gru_layers = 3
    shared_backbone = False
    slow_fast_gru = False
    

