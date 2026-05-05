#!/usr/bin/env python3
import os
import torch
from torch_dbs_extension import DBSOptions, final_scores


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    # The optimized score-only path is explicit opt-in. Omit this line to use
    # the exact custom CUDA kernel.
    os.environ.setdefault("DBS_ENABLE_SCORE_ONLY_FAST_PATH", "1")
    torch.manual_seed(0)
    B, T, K, V = 2, 8, 4, 32000
    x = torch.randn(B, T, K, V, device="cuda")
    x = torch.log_softmax(x, dim=-1)
    y = final_scores(x, DBSOptions(beam_size=K, eos_token=-1))
    print("cuda final scores shape", tuple(y.shape))
    print(y)


if __name__ == "__main__":
    main()
