#!/usr/bin/env python3
import torch
from torch_dbs_extension import DBSOptions, final_scores


def main():
    torch.manual_seed(0)
    B, T, K, V = 2, 4, 3, 16
    x = torch.randn(B, T, K, V)
    x = torch.log_softmax(x, dim=-1)
    y = final_scores(x, DBSOptions(beam_size=K, eos_token=-1))
    print("cpu final scores", y)


if __name__ == "__main__":
    main()
