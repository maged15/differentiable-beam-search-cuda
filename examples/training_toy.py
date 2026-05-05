"""End-to-end surrogate training example.

Run after building/installing the native extension:

    LD_LIBRARY_PATH=$PWD/build PYTHONPATH=$PWD/python python examples/training_toy.py
"""

from __future__ import annotations

import torch
from torch import nn

from torch_dbs_extension import DBSOptions, final_scores


class TinyStepModel(nn.Module):
    def __init__(self, hidden: int, beam_size: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden, beam_size * vocab_size)
        self.beam_size = beam_size
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        # Produces [T, K, V] logits/log-probs for the decoder.
        rows = []
        h = x
        for _ in range(steps):
            logits = self.proj(h).view(self.beam_size, self.vocab_size)
            rows.append(torch.log_softmax(logits, dim=-1))
            h = torch.tanh(h + 0.01)
        return torch.stack(rows, dim=0)


def main() -> None:
    torch.manual_seed(7)
    steps, beam_size, vocab_size, hidden = 4, 3, 32, 16
    model = TinyStepModel(hidden, beam_size, vocab_size)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    decoder_options = DBSOptions(beam_size=beam_size, validate_inputs=0)

    x = torch.randn(hidden)
    for update in range(5):
        opt.zero_grad(set_to_none=True)
        log_probs = model(x, steps)
        scores = final_scores(log_probs, decoder_options)
        # Toy objective: increase score of the best returned beam under the surrogate backward path.
        loss = -scores[0]
        loss.backward()
        opt.step()
        print(f"step={update} loss={loss.item():.4f} grad_norm={model.proj.weight.grad.norm().item():.4f}")


if __name__ == "__main__":
    main()
