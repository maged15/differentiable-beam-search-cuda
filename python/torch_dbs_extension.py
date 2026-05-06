# SPDX-License-Identifier: MIT
"""Native PyTorch wrapper for DBS.

Public tensor API:
  - unbatched: [T, K, V] -> [K]
  - batched:   [B, T, K, V] -> [B, K]

CPU supports surrogate autograd for both forms. CUDA supports hard forward
decoding and limited selected-path sparse surrogate backward when the optional
CUDA extension is built and beam_size <= 32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

try:
    import dbs_torch_ext as _ext  # noqa: F401 - loads TORCH_LIBRARY(dbs, ...)
except ImportError as exc:  # pragma: no cover
    raise ImportError("dbs_torch_ext is not built. Install with `pip install .` from the package root.") from exc

try:  # Optional CUDA extension built with DBS_BUILD_TORCH_CUDA=1.
    import dbs_torch_cuda_ext as _cuda_ext
except ImportError:  # pragma: no cover
    _cuda_ext = None

_INT_MAX = 2_147_483_647
_CUDA_FAST_MAX_BEAM = 32


@dataclass(frozen=True)
class DBSOptions:
    beam_size: int
    eos_token: int = -1
    selected_temperature: float = 1.0
    soft_topk_temperature: float = 0.25
    relaxed_pool_multiplier: int = 8
    vocab_block: int = 4096
    length_penalty_alpha: float = 0.0
    soft_topk_tolerance: float = 1.0e-4
    soft_topk_max_iters: int = 48
    min_length: int = 0
    validate_inputs: int = 1
    max_dense_gradient_elements: int = 100_000_000

    def args(self) -> Tuple[object, ...]:
        return (
            self.beam_size,
            self.eos_token,
            self.selected_temperature,
            self.soft_topk_temperature,
            self.relaxed_pool_multiplier,
            self.vocab_block,
            self.length_penalty_alpha,
            self.soft_topk_tolerance,
            self.soft_topk_max_iters,
            self.min_length,
            self.validate_inputs,
            self.max_dense_gradient_elements,
        )


def _validate_positive_int_dim(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if value > _INT_MAX:
        raise ValueError(f"{name} exceeds INT_MAX")


def _validate_eos_token(options: DBSOptions, vocab_size: int) -> None:
    if options.eos_token < -1:
        raise ValueError("eos_token must be -1 or non-negative")
    if options.eos_token >= vocab_size:
        raise ValueError(f"eos_token={options.eos_token} must be < vocab_size={vocab_size}")


def _validate_public_shape(log_probs: torch.Tensor, options: DBSOptions) -> bool:
    """Validate public shape and return True if input is unbatched."""
    _validate_positive_int_dim(options.beam_size, "beam_size")
    if log_probs.dim() == 3:
        _validate_positive_int_dim(log_probs.size(0), "T")
        _validate_positive_int_dim(log_probs.size(1), "K")
        _validate_positive_int_dim(log_probs.size(2), "V")
        if log_probs.size(1) != options.beam_size:
            raise ValueError(f"beam_size={options.beam_size} must equal log_probs.shape[1]={log_probs.size(1)}")
        _validate_eos_token(options, log_probs.size(2))
        return True
    if log_probs.dim() == 4:
        _validate_positive_int_dim(log_probs.size(0), "B")
        _validate_positive_int_dim(log_probs.size(1), "T")
        _validate_positive_int_dim(log_probs.size(2), "K")
        _validate_positive_int_dim(log_probs.size(3), "V")
        if log_probs.size(2) != options.beam_size:
            raise ValueError(f"beam_size={options.beam_size} must equal log_probs.shape[2]={log_probs.size(2)}")
        _validate_eos_token(options, log_probs.size(3))
        return False
    raise ValueError("log_probs must have shape [T,K,V] or [B,T,K,V]")




def _validate_cuda_supported_options(options: DBSOptions) -> None:
    """Validate options supported by the CUDA tensor API.

    It supports beam_size/eos_token/min_length plus input validation. Options
    that change other CPU decoding semantics are rejected instead of silently ignored.
    """
    default = DBSOptions(beam_size=options.beam_size, eos_token=options.eos_token)
    unsupported = []
    for name in (
        "selected_temperature",
        "soft_topk_temperature",
        "relaxed_pool_multiplier",
        "vocab_block",
        "length_penalty_alpha",
        "soft_topk_tolerance",
        "soft_topk_max_iters",
    ):
        if getattr(options, name) != getattr(default, name):
            unsupported.append(f"{name}={getattr(options, name)!r}")
    if unsupported:
        raise ValueError(
            "CUDA currently supports only beam_size/eos_token/min_length and default "
            "decoder-shaping options; unsupported CUDA options: " + ", ".join(unsupported)
        )


def _cpu_forward_batched(x: torch.Tensor, options: DBSOptions) -> torch.Tensor:
    if x.dim() == 3:
        return torch.ops.dbs.final_scores_forward(x, *options.args())
    rows = [torch.ops.dbs.final_scores_forward(x[b], *options.args()) for b in range(x.size(0))]
    return torch.stack(rows, dim=0)


def _cpu_backward_batched(x: torch.Tensor, grad_output: torch.Tensor, options: DBSOptions) -> torch.Tensor:
    if x.dim() == 3:
        if grad_output.dim() != 1 or grad_output.size(0) != options.beam_size:
            raise RuntimeError("grad_output for [T,K,V] input must have shape [K]")
        return torch.ops.dbs.final_scores_backward(x, grad_output.contiguous(), *options.args())

    if grad_output.dim() != 2 or grad_output.size(0) != x.size(0) or grad_output.size(1) != options.beam_size:
        raise RuntimeError("grad_output for [B,T,K,V] input must have shape [B,K]")
    rows = [torch.ops.dbs.final_scores_backward(x[b], grad_output[b].contiguous(), *options.args()) for b in range(x.size(0))]
    return torch.stack(rows, dim=0)


class _DBSFinalScores(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs: torch.Tensor, options: DBSOptions):
        input_was_unbatched = _validate_public_shape(log_probs, options)
        ctx.options = options
        ctx.input_dtype = log_probs.dtype
        ctx.was_cuda = log_probs.is_cuda
        ctx.input_was_unbatched = input_was_unbatched

        # Native operators accumulate in fp32. Half/bfloat16 inputs are explicitly promoted.
        x = log_probs.detach().to(dtype=torch.float32).contiguous()

        if log_probs.is_cuda:
            if _cuda_ext is None:
                raise RuntimeError(
                    "CUDA tensor received, but dbs_torch_cuda_ext was not built. "
                    "Reinstall with DBS_BUILD_TORCH_CUDA=1 and run python/tests/test_cuda_parity.py."
                )
            _validate_cuda_supported_options(options)
            # Run the full forward so parents/from_logprob are available for the
            # sparse surrogate backward. Large beams use scores-only CUDA forward
            # because the trace-emitting full kernel is bounded by DBS_CUDA_FAST_MAX_BEAM.
            x4 = x.unsqueeze(0) if input_was_unbatched else x
            if options.beam_size <= _CUDA_FAST_MAX_BEAM:
                _toks, scores, parents, from_logprob = _cuda_ext.decode_forward_full_cuda(
                    x4, options.beam_size, options.eos_token, options.min_length, options.validate_inputs
                )
                ctx.save_for_backward(x4, _toks, parents, from_logprob, scores)
                ctx.vocab_size = x4.size(-1)
            else:
                ctx.save_for_backward(x4)
                ctx.vocab_size = -1
                scores = _cuda_ext.final_scores_forward_cuda(
                    x4, options.beam_size, options.eos_token, options.min_length, options.validate_inputs
                )
            return scores.squeeze(0) if input_was_unbatched else scores

        ctx.save_for_backward(x)
        ctx.vocab_size = -1
        return _cpu_forward_batched(x, options)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        options: DBSOptions = ctx.options

        if ctx.was_cuda:
            if _cuda_ext is None or ctx.vocab_size < 0:
                raise RuntimeError(
                    "CUDA autograd backward requires dbs_torch_cuda_ext built with "
                    "DBS_BUILD_TORCH_CUDA=1 and beam_size <= DBS_CUDA_FAST_MAX_BEAM."
                )
            # saved = (x4, tokens, parents, from_logprob, final_scores)
            x4, _toks, parents, from_logprob, final_scores = ctx.saved_tensors
            B, T, K, V = x4.size(0), x4.size(1), x4.size(2), ctx.vocab_size

            g = grad_output.contiguous().to(dtype=torch.float32)
            if g.dim() == 1:
                g = g.unsqueeze(0)  # [1, K] for unbatched

            out_idx, out_val = _cuda_ext.backward_build_sparse_cuda(
                _toks, parents, from_logprob, final_scores, g, V
            )
            flat_size = B * T * K * V
            grad_flat = _cuda_ext._test_sparse_backward_scatter(
                out_idx.reshape(-1), out_val.reshape(-1), flat_size
            )
            grad = grad_flat.view(B, T, K, V)
            if ctx.input_was_unbatched:
                grad = grad.squeeze(0)
            return grad.to(dtype=ctx.input_dtype), None

        (log_probs_f32,) = ctx.saved_tensors
        g = grad_output.contiguous().to(dtype=torch.float32)
        grad = _cpu_backward_batched(log_probs_f32, g, options)
        return grad.to(dtype=ctx.input_dtype), None


def final_scores(log_probs: torch.Tensor, options: DBSOptions) -> torch.Tensor:
    """Return final beam scores for [T,K,V] or [B,T,K,V] log-prob tensors.

    CPU inputs support surrogate autograd. CUDA inputs support hard forward and
    limited selected-path sparse surrogate backward for beam_size <= 32.
    """
    return _DBSFinalScores.apply(log_probs, options)


def decode(log_probs: torch.Tensor, options: DBSOptions) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return CUDA hard-decode token traces and final scores.

    This debugging API keeps ``final_scores()`` unchanged. CUDA inputs return
    ``(tokens, scores)`` with shapes ``[T,K], [K]`` or ``[B,T,K], [B,K]``.
    CPU token-trace exposure remains available through the C ABI.
    """
    input_was_unbatched = _validate_public_shape(log_probs, options)
    if not log_probs.is_cuda:
        raise RuntimeError("decode() currently exposes CUDA token traces only; use final_scores() or the C ABI for CPU")
    if _cuda_ext is None:
        raise RuntimeError(
            "CUDA tensor received, but dbs_torch_cuda_ext was not built. "
            "Reinstall with DBS_BUILD_TORCH_CUDA=1 and run python/tests/test_cuda_parity.py."
        )
    _validate_cuda_supported_options(options)
    x = log_probs.detach().to(dtype=torch.float32).contiguous()
    if input_was_unbatched:
        tokens, scores = _cuda_ext.decode_forward_cuda(x.unsqueeze(0), options.beam_size, options.eos_token, options.min_length, options.validate_inputs)
        return tokens.squeeze(0), scores.squeeze(0)
    return _cuda_ext.decode_forward_cuda(x, options.beam_size, options.eos_token, options.min_length, options.validate_inputs)
