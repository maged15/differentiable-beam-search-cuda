# SPDX-License-Identifier: MIT
"""Native PyTorch wrapper for DBS.

Public tensor API:
  - unbatched: [T, K, V] -> [K]
  - batched:   [B, T, K, V] -> [B, K]

CPU supports surrogate autograd for both forms. CUDA tensors use native CUDA
forward when the requested options are supported, and otherwise fall back to the
CPU semantic implementation while returning CUDA outputs/gradients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
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
_NATIVE_CUDA_FAST_MAX_BEAM = 32
_NATIVE_CUDA_OPTION_NAMES = frozenset({
    "beam_size",
    "eos_token",
    "min_length",
    "validate_inputs",
})


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


def _validate_finite_float(value: float, name: str) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _validate_public_options(options: DBSOptions) -> None:
    _validate_finite_float(options.selected_temperature, "selected_temperature")
    if not options.selected_temperature > 0.0:
        raise ValueError("selected_temperature must be positive")
    _validate_finite_float(options.soft_topk_temperature, "soft_topk_temperature")
    if not options.soft_topk_temperature > 0.0:
        raise ValueError("soft_topk_temperature must be positive")
    if options.relaxed_pool_multiplier <= 0:
        raise ValueError("relaxed_pool_multiplier must be positive")
    if options.vocab_block <= 0:
        raise ValueError("vocab_block must be positive")
    _validate_finite_float(options.length_penalty_alpha, "length_penalty_alpha")
    if options.length_penalty_alpha < 0.0:
        raise ValueError("length_penalty_alpha cannot be negative")
    _validate_finite_float(options.soft_topk_tolerance, "soft_topk_tolerance")
    if not options.soft_topk_tolerance > 0.0:
        raise ValueError("soft_topk_tolerance must be positive")
    if options.soft_topk_max_iters <= 0:
        raise ValueError("soft_topk_max_iters must be positive")
    if options.min_length < 0:
        raise ValueError("min_length cannot be negative")
    if options.validate_inputs not in (0, 1):
        raise ValueError("validate_inputs must be 0 or 1")
    if options.max_dense_gradient_elements <= 0:
        raise ValueError("max_dense_gradient_elements must be positive")


def _validate_eos_token(options: DBSOptions, vocab_size: int) -> None:
    if options.eos_token < -1:
        raise ValueError("eos_token must be -1 or non-negative")
    if options.eos_token >= vocab_size:
        raise ValueError(f"eos_token={options.eos_token} must be < vocab_size={vocab_size}")


def _validate_public_shape(log_probs: torch.Tensor, options: DBSOptions) -> bool:
    """Validate public shape and return True if input is unbatched."""
    _validate_positive_int_dim(options.beam_size, "beam_size")
    _validate_public_options(options)
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


def _native_cuda_reference_options(options: DBSOptions) -> DBSOptions:
    return DBSOptions(
        beam_size=options.beam_size,
        eos_token=options.eos_token,
        min_length=options.min_length,
        validate_inputs=options.validate_inputs,
    )


def _option_value_matches(value: object, default: object) -> bool:
    if isinstance(value, float) or isinstance(default, float):
        return math.isclose(float(value), float(default), rel_tol=0.0, abs_tol=0.0)
    return value == default


def _native_cuda_unsupported_options(options: DBSOptions) -> list[str]:
    default = _native_cuda_reference_options(options)
    unsupported = []
    for field in fields(DBSOptions):
        name = field.name
        if name in _NATIVE_CUDA_OPTION_NAMES:
            continue
        if not _option_value_matches(getattr(options, name), getattr(default, name)):
            unsupported.append(f"{name}={getattr(options, name)!r}")
    return unsupported


def _native_cuda_forward_supported(options: DBSOptions) -> bool:
    """Return True when the native CUDA decoder is semantically equivalent.

    The CUDA C kernels implement hard final-score decoding with EOS/min-length
    support. Options that are validated or interpreted only by the CPU decoder
    deliberately use the CPU semantic fallback for the public tensor API. K > 32
    also falls back so public CUDA tensors do not hit the serial direct-C kernel.
    """
    return (
        _cuda_ext is not None
        and options.beam_size <= _NATIVE_CUDA_FAST_MAX_BEAM
        and not _native_cuda_unsupported_options(options)
    )


def _native_cuda_forward(x4: torch.Tensor, options: DBSOptions) -> torch.Tensor:
    if options.beam_size > _NATIVE_CUDA_FAST_MAX_BEAM:
        raise RuntimeError(
            "internal error: beam_size reached native CUDA fast path above "
            f"{_NATIVE_CUDA_FAST_MAX_BEAM}"
        )
    unsupported = _native_cuda_unsupported_options(options)
    if unsupported:
        raise RuntimeError(
            "internal error: unsupported options reached native CUDA forward: "
            + ", ".join(unsupported)
        )
    if _cuda_ext is None:
        raise RuntimeError("dbs_torch_cuda_ext is not built")
    return _cuda_ext.final_scores_forward_cuda(
        x4, options.beam_size, options.eos_token, options.min_length, options.validate_inputs
    )


def _validate_native_cuda_decode_options(options: DBSOptions) -> None:
    """Fail closed for decode(), which exposes native CUDA token traces."""
    unsupported = _native_cuda_unsupported_options(options)
    if unsupported:
        raise ValueError(
            "CUDA decode() exposes native token traces only for beam_size/eos_token/min_length "
            "and default decoder-shaping options; unsupported CUDA options: " + ", ".join(unsupported)
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
            x4 = x.unsqueeze(0) if input_was_unbatched else x
            x4_cpu = x4.detach().cpu().contiguous()
            ctx.save_for_backward(x4_cpu)

            if _native_cuda_forward_supported(options):
                scores = _native_cuda_forward(x4, options)
            else:
                scores = _cpu_forward_batched(x4_cpu, options).to(device=log_probs.device)
            return scores.squeeze(0) if input_was_unbatched else scores

        ctx.save_for_backward(x)
        return _cpu_forward_batched(x, options)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        options: DBSOptions = ctx.options

        if ctx.was_cuda:
            (x4_cpu,) = ctx.saved_tensors
            g = grad_output.detach().cpu().contiguous().to(dtype=torch.float32)
            if g.dim() == 1:
                g = g.unsqueeze(0)  # [1, K] for unbatched

            grad = _cpu_backward_batched(x4_cpu, g, options)
            if ctx.input_was_unbatched:
                grad = grad.squeeze(0)
            return grad.to(device=grad_output.device, dtype=ctx.input_dtype), None

        (log_probs_f32,) = ctx.saved_tensors
        g = grad_output.contiguous().to(dtype=torch.float32)
        grad = _cpu_backward_batched(log_probs_f32, g, options)
        return grad.to(dtype=ctx.input_dtype), None


def final_scores(log_probs: torch.Tensor, options: DBSOptions) -> torch.Tensor:
    """Return final beam scores for [T,K,V] or [B,T,K,V] log-prob tensors.

    CPU inputs support surrogate autograd. CUDA inputs return CUDA tensors and
    gradients with CPU-equivalent semantics; unsupported native CUDA options use
    a CPU semantic fallback internally.
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
    _validate_native_cuda_decode_options(options)
    x = log_probs.detach().to(dtype=torch.float32).contiguous()
    if input_was_unbatched:
        tokens, scores = _cuda_ext.decode_forward_cuda(x.unsqueeze(0), options.beam_size, options.eos_token, options.min_length, options.validate_inputs)
        return tokens.squeeze(0), scores.squeeze(0)
    return _cuda_ext.decode_forward_cuda(x, options.beam_size, options.eos_token, options.min_length, options.validate_inputs)
