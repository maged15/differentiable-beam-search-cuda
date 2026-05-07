# SPDX-License-Identifier: MIT
"""PyTorch CPU integration for libdbs.

The forward pass returns final beam scores. The backward pass calls libdbs' sparse C
backward path and scatters sparse entries into a PyTorch dense gradient tensor for
compatibility with standard autograd optimizers.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
import os
from dataclasses import dataclass
from typing import Optional
import weakref

import torch

_DBS_ABI_VERSION = 10
_DBS_OPTIONS_C_SIZE = 64
_DBS_OPTIONS_C_OFFSETS = {
    "beam_size": 0,
    "eos_token": 4,
    "selected_temperature": 8,
    "soft_topk_temperature": 12,
    "relaxed_pool_multiplier": 16,
    "vocab_block": 20,
    "length_penalty_alpha": 24,
    "soft_topk_tolerance": 28,
    "soft_topk_max_iters": 32,
    "min_length": 36,
    "validate_inputs": 40,
    "_pad0": 44,
    "max_dense_gradient_elements": 48,
    "reserved0": 56,
    "reserved1": 60,
}


class DBSOptionsC(ctypes.Structure):
    # Mirrors DBSOptionsC from include/dbs.h exactly.
    # max_dense_gradient_elements is int64_t; _pad0 makes the C compiler's
    # natural 4-byte alignment padding explicit so offset checks can fail fast.
    _fields_ = [
        ("beam_size", ctypes.c_int),
        ("eos_token", ctypes.c_int),
        ("selected_temperature", ctypes.c_float),
        ("soft_topk_temperature", ctypes.c_float),
        ("relaxed_pool_multiplier", ctypes.c_int),
        ("vocab_block", ctypes.c_int),
        ("length_penalty_alpha", ctypes.c_float),
        ("soft_topk_tolerance", ctypes.c_float),
        ("soft_topk_max_iters", ctypes.c_int),
        ("min_length", ctypes.c_int),
        ("validate_inputs", ctypes.c_int),
        ("_pad0", ctypes.c_int),
        ("max_dense_gradient_elements", ctypes.c_longlong),
        ("reserved0", ctypes.c_int),
        ("reserved1", ctypes.c_int),
    ]


def _assert_options_c_layout() -> None:
    actual_size = ctypes.sizeof(DBSOptionsC)
    if actual_size != _DBS_OPTIONS_C_SIZE:
        raise RuntimeError(f"DBSOptionsC ctypes size {actual_size} != ABI {_DBS_OPTIONS_C_SIZE}")
    for name, expected_offset in _DBS_OPTIONS_C_OFFSETS.items():
        actual_offset = getattr(DBSOptionsC, name).offset
        if actual_offset != expected_offset:
            raise RuntimeError(
                f"DBSOptionsC.{name} offset {actual_offset} != ABI {expected_offset}"
            )


_assert_options_c_layout()


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

    def as_c(self) -> DBSOptionsC:
        return DBSOptionsC(
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
            0,  # _pad0: implicit C alignment padding
            self.max_dense_gradient_elements,
            0,
            0,
        )


class _DBSLib:
    def __init__(self, path: Optional[str] = None) -> None:
        if path is None:
            raise ValueError("path must be resolved before loading libdbs")
        try:
            self.lib = ctypes.CDLL(path)
        except OSError as exc:
            raise _library_load_error(path, exc) from exc
        self._bind()
        abi = self.lib.dbs_abi_version()
        if abi != _DBS_ABI_VERSION:
            raise RuntimeError(f"libdbs ABI {abi} != expected {_DBS_ABI_VERSION}")

    def _bind(self) -> None:
        lib = self.lib
        lib.dbs_abi_version.argtypes = []
        lib.dbs_abi_version.restype = ctypes.c_int
        lib.dbs_create.argtypes = [DBSOptionsC]
        lib.dbs_create.restype = ctypes.c_void_p
        lib.dbs_destroy.argtypes = [ctypes.c_void_p]
        lib.dbs_last_error.argtypes = [ctypes.c_void_p]
        lib.dbs_last_error.restype = ctypes.c_char_p
        lib.dbs_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.dbs_decode.restype = ctypes.c_int
        lib.dbs_backward.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.dbs_backward.restype = ctypes.c_int
        lib.dbs_free_result.argtypes = [ctypes.c_void_p]
        lib.dbs_free_backward.argtypes = [ctypes.c_void_p]
        lib.dbs_result_final_scores.argtypes = [ctypes.c_void_p]
        lib.dbs_result_final_scores.restype = ctypes.POINTER(ctypes.c_float)
        lib.dbs_backward_sparse_logprob_count.argtypes = [ctypes.c_void_p]
        lib.dbs_backward_sparse_logprob_count.restype = ctypes.c_longlong
        lib.dbs_backward_sparse_logprob_indices.argtypes = [ctypes.c_void_p]
        lib.dbs_backward_sparse_logprob_indices.restype = ctypes.POINTER(ctypes.c_longlong)
        lib.dbs_backward_sparse_logprob_values.argtypes = [ctypes.c_void_p]
        lib.dbs_backward_sparse_logprob_values.restype = ctypes.POINTER(ctypes.c_float)

    def check(self, handle: int, code: int) -> None:
        if code == 0:
            return
        msg = self.lib.dbs_last_error(handle)
        raise RuntimeError((msg or b"libdbs call failed").decode("utf-8", errors="replace"))


def _resolve_library_path(path: Optional[str]) -> str:
    return path if path is not None else os.environ.get("DBS_LIBRARY", "libdbs.so")


def _library_load_error(path: str, exc: OSError) -> RuntimeError:
    return RuntimeError(
        f"Unable to load libdbs from {path!r}: {exc}. Build libdbs first, then set "
        "DBS_LIBRARY to the full shared-library path or add the build directory to "
        "LD_LIBRARY_PATH, DYLD_LIBRARY_PATH, or PATH."
    )


@lru_cache(maxsize=None)
def _get_dbs_lib(path: str) -> _DBSLib:
    return _DBSLib(path)


def _release_c_state(dbs: _DBSLib, handle: int, result: int) -> None:
    if result:
        dbs.lib.dbs_free_result(result)
    if handle:
        dbs.lib.dbs_destroy(handle)


class _CState:
    def __init__(self, dbs: _DBSLib, handle: int, result: int) -> None:
        self.dbs = dbs
        self.handle = handle
        self.result = result
        self._finalizer = weakref.finalize(self, _release_c_state, dbs, handle, result)

    def close(self) -> None:
        self._finalizer()

    @property
    def closed(self) -> bool:
        return not self._finalizer.alive


class _DBSFinalScores(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs: torch.Tensor, options: DBSOptions, lib_path: Optional[str]):
        if log_probs.device.type != "cpu":
            raise ValueError("torch_dbs currently supports CPU tensors; use the C ABI for GPU-side integration")
        if log_probs.dtype != torch.float32:
            raise ValueError("log_probs must be torch.float32")
        if log_probs.ndim != 3:
            raise ValueError("log_probs must have shape [T, K, V]")
        if log_probs.shape[1] != options.beam_size:
            raise ValueError("log_probs.shape[1] must equal options.beam_size")

        x = log_probs.contiguous()
        T, _, V = x.shape
        if options.eos_token < -1:
            raise ValueError("eos_token must be -1 or non-negative")
        if options.eos_token >= V:
            raise ValueError("eos_token must be less than log_probs.shape[2]")
        dbs = _get_dbs_lib(_resolve_library_path(lib_path))
        handle = dbs.lib.dbs_create(options.as_c())
        if not handle:
            raise RuntimeError("dbs_create failed")

        result = ctypes.c_void_p()
        try:
            ptr = x.data_ptr()
            code = dbs.lib.dbs_decode(handle, ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)), T, V, ctypes.byref(result))
            dbs.check(handle, code)

            final_ptr = dbs.lib.dbs_result_final_scores(result)
            if not final_ptr:
                raise RuntimeError("dbs_result_final_scores returned null")
            final_buf = (ctypes.c_float * options.beam_size).from_address(ctypes.addressof(final_ptr.contents))
            out = torch.frombuffer(final_buf, dtype=torch.float32, count=options.beam_size).clone()

            state = _CState(dbs, handle, result.value)
            handle = 0
            result = ctypes.c_void_p()
            ctx.state = state
            ctx.shape = tuple(x.shape)
            ctx.options = options
            return out
        finally:
            if result.value:
                dbs.lib.dbs_free_result(result)
            if handle:
                dbs.lib.dbs_destroy(handle)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        state: _CState = ctx.state
        dbs = state.dbs
        grad_output = grad_output.detach().contiguous().to(dtype=torch.float32, device="cpu")
        grad_final = ctypes.cast(grad_output.data_ptr(), ctypes.POINTER(ctypes.c_float))
        backward = ctypes.c_void_p()
        try:
            code = dbs.lib.dbs_backward(state.handle, state.result, None, None, grad_final, ctypes.byref(backward))
            dbs.check(state.handle, code)

            T, K, V = ctx.shape
            grad = torch.zeros((T * K * V,), dtype=torch.float32)
            grad_numel = grad.numel()
            n = dbs.lib.dbs_backward_sparse_logprob_count(backward)
            if n < 0:
                raise RuntimeError("sparse gradient count must be non-negative")
            idx = dbs.lib.dbs_backward_sparse_logprob_indices(backward)
            val = dbs.lib.dbs_backward_sparse_logprob_values(backward)
            if n and (not idx or not val):
                raise RuntimeError("sparse gradient buffers are null")
            if n:
                idx_buf = (ctypes.c_longlong * n).from_address(ctypes.addressof(idx.contents))
                val_buf = (ctypes.c_float * n).from_address(ctypes.addressof(val.contents))
                idx_tensor = torch.frombuffer(idx_buf, dtype=torch.int64, count=n)
                val_tensor = torch.frombuffer(val_buf, dtype=torch.float32, count=n)
                invalid = (idx_tensor < 0) | (idx_tensor >= grad_numel)
                if torch.any(invalid).item():
                    raise RuntimeError("sparse gradient index out of bounds")
                grad.scatter_add_(0, idx_tensor, val_tensor)
            return grad.reshape((T, K, V)), None, None
        finally:
            if backward.value:
                dbs.lib.dbs_free_backward(backward)
            state.close()


def final_scores(log_probs: torch.Tensor, options: DBSOptions, lib_path: Optional[str] = None) -> torch.Tensor:
    """Return final beam scores for a [T, K, V] CPU float32 log-prob tensor. Backward uses the sparse/default C ABI path."""
    return _DBSFinalScores.apply(log_probs, options, lib_path)
