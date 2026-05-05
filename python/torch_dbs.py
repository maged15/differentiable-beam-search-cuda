# SPDX-License-Identifier: MIT
"""PyTorch CPU integration for libdbs.

The forward pass returns final beam scores. The backward pass calls libdbs' sparse C
backward path and scatters sparse entries into a PyTorch dense gradient tensor for
compatibility with standard autograd optimizers.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Optional

import torch


class DBSOptionsC(ctypes.Structure):
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
        ("max_dense_gradient_elements", ctypes.c_int),
        ("reserved0", ctypes.c_int),
        ("reserved1", ctypes.c_int),
    ]


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
            self.max_dense_gradient_elements,
            0,
            0,
        )


class _DBSLib:
    def __init__(self, path: Optional[str] = None) -> None:
        if path is None:
            path = os.environ.get("DBS_LIBRARY", "libdbs.so")
        self.lib = ctypes.CDLL(path)
        self._bind()

    def _bind(self) -> None:
        lib = self.lib
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


class _CState:
    def __init__(self, dbs: _DBSLib, handle: int, result: int) -> None:
        self.dbs = dbs
        self.handle = handle
        self.result = result
        self.closed = False

    def close(self) -> None:
        if not self.closed:
            if self.result:
                self.dbs.lib.dbs_free_result(self.result)
            if self.handle:
                self.dbs.lib.dbs_destroy(self.handle)
            self.closed = True

    def __del__(self) -> None:
        self.close()


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
        dbs = _DBSLib(lib_path)
        handle = dbs.lib.dbs_create(options.as_c())
        if not handle:
            raise RuntimeError("dbs_create failed")

        result = ctypes.c_void_p()
        ptr = x.data_ptr()
        code = dbs.lib.dbs_decode(handle, ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)), T, V, ctypes.byref(result))
        dbs.check(handle, code)

        final_ptr = dbs.lib.dbs_result_final_scores(result)
        out = torch.empty((options.beam_size,), dtype=torch.float32)
        for i in range(options.beam_size):
            out[i] = final_ptr[i]

        ctx.state = _CState(dbs, handle, result.value)
        ctx.shape = tuple(x.shape)
        ctx.options = options
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        state: _CState = ctx.state
        dbs = state.dbs
        grad_output = grad_output.detach().contiguous().to(dtype=torch.float32, device="cpu")
        grad_final = ctypes.cast(grad_output.data_ptr(), ctypes.POINTER(ctypes.c_float))
        backward = ctypes.c_void_p()
        code = dbs.lib.dbs_backward(state.handle, state.result, None, None, grad_final, ctypes.byref(backward))
        dbs.check(state.handle, code)

        T, K, V = ctx.shape
        grad = torch.zeros((T * K * V,), dtype=torch.float32)
        grad_numel = grad.numel()
        n = dbs.lib.dbs_backward_sparse_logprob_count(backward)
        idx = dbs.lib.dbs_backward_sparse_logprob_indices(backward)
        val = dbs.lib.dbs_backward_sparse_logprob_values(backward)
        try:
            for i in range(n):
                j = int(idx[i])
                if j < 0 or j >= grad_numel:
                    raise RuntimeError("sparse gradient index out of bounds")
                grad[j] += float(val[i])
        finally:
            dbs.lib.dbs_free_backward(backward)
            state.close()
        return grad.reshape((T, K, V)), None, None


def final_scores(log_probs: torch.Tensor, options: DBSOptions, lib_path: Optional[str] = None) -> torch.Tensor:
    """Return final beam scores for a [T, K, V] CPU float32 log-prob tensor. Backward uses the sparse/default C ABI path."""
    return _DBSFinalScores.apply(log_probs, options, lib_path)
