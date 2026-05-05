# SPDX-License-Identifier: MIT
"""JAX custom VJP wrapper for libdbs.

This is a CPU host-callback integration. It is useful for validation and research flows;
production JAX/XLA deployment should replace it with a registered XLA custom call.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


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


class _Lib:
    def __init__(self, path: Optional[str] = None) -> None:
        self.lib = ctypes.CDLL(path or os.environ.get("DBS_LIBRARY", "libdbs.so"))
        self.lib.dbs_create.argtypes = [DBSOptionsC]
        self.lib.dbs_create.restype = ctypes.c_void_p
        self.lib.dbs_destroy.argtypes = [ctypes.c_void_p]
        self.lib.dbs_decode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.dbs_decode.restype = ctypes.c_int
        self.lib.dbs_backward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_void_p)]
        self.lib.dbs_backward.restype = ctypes.c_int
        self.lib.dbs_free_result.argtypes = [ctypes.c_void_p]
        self.lib.dbs_free_backward.argtypes = [ctypes.c_void_p]
        self.lib.dbs_result_final_scores.argtypes = [ctypes.c_void_p]
        self.lib.dbs_result_final_scores.restype = ctypes.POINTER(ctypes.c_float)
        self.lib.dbs_backward_sparse_logprob_count.argtypes = [ctypes.c_void_p]
        self.lib.dbs_backward_sparse_logprob_count.restype = ctypes.c_longlong
        self.lib.dbs_backward_sparse_logprob_indices.argtypes = [ctypes.c_void_p]
        self.lib.dbs_backward_sparse_logprob_indices.restype = ctypes.POINTER(ctypes.c_longlong)
        self.lib.dbs_backward_sparse_logprob_values.argtypes = [ctypes.c_void_p]
        self.lib.dbs_backward_sparse_logprob_values.restype = ctypes.POINTER(ctypes.c_float)
        self.lib.dbs_last_error.argtypes = [ctypes.c_void_p]
        self.lib.dbs_last_error.restype = ctypes.c_char_p

    def check(self, h: int, rc: int) -> None:
        if rc == 0:
            return
        msg = self.lib.dbs_last_error(h)
        raise RuntimeError((msg or b"libdbs call failed").decode("utf-8", errors="replace"))


def _forward_np(x: np.ndarray, options: DBSOptions, lib_path: Optional[str]) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    t, k, v = x.shape
    if k != options.beam_size:
        raise ValueError("shape[1] must equal options.beam_size")
    dbs = _Lib(lib_path)
    h = dbs.lib.dbs_create(options.as_c())
    if not h:
        raise RuntimeError("dbs_create failed")
    r = ctypes.c_void_p()
    try:
        dbs.check(h, dbs.lib.dbs_decode(h, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), t, v, ctypes.byref(r)))
        ptr = dbs.lib.dbs_result_final_scores(r)
        return np.array([ptr[i] for i in range(k)], dtype=np.float32)
    finally:
        if r.value:
            dbs.lib.dbs_free_result(r)
        dbs.lib.dbs_destroy(h)


def _backward_np(x: np.ndarray, g: np.ndarray, options: DBSOptions, lib_path: Optional[str]) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    g = np.ascontiguousarray(g, dtype=np.float32)
    t, k, v = x.shape
    dbs = _Lib(lib_path)
    h = dbs.lib.dbs_create(options.as_c())
    if not h:
        raise RuntimeError("dbs_create failed")
    r = ctypes.c_void_p()
    b = ctypes.c_void_p()
    try:
        dbs.check(h, dbs.lib.dbs_decode(h, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), t, v, ctypes.byref(r)))
        dbs.check(h, dbs.lib.dbs_backward(h, r, None, None, g.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.byref(b)))
        out = np.zeros((t * k * v,), dtype=np.float32)
        n = dbs.lib.dbs_backward_sparse_logprob_count(b)
        idx = dbs.lib.dbs_backward_sparse_logprob_indices(b)
        val = dbs.lib.dbs_backward_sparse_logprob_values(b)
        for i in range(n):
            out[int(idx[i])] += float(val[i])
        return out.reshape((t, k, v))
    finally:
        if b.value:
            dbs.lib.dbs_free_backward(b)
        if r.value:
            dbs.lib.dbs_free_result(r)
        dbs.lib.dbs_destroy(h)


def final_scores(log_probs, options: DBSOptions, lib_path: Optional[str] = None):
    @jax.custom_vjp
    def _impl(x):
        shape_dtype = jax.ShapeDtypeStruct((options.beam_size,), jnp.float32)
        return jax.pure_callback(lambda y: _forward_np(y, options, lib_path), shape_dtype, x)

    def fwd(x):
        y = _impl(x)
        return y, x

    def bwd(x, g):
        shape_dtype = jax.ShapeDtypeStruct(x.shape, jnp.float32)
        grad = jax.pure_callback(lambda y, dy: _backward_np(y, dy, options, lib_path), shape_dtype, x, g)
        return (grad,)

    _impl.defvjp(fwd, bwd)
    return _impl(log_probs)
