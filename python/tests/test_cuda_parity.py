"""Hardware parity tests for the optional CUDA backend.

Run after building with:
  cmake -S . -B build-cuda -DDBS_ENABLE_CUDA=ON
  cmake --build build-cuda --parallel
  DBS_BUILD_TORCH_CUDA=1 pip install -e .
"""

import importlib
import os

import pytest
import torch

import torch_dbs_extension as dbs_ext
from torch_dbs_extension import DBSOptions, decode, final_scores


INVALID_ARGUMENT_MSG = "invalid argument"
CPU_SEMANTIC_FALLBACK_OPTIONS = (
    ("selected_temperature", 0.7),
    ("soft_topk_temperature", 0.4),
    ("relaxed_pool_multiplier", 3),
    ("vocab_block", 17),
    ("length_penalty_alpha", 0.2),
    ("soft_topk_tolerance", 1.0e-5),
    ("soft_topk_max_iters", 64),
    ("max_dense_gradient_elements", 1_000_000),
)
INVALID_CPU_SEMANTIC_OPTIONS = (
    ({"selected_temperature": 0.0}, "selected_temperature must be positive"),
    ({"soft_topk_temperature": 0.0}, "soft_topk_temperature must be positive"),
    ({"relaxed_pool_multiplier": 0}, "relaxed_pool_multiplier must be positive"),
    ({"vocab_block": 0}, "vocab_block must be positive"),
    ({"length_penalty_alpha": -0.1}, "length_penalty_alpha cannot be negative"),
    ({"soft_topk_tolerance": 0.0}, "soft_topk_tolerance must be positive"),
    ({"soft_topk_max_iters": 0}, "soft_topk_max_iters must be positive"),
    ({"max_dense_gradient_elements": 0}, "max_dense_gradient_elements must be positive"),
)


def _require_bool_availability(value, source):
    if value is None:
        pytest.fail(f"{source} returned None; expected a boolean availability result")
    return bool(value)


def _require_cuda_ext():
    torch_cuda_available = _require_bool_availability(
        torch.cuda.is_available(), "torch.cuda.is_available()"
    )
    if not torch_cuda_available:
        pytest.skip("CUDA device unavailable")
    try:
        ext = importlib.import_module("dbs_torch_cuda_ext")
    except ImportError:
        pytest.skip("dbs_torch_cuda_ext not built")
    ext_cuda_available = _require_bool_availability(
        ext.cuda_available(), "dbs_torch_cuda_ext.cuda_available()"
    )
    if not ext_cuda_available:
        pytest.skip("dbs_torch_cuda_ext reports CUDA unavailable")
    return ext


def test_cuda_requirement_fails_on_unknown_torch_availability(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: None)
    with pytest.raises(pytest.fail.Exception, match="torch.cuda.is_available"):
        _require_cuda_ext()


def test_cuda_requirement_fails_on_unknown_extension_availability(monkeypatch):
    class FakeCudaExt:
        @staticmethod
        def cuda_available():
            return None

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(importlib, "import_module", lambda _: FakeCudaExt())
    with pytest.raises(pytest.fail.Exception, match="dbs_torch_cuda_ext.cuda_available"):
        _require_cuda_ext()


@pytest.mark.parametrize("shape", [(1, 4, 2, 128), (3, 6, 4, 1024), (2, 8, 8, 4096)])
@pytest.mark.parametrize("eos", [-1, 2])
def test_cuda_forward_matches_cpu_final_scores(shape, eos):
    _require_cuda_ext()
    torch.manual_seed(1234)
    x_cpu = torch.randn(*shape, dtype=torch.float32)
    x_cpu = torch.log_softmax(x_cpu, dim=-1)
    opts = DBSOptions(beam_size=shape[2], eos_token=eos, validate_inputs=1)

    # CPU operator takes [T,K,V]; compare each batch item to CUDA [B,T,K,V].
    cpu_scores = torch.stack([final_scores(x_cpu[b], opts).detach() for b in range(shape[0])], dim=0)
    cuda_scores = final_scores(x_cpu.cuda(), opts).detach().cpu()
    torch.testing.assert_close(cuda_scores, cpu_scores, rtol=1e-5, atol=1e-5)


def test_cuda_sparse_scatter_module_available():
    ext = _require_cuda_ext()
    assert ext.cuda_available()


def test_cuda_sparse_scatter_duplicate_and_empty_indices():
    ext = _require_cuda_ext()
    indices = torch.tensor([0, 1, 1, 4], device="cuda", dtype=torch.int64)
    values = torch.tensor([1.0, 2.0, 3.0, 5.0], device="cuda", dtype=torch.float32)
    grad = ext._test_sparse_backward_scatter(indices, values, 5).detach().cpu()
    torch.testing.assert_close(grad, torch.tensor([1.0, 5.0, 0.0, 0.0, 5.0]))

    empty_indices = torch.empty((0,), device="cuda", dtype=torch.int64)
    empty_values = torch.empty((0,), device="cuda", dtype=torch.float32)
    empty_grad = ext._test_sparse_backward_scatter(empty_indices, empty_values, 3).detach().cpu()
    torch.testing.assert_close(empty_grad, torch.zeros(3))


def test_cuda_sparse_scatter_skips_invalid_indices():
    # -1 sentinels (from backward builder) and out-of-range indices are silently skipped.
    ext = _require_cuda_ext()
    values = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32)

    negative = torch.tensor([0, -1], device="cuda", dtype=torch.int64)
    grad = ext._test_sparse_backward_scatter(negative, values, 3).detach().cpu()
    torch.testing.assert_close(grad, torch.tensor([1.0, 0.0, 0.0]))

    too_large = torch.tensor([0, 3], device="cuda", dtype=torch.int64)
    grad2 = ext._test_sparse_backward_scatter(too_large, values, 3).detach().cpu()
    torch.testing.assert_close(grad2, torch.tensor([1.0, 0.0, 0.0]))


def test_direct_cuda_decode_rejects_invalid_eos_values():
    ext = _require_cuda_ext()
    x = torch.log_softmax(torch.randn(1, 2, 8, device="cuda", dtype=torch.float32), dim=-1)
    invalid_argument = 2
    assert ext._test_decode_forward_status(x, 2, -2, False) == invalid_argument
    assert ext._test_decode_forward_status(x, 2, -2, True) == invalid_argument
    assert ext._test_decode_forward_status(x, 2, 8, False) == invalid_argument
    assert ext._test_decode_forward_status(x, 2, 8, True) == invalid_argument


def test_cuda_fast_kernel_enforces_min_length_for_eos():
    ext = _require_cuda_ext()
    x_cpu = torch.full((1, 3, 2, 6), -10.0, dtype=torch.float32)
    x_cpu[:, :, :, 0] = 0.0      # EOS would win immediately without min_length.
    x_cpu[:, :, :, 1] = -0.1
    x_cpu[:, :, :, 2] = -0.2

    tokens, scores = ext._test_decode_forward_fast_ex(x_cpu.cuda(), 2, 0, 2)
    tokens_cpu = tokens.detach().cpu()
    assert 0 not in tokens_cpu[0, 0].tolist()
    assert 0 in tokens_cpu[0, 1].tolist()

    opts = DBSOptions(beam_size=2, eos_token=0, min_length=2, validate_inputs=1)
    cpu_scores = final_scores(x_cpu[0], opts).detach()
    torch.testing.assert_close(scores.detach().cpu()[0], cpu_scores, rtol=1e-5, atol=1e-5)


def test_cuda_public_min_length_matches_cpu():
    _require_cuda_ext()
    x_cpu = torch.full((1, 3, 2, 6), -10.0, dtype=torch.float32)
    x_cpu[:, :, :, 0] = 0.0
    x_cpu[:, :, :, 1] = -0.1
    x_cpu[:, :, :, 2] = -0.2
    opts = DBSOptions(beam_size=2, eos_token=0, min_length=2, validate_inputs=1)
    cpu_scores = final_scores(x_cpu[0], opts).detach()
    cuda_scores = final_scores(x_cpu.cuda(), opts).detach().cpu()[0]
    torch.testing.assert_close(cuda_scores, cpu_scores, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("eos", [-1, 0])
def test_cuda_equal_score_edge_cases_match_cpu(eos):
    _require_cuda_ext()
    x_cpu = torch.full((2, 3, 3, 9), -2.0, dtype=torch.float32)
    x_cpu[:, :, :, 0] = -0.1
    x_cpu[:, :, :, 1] = -0.1
    x_cpu[:, :, :, 2] = -0.1
    if eos >= 0:
        x_cpu[:, :, :, eos] = -0.1
    opts = DBSOptions(beam_size=3, eos_token=eos, validate_inputs=1)
    cpu_scores = torch.stack([final_scores(x_cpu[b], opts).detach() for b in range(x_cpu.size(0))], dim=0)
    cuda_scores = final_scores(x_cpu.cuda(), opts).detach().cpu()
    torch.testing.assert_close(cuda_scores, cpu_scores, rtol=1e-6, atol=1e-6)


def test_cuda_equal_score_direct_tokens_are_deterministic():
    ext = _require_cuda_ext()
    x_cpu = torch.full((1, 2, 3, 6), -1.0, dtype=torch.float32)
    tokens, scores = ext._test_decode_forward_fast_ex(x_cpu.cuda(), 3, -1, 0)
    tokens_cpu = tokens.detach().cpu()
    assert tokens_cpu[0, 0].tolist() == [0, 1, 2]
    assert tokens_cpu[0, 1].tolist() == [0, 1, 2]
    torch.testing.assert_close(scores.detach().cpu()[0], torch.full((3,), -2.0))


def test_cuda_eos_carry_tie_prefers_carry_forward_order():
    ext = _require_cuda_ext()
    x_cpu = torch.full((1, 2, 2, 5), -4.0, dtype=torch.float32)
    x_cpu[:, :, :, 0] = 0.0  # EOS ties with carry-forward at step 1.
    x_cpu[:, :, :, 1] = 0.0
    tokens, _ = ext._test_decode_forward_fast_ex(x_cpu.cuda(), 2, 0, 0)
    tokens_cpu = tokens.detach().cpu()
    assert tokens_cpu[0, 0].tolist() == [0, 1]
    assert tokens_cpu[0, 1].tolist() == [0, 0]


def test_cuda_all_negative_infinity_rows_match_cpu():
    _require_cuda_ext()
    x_cpu = torch.full((2, 2, 2, 5), -float("inf"), dtype=torch.float32)
    opts = DBSOptions(beam_size=2, eos_token=-1, validate_inputs=0)
    cpu_scores = torch.stack([final_scores(x_cpu[b], opts).detach() for b in range(x_cpu.size(0))], dim=0)
    cuda_scores = final_scores(x_cpu.cuda(), opts).detach().cpu()
    torch.testing.assert_close(cuda_scores, cpu_scores, equal_nan=True)


def test_cuda_variable_decode_rejects_invalid_per_example_metadata():
    ext = _require_cuda_ext()
    torch.manual_seed(707)
    x_cpu = torch.randn(2, 3, 2, 16, dtype=torch.float32)
    x_cpu = torch.log_softmax(x_cpu, dim=-1)
    x = x_cpu.cuda()
    steps = torch.tensor([3, 3], device="cuda", dtype=torch.int32)
    beams = torch.tensor([2, 2], device="cuda", dtype=torch.int32)
    eos = torch.tensor([-1, -1], device="cuda", dtype=torch.int32)
    min_lengths = torch.tensor([0, 0], device="cuda", dtype=torch.int32)

    bad_cases = [
        (torch.tensor([0, 3], device="cuda", dtype=torch.int32), beams, eos, min_lengths),
        (steps, torch.tensor([0, 2], device="cuda", dtype=torch.int32), eos, min_lengths),
        (steps, beams, torch.tensor([999, -1], device="cuda", dtype=torch.int32), min_lengths),
        (steps, beams, torch.tensor([-2, -1], device="cuda", dtype=torch.int32), min_lengths),
        (steps, beams, eos, torch.tensor([-1, 0], device="cuda", dtype=torch.int32)),
    ]
    for bad_steps, bad_beams, bad_eos, bad_min_lengths in bad_cases:
        with pytest.raises(RuntimeError, match=INVALID_ARGUMENT_MSG):
            ext._test_decode_forward_variable(x, bad_steps, bad_beams, bad_eos, bad_min_lengths)


def test_cuda_variable_decode_initializes_ragged_trailing_outputs():
    ext = _require_cuda_ext()
    x_cpu = torch.full((2, 3, 2, 8), -3.0, dtype=torch.float32)
    x_cpu[:, :, :, 0] = -0.1
    x_cpu[:, :, :, 1] = -0.2
    x = x_cpu.cuda()
    steps = torch.tensor([2, 3], device="cuda", dtype=torch.int32)
    beams = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    eos = torch.tensor([-1, -1], device="cuda", dtype=torch.int32)
    min_lengths = torch.tensor([0, 0], device="cuda", dtype=torch.int32)

    tokens, scores = ext._test_decode_forward_variable(x, steps, beams, eos, min_lengths)
    tokens_cpu = tokens.detach().cpu()
    scores_cpu = scores.detach().cpu()
    assert torch.equal(tokens_cpu[0, :2, 1], torch.full((2,), -1, dtype=torch.int32))
    assert torch.equal(tokens_cpu[0, 2], torch.full((2,), -1, dtype=torch.int32))
    assert torch.isneginf(scores_cpu[0, 1])
    assert torch.isfinite(scores_cpu[0, 0])
    assert torch.isfinite(scores_cpu[1]).all()


@pytest.mark.skipif(os.environ.get("DBS_CUDA_LARGE_SCATTER_TEST") != "1", reason="large scatter grid-stride test is release-hardware gated")
def test_cuda_sparse_scatter_large_nnz_grid_stride():
    ext = _require_cuda_ext()
    grad_count = int(os.environ.get("DBS_CUDA_LARGE_SCATTER_GRAD_COUNT", "1024"))
    nnz = int(os.environ.get("DBS_CUDA_LARGE_SCATTER_NNZ", str(65535 * 256 + 4096)))
    indices = torch.arange(nnz, device="cuda", dtype=torch.int64).remainder_(grad_count)
    values = torch.ones((nnz,), device="cuda", dtype=torch.float32)
    grad = ext._test_sparse_backward_scatter(indices, values, grad_count).detach().cpu()
    expected = torch.full((grad_count,), float(nnz // grad_count), dtype=torch.float32)
    expected[: nnz % grad_count] += 1.0
    torch.testing.assert_close(grad, expected, rtol=0.0, atol=0.0)


def test_cuda_exact_kernel_uses_current_stream():
    _require_cuda_ext()
    torch.manual_seed(2026)
    x_cpu = torch.randn(2, 5, 3, 257, dtype=torch.float32)
    x_cpu = torch.log_softmax(x_cpu, dim=-1)
    opts = DBSOptions(beam_size=3, eos_token=7, validate_inputs=1)
    x = x_cpu.cuda()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        y = final_scores(x, opts)
        marker = torch.empty((), device="cuda").fill_(1.0)
    stream.synchronize()
    cpu_scores = torch.stack([final_scores(x_cpu[b], opts).detach() for b in range(x_cpu.size(0))], dim=0)
    torch.testing.assert_close(y.cpu(), cpu_scores, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(marker.cpu(), torch.tensor(1.0))


def test_cuda_tie_and_near_tie_exact_parity(monkeypatch):
    _require_cuda_ext()
    # Exact path is default. Keep this adversarial case deterministic: ties, near ties,
    # and beam-specific rows across time. This catches unsafe score-only shortcuts.
    monkeypatch.delenv("DBS_ENABLE_SCORE_ONLY_FAST_PATH", raising=False)
    x_cpu = torch.full((2, 4, 4, 64), -20.0, dtype=torch.float32)
    x_cpu[:, :, :, 0] = -0.1
    x_cpu[:, :, :, 1] = -0.1
    x_cpu[:, :, :, 2] = -0.100001
    x_cpu[:, 1, 2, 3] = -0.05
    x_cpu[:, 2, 1, 4] = -0.05
    x_cpu[:, 3, 3, 5] = -0.05
    opts = DBSOptions(beam_size=4, eos_token=-1, validate_inputs=1)
    cpu_scores = torch.stack([final_scores(x_cpu[b], opts).detach() for b in range(x_cpu.size(0))], dim=0)
    cuda_scores = final_scores(x_cpu.cuda(), opts).detach().cpu()
    torch.testing.assert_close(cuda_scores, cpu_scores, rtol=1e-6, atol=1e-6)


def test_cuda_score_only_fast_path_is_explicit(monkeypatch):
    ext = _require_cuda_ext()
    torch.manual_seed(55)
    x_cpu = torch.randn(2, 5, 4, 512, dtype=torch.float32)
    x_cpu = torch.log_softmax(x_cpu, dim=-1)
    opts = DBSOptions(beam_size=4, eos_token=-1, validate_inputs=1)
    monkeypatch.setenv("DBS_ENABLE_SCORE_ONLY_FAST_PATH", "1")
    fast = final_scores(x_cpu.cuda(), opts).detach().cpu()
    monkeypatch.setenv("DBS_FORCE_EXACT_CUDA_KERNEL", "1")
    exact = ext.final_scores_forward_cuda_exact_kernel(x_cpu.cuda(), opts.beam_size, opts.eos_token).detach().cpu()
    torch.testing.assert_close(fast, exact, rtol=1e-5, atol=1e-5)


def test_cuda_public_api_accepts_unbatched_shape():
    _require_cuda_ext()
    torch.manual_seed(99)
    x_cpu = torch.randn(5, 3, 257, dtype=torch.float32)
    x_cpu = torch.log_softmax(x_cpu, dim=-1)
    opts = DBSOptions(beam_size=3, eos_token=-1, validate_inputs=1)
    cpu_scores = final_scores(x_cpu, opts).detach()
    cuda_scores = final_scores(x_cpu.cuda(), opts).detach().cpu()
    assert cuda_scores.shape == cpu_scores.shape == (3,)
    torch.testing.assert_close(cuda_scores, cpu_scores, rtol=1e-5, atol=1e-5)


def test_cuda_debug_sync_check_mode(monkeypatch):
    _require_cuda_ext()
    torch.manual_seed(123)
    monkeypatch.setenv("DBS_CUDA_SYNC_CHECK", "1")
    x_cpu = torch.randn(4, 2, 128, dtype=torch.float32)
    x_cpu = torch.log_softmax(x_cpu, dim=-1)
    opts = DBSOptions(beam_size=2, eos_token=-1, validate_inputs=1)
    cpu_scores = final_scores(x_cpu, opts).detach()
    cuda_scores = final_scores(x_cpu.cuda(), opts).detach().cpu()
    torch.testing.assert_close(cuda_scores, cpu_scores, rtol=1e-5, atol=1e-5)


def test_cuda_rejects_nan_and_positive_infinity_when_validation_enabled():
    _require_cuda_ext()
    x = torch.log_softmax(torch.randn(4, 2, 16, device="cuda", dtype=torch.float32), dim=-1)
    opts = DBSOptions(beam_size=2, validate_inputs=1)

    with_nan = x.clone()
    with_nan[0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="NaN or \\+Inf"):
        final_scores(with_nan, opts)

    with_pos_inf = x.clone()
    with_pos_inf[0, 0, 0] = float("inf")
    with pytest.raises(RuntimeError, match="NaN or \\+Inf"):
        final_scores(with_pos_inf, opts)

    with_neg_inf = x.clone()
    with_neg_inf[0, 0, 0] = -float("inf")
    y = final_scores(with_neg_inf, opts).detach()
    assert torch.isfinite(y).all()


def test_native_cuda_op_accepts_unbatched_shape_directly():
    ext = _require_cuda_ext()
    torch.manual_seed(101)
    x_cpu = torch.randn(5, 3, 257, dtype=torch.float32)
    x_cpu = torch.log_softmax(x_cpu, dim=-1)
    opts = DBSOptions(beam_size=3, eos_token=-1, validate_inputs=1)
    cpu_scores = final_scores(x_cpu, opts).detach()
    cuda_scores = ext.final_scores_forward_cuda(x_cpu.cuda(), opts.beam_size, opts.eos_token).detach().cpu()
    assert cuda_scores.shape == cpu_scores.shape == (3,)
    torch.testing.assert_close(cuda_scores, cpu_scores, rtol=1e-5, atol=1e-5)


def test_cuda_public_api_supports_non_default_options_with_cpu_semantic_parity():
    _require_cuda_ext()
    torch.manual_seed(616)
    x_cpu = torch.randn(4, 2, 128, dtype=torch.float32)
    x_cpu = torch.log_softmax(x_cpu, dim=-1)
    opts = DBSOptions(
        beam_size=2,
        eos_token=-1,
        selected_temperature=0.7,
        soft_topk_temperature=0.4,
        relaxed_pool_multiplier=3,
        vocab_block=17,
        length_penalty_alpha=0.2,
        soft_topk_tolerance=1.0e-5,
        soft_topk_max_iters=64,
    )
    cpu_scores = final_scores(x_cpu, opts).detach()
    cuda_scores = final_scores(x_cpu.cuda(), opts).detach().cpu()
    torch.testing.assert_close(cuda_scores, cpu_scores, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("option_name", "option_value"), CPU_SEMANTIC_FALLBACK_OPTIONS)
def test_cuda_public_api_supports_each_cpu_semantic_fallback_option(option_name, option_value):
    _require_cuda_ext()
    torch.manual_seed(617)
    x_cpu = torch.randn(3, 2, 19, dtype=torch.float32)
    x_cpu = torch.log_softmax(x_cpu, dim=-1)
    opts = DBSOptions(beam_size=2, eos_token=4, **{option_name: option_value})

    assert not dbs_ext._native_cuda_forward_supported(opts)
    cpu_scores = final_scores(x_cpu, opts).detach()
    cuda_scores = final_scores(x_cpu.cuda(), opts).detach().cpu()
    torch.testing.assert_close(cuda_scores, cpu_scores, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("beam_size", "expected_native"), [(32, True), (33, False), (64, False), (65, False)])
def test_cuda_public_api_routes_beam_limit_edges(beam_size, expected_native):
    _require_cuda_ext()
    torch.manual_seed(618 + beam_size)
    x_cpu = torch.randn(1, 2, beam_size, 17, dtype=torch.float32)
    x_cpu = torch.log_softmax(x_cpu, dim=-1)
    opts = DBSOptions(beam_size=beam_size, eos_token=-1, validate_inputs=1)

    assert dbs_ext._native_cuda_forward_supported(opts) is expected_native
    cpu_scores = final_scores(x_cpu, opts).detach()
    cuda_scores = final_scores(x_cpu.cuda(), opts).detach().cpu()
    torch.testing.assert_close(cuda_scores, cpu_scores, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("option_name", "option_value"), CPU_SEMANTIC_FALLBACK_OPTIONS)
def test_cuda_public_decode_rejects_cpu_semantic_only_options(option_name, option_value):
    _require_cuda_ext()
    x_cpu = torch.log_softmax(torch.randn(3, 2, 19, dtype=torch.float32), dim=-1)
    opts = DBSOptions(beam_size=2, eos_token=4, **{option_name: option_value})

    with pytest.raises(ValueError, match=option_name):
        decode(x_cpu.cuda(), opts)


@pytest.mark.parametrize(("option_kwargs", "message"), INVALID_CPU_SEMANTIC_OPTIONS)
def test_cuda_public_api_rejects_invalid_cpu_semantic_options(option_kwargs, message):
    _require_cuda_ext()
    x = torch.randn(4, 2, 16, device="cuda", dtype=torch.float32)
    with pytest.raises((ValueError, RuntimeError), match=message):
        final_scores(x, DBSOptions(beam_size=2, **option_kwargs))



def test_cuda_public_api_rejects_invalid_shapes_and_option_values():
    _require_cuda_ext()
    bad = torch.randn(0, 2, 8, device="cuda", dtype=torch.float32)
    with pytest.raises((ValueError, RuntimeError)):
        final_scores(bad, DBSOptions(beam_size=2))
    x = torch.randn(4, 2, 16, device="cuda", dtype=torch.float32)
    with pytest.raises((ValueError, RuntimeError), match="selected_temperature must be positive"):
        final_scores(x, DBSOptions(beam_size=2, selected_temperature=-0.7))


def test_cuda_rejects_beam_size_above_backend_limit():
    ext = _require_cuda_ext()
    x = torch.randn(1, 65, 2, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="beam_size exceeds CUDA backend maximum"):
        ext.final_scores_forward_cuda(x, 65, -1)


def test_cuda_public_api_accepts_non_contiguous_and_half_inputs():
    _require_cuda_ext()
    torch.manual_seed(314)
    base = torch.randn(3, 5, 2, 257, dtype=torch.float32)
    base = torch.log_softmax(base, dim=-1)
    x_cpu = base.transpose(0, 1)  # [5,3,2,257], non-contiguous batched input
    opts = DBSOptions(beam_size=2, eos_token=-1)
    cpu_scores = torch.stack([final_scores(x_cpu[b], opts).detach() for b in range(x_cpu.size(0))], dim=0)
    cuda_scores = final_scores(x_cpu.cuda(), opts).detach().cpu()
    torch.testing.assert_close(cuda_scores, cpu_scores, rtol=1e-5, atol=1e-5)

    half_scores = final_scores(x_cpu.cuda().to(torch.float16), opts).detach().cpu()
    assert half_scores.shape == cpu_scores.shape
    assert torch.isfinite(half_scores).all()


def test_cuda_public_decode_returns_tokens_and_scores():
    _require_cuda_ext()
    x_cpu = torch.full((2, 3, 2, 8), -4.0, dtype=torch.float32)
    x_cpu[:, :, :, 0] = -0.1
    x_cpu[:, :, :, 1] = -0.2
    opts = DBSOptions(beam_size=2, eos_token=-1, validate_inputs=1)
    tokens, scores = decode(x_cpu.cuda(), opts)
    assert tokens.shape == (2, 3, 2)
    assert scores.shape == (2, 2)
    assert tokens.dtype == torch.int32
    torch.testing.assert_close(scores.detach().cpu(), final_scores(x_cpu.cuda(), opts).detach().cpu())
