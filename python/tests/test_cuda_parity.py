"""Hardware parity tests for the optional CUDA backend.

Run after building with:
  cmake -S . -B build-cuda -DDBS_ENABLE_CUDA=ON
  cmake --build build-cuda --parallel
  DBS_BUILD_TORCH_CUDA=1 pip install -e .
"""

import importlib

import pytest
import torch

from torch_dbs_extension import DBSOptions, final_scores


def _require_cuda_ext():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    try:
        return importlib.import_module("dbs_torch_cuda_ext")
    except ImportError:
        pytest.skip("dbs_torch_cuda_ext not built")


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
    assert marker.item() == 1.0


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


def test_cuda_rejects_options_that_would_be_silently_ignored():
    _require_cuda_ext()
    x = torch.randn(4, 2, 128, device="cuda", dtype=torch.float32)
    x = torch.log_softmax(x, dim=-1)
    opts = DBSOptions(beam_size=2, eos_token=-1, min_length=2)
    with pytest.raises(ValueError, match="unsupported CUDA options"):
        final_scores(x, opts)



def test_cuda_public_api_rejects_invalid_shapes_and_options():
    _require_cuda_ext()
    bad = torch.randn(0, 2, 8, device="cuda", dtype=torch.float32)
    with pytest.raises((ValueError, RuntimeError)):
        final_scores(bad, DBSOptions(beam_size=2))
    x = torch.randn(4, 2, 16, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="unsupported CUDA options"):
        final_scores(x, DBSOptions(beam_size=2, min_length=1))


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
