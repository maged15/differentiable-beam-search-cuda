"""CUDA autograd smoke tests for the optional torch CUDA extension."""

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


def test_cuda_selected_path_backward_batched_smoke():
    _require_cuda_ext()
    torch.manual_seed(2027)
    x_cpu = torch.randn(2, 3, 2, 17, dtype=torch.float32)
    x_cpu = torch.log_softmax(x_cpu, dim=-1)
    x = x_cpu.cuda().requires_grad_(True)

    y = final_scores(x, DBSOptions(beam_size=2, eos_token=-1, validate_inputs=1))
    y.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert x.grad.is_cuda
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad).item() > 0
    assert torch.count_nonzero(x.grad).item() <= 2 * 3 * 2


def test_cuda_selected_path_backward_unbatched_smoke():
    _require_cuda_ext()
    torch.manual_seed(2028)
    x = torch.randn(3, 2, 19, device="cuda", dtype=torch.float32)
    x = torch.log_softmax(x, dim=-1).detach().requires_grad_(True)

    y = final_scores(x, DBSOptions(beam_size=2, eos_token=3, min_length=2, validate_inputs=1))
    y[0].backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


def test_cuda_large_beam_backward_raises_cleanly():
    _require_cuda_ext()
    x = torch.randn(1, 2, 33, 11, device="cuda", dtype=torch.float32)
    x = torch.log_softmax(x, dim=-1).detach().requires_grad_(True)
    y = final_scores(x, DBSOptions(beam_size=33, eos_token=-1, validate_inputs=1))

    with pytest.raises(RuntimeError, match="beam_size <= DBS_CUDA_FAST_MAX_BEAM"):
        y.sum().backward()
