"""CUDA autograd smoke tests for the optional torch CUDA extension."""

import importlib

import pytest
import torch

import torch_dbs_extension as dbs_ext
from torch_dbs_extension import DBSOptions, final_scores

CPU_SEMANTIC_FALLBACK_OPTIONS = (
    ("selected_temperature", 0.7),
    ("soft_topk_temperature", 0.4),
    ("relaxed_pool_multiplier", 3),
    ("vocab_block", 7),
    ("length_penalty_alpha", 0.35),
    ("soft_topk_tolerance", 1.0e-5),
    ("soft_topk_max_iters", 64),
    ("max_dense_gradient_elements", 1_000_000),
)


def _require_cuda_ext():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    try:
        return importlib.import_module("dbs_torch_cuda_ext")
    except ImportError:
        pytest.skip("dbs_torch_cuda_ext not built")


def _assert_cuda_matches_cpu_forward_backward(x_cpu, opts, grad_out):
    cpu_x = x_cpu.detach().clone().requires_grad_(True)
    cuda_x = x_cpu.detach().clone().cuda().requires_grad_(True)

    cpu_y = final_scores(cpu_x, opts)
    cuda_y = final_scores(cuda_x, opts)
    torch.testing.assert_close(cuda_y.detach().cpu(), cpu_y.detach(), rtol=1e-5, atol=1e-5)

    cpu_y.backward(grad_out)
    cuda_y.backward(grad_out.cuda())
    assert cuda_x.grad is not None
    assert cuda_x.grad.is_cuda
    torch.testing.assert_close(cuda_x.grad.detach().cpu(), cpu_x.grad.detach(), rtol=1e-5, atol=1e-5)


def test_cuda_backward_matches_cpu_batched_default_options():
    _require_cuda_ext()
    torch.manual_seed(2027)
    x_cpu = torch.randn(2, 3, 2, 17, dtype=torch.float32)
    x_cpu = torch.log_softmax(x_cpu, dim=-1)
    grad_out = torch.tensor([[1.0, -0.25], [0.5, 0.75]], dtype=torch.float32)
    _assert_cuda_matches_cpu_forward_backward(
        x_cpu,
        DBSOptions(beam_size=2, eos_token=-1, validate_inputs=1),
        grad_out,
    )


def test_cuda_backward_matches_cpu_unbatched_eos_min_length():
    _require_cuda_ext()
    torch.manual_seed(2028)
    x_cpu = torch.randn(3, 2, 19, dtype=torch.float32)
    x_cpu = torch.log_softmax(x_cpu, dim=-1)
    grad_out = torch.tensor([0.5, -1.25], dtype=torch.float32)
    _assert_cuda_matches_cpu_forward_backward(
        x_cpu,
        DBSOptions(beam_size=2, eos_token=3, min_length=2, validate_inputs=1),
        grad_out,
    )


def test_cuda_backward_matches_cpu_non_default_options():
    _require_cuda_ext()
    torch.manual_seed(2029)
    x_cpu = torch.randn(2, 4, 3, 23, dtype=torch.float32)
    x_cpu = torch.log_softmax(x_cpu, dim=-1)
    grad_out = torch.tensor([[0.25, -0.5, 1.0], [1.5, -1.0, 0.125]], dtype=torch.float32)
    opts = DBSOptions(
        beam_size=3,
        eos_token=5,
        selected_temperature=0.7,
        soft_topk_temperature=0.4,
        relaxed_pool_multiplier=3,
        vocab_block=7,
        length_penalty_alpha=0.35,
        soft_topk_tolerance=1.0e-5,
        soft_topk_max_iters=64,
        min_length=2,
        validate_inputs=1,
    )
    _assert_cuda_matches_cpu_forward_backward(x_cpu, opts, grad_out)


@pytest.mark.parametrize(("option_name", "option_value"), CPU_SEMANTIC_FALLBACK_OPTIONS)
def test_cuda_backward_matches_cpu_for_each_cpu_semantic_fallback_option(option_name, option_value):
    _require_cuda_ext()
    torch.manual_seed(2029)
    x_cpu = torch.randn(2, 3, 2, 13, dtype=torch.float32)
    x_cpu = torch.log_softmax(x_cpu, dim=-1)
    grad_out = torch.tensor([[0.25, -0.5], [1.0, 0.125]], dtype=torch.float32)
    opts = DBSOptions(beam_size=2, eos_token=3, **{option_name: option_value})
    assert not dbs_ext._native_cuda_forward_supported(opts)
    _assert_cuda_matches_cpu_forward_backward(x_cpu, opts, grad_out)


def test_cuda_large_beam_backward_matches_cpu():
    _require_cuda_ext()
    torch.manual_seed(2030)
    x_cpu = torch.randn(1, 2, 33, 11, dtype=torch.float32)
    x_cpu = torch.log_softmax(x_cpu, dim=-1)
    grad_out = torch.linspace(-1.0, 1.0, 33, dtype=torch.float32).unsqueeze(0)
    _assert_cuda_matches_cpu_forward_backward(
        x_cpu,
        DBSOptions(beam_size=33, eos_token=-1, validate_inputs=1),
        grad_out,
    )
