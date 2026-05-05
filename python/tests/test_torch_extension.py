import torch

from torch_dbs_extension import DBSOptions, final_scores


def test_autograd_cpu_float32():
    opts = DBSOptions(beam_size=2)
    x = torch.randn(3, 2, 8, dtype=torch.float32, requires_grad=True)
    y = final_scores(x, opts).sum()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_autograd_cpu_bfloat16_promotes_and_returns_bfloat16_grad():
    opts = DBSOptions(beam_size=2, validate_inputs=0)
    x = torch.randn(2, 2, 8, dtype=torch.bfloat16, requires_grad=True)
    y = final_scores(x, opts).sum()
    y.backward()
    assert x.grad is not None
    assert x.grad.dtype == torch.bfloat16


def test_cuda_forward_parity_if_extension_available():
    if not torch.cuda.is_available():
        return
    try:
        import dbs_torch_cuda_ext  # noqa: F401
    except ImportError:
        return
    opts = DBSOptions(beam_size=2, eos_token=-1)
    x = torch.randn(4, 2, 64, dtype=torch.float32)
    x = torch.log_softmax(x, dim=-1)
    cpu = final_scores(x, opts).detach()
    cuda = final_scores(x.cuda(), opts).detach().cpu()
    assert cuda.shape == cpu.shape == (2,)
    torch.testing.assert_close(cuda, cpu, rtol=1e-5, atol=1e-5)
