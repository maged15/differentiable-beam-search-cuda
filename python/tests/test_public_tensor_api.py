import pytest
import torch

from torch_dbs_extension import DBSOptions, final_scores


def test_cpu_accepts_unbatched_and_batched_shapes():
    torch.manual_seed(1)
    opts = DBSOptions(beam_size=3)
    x = torch.randn(2, 4, 3, 32, dtype=torch.float32)
    x = torch.log_softmax(x, dim=-1)
    batched = final_scores(x, opts)
    manual = torch.stack([final_scores(x[b], opts) for b in range(x.size(0))], dim=0)
    assert batched.shape == (2, 3)
    torch.testing.assert_close(batched, manual)


def test_cpu_batched_autograd_shape():
    opts = DBSOptions(beam_size=2)
    x = torch.randn(3, 4, 2, 16, dtype=torch.float32, requires_grad=True)
    y = final_scores(x, opts)
    assert y.shape == (3, 2)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


def test_rejects_bad_public_shape():
    opts = DBSOptions(beam_size=2)
    with pytest.raises(ValueError):
        final_scores(torch.randn(2, 3), opts)
    with pytest.raises(ValueError):
        final_scores(torch.randn(4, 3, 16), opts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_accepts_unbatched_shape_if_extension_built():
    pytest.importorskip("dbs_torch_cuda_ext")
    opts = DBSOptions(beam_size=2)
    x = torch.randn(4, 2, 64, device="cuda", dtype=torch.float32)
    x = torch.log_softmax(x, dim=-1)
    y = final_scores(x, opts)
    assert y.shape == (2,)


def test_rejects_zero_dimensions_and_bad_beam():
    opts = DBSOptions(beam_size=2)
    with pytest.raises(ValueError):
        final_scores(torch.empty(0, 2, 8), opts)
    with pytest.raises(ValueError):
        final_scores(torch.empty(4, 0, 8), DBSOptions(beam_size=0))
    # [B,T,K,V] with B=1 is valid; reject only a real beam-size mismatch.
    with pytest.raises(ValueError):
        final_scores(torch.empty(1, 4, 3, 8), opts)

    y = final_scores(torch.log_softmax(torch.randn(1, 4, 2, 8), dim=-1), opts)
    assert y.shape == (1, 2)


def test_rejects_eos_token_outside_vocab_before_native_decode():
    x = torch.log_softmax(torch.randn(3, 2, 8), dim=-1)
    with pytest.raises(ValueError, match="eos_token"):
        final_scores(x, DBSOptions(beam_size=2, eos_token=8))
    with pytest.raises(ValueError, match="eos_token"):
        final_scores(x, DBSOptions(beam_size=2, eos_token=-2))


def test_cpu_unbatched_and_batched_contract_shapes():
    opts = DBSOptions(beam_size=2)
    x3 = torch.randn(3, 2, 16, dtype=torch.float32)
    x4 = torch.randn(5, 3, 2, 16, dtype=torch.float32)
    assert final_scores(torch.log_softmax(x3, dim=-1), opts).shape == (2,)
    assert final_scores(torch.log_softmax(x4, dim=-1), opts).shape == (5, 2)


def test_cuda_unbatched_public_contract_if_extension_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("dbs_torch_cuda_ext")
    opts = DBSOptions(beam_size=3, eos_token=-1)
    x = torch.randn(5, 3, 257, dtype=torch.float32)
    x = torch.log_softmax(x, dim=-1)
    y = final_scores(x.cuda(), opts).detach().cpu()
    ref = final_scores(x, opts).detach()
    assert y.shape == (3,)
    torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-5)


def test_cuda_min_length_supported_if_extension_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("dbs_torch_cuda_ext")
    x = torch.randn(4, 2, 64, device="cuda", dtype=torch.float32)
    x = torch.log_softmax(x, dim=-1)
    y = final_scores(x, DBSOptions(beam_size=2, min_length=1))
    assert y.shape == (2,)
