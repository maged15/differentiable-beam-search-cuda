// SPDX-License-Identifier: MIT
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "dbs_cuda.h"

#include <cstdlib>
#include <limits>
#include <cmath>
#include <vector>

static void validate_int_bound(int64_t value, const char* name) {
    TORCH_CHECK(value > 0, name, " must be positive");
    TORCH_CHECK(value <= static_cast<int64_t>(std::numeric_limits<int>::max()), name, " exceeds INT_MAX");
}

static bool validate_cuda_log_probs_public(const torch::Tensor& log_probs, int64_t beam_size) {
    TORCH_CHECK(log_probs.is_cuda(), "log_probs must be CUDA");
    TORCH_CHECK(log_probs.dtype() == torch::kFloat32, "CUDA op currently requires float32");
    TORCH_CHECK(log_probs.dim() == 3 || log_probs.dim() == 4, "expected [T,K,V] or [B,T,K,V]");
    TORCH_CHECK(beam_size > 0 && beam_size <= static_cast<int64_t>(std::numeric_limits<int>::max()), "beam_size must be in [1, INT_MAX]");

    if (log_probs.dim() == 3) {
        validate_int_bound(log_probs.size(0), "T");
        validate_int_bound(log_probs.size(1), "K");
        validate_int_bound(log_probs.size(2), "V");
        TORCH_CHECK(log_probs.size(1) == beam_size, "beam_size must match log_probs.size(1)");
        return true;
    }

    validate_int_bound(log_probs.size(0), "B");
    validate_int_bound(log_probs.size(1), "T");
    validate_int_bound(log_probs.size(2), "K");
    validate_int_bound(log_probs.size(3), "V");
    TORCH_CHECK(log_probs.size(2) == beam_size, "beam_size must match log_probs.size(2)");
    return false;
}


static void validate_eos_token_bound(int64_t eos_token) {
    TORCH_CHECK(eos_token >= -1 && eos_token <= static_cast<int64_t>(std::numeric_limits<int>::max()), "eos_token must be -1 or fit in int");
}

static torch::Tensor final_scores_forward_cuda_exact_kernel(torch::Tensor log_probs, int64_t beam_size, int64_t eos_token) {
    validate_eos_token_bound(eos_token);
    const bool unbatched = validate_cuda_log_probs_public(log_probs, beam_size);
    c10::cuda::CUDAGuard device_guard(log_probs.device());

    auto x4 = unbatched ? log_probs.unsqueeze(0) : log_probs;
    auto x = x4.contiguous();
    const int B = static_cast<int>(x.size(0));
    const int T = static_cast<int>(x.size(1));
    const int K = static_cast<int>(x.size(2));
    const int V = static_cast<int>(x.size(3));

    auto tokens = torch::empty({B, T, K}, torch::TensorOptions().device(log_probs.device()).dtype(torch::kInt32));
    auto scores = torch::empty({B, K}, log_probs.options());
    const auto stream = at::cuda::getCurrentCUDAStream(log_probs.get_device());
    const int rc = dbs_cuda_decode_forward_fast(
        x.data_ptr<float>(), B, T, K, V, static_cast<int>(eos_token),
        tokens.data_ptr<int32_t>(), scores.data_ptr<float>(), stream.stream());
    TORCH_CHECK(rc == DBS_CUDA_STATUS_OK, dbs_cuda_status_string(rc));
    return unbatched ? scores.squeeze(0) : scores;
}

static torch::Tensor final_scores_forward_cuda_aten_topk(torch::Tensor log_probs, int64_t beam_size) {
    const bool unbatched = validate_cuda_log_probs_public(log_probs, beam_size);
    c10::cuda::CUDAGuard device_guard(log_probs.device());

    auto x4 = unbatched ? log_probs.unsqueeze(0) : log_probs;
    auto x = x4.contiguous();
    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    const int64_t K = x.size(2);
    const int64_t V = x.size(3);

    // DBS initialization semantics: only beam 0 is live at t=0. This is different
    // from a naive PyTorch reference that initializes every beam to zero.
    auto scores = torch::full({B, K}, -std::numeric_limits<float>::infinity(), x.options());
    scores.select(1, 0).fill_(0.0f);

    for (int64_t t = 0; t < T; ++t) {
        auto step = x.select(1, t);                 // [B,K,V]
        auto cand = step + scores.view({B, K, 1});  // [B,K,V]
        auto flat = cand.reshape({B, K * V});       // [B,K*V]
        scores = std::get<0>(flat.topk(K, 1, true, true));
    }
    return unbatched ? scores.squeeze(0) : scores;
}

static torch::Tensor final_scores_forward_cuda(torch::Tensor log_probs, int64_t beam_size, int64_t eos_token) {
    validate_eos_token_bound(eos_token);
    validate_cuda_log_probs_public(log_probs, beam_size);

    // The exact custom kernel is the default because it preserves the C ABI decoder's
    // deterministic rank-aligned beam semantics and uses PyTorch's current CUDA stream.
    // The ATen topk path is an opt-in score-only fast path for no-EOS, rank-aligned
    // log-prob tensors. Enable it explicitly with DBS_ENABLE_SCORE_ONLY_FAST_PATH=1.
    const char* enable_fast = std::getenv("DBS_ENABLE_SCORE_ONLY_FAST_PATH");
    const char* legacy_fast = std::getenv("DBS_ASSUME_RANK_ALIGNED_LOG_PROBS");
    const char* force_exact = std::getenv("DBS_FORCE_EXACT_CUDA_KERNEL");
    if (eos_token < 0 && !force_exact && ((enable_fast && *enable_fast == '1') || (legacy_fast && *legacy_fast == '1'))) {
        return final_scores_forward_cuda_aten_topk(log_probs, beam_size);
    }

    return final_scores_forward_cuda_exact_kernel(log_probs, beam_size, eos_token);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("final_scores_forward_cuda", &final_scores_forward_cuda, "DBS CUDA final scores forward for [T,K,V] or [B,T,K,V]");
    m.def("final_scores_forward_cuda_exact_kernel", &final_scores_forward_cuda_exact_kernel, "DBS exact custom CUDA kernel final scores forward");
    m.def("final_scores_forward_cuda_aten_topk", &final_scores_forward_cuda_aten_topk, "DBS ATen topk CUDA final scores forward");
    m.def("cuda_available", []() { return dbs_cuda_available() != 0; });
}
