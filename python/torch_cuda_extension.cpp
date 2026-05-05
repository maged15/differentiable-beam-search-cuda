// SPDX-License-Identifier: MIT
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "dbs_cuda.h"

#include <cstdlib>
#include <limits>
#include <cmath>
#include <tuple>
#include <vector>

namespace py = pybind11;

static void validate_int_bound(int64_t value, const char* name) {
    TORCH_CHECK(value > 0, name, " must be positive");
    TORCH_CHECK(value <= static_cast<int64_t>(std::numeric_limits<int>::max()), name, " exceeds INT_MAX");
}

static bool validate_cuda_log_probs_public(const torch::Tensor& log_probs, int64_t beam_size) {
    TORCH_CHECK(log_probs.is_cuda(), "log_probs must be CUDA");
    TORCH_CHECK(log_probs.dtype() == torch::kFloat32, "CUDA op currently requires float32");
    TORCH_CHECK(log_probs.dim() == 3 || log_probs.dim() == 4, "expected [T,K,V] or [B,T,K,V]");
    TORCH_CHECK(beam_size > 0 && beam_size <= static_cast<int64_t>(std::numeric_limits<int>::max()), "beam_size must be in [1, INT_MAX]");
    TORCH_CHECK(beam_size <= DBS_CUDA_MAX_BEAM, "beam_size exceeds CUDA backend maximum");

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

static void validate_min_length_bound(int64_t min_length) {
    TORCH_CHECK(min_length >= 0 && min_length <= static_cast<int64_t>(std::numeric_limits<int>::max()), "min_length must be in [0, INT_MAX]");
}

static void validate_cuda_log_prob_values(const torch::Tensor& log_probs) {
    TORCH_CHECK(!log_probs.isnan().any().item<bool>(), "log_probs contains NaN or +Inf");
    TORCH_CHECK(!log_probs.eq(std::numeric_limits<float>::infinity()).any().item<bool>(), "log_probs contains NaN or +Inf");
}

static std::tuple<torch::Tensor, torch::Tensor> decode_forward_cuda_exact_kernel(
    torch::Tensor log_probs,
    int64_t beam_size,
    int64_t eos_token,
    int64_t min_length,
    int64_t validate_inputs) {
    validate_eos_token_bound(eos_token);
    validate_min_length_bound(min_length);
    TORCH_CHECK(validate_inputs == 0 || validate_inputs == 1, "validate_inputs must be 0 or 1");
    const bool unbatched = validate_cuda_log_probs_public(log_probs, beam_size);
    c10::cuda::CUDAGuard device_guard(log_probs.device());
    if (validate_inputs != 0) {
        validate_cuda_log_prob_values(log_probs);
    }

    auto x4 = unbatched ? log_probs.unsqueeze(0) : log_probs;
    auto x = x4.contiguous();
    const int B = static_cast<int>(x.size(0));
    const int T = static_cast<int>(x.size(1));
    const int K = static_cast<int>(x.size(2));
    const int V = static_cast<int>(x.size(3));

    auto tokens = torch::empty({B, T, K}, torch::TensorOptions().device(log_probs.device()).dtype(torch::kInt32));
    auto scores = torch::empty({B, K}, log_probs.options());
    const auto stream = at::cuda::getCurrentCUDAStream(log_probs.get_device());
    const int rc = dbs_cuda_decode_forward_fast_ex(
        x.data_ptr<float>(), B, T, K, V, static_cast<int>(eos_token), static_cast<int>(min_length),
        tokens.data_ptr<int32_t>(), scores.data_ptr<float>(), stream.stream());
    TORCH_CHECK(rc == DBS_CUDA_STATUS_OK, dbs_cuda_status_string(rc));
    return std::make_tuple(unbatched ? tokens.squeeze(0) : tokens, unbatched ? scores.squeeze(0) : scores);
}

static torch::Tensor final_scores_forward_cuda_exact_kernel(
    torch::Tensor log_probs,
    int64_t beam_size,
    int64_t eos_token,
    int64_t min_length,
    int64_t validate_inputs) {
    return std::get<1>(decode_forward_cuda_exact_kernel(log_probs, beam_size, eos_token, min_length, validate_inputs));
}

static torch::Tensor final_scores_forward_cuda_aten_topk(torch::Tensor log_probs, int64_t beam_size, int64_t validate_inputs) {
    const bool unbatched = validate_cuda_log_probs_public(log_probs, beam_size);
    TORCH_CHECK(validate_inputs == 0 || validate_inputs == 1, "validate_inputs must be 0 or 1");
    c10::cuda::CUDAGuard device_guard(log_probs.device());
    if (validate_inputs != 0) {
        validate_cuda_log_prob_values(log_probs);
    }

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

static torch::Tensor final_scores_forward_cuda(
    torch::Tensor log_probs,
    int64_t beam_size,
    int64_t eos_token,
    int64_t min_length,
    int64_t validate_inputs) {
    validate_eos_token_bound(eos_token);
    validate_min_length_bound(min_length);
    TORCH_CHECK(validate_inputs == 0 || validate_inputs == 1, "validate_inputs must be 0 or 1");
    validate_cuda_log_probs_public(log_probs, beam_size);
    c10::cuda::CUDAGuard device_guard(log_probs.device());
    if (validate_inputs != 0) {
        validate_cuda_log_prob_values(log_probs);
    }

    // The exact custom kernel is the default because it preserves the C ABI decoder's
    // deterministic rank-aligned beam semantics and uses PyTorch's current CUDA stream.
    // The ATen topk path is an opt-in score-only fast path for no-EOS, rank-aligned
    // log-prob tensors. Enable it explicitly with DBS_ENABLE_SCORE_ONLY_FAST_PATH=1.
    const char* enable_fast = std::getenv("DBS_ENABLE_SCORE_ONLY_FAST_PATH");
    const char* legacy_fast = std::getenv("DBS_ASSUME_RANK_ALIGNED_LOG_PROBS");
    const char* force_exact = std::getenv("DBS_FORCE_EXACT_CUDA_KERNEL");
    if (eos_token < 0 && !force_exact && ((enable_fast && *enable_fast == '1') || (legacy_fast && *legacy_fast == '1'))) {
        TORCH_WARN_ONCE("DBS_ENABLE_SCORE_ONLY_FAST_PATH enables the score-only ATen helper; it does not expose parent/token traces and is not a semantic substitute for exact decode.");
        return final_scores_forward_cuda_aten_topk(log_probs, beam_size, 0);
    }

    return final_scores_forward_cuda_exact_kernel(log_probs, beam_size, eos_token, min_length, 0);
}

static torch::Tensor test_sparse_backward_scatter(torch::Tensor indices, torch::Tensor values, int64_t grad_count) {
    TORCH_CHECK(indices.is_cuda(), "indices must be CUDA");
    TORCH_CHECK(values.is_cuda(), "values must be CUDA");
    TORCH_CHECK(indices.device() == values.device(), "indices and values must be on the same CUDA device");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(values.dtype() == torch::kFloat32, "values must be float32");
    TORCH_CHECK(indices.numel() == values.numel(), "indices and values must have the same number of elements");
    TORCH_CHECK(grad_count > 0, "grad_count must be positive");

    c10::cuda::CUDAGuard device_guard(values.device());
    auto idx = indices.contiguous();
    auto val = values.contiguous();
    auto grad = torch::zeros({grad_count}, values.options());
    const auto stream = at::cuda::getCurrentCUDAStream(values.get_device());
    const int rc = dbs_cuda_sparse_backward_scatter(
        idx.data_ptr<int64_t>(),
        val.data_ptr<float>(),
        static_cast<int64_t>(idx.numel()),
        grad.data_ptr<float>(),
        grad_count,
        stream.stream());
    TORCH_CHECK(rc == DBS_CUDA_STATUS_OK, dbs_cuda_status_string(rc));
    return grad;
}

static std::tuple<torch::Tensor, torch::Tensor> test_decode_forward_fast_ex(
    torch::Tensor log_probs,
    int64_t beam_size,
    int64_t eos_token,
    int64_t min_length) {
    validate_eos_token_bound(eos_token);
    return decode_forward_cuda_exact_kernel(log_probs, beam_size, eos_token, min_length, 1);
}

static void validate_cuda_i32_vector(const torch::Tensor& tensor, int64_t expected, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.dtype() == torch::kInt32, name, " must be int32");
    TORCH_CHECK(tensor.numel() == expected, name, " must have one element per batch item");
}

static std::tuple<torch::Tensor, torch::Tensor> test_decode_forward_variable(
    torch::Tensor log_probs,
    torch::Tensor steps_per_example,
    torch::Tensor beam_sizes_per_example,
    torch::Tensor eos_tokens_per_example,
    torch::Tensor min_lengths_per_example) {
    TORCH_CHECK(log_probs.is_cuda(), "log_probs must be CUDA");
    TORCH_CHECK(log_probs.dtype() == torch::kFloat32, "log_probs must be float32");
    TORCH_CHECK(log_probs.dim() == 4, "log_probs must have shape [B,T,K,V]");
    validate_int_bound(log_probs.size(0), "B");
    validate_int_bound(log_probs.size(1), "T");
    validate_int_bound(log_probs.size(2), "K");
    validate_int_bound(log_probs.size(3), "V");
    TORCH_CHECK(log_probs.size(2) <= DBS_CUDA_MAX_BEAM, "K exceeds CUDA backend maximum");

    const int64_t B64 = log_probs.size(0);
    validate_cuda_i32_vector(steps_per_example, B64, "steps_per_example");
    validate_cuda_i32_vector(beam_sizes_per_example, B64, "beam_sizes_per_example");
    validate_cuda_i32_vector(eos_tokens_per_example, B64, "eos_tokens_per_example");
    validate_cuda_i32_vector(min_lengths_per_example, B64, "min_lengths_per_example");
    TORCH_CHECK(
        steps_per_example.device() == log_probs.device() &&
        beam_sizes_per_example.device() == log_probs.device() &&
        eos_tokens_per_example.device() == log_probs.device() &&
        min_lengths_per_example.device() == log_probs.device(),
        "variable metadata tensors must be on log_probs.device()");

    c10::cuda::CUDAGuard device_guard(log_probs.device());
    auto x = log_probs.contiguous();
    auto steps = steps_per_example.contiguous();
    auto beams = beam_sizes_per_example.contiguous();
    auto eos = eos_tokens_per_example.contiguous();
    auto mins = min_lengths_per_example.contiguous();
    const int B = static_cast<int>(x.size(0));
    const int T = static_cast<int>(x.size(1));
    const int K = static_cast<int>(x.size(2));
    const int V = static_cast<int>(x.size(3));

    auto tokens = torch::empty({B, T, K}, torch::TensorOptions().device(log_probs.device()).dtype(torch::kInt32));
    auto scores = torch::empty({B, K}, log_probs.options());
    const auto stream = at::cuda::getCurrentCUDAStream(log_probs.get_device());
    const int rc = dbs_cuda_decode_forward_variable(
        x.data_ptr<float>(), B, T, K, V,
        steps.data_ptr<int32_t>(),
        beams.data_ptr<int32_t>(),
        eos.data_ptr<int32_t>(),
        mins.data_ptr<int32_t>(),
        tokens.data_ptr<int32_t>(),
        scores.data_ptr<float>(),
        stream.stream());
    TORCH_CHECK(rc == DBS_CUDA_STATUS_OK, dbs_cuda_status_string(rc));
    return std::make_tuple(tokens, scores);
}

static int test_decode_forward_status(torch::Tensor log_probs, int64_t beam_size, int64_t eos_token, bool fast_path) {
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
    if (fast_path) {
        return dbs_cuda_decode_forward_fast_ex(
            x.data_ptr<float>(), B, T, K, V, static_cast<int>(eos_token), 0,
            tokens.data_ptr<int32_t>(), scores.data_ptr<float>(), stream.stream());
    }
    return dbs_cuda_decode_forward(
        x.data_ptr<float>(), B, T, K, V, static_cast<int>(eos_token),
        tokens.data_ptr<int32_t>(), scores.data_ptr<float>(), stream.stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("final_scores_forward_cuda", &final_scores_forward_cuda, "DBS CUDA final scores forward for [T,K,V] or [B,T,K,V]", py::arg("log_probs"), py::arg("beam_size"), py::arg("eos_token"), py::arg("min_length") = 0, py::arg("validate_inputs") = 1);
    m.def("decode_forward_cuda", &decode_forward_cuda_exact_kernel, "DBS CUDA decode forward returning tokens and final scores", py::arg("log_probs"), py::arg("beam_size"), py::arg("eos_token"), py::arg("min_length") = 0, py::arg("validate_inputs") = 1);
    m.def("final_scores_forward_cuda_exact_kernel", &final_scores_forward_cuda_exact_kernel, "DBS exact custom CUDA kernel final scores forward", py::arg("log_probs"), py::arg("beam_size"), py::arg("eos_token"), py::arg("min_length") = 0, py::arg("validate_inputs") = 1);
    m.def("final_scores_forward_cuda_aten_topk", &final_scores_forward_cuda_aten_topk, "DBS score-only ATen topk CUDA helper", py::arg("log_probs"), py::arg("beam_size"), py::arg("validate_inputs") = 1);
    m.def("_test_sparse_backward_scatter", &test_sparse_backward_scatter, "Test-only wrapper for DBS CUDA sparse backward scatter");
    m.def("_test_decode_forward_fast_ex", &test_decode_forward_fast_ex, "Test-only wrapper for DBS CUDA fast decode with min_length");
    m.def("_test_decode_forward_variable", &test_decode_forward_variable, "Test-only wrapper for DBS CUDA variable decode");
    m.def("_test_decode_forward_status", &test_decode_forward_status, "Test-only direct CUDA decode status wrapper");
    m.def("cuda_available", []() { return dbs_cuda_available() != 0; });
}
