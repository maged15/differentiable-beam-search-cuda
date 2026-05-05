// SPDX-License-Identifier: MIT
#include <torch/extension.h>
#include "dbs.h"

#include <stdexcept>
#include <string>
#include <vector>
#include <limits>
#include <cmath>
#include <cstring>



static void validate_finite(double value, const char* name) {
    TORCH_CHECK(std::isfinite(value), name, " must be finite");
}

static void validate_int_bound(int64_t value, const char* name) {
    TORCH_CHECK(value > 0, name, " must be positive");
    TORCH_CHECK(value <= static_cast<int64_t>(std::numeric_limits<int>::max()), name, " exceeds INT_MAX");
}

static void validate_cpu_log_probs_3d(const torch::Tensor& log_probs, int64_t beam_size) {
    TORCH_CHECK(log_probs.device().is_cpu(), "dbs extension currently supports CPU tensors");
    TORCH_CHECK(log_probs.scalar_type() == torch::kFloat32, "log_probs must be float32");
    TORCH_CHECK(log_probs.dim() == 3, "log_probs must have shape [T, K, V]");
    TORCH_CHECK(beam_size > 0 && beam_size <= static_cast<int64_t>(std::numeric_limits<int>::max()), "beam_size must be in [1, INT_MAX]");
    validate_int_bound(log_probs.size(0), "T");
    validate_int_bound(log_probs.size(1), "K");
    validate_int_bound(log_probs.size(2), "V");
    TORCH_CHECK(log_probs.size(1) == beam_size, "beam_size must equal log_probs.size(1)");
}

static DBSOptionsC make_options(
    int64_t beam_size,
    int64_t eos_token,
    double selected_temperature,
    double soft_topk_temperature,
    int64_t relaxed_pool_multiplier,
    int64_t vocab_block,
    double length_penalty_alpha,
    double soft_topk_tolerance,
    int64_t soft_topk_max_iters,
    int64_t min_length,
    int64_t validate_inputs,
    int64_t max_dense_gradient_elements) {
    TORCH_CHECK(beam_size > 0 && beam_size <= static_cast<int64_t>(std::numeric_limits<int>::max()), "beam_size must be in [1, INT_MAX]");
    TORCH_CHECK(eos_token >= -1 && eos_token <= static_cast<int64_t>(std::numeric_limits<int>::max()), "eos_token must be -1 or fit in int");
    TORCH_CHECK(relaxed_pool_multiplier > 0 && relaxed_pool_multiplier <= static_cast<int64_t>(std::numeric_limits<int>::max()), "relaxed_pool_multiplier must fit in int");
    TORCH_CHECK(vocab_block > 0 && vocab_block <= static_cast<int64_t>(std::numeric_limits<int>::max()), "vocab_block must fit in int");
    TORCH_CHECK(soft_topk_max_iters > 0 && soft_topk_max_iters <= static_cast<int64_t>(std::numeric_limits<int>::max()), "soft_topk_max_iters must fit in int");
    TORCH_CHECK(min_length >= 0 && min_length <= static_cast<int64_t>(std::numeric_limits<int>::max()), "min_length must fit in int");
    TORCH_CHECK(validate_inputs == 0 || validate_inputs == 1, "validate_inputs must be 0 or 1");
    TORCH_CHECK(max_dense_gradient_elements >= 0 && max_dense_gradient_elements <= static_cast<int64_t>(std::numeric_limits<int>::max()), "max_dense_gradient_elements must fit in int");
    validate_finite(selected_temperature, "selected_temperature");
    validate_finite(soft_topk_temperature, "soft_topk_temperature");
    validate_finite(length_penalty_alpha, "length_penalty_alpha");
    validate_finite(soft_topk_tolerance, "soft_topk_tolerance");
    TORCH_CHECK(selected_temperature > 0.0, "selected_temperature must be positive");
    TORCH_CHECK(soft_topk_temperature > 0.0, "soft_topk_temperature must be positive");
    TORCH_CHECK(soft_topk_tolerance > 0.0, "soft_topk_tolerance must be positive");
    DBSOptionsC opt{};
    opt.beam_size = static_cast<int>(beam_size);
    opt.eos_token = static_cast<int>(eos_token);
    opt.selected_temperature = static_cast<float>(selected_temperature);
    opt.soft_topk_temperature = static_cast<float>(soft_topk_temperature);
    opt.relaxed_pool_multiplier = static_cast<int>(relaxed_pool_multiplier);
    opt.vocab_block = static_cast<int>(vocab_block);
    opt.length_penalty_alpha = static_cast<float>(length_penalty_alpha);
    opt.soft_topk_tolerance = static_cast<float>(soft_topk_tolerance);
    opt.soft_topk_max_iters = static_cast<int>(soft_topk_max_iters);
    opt.min_length = static_cast<int>(min_length);
    opt.validate_inputs = static_cast<int>(validate_inputs);
    opt.max_dense_gradient_elements = static_cast<int>(max_dense_gradient_elements);
    return opt;
}

static void throw_if_failed(DBSDecoderHandle* h, int rc, const char* what) {
    if (rc == 0) return;
    const char* msg = h ? dbs_last_error(h) : dbs_last_global_error();
    std::string text = std::string(what) + " failed";
    if (msg && *msg) text += std::string(": ") + msg;
    throw std::runtime_error(text);
}

torch::Tensor final_scores_forward(
    torch::Tensor log_probs,
    int64_t beam_size,
    int64_t eos_token,
    double selected_temperature,
    double soft_topk_temperature,
    int64_t relaxed_pool_multiplier,
    int64_t vocab_block,
    double length_penalty_alpha,
    double soft_topk_tolerance,
    int64_t soft_topk_max_iters,
    int64_t min_length,
    int64_t validate_inputs,
    int64_t max_dense_gradient_elements) {
    validate_cpu_log_probs_3d(log_probs, beam_size);

    auto x = log_probs.contiguous();
    const int T = static_cast<int>(x.size(0));
    const int V = static_cast<int>(x.size(2));

    DBSOptionsC opt = make_options(beam_size, eos_token, selected_temperature, soft_topk_temperature,
                                   relaxed_pool_multiplier, vocab_block, length_penalty_alpha,
                                   soft_topk_tolerance, soft_topk_max_iters, min_length,
                                   validate_inputs, max_dense_gradient_elements);
    DBSDecoderHandle* h = nullptr;
    throw_if_failed(nullptr, dbs_create_ex(opt, &h), "dbs_create_ex");

    DBSResultHandle* r = nullptr;
    try {
        throw_if_failed(h, dbs_decode(h, x.data_ptr<float>(), T, V, &r), "dbs_decode");
        auto out = torch::empty({beam_size}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        const float* scores = dbs_result_final_scores(r);
        TORCH_CHECK(scores != nullptr, "dbs_result_final_scores returned null");
        std::memcpy(out.data_ptr<float>(), scores, static_cast<size_t>(beam_size) * sizeof(float));
        dbs_free_result(r);
        dbs_destroy(h);
        return out;
    } catch (...) {
        if (r) dbs_free_result(r);
        dbs_destroy(h);
        throw;
    }
}

torch::Tensor final_scores_backward(
    torch::Tensor log_probs,
    torch::Tensor grad_final_scores,
    int64_t beam_size,
    int64_t eos_token,
    double selected_temperature,
    double soft_topk_temperature,
    int64_t relaxed_pool_multiplier,
    int64_t vocab_block,
    double length_penalty_alpha,
    double soft_topk_tolerance,
    int64_t soft_topk_max_iters,
    int64_t min_length,
    int64_t validate_inputs,
    int64_t max_dense_gradient_elements) {
    validate_cpu_log_probs_3d(log_probs, beam_size);
    TORCH_CHECK(grad_final_scores.device().is_cpu(), "grad_final_scores must be CPU");
    TORCH_CHECK(grad_final_scores.scalar_type() == torch::kFloat32, "grad_final_scores must be float32");
    TORCH_CHECK(grad_final_scores.dim() == 1, "grad_final_scores must have shape [K]");
    TORCH_CHECK(grad_final_scores.size(0) == beam_size, "grad_final_scores size must equal beam_size");
    validate_int_bound(grad_final_scores.size(0), "grad_final_scores.K");

    auto x = log_probs.contiguous();
    auto g = grad_final_scores.contiguous();
    const int T = static_cast<int>(x.size(0));
    const int V = static_cast<int>(x.size(2));

    DBSOptionsC opt = make_options(beam_size, eos_token, selected_temperature, soft_topk_temperature,
                                   relaxed_pool_multiplier, vocab_block, length_penalty_alpha,
                                   soft_topk_tolerance, soft_topk_max_iters, min_length,
                                   validate_inputs, max_dense_gradient_elements);
    DBSDecoderHandle* h = nullptr;
    throw_if_failed(nullptr, dbs_create_ex(opt, &h), "dbs_create_ex");
    DBSResultHandle* r = nullptr;
    DBSBackwardHandle* b = nullptr;
    try {
        throw_if_failed(h, dbs_decode(h, x.data_ptr<float>(), T, V, &r), "dbs_decode");
        throw_if_failed(h, dbs_backward(h, r, nullptr, nullptr, g.data_ptr<float>(), &b), "dbs_backward");
        auto grad = torch::zeros_like(x).contiguous().view({-1});
        const int64_t n = dbs_backward_sparse_logprob_count(b);
        const int64_t* idx = dbs_backward_sparse_logprob_indices(b);
        const float* val = dbs_backward_sparse_logprob_values(b);
        TORCH_CHECK(n >= 0, "sparse gradient count must be non-negative");
        TORCH_CHECK((n == 0) || (idx != nullptr && val != nullptr), "sparse gradient buffers are null");
        auto* gp = grad.data_ptr<float>();
        const int64_t grad_numel = grad.numel();
        for (int64_t i = 0; i < n; ++i) {
            TORCH_CHECK(idx[i] >= 0 && idx[i] < grad_numel, "sparse gradient index out of bounds");
            gp[idx[i]] += val[i];
        }
        dbs_free_backward(b);
        dbs_free_result(r);
        dbs_destroy(h);
        return grad.view_as(x);
    } catch (...) {
        if (b) dbs_free_backward(b);
        if (r) dbs_free_result(r);
        dbs_destroy(h);
        throw;
    }
}


TORCH_LIBRARY(dbs, m) {
    m.def("final_scores_forward(Tensor log_probs, int beam_size, int eos_token, float selected_temperature, float soft_topk_temperature, int relaxed_pool_multiplier, int vocab_block, float length_penalty_alpha, float soft_topk_tolerance, int soft_topk_max_iters, int min_length, int validate_inputs, int max_dense_gradient_elements) -> Tensor");
    m.def("final_scores_backward(Tensor log_probs, Tensor grad_final_scores, int beam_size, int eos_token, float selected_temperature, float soft_topk_temperature, int relaxed_pool_multiplier, int vocab_block, float length_penalty_alpha, float soft_topk_tolerance, int soft_topk_max_iters, int min_length, int validate_inputs, int max_dense_gradient_elements) -> Tensor");
}

TORCH_LIBRARY_IMPL(dbs, CPU, m) {
    m.impl("final_scores_forward", &final_scores_forward);
    m.impl("final_scores_backward", &final_scores_backward);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("final_scores_forward", &final_scores_forward, "DBS final scores forward (CPU)");
    m.def("final_scores_backward", &final_scores_backward, "DBS final scores backward via sparse surrogate (CPU)");
    m.def("has_torch_ops", []() { return true; });
}
