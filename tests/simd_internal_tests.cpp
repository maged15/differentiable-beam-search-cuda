#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "../src/differentiable_beam.cpp"

static void assert_close(float actual, float expected, float atol, float rtol) {
    const float diff = std::fabs(actual - expected);
    const float scale = std::max(1.0f, std::fabs(expected));
    if (!(diff <= atol + rtol * scale)) {
        std::cerr << "mismatch actual=" << actual << " expected=" << expected
                  << " diff=" << diff << "\n";
        std::abort();
    }
}

static void soft_topk_scalar_reference(
    const std::vector<float>& scores,
    std::vector<float>& out,
    int target_k,
    float temperature,
    float tolerance,
    int max_iters
) {
    constexpr float NEG_GUARD = -1.0e30f;
    std::fill(out.begin(), out.end(), 0.0f);

    int active = 0;
    float min_s = std::numeric_limits<float>::infinity();
    float max_s = -std::numeric_limits<float>::infinity();
    for (float score : scores) {
        if (score > NEG_GUARD) {
            ++active;
            min_s = std::min(min_s, score);
            max_s = std::max(max_s, score);
        }
    }
    if (active == 0 || target_k <= 0) return;
    if (target_k >= active) {
        for (size_t i = 0; i < scores.size(); ++i) {
            if (scores[i] > NEG_GUARD) out[i] = 1.0f;
        }
        return;
    }

    float lo = min_s - 80.0f * temperature;
    float hi = max_s + 80.0f * temperature;
    for (int it = 0; it < max_iters; ++it) {
        const float mid = 0.5f * (lo + hi);
        const float s = dbs::sum_sigmoid_shifted_scalar(scores.data(), static_cast<int>(scores.size()), mid, temperature);
        const float err = s - static_cast<float>(target_k);
        if (std::fabs(err) <= tolerance || std::fabs(hi - lo) <= tolerance * std::max(1.0f, std::fabs(mid))) {
            lo = hi = mid;
            break;
        }
        if (err > 0.0f) lo = mid;
        else hi = mid;
    }

    const float theta = 0.5f * (lo + hi);
    dbs::soft_topk_write_scalar(scores.data(), out.data(), static_cast<int>(scores.size()), theta, temperature);
}

#if DBS_CAN_COMPILE_AVX512
DBS_AVX512_TARGET static void test_avx512_exp512_ps_matches_scalar_exp() {
    alignas(64) float input[16] = {
        -12.0f, -8.0f, -4.0f, -2.0f, -1.0f, -0.5f, -0.125f, 0.0f,
         0.125f, 0.5f, 1.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f,
    };
    alignas(64) float output[16] = {};
    const __m512 y = dbs::avx512::exp512_ps(_mm512_load_ps(input));
    _mm512_store_ps(output, y);

    for (int i = 0; i < 16; ++i) {
        assert_close(output[i], dbs::safe_exp_scalar(input[i]), 2.5e-4f, 2.5e-4f);
    }
}

DBS_AVX512_TARGET static void test_avx512_selected_softmax_matches_scalar() {
    std::vector<float> scores = {
        -4.0f, -1.5f, -0.25f, 0.0f, 0.125f, 0.5f, 1.0f, 2.0f,
        -1.0e31f, -3.0f, 3.5f, -2.25f, 0.75f, -0.75f, 1.5f, -5.0f,
        2.75f, -1.25f, 0.25f,
    };
    std::vector<float> scalar(scores.size(), 0.0f);
    std::vector<float> simd(scores.size(), 0.0f);
    dbs::softmax_selected_scalar(scores.data(), scalar.data(), static_cast<int>(scores.size()), 0.7f);
    dbs::avx512::softmax_selected(scores.data(), simd.data(), static_cast<int>(scores.size()), 0.7f);

    float scalar_sum = 0.0f;
    float simd_sum = 0.0f;
    for (size_t i = 0; i < scores.size(); ++i) {
        assert_close(simd[i], scalar[i], 8.0e-4f, 8.0e-4f);
        scalar_sum += scalar[i];
        simd_sum += simd[i];
    }
    assert_close(simd_sum, scalar_sum, 1.0e-4f, 1.0e-4f);
}

DBS_AVX512_TARGET static void soft_topk_avx512_reference(
    const std::vector<float>& scores,
    std::vector<float>& out,
    int target_k,
    float temperature,
    float tolerance,
    int max_iters
) {
    constexpr float NEG_GUARD = -1.0e30f;
    std::fill(out.begin(), out.end(), 0.0f);

    int active = 0;
    float min_s = std::numeric_limits<float>::infinity();
    float max_s = -std::numeric_limits<float>::infinity();
    for (float score : scores) {
        if (score > NEG_GUARD) {
            ++active;
            min_s = std::min(min_s, score);
            max_s = std::max(max_s, score);
        }
    }
    if (active == 0 || target_k <= 0) return;
    if (target_k >= active) {
        for (size_t i = 0; i < scores.size(); ++i) {
            if (scores[i] > NEG_GUARD) out[i] = 1.0f;
        }
        return;
    }

    float lo = min_s - 80.0f * temperature;
    float hi = max_s + 80.0f * temperature;
    for (int it = 0; it < max_iters; ++it) {
        const float mid = 0.5f * (lo + hi);
        const float s = dbs::avx512::sum_sigmoid_shifted(scores.data(), static_cast<int>(scores.size()), mid, temperature);
        const float err = s - static_cast<float>(target_k);
        if (std::fabs(err) <= tolerance || std::fabs(hi - lo) <= tolerance * std::max(1.0f, std::fabs(mid))) {
            lo = hi = mid;
            break;
        }
        if (err > 0.0f) lo = mid;
        else hi = mid;
    }

    const float theta = 0.5f * (lo + hi);
    dbs::avx512::soft_topk_write(scores.data(), out.data(), static_cast<int>(scores.size()), theta, temperature);
}

DBS_AVX512_TARGET static void test_avx512_relaxed_topk_matches_scalar() {
    std::vector<float> scores = {
        3.0f, 2.75f, 2.0f, 1.25f, 0.5f, 0.25f, -0.25f, -0.75f,
        -1.0f, -1.5f, -2.0f, -3.5f, -1.0e31f, 1.0f, 1.5f, -4.0f,
        0.0f, 2.25f, -2.5f, 0.875f,
    };
    std::vector<float> scalar(scores.size(), 0.0f);
    std::vector<float> simd(scores.size(), 0.0f);
    soft_topk_scalar_reference(scores, scalar, 5, 0.35f, 1.0e-5f, 80);
    soft_topk_avx512_reference(scores, simd, 5, 0.35f, 1.0e-5f, 80);

    float scalar_sum = 0.0f;
    float simd_sum = 0.0f;
    for (size_t i = 0; i < scores.size(); ++i) {
        assert_close(simd[i], scalar[i], 1.5e-3f, 1.5e-3f);
        scalar_sum += scalar[i];
        simd_sum += simd[i];
    }
    assert_close(simd_sum, scalar_sum, 3.0e-3f, 3.0e-3f);
}
#endif

int main() {
#if DBS_CAN_COMPILE_AVX2
    assert(DBS_CAN_COMPILE_AVX2 == 1);
#endif
#if DBS_CAN_COMPILE_SSE42
    assert(DBS_CAN_COMPILE_SSE42 == 1);
#endif

#if DBS_CAN_COMPILE_AVX512
    if (dbs::runtime_has_avx512()) {
        test_avx512_exp512_ps_matches_scalar_exp();
        test_avx512_selected_softmax_matches_scalar();
        test_avx512_relaxed_topk_matches_scalar();
    }
#endif

    std::cout << "dbs_simd_internal_tests passed\n";
    return 0;
}
