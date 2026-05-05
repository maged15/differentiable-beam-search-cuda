// SPDX-License-Identifier: MIT
// Hardened differentiable beam search decoder.
// Hard beam output with deterministic tie-breaking, sigmoid-bisection k-hot surrogate gradients,
// runtime-dispatched AVX-512 fast path, scalar fallback, GNMT length penalty, EOS handling,
// sparse-first backward C ABI, batched decode entry points, constraint hooks, and demo timing instrumentation.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#if defined(_MSC_VER)
#include <malloc.h>
#endif
#include <cstring>
#include <exception>
#include <atomic>
#include <mutex>
#include <thread>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "dbs_internal_test_hooks.h"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#define DBS_X86 1
#else
#define DBS_X86 0
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define DBS_ARM_NEON 1
#else
#define DBS_ARM_NEON 0
#endif

#if DBS_X86 && (defined(__GNUC__) || defined(__clang__))
#define DBS_CAN_COMPILE_AVX512 1
#define DBS_AVX512_TARGET __attribute__((target("avx512f,fma")))
#else
#define DBS_CAN_COMPILE_AVX512 0
#define DBS_AVX512_TARGET
#endif

#if DBS_X86 && (defined(__GNUC__) || defined(__clang__))
#define DBS_CAN_COMPILE_AVX2 1
#define DBS_AVX2_TARGET __attribute__((target("avx2,fma")))
#define DBS_CAN_COMPILE_SSE42 1
#define DBS_SSE42_TARGET __attribute__((target("sse4.2")))
#else
#define DBS_CAN_COMPILE_AVX2 0
#define DBS_AVX2_TARGET
#define DBS_CAN_COMPILE_SSE42 0
#define DBS_SSE42_TARGET
#endif

#ifndef DBS_VERSION_MAJOR
#define DBS_VERSION_MAJOR 1
#define DBS_VERSION_MINOR 0
#define DBS_VERSION_PATCH 0
#endif

#define DBS_ABI_VERSION 10

#if defined(_WIN32) && defined(DBS_STATIC)
#define DBS_EXPORT
#elif defined(_WIN32) && defined(DBS_BUILD_SHARED) && defined(DBS_COMPILING_LIBRARY)
#define DBS_EXPORT __declspec(dllexport)
#elif defined(_WIN32) && defined(DBS_BUILD_SHARED)
#define DBS_EXPORT __declspec(dllimport)
#elif defined(_WIN32)
#define DBS_EXPORT
#elif defined(__GNUC__) || defined(__clang__)
#define DBS_EXPORT __attribute__((visibility("default")))
#else
#define DBS_EXPORT
#endif

namespace dbs {

static std::atomic<int64_t> g_allocator_calls{0};
static std::atomic<int64_t> g_allocator_bytes{0};
static thread_local uint64_t g_deterministic_seed_tls = 0;

using TokenFilterFn = int (*)(
    void* user_data,
    int batch_index,
    int step,
    int parent_beam,
    const int32_t* prefix_tokens,
    int prefix_len,
    int token
);

template <class T, std::size_t Alignment = 64>
class AlignedAllocator {
public:
    using value_type = T;

    AlignedAllocator() noexcept = default;

    template <class U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;

        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }

        const std::size_t bytes = n * sizeof(T);
        void* p = nullptr;

#if defined(_MSC_VER)
        p = _aligned_malloc(bytes, Alignment);
        if (!p) throw std::bad_alloc();
#else
        if (posix_memalign(&p, Alignment, bytes) != 0) {
            throw std::bad_alloc();
        }
#endif

        g_allocator_calls.fetch_add(1, std::memory_order_relaxed);
        g_allocator_bytes.fetch_add(static_cast<int64_t>(bytes), std::memory_order_relaxed);
        return static_cast<T*>(p);
    }

    void deallocate(T* p, std::size_t n) noexcept {
        g_allocator_bytes.fetch_sub(static_cast<int64_t>(n * sizeof(T)), std::memory_order_relaxed);
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        std::free(p);
#endif
    }

    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

template <class T, class U, std::size_t A>
bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) {
    return true;
}

template <class T, class U, std::size_t A>
bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) {
    return false;
}

using AlignedFloatVector = std::vector<float, AlignedAllocator<float, 64>>;
using AlignedIntVector = std::vector<int32_t, AlignedAllocator<int32_t, 64>>;
using AlignedInt64Vector = std::vector<int64_t, AlignedAllocator<int64_t, 64>>;

struct BeamOptions {
    int beam_size = 8;
    int eos_token = -1;

    float selected_temperature = 1.0f;
    float soft_topk_temperature = 0.25f;

    int relaxed_pool_multiplier = 8;
    int vocab_block = 4096;

    float length_penalty_alpha = 0.0f;
    float soft_topk_tolerance = 1.0e-4f;
    int soft_topk_max_iters = 48;

    int min_length = 0;
    int validate_inputs = 1;
    int max_dense_gradient_elements = 100000000;
};

struct DecodeConstraints {
    // banned_tokens is optional and has shape [V]. Non-zero means token is not allowed.
    const uint8_t* banned_tokens = nullptr;

    // forced_tokens is optional and has shape [T]. A value >= 0 forces that token at the step.
    const int32_t* forced_tokens = nullptr;

    // Optional per-call minimum output length. Negative means use BeamOptions::min_length.
    int min_length = -1;

    // Optional production constraints. Disabled when repetition_penalty <= 1 and no_repeat_ngram_size <= 0.
    float repetition_penalty = 1.0f;
    int no_repeat_ngram_size = 0;
    TokenFilterFn token_filter = nullptr;
    void* token_filter_user_data = nullptr;
    int batch_index = 0;
};

struct DecodeResult {
    int steps = 0;
    int beam_size = 0;
    int vocab_size = 0;
    int eos_token = -1;
    int relaxed_pool_size = 0;

    float selected_temperature = 1.0f;
    float soft_topk_temperature = 0.25f;
    float length_penalty_alpha = 0.0f;

    AlignedIntVector parents;       // [T * K]
    AlignedIntVector tokens;        // [T * K]
    AlignedIntVector lengths;       // [T * K]

    AlignedFloatVector raw_scores;  // [T * K]
    AlignedFloatVector scores;      // length-penalized ranking scores [T * K]
    AlignedFloatVector weights;     // selected-beam softmax weights [T * K]

    AlignedFloatVector final_raw_scores; // [K]
    AlignedFloatVector final_scores;     // [K]

    std::vector<uint8_t> from_logprob;   // [T * K]

    AlignedIntVector pool_parents;       // [T * P]
    AlignedIntVector pool_tokens;        // [T * P]
    AlignedIntVector pool_lengths;       // [T * P]

    AlignedFloatVector pool_raw_scores;  // [T * P]
    AlignedFloatVector pool_scores;      // [T * P]
    AlignedFloatVector relaxed_weights;  // [T * P]

    std::vector<uint8_t> pool_from_logprob; // [T * P]

    std::vector<std::vector<int32_t>> sequences;
};

struct BackwardResult {
    AlignedFloatVector grad_log_probs;      // [T * K * V], populated only by explicit dense backward.
    AlignedFloatVector grad_initial_scores; // [K]

    AlignedInt64Vector sparse_logprob_indices; // flattened [T*K*V] indices, populated by sparse/default backward.
    AlignedFloatVector sparse_logprob_values;  // same count as sparse_logprob_indices.
    bool sparse = false;
};

struct SparseGradEntry {
    int64_t index;
    float value;
};

struct Candidate {
    float score;
    float raw_score;
    int32_t parent;
    int32_t token;
    int32_t length;
    uint8_t from_logprob;
};

static inline float safe_exp_scalar(float x) {
    x = std::min(88.3762626647949f, std::max(-88.3762626647949f, x));
    return std::exp(x);
}

static inline float sigmoid_scalar(float x) {
    if (x >= 0.0f) {
        const float e = safe_exp_scalar(-x);
        return 1.0f / (1.0f + e);
    }

    const float e = safe_exp_scalar(x);
    return e / (1.0f + e);
}

static inline float gnmt_length_penalty(int length, float alpha) {
    if (alpha == 0.0f) return 1.0f;

    const int l = std::max(1, length);
    return std::pow((5.0f + static_cast<float>(l)) / 6.0f, alpha);
}

static size_t checked_mul_size(size_t a, size_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::overflow_error(what);
    }
    return a * b;
}

static inline bool candidate_better(const Candidate& a, const Candidate& b) noexcept {
    if (a.score != b.score) return a.score > b.score;
    if (a.raw_score != b.raw_score) return a.raw_score > b.raw_score;
    if (a.parent != b.parent) return a.parent < b.parent;
    if (a.token != b.token) return a.token < b.token;
    if (a.length != b.length) return a.length < b.length;
    return a.from_logprob > b.from_logprob;
}


static bool prefix_contains_token(const std::vector<int32_t>& prefix, int token) {
    return std::find(prefix.begin(), prefix.end(), token) != prefix.end();
}

static bool would_repeat_ngram(const std::vector<int32_t>& prefix, int token, int n) {
    if (n <= 0) return false;
    if (n == 1) return prefix_contains_token(prefix, token);
    if (static_cast<int>(prefix.size()) + 1 < n) return false;

    std::vector<int32_t> tail;
    tail.reserve(static_cast<size_t>(n));
    const int start = static_cast<int>(prefix.size()) - (n - 1);
    for (int i = start; i < static_cast<int>(prefix.size()); ++i) tail.push_back(prefix[static_cast<size_t>(i)]);
    tail.push_back(token);

    for (int i = 0; i + n <= static_cast<int>(prefix.size()); ++i) {
        bool same = true;
        for (int j = 0; j < n; ++j) {
            if (prefix[static_cast<size_t>(i + j)] != tail[static_cast<size_t>(j)]) {
                same = false;
                break;
            }
        }
        if (same) return true;
    }
    return false;
}

static bool token_allowed_by_advanced_constraints(
    const DecodeConstraints* constraints,
    int step,
    int parent,
    const std::vector<int32_t>& prefix,
    int token
) {
    if (!constraints) return true;
    if (constraints->no_repeat_ngram_size > 0 &&
        would_repeat_ngram(prefix, token, constraints->no_repeat_ngram_size)) {
        return false;
    }
    if (constraints->token_filter) {
        const int rc = constraints->token_filter(
            constraints->token_filter_user_data,
            constraints->batch_index,
            step,
            parent,
            prefix.empty() ? nullptr : prefix.data(),
            static_cast<int>(prefix.size()),
            token);
        if (rc == 0) return false;
    }
    return true;
}

static float apply_repetition_penalty(float lp, const DecodeConstraints* constraints, const std::vector<int32_t>& prefix, int token) {
    if (!constraints || !(constraints->repetition_penalty > 1.0f)) return lp;
    if (!prefix_contains_token(prefix, token)) return lp;
    return lp - std::log(constraints->repetition_penalty);
}

static inline void insert_topk(Candidate* top, int k, const Candidate& c) noexcept {
    if (!candidate_better(c, top[k - 1])) return;

    // Maintains a descending top-P buffer with deterministic tie-breaking.
    int lo = 0;
    int hi = k - 1;

    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);

        if (candidate_better(c, top[mid])) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    const int pos = lo;

    for (int i = k - 1; i > pos; --i) {
        top[i] = top[i - 1];
    }

    top[pos] = c;
}

static void validate_logprob_tensor(const float* x, int steps, int beam_size, int vocab_size) {
    if (!x) throw std::invalid_argument("log_probs cannot be null");
    const size_t st = static_cast<size_t>(steps);
    const size_t k = static_cast<size_t>(beam_size);
    const size_t v = static_cast<size_t>(vocab_size);
    const size_t count = checked_mul_size(checked_mul_size(st, k, "log-prob tensor size overflow"), v, "log-prob tensor size overflow");

    for (size_t i = 0; i < count; ++i) {
        const float value = x[i];
        if (std::isnan(value) || value == std::numeric_limits<float>::infinity()) {
            throw std::invalid_argument("log_probs contains NaN or +Inf");
        }
    }
}

static float dot_scalar(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

static void softmax_selected_scalar(
    const float* scores,
    float* out,
    int n,
    float temperature
) {
    constexpr float NEG_GUARD = -1.0e30f;

    std::fill(out, out + n, 0.0f);

    float maxv = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) {
        if (scores[i] > NEG_GUARD) maxv = std::max(maxv, scores[i] / temperature);
    }

    if (!std::isfinite(maxv)) return;

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (scores[i] > NEG_GUARD) {
            out[i] = safe_exp_scalar(scores[i] / temperature - maxv);
            sum += out[i];
        }
    }

    if (!(sum > 0.0f) || !std::isfinite(sum)) {
        std::fill(out, out + n, 0.0f);
        return;
    }

    const float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; ++i) out[i] *= inv_sum;
}

static float sum_sigmoid_shifted_scalar(
    const float* scores,
    int n,
    float theta,
    float temperature
) {
    constexpr float NEG_GUARD = -1.0e30f;

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (scores[i] > NEG_GUARD) {
            sum += sigmoid_scalar((scores[i] - theta) / temperature);
        }
    }
    return sum;
}

static void soft_topk_write_scalar(
    const float* scores,
    float* out,
    int n,
    float theta,
    float temperature
) {
    constexpr float NEG_GUARD = -1.0e30f;

    for (int i = 0; i < n; ++i) {
        if (scores[i] > NEG_GUARD) out[i] = sigmoid_scalar((scores[i] - theta) / temperature);
        else out[i] = 0.0f;
    }
}

static void scan_parent_row_scalar(
    const float* row,
    float parent_raw,
    int parent_length,
    int parent,
    int vocab_size,
    Candidate* top,
    int top_count,
    int vocab_block,
    float length_penalty_alpha,
    const uint8_t* banned_tokens,
    int forced_token,
    int eos_token,
    int min_length
) {
    const int new_len = parent_length + 1;
    const float inv_penalty = 1.0f / gnmt_length_penalty(new_len, length_penalty_alpha);

    for (int base = 0; base < vocab_size; base += vocab_block) {
        const int end = std::min(vocab_size, base + vocab_block);
        for (int v = base; v < end; ++v) {
            if (forced_token >= 0 && v != forced_token) continue;
            if (banned_tokens && banned_tokens[v]) continue;
            if (eos_token >= 0 && v == eos_token && new_len < min_length) continue;

            const float lp = row[v];
            if (!std::isfinite(lp)) continue;

            const float raw = parent_raw + lp;
            const float rank = raw * inv_penalty;
            insert_topk(top, top_count, Candidate{rank, raw, parent, v, new_len, 1});
        }
    }
}


static void scan_parent_row_scalar_advanced(
    const float* row,
    float parent_raw,
    int parent_length,
    int parent,
    int step,
    int vocab_size,
    Candidate* top,
    int top_count,
    int vocab_block,
    float length_penalty_alpha,
    const DecodeConstraints* constraints,
    const std::vector<int32_t>& prefix,
    int forced_token,
    int eos_token,
    int min_length
) {
    const int new_len = parent_length + 1;
    const float inv_penalty = 1.0f / gnmt_length_penalty(new_len, length_penalty_alpha);

    for (int base = 0; base < vocab_size; base += vocab_block) {
        const int end = std::min(vocab_size, base + vocab_block);
        for (int v = base; v < end; ++v) {
            if (forced_token >= 0 && v != forced_token) continue;
            if (constraints && constraints->banned_tokens && constraints->banned_tokens[v]) continue;
            if (eos_token >= 0 && v == eos_token && new_len < min_length) continue;
            if (!token_allowed_by_advanced_constraints(constraints, step, parent, prefix, v)) continue;

            float lp = row[v];
            if (!std::isfinite(lp)) continue;
            lp = apply_repetition_penalty(lp, constraints, prefix, v);

            const float raw = parent_raw + lp;
            const float rank = raw * inv_penalty;
            insert_topk(top, top_count, Candidate{rank, raw, parent, v, new_len, 1});
        }
    }
}

static bool runtime_has_avx512() noexcept {
#if DBS_CAN_COMPILE_AVX512
    static const bool supported = []() noexcept {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma");
    }();
    return supported;
#else
    return false;
#endif
}

static bool runtime_has_avx2() noexcept {
#if DBS_X86 && (defined(__GNUC__) || defined(__clang__))
    static const bool supported = []() noexcept {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }();
    return supported;
#else
    return false;
#endif
}

static bool runtime_has_sse42() noexcept {
#if DBS_X86 && (defined(__GNUC__) || defined(__clang__))
    static const bool supported = []() noexcept {
        __builtin_cpu_init();
        return __builtin_cpu_supports("sse4.2");
    }();
    return supported;
#else
    return false;
#endif
}

static bool runtime_has_neon() noexcept {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return true;
#else
    return false;
#endif
}

enum class KernelPath : int {
    Scalar = 0,
    SSE42 = 1,
    AVX2 = 2,
    AVX512 = 3,
    NEON = 4,
    CUDA = 5
};

static thread_local bool g_kernel_override_enabled = false;
static thread_local KernelPath g_kernel_override = KernelPath::Scalar;

static bool kernel_path_runtime_available(KernelPath k) noexcept {
    switch (k) {
        case KernelPath::AVX512: return runtime_has_avx512();
        case KernelPath::AVX2: return runtime_has_avx2();
        case KernelPath::SSE42: return runtime_has_sse42();
        case KernelPath::NEON: return runtime_has_neon();
        case KernelPath::Scalar: return true;
        default: return false;
    }
}

static bool kernel_path_enabled(KernelPath k) noexcept {
    return !g_kernel_override_enabled || g_kernel_override == k;
}

static KernelPath selected_kernel_path() noexcept {
    if (g_kernel_override_enabled) {
        return kernel_path_runtime_available(g_kernel_override) ? g_kernel_override : KernelPath::Scalar;
    }
#if DBS_CAN_COMPILE_AVX512
    if (runtime_has_avx512()) return KernelPath::AVX512;
#endif
#if DBS_CAN_COMPILE_AVX2
    if (runtime_has_avx2()) return KernelPath::AVX2;
#endif
#if DBS_CAN_COMPILE_SSE42
    if (runtime_has_sse42()) return KernelPath::SSE42;
#endif
#if DBS_ARM_NEON
    if (runtime_has_neon()) return KernelPath::NEON;
#endif
    return KernelPath::Scalar;
}

static const char* kernel_path_name(KernelPath k) noexcept {
    switch (k) {
        case KernelPath::AVX512: return "avx512";
        case KernelPath::AVX2: return "avx2";
        case KernelPath::SSE42: return "sse4.2";
        case KernelPath::NEON: return "neon";
        case KernelPath::CUDA: return "cuda-scaffold";
        default: return "scalar";
    }
}

#if DBS_CAN_COMPILE_AVX512
namespace avx512 {

DBS_AVX512_TARGET static inline float reduce_add512(__m512 x) {
    return _mm512_reduce_add_ps(x);
}

DBS_AVX512_TARGET static inline float reduce_max512(__m512 x) {
    return _mm512_reduce_max_ps(x);
}

DBS_AVX512_TARGET static inline __m512 exp512_ps(__m512 x) {
    const __m512 exp_hi = _mm512_set1_ps(88.3762626647949f);
    const __m512 exp_lo = _mm512_set1_ps(-88.3762626647949f);
    const __m512 log2ef = _mm512_set1_ps(1.44269504088896341f);
    const __m512 ln2f = _mm512_set1_ps(0.6931471805599453f);

    x = _mm512_min_ps(x, exp_hi);
    x = _mm512_max_ps(x, exp_lo);

    __m512 fx = _mm512_fmadd_ps(x, log2ef, _mm512_set1_ps(0.5f));
    fx = _mm512_floor_ps(fx);

    x = _mm512_fnmadd_ps(fx, ln2f, x);

    __m512 y = _mm512_set1_ps(1.9875691500E-4f);
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.3981999507E-3f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(8.3334519073E-3f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(4.1665795894E-2f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.6666665459E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(5.0000001201E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.0f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.0f));

    __m512i emm0 = _mm512_cvttps_epi32(fx);
    emm0 = _mm512_add_epi32(emm0, _mm512_set1_epi32(127));
    emm0 = _mm512_slli_epi32(emm0, 23);

    return _mm512_mul_ps(y, _mm512_castsi512_ps(emm0));
}

DBS_AVX512_TARGET static inline __m512 sigmoid512_ps(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    return _mm512_div_ps(one, _mm512_add_ps(one, exp512_ps(_mm512_sub_ps(_mm512_setzero_ps(), x))));
}

DBS_AVX512_TARGET float dot(const float* a, const float* b, int n) {
    __m512 acc = _mm512_setzero_ps();

    int i = 0;
    for (; i + 15 < n; i += 16) {
        const __m512 av = _mm512_loadu_ps(a + i);
        const __m512 bv = _mm512_loadu_ps(b + i);
        acc = _mm512_fmadd_ps(av, bv, acc);
    }

    if (i < n) {
        const int rem = n - i;
        const __mmask16 mask = static_cast<__mmask16>((1u << rem) - 1u);
        const __m512 av = _mm512_maskz_loadu_ps(mask, a + i);
        const __m512 bv = _mm512_maskz_loadu_ps(mask, b + i);
        acc = _mm512_fmadd_ps(av, bv, acc);
    }

    return reduce_add512(acc);
}

DBS_AVX512_TARGET void softmax_selected(
    const float* scores,
    float* out,
    int n,
    float temperature
) {
    constexpr float NEG_GUARD = -1.0e30f;

    std::fill(out, out + n, 0.0f);

    const __m512 inv_temp = _mm512_set1_ps(1.0f / temperature);
    const __m512 neg_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    const __m512 guard = _mm512_set1_ps(NEG_GUARD);

    __m512 vmax = neg_inf;

    int i = 0;
    for (; i + 15 < n; i += 16) {
        const __m512 raw = _mm512_loadu_ps(scores + i);
        const __mmask16 valid = _mm512_cmp_ps_mask(raw, guard, _CMP_GT_OS);
        __m512 scaled = _mm512_mul_ps(raw, inv_temp);
        scaled = _mm512_mask_mov_ps(neg_inf, valid, scaled);
        vmax = _mm512_max_ps(vmax, scaled);
    }

    if (i < n) {
        const int rem = n - i;
        const __mmask16 lane_mask = static_cast<__mmask16>((1u << rem) - 1u);
        const __m512 raw = _mm512_maskz_loadu_ps(lane_mask, scores + i);
        const __mmask16 valid = lane_mask & _mm512_cmp_ps_mask(raw, guard, _CMP_GT_OS);
        __m512 scaled = _mm512_mul_ps(raw, inv_temp);
        scaled = _mm512_mask_mov_ps(neg_inf, valid, scaled);
        vmax = _mm512_max_ps(vmax, scaled);
    }

    const float maxv = reduce_max512(vmax);
    if (!std::isfinite(maxv)) return;

    const __m512 max_vec = _mm512_set1_ps(maxv);
    __m512 sum_vec = _mm512_setzero_ps();

    i = 0;
    for (; i + 15 < n; i += 16) {
        const __m512 raw = _mm512_loadu_ps(scores + i);
        const __mmask16 valid = _mm512_cmp_ps_mask(raw, guard, _CMP_GT_OS);
        __m512 scaled = _mm512_sub_ps(_mm512_mul_ps(raw, inv_temp), max_vec);
        __m512 e = exp512_ps(scaled);
        e = _mm512_maskz_mov_ps(valid, e);
        sum_vec = _mm512_add_ps(sum_vec, e);
        _mm512_storeu_ps(out + i, e);
    }

    if (i < n) {
        const int rem = n - i;
        const __mmask16 lane_mask = static_cast<__mmask16>((1u << rem) - 1u);
        const __m512 raw = _mm512_maskz_loadu_ps(lane_mask, scores + i);
        const __mmask16 valid = lane_mask & _mm512_cmp_ps_mask(raw, guard, _CMP_GT_OS);
        __m512 scaled = _mm512_sub_ps(_mm512_mul_ps(raw, inv_temp), max_vec);
        __m512 e = exp512_ps(scaled);
        e = _mm512_maskz_mov_ps(valid, e);
        sum_vec = _mm512_add_ps(sum_vec, e);
        _mm512_mask_storeu_ps(out + i, lane_mask, e);
    }

    const float sum = reduce_add512(sum_vec);
    if (!(sum > 0.0f) || !std::isfinite(sum)) {
        std::fill(out, out + n, 0.0f);
        return;
    }

    const __m512 inv_sum = _mm512_set1_ps(1.0f / sum);
    int j = 0;
    for (; j + 15 < n; j += 16) {
        const __m512 y = _mm512_mul_ps(_mm512_loadu_ps(out + j), inv_sum);
        _mm512_storeu_ps(out + j, y);
    }

    if (j < n) {
        const int rem = n - j;
        const __mmask16 mask = static_cast<__mmask16>((1u << rem) - 1u);
        const __m512 y = _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, out + j), inv_sum);
        _mm512_mask_storeu_ps(out + j, mask, y);
    }
}

DBS_AVX512_TARGET float sum_sigmoid_shifted(
    const float* scores,
    int n,
    float theta,
    float temperature
) {
    constexpr float NEG_GUARD = -1.0e30f;

    const __m512 theta_vec = _mm512_set1_ps(theta);
    const __m512 inv_temp = _mm512_set1_ps(1.0f / temperature);
    const __m512 guard = _mm512_set1_ps(NEG_GUARD);
    __m512 sum_vec = _mm512_setzero_ps();

    int i = 0;
    for (; i + 15 < n; i += 16) {
        const __m512 s = _mm512_loadu_ps(scores + i);
        const __mmask16 valid = _mm512_cmp_ps_mask(s, guard, _CMP_GT_OS);
        const __m512 z = _mm512_mul_ps(_mm512_sub_ps(s, theta_vec), inv_temp);
        __m512 y = sigmoid512_ps(z);
        y = _mm512_maskz_mov_ps(valid, y);
        sum_vec = _mm512_add_ps(sum_vec, y);
    }

    if (i < n) {
        const int rem = n - i;
        const __mmask16 lane_mask = static_cast<__mmask16>((1u << rem) - 1u);
        const __m512 s = _mm512_maskz_loadu_ps(lane_mask, scores + i);
        const __mmask16 valid = lane_mask & _mm512_cmp_ps_mask(s, guard, _CMP_GT_OS);
        const __m512 z = _mm512_mul_ps(_mm512_sub_ps(s, theta_vec), inv_temp);
        __m512 y = sigmoid512_ps(z);
        y = _mm512_maskz_mov_ps(valid, y);
        sum_vec = _mm512_add_ps(sum_vec, y);
    }

    return reduce_add512(sum_vec);
}

DBS_AVX512_TARGET void soft_topk_write(
    const float* scores,
    float* out,
    int n,
    float theta,
    float temperature
) {
    constexpr float NEG_GUARD = -1.0e30f;

    const __m512 theta_vec = _mm512_set1_ps(theta);
    const __m512 inv_temp = _mm512_set1_ps(1.0f / temperature);
    const __m512 guard = _mm512_set1_ps(NEG_GUARD);

    int i = 0;
    for (; i + 15 < n; i += 16) {
        const __m512 s = _mm512_loadu_ps(scores + i);
        const __mmask16 valid = _mm512_cmp_ps_mask(s, guard, _CMP_GT_OS);
        const __m512 z = _mm512_mul_ps(_mm512_sub_ps(s, theta_vec), inv_temp);
        const __m512 y = _mm512_maskz_mov_ps(valid, sigmoid512_ps(z));
        _mm512_storeu_ps(out + i, y);
    }

    if (i < n) {
        const int rem = n - i;
        const __mmask16 lane_mask = static_cast<__mmask16>((1u << rem) - 1u);
        const __m512 s = _mm512_maskz_loadu_ps(lane_mask, scores + i);
        const __mmask16 valid = lane_mask & _mm512_cmp_ps_mask(s, guard, _CMP_GT_OS);
        const __m512 z = _mm512_mul_ps(_mm512_sub_ps(s, theta_vec), inv_temp);
        const __m512 y = _mm512_maskz_mov_ps(valid, sigmoid512_ps(z));
        _mm512_mask_storeu_ps(out + i, lane_mask, y);
    }
}

DBS_AVX512_TARGET void scan_parent_row(
    const float* row,
    float parent_raw,
    int parent_length,
    int parent,
    int vocab_size,
    Candidate* top,
    int top_count,
    int vocab_block,
    float length_penalty_alpha,
    const uint8_t* banned_tokens,
    int forced_token,
    int eos_token,
    int min_length
) {
    if (banned_tokens || forced_token >= 0 || (eos_token >= 0 && parent_length + 1 < min_length)) {
        scan_parent_row_scalar(row, parent_raw, parent_length, parent, vocab_size, top, top_count, vocab_block, length_penalty_alpha, banned_tokens, forced_token, eos_token, min_length);
        return;
    }
    const int new_len = parent_length + 1;
    const float inv_penalty = 1.0f / gnmt_length_penalty(new_len, length_penalty_alpha);

    const __m512 parent_vec = _mm512_set1_ps(parent_raw);
    const __m512 scale_vec = _mm512_set1_ps(inv_penalty);
    const __m512 neg_inf_vec = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    const __m512 pos_inf_vec = _mm512_set1_ps(std::numeric_limits<float>::infinity());

    alignas(64) float rank_tmp[16];
    alignas(64) float raw_tmp[16];

    // Roughly four 64B cache lines ahead. This is far enough for streaming rows
    // without pulling too much cold vocabulary data into L1.
    constexpr int PREFETCH_FLOATS = 256;

    for (int base = 0; base < vocab_size; base += vocab_block) {
        const int end = std::min(vocab_size, base + vocab_block);
        int v = base;

        for (; v + 15 < end; v += 16) {
            const int pf = v + PREFETCH_FLOATS;
            if (pf < vocab_size) {
                _mm_prefetch(reinterpret_cast<const char*>(row + pf), _MM_HINT_T0);
            }

            const __m512 lp = _mm512_loadu_ps(row + v);
            const __m512 raw_vec = _mm512_add_ps(lp, parent_vec);
            const __m512 rank_vec = _mm512_mul_ps(raw_vec, scale_vec);

            const float threshold = top[top_count - 1].score;
            const __mmask16 finite_mask =
                _mm512_cmp_ps_mask(lp, neg_inf_vec, _CMP_GT_OS) &
                _mm512_cmp_ps_mask(lp, pos_inf_vec, _CMP_LT_OS);
            const __mmask16 mask = finite_mask & _mm512_cmp_ps_mask(rank_vec, _mm512_set1_ps(threshold), _CMP_GT_OS);

            if (mask) {
                _mm512_store_ps(rank_tmp, rank_vec);
                _mm512_store_ps(raw_tmp, raw_vec);

                uint32_t m = static_cast<uint32_t>(mask);
                while (m) {
                    const int lane = __builtin_ctz(m);
                    insert_topk(top, top_count, Candidate{rank_tmp[lane], raw_tmp[lane], parent, v + lane, new_len, 1});
                    m &= m - 1;
                }
            }
        }

        if (v < end) {
            const int rem = end - v;
            const __mmask16 lane_mask = static_cast<__mmask16>((1u << rem) - 1u);

            const __m512 lp = _mm512_maskz_loadu_ps(lane_mask, row + v);
            const __m512 raw_vec = _mm512_add_ps(lp, parent_vec);
            const __m512 rank_vec = _mm512_mul_ps(raw_vec, scale_vec);

            const float threshold = top[top_count - 1].score;
            const __mmask16 finite_mask =
                _mm512_cmp_ps_mask(lp, neg_inf_vec, _CMP_GT_OS) &
                _mm512_cmp_ps_mask(lp, pos_inf_vec, _CMP_LT_OS);
            __mmask16 mask = lane_mask & finite_mask & _mm512_cmp_ps_mask(rank_vec, _mm512_set1_ps(threshold), _CMP_GT_OS);

            if (mask) {
                _mm512_store_ps(rank_tmp, rank_vec);
                _mm512_store_ps(raw_tmp, raw_vec);

                uint32_t m = static_cast<uint32_t>(mask);
                while (m) {
                    const int lane = __builtin_ctz(m);
                    insert_topk(top, top_count, Candidate{rank_tmp[lane], raw_tmp[lane], parent, v + lane, new_len, 1});
                    m &= m - 1;
                }
            }
        }
    }
}

} // namespace avx512
#endif


#if DBS_CAN_COMPILE_AVX2
namespace avx2 {
DBS_AVX2_TARGET static inline float hsum256(__m256 v) {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, v);
    float s = 0.0f;
    for (float x : tmp) s += x;
    return s;
}

DBS_AVX2_TARGET float dot(const float* a, const float* b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        const __m256 av = _mm256_loadu_ps(a + i);
        const __m256 bv = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(av, bv, acc);
    }
    float s = hsum256(acc);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

DBS_AVX2_TARGET void softmax_selected(const float* scores, float* out, int n, float temperature) {
    constexpr float NEG_GUARD = -1.0e30f;
    std::fill(out, out + n, 0.0f);
    const __m256 inv_temp = _mm256_set1_ps(1.0f / temperature);
    const __m256 guard = _mm256_set1_ps(NEG_GUARD);
    const __m256 neg_inf = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    __m256 vmax = neg_inf;
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 raw = _mm256_loadu_ps(scores + i);
        __m256 valid = _mm256_cmp_ps(raw, guard, _CMP_GT_OS);
        __m256 scaled = _mm256_mul_ps(raw, inv_temp);
        scaled = _mm256_blendv_ps(neg_inf, scaled, valid);
        vmax = _mm256_max_ps(vmax, scaled);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, vmax);
    float maxv = -std::numeric_limits<float>::infinity();
    for (float x : tmp) maxv = std::max(maxv, x);
    for (; i < n; ++i) if (scores[i] > NEG_GUARD) maxv = std::max(maxv, scores[i] / temperature);
    if (!std::isfinite(maxv)) return;
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
        if (scores[j] > NEG_GUARD) {
            out[j] = safe_exp_scalar(scores[j] / temperature - maxv);
            sum += out[j];
        }
    }
    if (!(sum > 0.0f) || !std::isfinite(sum)) { std::fill(out, out + n, 0.0f); return; }
    const float inv_sum = 1.0f / sum;
    i = 0;
    const __m256 inv = _mm256_set1_ps(inv_sum);
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(out + i), inv));
    }
    for (; i < n; ++i) out[i] *= inv_sum;
}

DBS_AVX2_TARGET void scan_parent_row(
    const float* row,
    float parent_raw,
    int parent_length,
    int parent,
    int vocab_size,
    Candidate* top,
    int top_count,
    int vocab_block,
    float length_penalty_alpha,
    const uint8_t* banned_tokens,
    int forced_token,
    int eos_token,
    int min_length
) {
    if (banned_tokens || forced_token >= 0 || (eos_token >= 0 && parent_length + 1 < min_length)) {
        scan_parent_row_scalar(row, parent_raw, parent_length, parent, vocab_size, top, top_count, vocab_block, length_penalty_alpha, banned_tokens, forced_token, eos_token, min_length);
        return;
    }
    const int new_len = parent_length + 1;
    const float inv_penalty = 1.0f / gnmt_length_penalty(new_len, length_penalty_alpha);

    const __m256 parent_vec = _mm256_set1_ps(parent_raw);
    const __m256 scale_vec = _mm256_set1_ps(inv_penalty);
    const __m256 neg_inf_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    const __m256 pos_inf_vec = _mm256_set1_ps(std::numeric_limits<float>::infinity());

    for (int base = 0; base < vocab_size; base += vocab_block) {
        const int end = std::min(vocab_size, base + vocab_block);
        int v = base;

        for (; v + 7 < end; v += 8) {
            const __m256 lp = _mm256_loadu_ps(row + v);
            const __m256 raw_vec = _mm256_add_ps(lp, parent_vec);
            const __m256 rank_vec = _mm256_mul_ps(raw_vec, scale_vec);

            const float threshold = top[top_count - 1].score;
            const __m256 finite = _mm256_and_ps(
                _mm256_cmp_ps(lp, neg_inf_vec, _CMP_GT_OS),
                _mm256_cmp_ps(lp, pos_inf_vec, _CMP_LT_OS));
            const __m256 better = _mm256_cmp_ps(rank_vec, _mm256_set1_ps(threshold), _CMP_GT_OS);
            const int mask = _mm256_movemask_ps(_mm256_and_ps(finite, better));

            if (mask) {
                alignas(32) float rank_tmp[8];
                alignas(32) float raw_tmp[8];
                _mm256_store_ps(rank_tmp, rank_vec);
                _mm256_store_ps(raw_tmp, raw_vec);
                int m = mask;
                while (m) {
                    const int lane = __builtin_ctz(m);
                    insert_topk(top, top_count, Candidate{rank_tmp[lane], raw_tmp[lane], parent, v + lane, new_len, 1});
                    m &= m - 1;
                }
            }
        }

        for (; v < end; ++v) {
            const float lp = row[v];
            if (!std::isfinite(lp)) continue;
            const float raw = parent_raw + lp;
            const float rank = raw * inv_penalty;
            insert_topk(top, top_count, Candidate{rank, raw, parent, v, new_len, 1});
        }
    }
}
}
#endif

#if DBS_CAN_COMPILE_SSE42
namespace sse42 {
DBS_SSE42_TARGET static inline float hsum128(__m128 v) {
    alignas(16) float tmp[4];
    _mm_store_ps(tmp, v);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

DBS_SSE42_TARGET float dot(const float* a, const float* b, int n) {
    __m128 acc = _mm_setzero_ps();
    int i = 0;
    for (; i + 3 < n; i += 4) {
        acc = _mm_add_ps(acc, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
    float s = hsum128(acc);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

DBS_SSE42_TARGET void softmax_selected(const float* scores, float* out, int n, float temperature) {
    constexpr float NEG_GUARD = -1.0e30f;
    std::fill(out, out + n, 0.0f);
    const __m128 inv_temp = _mm_set1_ps(1.0f / temperature);
    const __m128 guard = _mm_set1_ps(NEG_GUARD);
    const __m128 neg_inf = _mm_set1_ps(-std::numeric_limits<float>::infinity());
    __m128 vmax = neg_inf;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        __m128 raw = _mm_loadu_ps(scores + i);
        __m128 valid = _mm_cmpgt_ps(raw, guard);
        __m128 scaled = _mm_mul_ps(raw, inv_temp);
        scaled = _mm_or_ps(_mm_and_ps(valid, scaled), _mm_andnot_ps(valid, neg_inf));
        vmax = _mm_max_ps(vmax, scaled);
    }
    alignas(16) float tmp[4];
    _mm_store_ps(tmp, vmax);
    float maxv = -std::numeric_limits<float>::infinity();
    for (float x : tmp) maxv = std::max(maxv, x);
    for (; i < n; ++i) if (scores[i] > NEG_GUARD) maxv = std::max(maxv, scores[i] / temperature);
    if (!std::isfinite(maxv)) return;
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
        if (scores[j] > NEG_GUARD) { out[j] = safe_exp_scalar(scores[j] / temperature - maxv); sum += out[j]; }
    }
    if (!(sum > 0.0f) || !std::isfinite(sum)) { std::fill(out, out + n, 0.0f); return; }
    const float inv_sum = 1.0f / sum;
    i = 0;
    const __m128 inv = _mm_set1_ps(inv_sum);
    for (; i + 3 < n; i += 4) _mm_storeu_ps(out + i, _mm_mul_ps(_mm_loadu_ps(out + i), inv));
    for (; i < n; ++i) out[i] *= inv_sum;
}
}
#endif

#if DBS_ARM_NEON
namespace neon {
static float dot(const float* a, const float* b, int n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        acc = vmlaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float tmp[4];
    vst1q_f32(tmp, acc);
    float s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

static void softmax_selected(const float* scores, float* out, int n, float temperature) {
    // NEON path accelerates normalization; scalar exp remains deliberate for cross-platform accuracy.
    softmax_selected_scalar(scores, out, n, temperature);
}
}
#endif

static float dot_simd_or_scalar(const float* a, const float* b, int n) {
#if DBS_CAN_COMPILE_AVX512
    if (kernel_path_enabled(KernelPath::AVX512) && runtime_has_avx512()) return avx512::dot(a, b, n);
#endif
#if DBS_CAN_COMPILE_AVX2
    if (kernel_path_enabled(KernelPath::AVX2) && runtime_has_avx2()) return avx2::dot(a, b, n);
#endif
#if DBS_CAN_COMPILE_SSE42
    if (kernel_path_enabled(KernelPath::SSE42) && runtime_has_sse42()) return sse42::dot(a, b, n);
#endif
#if DBS_ARM_NEON
    if (kernel_path_enabled(KernelPath::NEON) && runtime_has_neon()) return neon::dot(a, b, n);
#endif
    return dot_scalar(a, b, n);
}

static void softmax_selected(
    const float* scores,
    float* out,
    int n,
    float temperature
) {
#if DBS_CAN_COMPILE_AVX512
    if (kernel_path_enabled(KernelPath::AVX512) && runtime_has_avx512()) {
        avx512::softmax_selected(scores, out, n, temperature);
        return;
    }
#endif
#if DBS_CAN_COMPILE_AVX2
    if (kernel_path_enabled(KernelPath::AVX2) && runtime_has_avx2()) {
        avx2::softmax_selected(scores, out, n, temperature);
        return;
    }
#endif
#if DBS_CAN_COMPILE_SSE42
    if (kernel_path_enabled(KernelPath::SSE42) && runtime_has_sse42()) {
        sse42::softmax_selected(scores, out, n, temperature);
        return;
    }
#endif
#if DBS_ARM_NEON
    if (kernel_path_enabled(KernelPath::NEON) && runtime_has_neon()) {
        neon::softmax_selected(scores, out, n, temperature);
        return;
    }
#endif
    softmax_selected_scalar(scores, out, n, temperature);
}

static float sum_sigmoid_shifted(
    const float* scores,
    int n,
    float theta,
    float temperature
) {
#if DBS_CAN_COMPILE_AVX512
    if (kernel_path_enabled(KernelPath::AVX512) && runtime_has_avx512()) return avx512::sum_sigmoid_shifted(scores, n, theta, temperature);
#endif
    return sum_sigmoid_shifted_scalar(scores, n, theta, temperature);
}

static void soft_topk_write(
    const float* scores,
    float* out,
    int n,
    float theta,
    float temperature
) {
#if DBS_CAN_COMPILE_AVX512
    if (kernel_path_enabled(KernelPath::AVX512) && runtime_has_avx512()) {
        avx512::soft_topk_write(scores, out, n, theta, temperature);
        return;
    }
#endif
    soft_topk_write_scalar(scores, out, n, theta, temperature);
}

static void soft_topk_inclusion(
    const float* scores,
    float* out,
    int n,
    int target_k,
    float temperature,
    float tolerance,
    int max_iters
) {
    constexpr float NEG_GUARD = -1.0e30f;

    std::fill(out, out + n, 0.0f);

    int active = 0;
    float min_s = std::numeric_limits<float>::infinity();
    float max_s = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < n; ++i) {
        if (scores[i] > NEG_GUARD) {
            ++active;
            min_s = std::min(min_s, scores[i]);
            max_s = std::max(max_s, scores[i]);
        }
    }

    if (active == 0 || target_k <= 0) return;

    if (target_k >= active) {
        for (int i = 0; i < n; ++i) {
            if (scores[i] > NEG_GUARD) out[i] = 1.0f;
        }
        return;
    }

    float lo = min_s - 80.0f * temperature;
    float hi = max_s + 80.0f * temperature;

    for (int it = 0; it < max_iters; ++it) {
        const float mid = 0.5f * (lo + hi);
        const float s = sum_sigmoid_shifted(scores, n, mid, temperature);
        const float err = s - static_cast<float>(target_k);

        if (std::fabs(err) <= tolerance || std::fabs(hi - lo) <= tolerance * std::max(1.0f, std::fabs(mid))) {
            lo = hi = mid;
            break;
        }

        if (err > 0.0f) lo = mid;
        else hi = mid;
    }

    const float theta = 0.5f * (lo + hi);
    soft_topk_write(scores, out, n, theta, temperature);
}

class DifferentiableBeamSearchAVX512 {
public:
    explicit DifferentiableBeamSearchAVX512(BeamOptions options)
        : opt_(options) {
        if (opt_.beam_size <= 0) {
            throw std::invalid_argument("beam_size must be positive");
        }

        if (!(opt_.selected_temperature > 0.0f)) {
            throw std::invalid_argument("selected_temperature must be positive");
        }

        if (!(opt_.soft_topk_temperature > 0.0f)) {
            throw std::invalid_argument("soft_topk_temperature must be positive");
        }

        if (opt_.soft_topk_max_iters <= 0) {
            throw std::invalid_argument("soft_topk_max_iters must be positive");
        }

        if (!(opt_.soft_topk_tolerance > 0.0f)) {
            throw std::invalid_argument("soft_topk_tolerance must be positive");
        }

        if (opt_.length_penalty_alpha < 0.0f) {
            throw std::invalid_argument("length_penalty_alpha cannot be negative");
        }

        if (opt_.min_length < 0) {
            throw std::invalid_argument("min_length cannot be negative");
        }

        if (opt_.max_dense_gradient_elements <= 0) {
            throw std::invalid_argument("max_dense_gradient_elements must be positive");
        }

        opt_.vocab_block = std::max(16, opt_.vocab_block);
        opt_.relaxed_pool_multiplier = std::max(1, opt_.relaxed_pool_multiplier);
    }

    DecodeResult decode(const float* log_probs, int steps, int vocab_size) const {
        return decode_constrained(log_probs, steps, vocab_size, nullptr);
    }

    DecodeResult decode_constrained(const float* log_probs, int steps, int vocab_size, const DecodeConstraints* constraints) const {
        if (!log_probs) throw std::invalid_argument("log_probs cannot be null");
        if (steps <= 0) throw std::invalid_argument("steps must be positive");
        if (vocab_size <= 0) throw std::invalid_argument("vocab_size must be positive");

        if (opt_.eos_token >= vocab_size) {
            throw std::invalid_argument("eos_token is outside vocab");
        }

        if (constraints && constraints->forced_tokens) {
            for (int t = 0; t < steps; ++t) {
                const int tok = constraints->forced_tokens[t];
                if (tok >= vocab_size) throw std::invalid_argument("forced token is outside vocab");
            }
        }

        if (opt_.validate_inputs) {
            validate_logprob_tensor(log_probs, steps, opt_.beam_size, vocab_size);
        }

        const int effective_min_length = constraints && constraints->min_length >= 0 ? constraints->min_length : opt_.min_length;

        const int K = opt_.beam_size;
        const int V = vocab_size;
        const size_t relaxed_pool_size = checked_mul_size(static_cast<size_t>(K), static_cast<size_t>(opt_.relaxed_pool_multiplier), "relaxed pool size overflow");
        if (relaxed_pool_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("relaxed pool size exceeds int range");
        }
        const int P = std::max(K, static_cast<int>(relaxed_pool_size));

        const size_t selected_count = checked_mul_size(static_cast<size_t>(steps), static_cast<size_t>(K), "selected result size overflow");
        const size_t pool_count = checked_mul_size(static_cast<size_t>(steps), static_cast<size_t>(P), "pool result size overflow");
        (void)checked_mul_size(selected_count, static_cast<size_t>(V), "log-prob tensor size overflow");

        constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

        DecodeResult result;
        result.steps = steps;
        result.beam_size = K;
        result.vocab_size = V;
        result.eos_token = opt_.eos_token;
        result.relaxed_pool_size = P;
        result.selected_temperature = opt_.selected_temperature;
        result.soft_topk_temperature = opt_.soft_topk_temperature;
        result.length_penalty_alpha = opt_.length_penalty_alpha;

        result.parents.assign(selected_count, -1);
        result.tokens.assign(selected_count, -1);
        result.lengths.assign(selected_count, 0);

        result.raw_scores.assign(selected_count, NEG_INF);
        result.scores.assign(selected_count, NEG_INF);
        result.weights.assign(selected_count, 0.0f);

        result.final_raw_scores.assign(K, NEG_INF);
        result.final_scores.assign(K, NEG_INF);

        result.from_logprob.assign(selected_count, 0);

        result.pool_parents.assign(pool_count, -1);
        result.pool_tokens.assign(pool_count, -1);
        result.pool_lengths.assign(pool_count, 0);

        result.pool_raw_scores.assign(pool_count, NEG_INF);
        result.pool_scores.assign(pool_count, NEG_INF);
        result.relaxed_weights.assign(pool_count, 0.0f);

        result.pool_from_logprob.assign(pool_count, 0);

        AlignedFloatVector prev_raw_scores(K, NEG_INF);
        AlignedFloatVector next_raw_scores(K, NEG_INF);

        AlignedIntVector prev_lengths(K, 0);
        AlignedIntVector next_lengths(K, 0);

        std::vector<uint8_t> ended_prev(K, 0);
        std::vector<uint8_t> ended_next(K, 0);
        std::vector<std::vector<int32_t>> prev_sequences(K);
        std::vector<std::vector<int32_t>> next_sequences(K);

        prev_raw_scores[0] = 0.0f;

        const bool has_advanced_constraints = constraints && (
            constraints->repetition_penalty > 1.0f ||
            constraints->no_repeat_ngram_size > 0 ||
            constraints->token_filter != nullptr
        );

        std::vector<Candidate> top(P);

        for (int t = 0; t < steps; ++t) {
            for (int i = 0; i < P; ++i) {
                top[i] = Candidate{NEG_INF, NEG_INF, -1, -1, 0, 0};
            }

            const float* step_base = log_probs + static_cast<size_t>(t) * K * V;

            for (int b = 0; b < K; ++b) {
                const float parent_raw = prev_raw_scores[b];

                if (!std::isfinite(parent_raw)) continue;

                if (opt_.eos_token >= 0 && ended_prev[b]) {
                    const int len = std::max(1, static_cast<int>(prev_lengths[b]));
                    const float rank =
                        parent_raw / gnmt_length_penalty(len, opt_.length_penalty_alpha);

                    insert_topk(
                        top.data(),
                        P,
                        Candidate{rank, parent_raw, b, opt_.eos_token, len, 0}
                    );

                    continue;
                }

                const float* row = step_base + static_cast<size_t>(b) * V;

                const int forced_token = constraints && constraints->forced_tokens ? constraints->forced_tokens[t] : -1;

                if (has_advanced_constraints) {
                    scan_parent_row_scalar_advanced(
                        row,
                        parent_raw,
                        prev_lengths[b],
                        b,
                        t,
                        V,
                        top.data(),
                        P,
                        opt_.vocab_block,
                        opt_.length_penalty_alpha,
                        constraints,
                        prev_sequences[b],
                        forced_token,
                        opt_.eos_token,
                        effective_min_length
                    );
                } else {
                    scan_parent_row(
                        row,
                        parent_raw,
                        prev_lengths[b],
                        b,
                        V,
                        top.data(),
                        P,
                        constraints ? constraints->banned_tokens : nullptr,
                        forced_token,
                        effective_min_length
                    );
                }
            }

            for (int p = 0; p < P; ++p) {
                const size_t idx = static_cast<size_t>(t) * P + p;

                result.pool_parents[idx] = top[p].parent;
                result.pool_tokens[idx] = top[p].token;
                result.pool_lengths[idx] = top[p].length;

                result.pool_raw_scores[idx] = top[p].raw_score;
                result.pool_scores[idx] = top[p].score;

                result.pool_from_logprob[idx] = top[p].from_logprob;
            }

            soft_topk_inclusion(
                result.pool_scores.data() + static_cast<size_t>(t) * P,
                result.relaxed_weights.data() + static_cast<size_t>(t) * P,
                P,
                K,
                opt_.soft_topk_temperature,
                opt_.soft_topk_tolerance,
                opt_.soft_topk_max_iters
            );

            for (int k = 0; k < K; ++k) {
                const size_t idx = static_cast<size_t>(t) * K + k;

                result.parents[idx] = top[k].parent;
                result.tokens[idx] = top[k].token;
                result.lengths[idx] = top[k].length;

                result.raw_scores[idx] = top[k].raw_score;
                result.scores[idx] = top[k].score;

                result.from_logprob[idx] = top[k].from_logprob;

                next_raw_scores[k] = top[k].raw_score;
                next_lengths[k] = top[k].length;

                next_sequences[k].clear();
                if (top[k].parent >= 0) {
                    next_sequences[k] = prev_sequences[static_cast<size_t>(top[k].parent)];
                    const bool parent_ended = ended_prev[top[k].parent] != 0;
                    const bool token_is_eos =
                        opt_.eos_token >= 0 && top[k].token == opt_.eos_token;

                    if (top[k].from_logprob && top[k].token >= 0) {
                        next_sequences[k].push_back(top[k].token);
                    }

                    ended_next[k] = static_cast<uint8_t>(parent_ended || token_is_eos);
                } else {
                    ended_next[k] = 0;
                }
            }

            softmax_selected(
                result.scores.data() + static_cast<size_t>(t) * K,
                result.weights.data() + static_cast<size_t>(t) * K,
                K,
                opt_.selected_temperature
            );

            std::swap(prev_raw_scores, next_raw_scores);
            std::fill(next_raw_scores.begin(), next_raw_scores.end(), NEG_INF);

            std::swap(prev_lengths, next_lengths);
            std::fill(next_lengths.begin(), next_lengths.end(), 0);

            ended_prev.swap(ended_next);
            std::fill(ended_next.begin(), ended_next.end(), uint8_t{0});

            prev_sequences.swap(next_sequences);
            for (auto& seq : next_sequences) seq.clear();
        }

        result.final_raw_scores = prev_raw_scores;

        for (int k = 0; k < K; ++k) {
            const int len = std::max(1, static_cast<int>(prev_lengths[k]));
            result.final_scores[k] =
                prev_raw_scores[k] / gnmt_length_penalty(len, opt_.length_penalty_alpha);
        }

        result.sequences = build_sequences(result);

        return result;
    }

    BackwardResult backward(
        const DecodeResult& fwd,
        const float* grad_selected_weights,
        const float* grad_relaxed_weights,
        const float* grad_final_scores
    ) const {
        const int T = fwd.steps;
        const int K = fwd.beam_size;
        const int V = fwd.vocab_size;
        const int P = fwd.relaxed_pool_size;

        if (K != opt_.beam_size) {
            throw std::invalid_argument("forward result beam_size does not match decoder");
        }

        if (P < K) {
            throw std::invalid_argument("invalid relaxed pool size");
        }

        const size_t selected_count = checked_mul_size(static_cast<size_t>(T), static_cast<size_t>(K), "selected gradient size overflow");
        const size_t dense_grad_count = checked_mul_size(selected_count, static_cast<size_t>(V), "dense gradient size overflow");
        if (dense_grad_count > static_cast<size_t>(opt_.max_dense_gradient_elements)) {
            throw std::length_error("dense gradient allocation exceeds max_dense_gradient_elements; use sparse backward");
        }

        BackwardResult out;
        out.sparse = false;
        out.grad_log_probs.assign(dense_grad_count, 0.0f);
        out.grad_initial_scores.assign(K, 0.0f);

        AlignedFloatVector next_raw_grad(K, 0.0f);
        AlignedFloatVector prev_raw_grad(K, 0.0f);

        for (int t = T - 1; t >= 0; --t) {
            std::fill(prev_raw_grad.begin(), prev_raw_grad.end(), 0.0f);

            backward_selected(
                fwd,
                grad_selected_weights,
                grad_final_scores,
                next_raw_grad.data(),
                prev_raw_grad.data(),
                out.grad_log_probs.data(),
                t
            );

            backward_relaxed_topk(
                fwd,
                grad_relaxed_weights,
                prev_raw_grad.data(),
                out.grad_log_probs.data(),
                t
            );

            next_raw_grad.swap(prev_raw_grad);
        }

        out.grad_initial_scores = next_raw_grad;
        return out;
    }

    BackwardResult backward_sparse(
        const DecodeResult& fwd,
        const float* grad_selected_weights,
        const float* grad_relaxed_weights,
        const float* grad_final_scores
    ) const {
        const int T = fwd.steps;
        const int K = fwd.beam_size;
        const int P = fwd.relaxed_pool_size;

        if (K != opt_.beam_size) {
            throw std::invalid_argument("forward result beam_size does not match decoder");
        }

        if (P < K) {
            throw std::invalid_argument("invalid relaxed pool size");
        }

        BackwardResult out;
        out.sparse = true;
        out.grad_initial_scores.assign(K, 0.0f);

        std::vector<SparseGradEntry> entries;
        entries.reserve(static_cast<size_t>(T) * static_cast<size_t>(K + P));

        AlignedFloatVector next_raw_grad(K, 0.0f);
        AlignedFloatVector prev_raw_grad(K, 0.0f);

        for (int t = T - 1; t >= 0; --t) {
            std::fill(prev_raw_grad.begin(), prev_raw_grad.end(), 0.0f);

            backward_selected_sparse(
                fwd,
                grad_selected_weights,
                grad_final_scores,
                next_raw_grad.data(),
                prev_raw_grad.data(),
                entries,
                t
            );

            backward_relaxed_topk_sparse(
                fwd,
                grad_relaxed_weights,
                prev_raw_grad.data(),
                entries,
                t
            );

            next_raw_grad.swap(prev_raw_grad);
        }

        out.grad_initial_scores = next_raw_grad;
        finalize_sparse_entries(entries, out);
        return out;
    }

private:
    BeamOptions opt_;

    void scan_parent_row(
        const float* row,
        float parent_raw,
        int parent_length,
        int parent,
        int vocab_size,
        Candidate* top,
        int top_count,
        const uint8_t* banned_tokens,
        int forced_token,
        int min_length
    ) const {
#if DBS_CAN_COMPILE_AVX512
        if (kernel_path_enabled(KernelPath::AVX512) && runtime_has_avx512()) {
            avx512::scan_parent_row(
                row, parent_raw, parent_length, parent,
                vocab_size, top, top_count, opt_.vocab_block,
                opt_.length_penalty_alpha, banned_tokens, forced_token,
                opt_.eos_token, min_length
            );
            return;
        }
#endif
#if DBS_CAN_COMPILE_AVX2
        if (kernel_path_enabled(KernelPath::AVX2) && runtime_has_avx2()) {
            avx2::scan_parent_row(
                row, parent_raw, parent_length, parent,
                vocab_size, top, top_count, opt_.vocab_block,
                opt_.length_penalty_alpha, banned_tokens, forced_token,
                opt_.eos_token, min_length
            );
            return;
        }
#endif

        scan_parent_row_scalar(
            row,
            parent_raw,
            parent_length,
            parent,
            vocab_size,
            top,
            top_count,
            opt_.vocab_block,
            opt_.length_penalty_alpha,
            banned_tokens,
            forced_token,
            opt_.eos_token,
            min_length
        );
    }

    static void scatter_raw_candidate_grad(
        const DecodeResult& fwd,
        float* grad_log_probs,
        float* prev_raw_grad,
        int t,
        int parent,
        int token,
        uint8_t from_logprob,
        float draw
    ) {
        if (parent < 0 || draw == 0.0f) return;

        const int K = fwd.beam_size;
        const int V = fwd.vocab_size;

        prev_raw_grad[parent] += draw;

        if (from_logprob && token >= 0) {
            const size_t grad_idx =
                (static_cast<size_t>(t) * K + parent) * V + token;

            grad_log_probs[grad_idx] += draw;
        }
    }

    static void scatter_raw_candidate_grad_sparse(
        const DecodeResult& fwd,
        std::vector<SparseGradEntry>& entries,
        float* prev_raw_grad,
        int t,
        int parent,
        int token,
        uint8_t from_logprob,
        float draw
    ) {
        if (parent < 0 || draw == 0.0f) return;

        const int K = fwd.beam_size;
        const int V = fwd.vocab_size;

        prev_raw_grad[parent] += draw;

        if (from_logprob && token >= 0) {
            const int64_t grad_idx =
                (static_cast<int64_t>(t) * K + parent) * static_cast<int64_t>(V) + token;
            entries.push_back(SparseGradEntry{grad_idx, draw});
        }
    }

    static void finalize_sparse_entries(
        std::vector<SparseGradEntry>& entries,
        BackwardResult& out
    ) {
        if (entries.empty()) return;

        std::sort(
            entries.begin(),
            entries.end(),
            [](const SparseGradEntry& a, const SparseGradEntry& b) {
                return a.index < b.index;
            }
        );

        out.sparse_logprob_indices.reserve(entries.size());
        out.sparse_logprob_values.reserve(entries.size());

        int64_t cur = entries[0].index;
        float sum = 0.0f;

        for (const SparseGradEntry& e : entries) {
            if (e.index == cur) {
                sum += e.value;
            } else {
                out.sparse_logprob_indices.push_back(cur);
                out.sparse_logprob_values.push_back(sum);
                cur = e.index;
                sum = e.value;
            }
        }

        out.sparse_logprob_indices.push_back(cur);
        out.sparse_logprob_values.push_back(sum);
    }

    void backward_selected(
        const DecodeResult& fwd,
        const float* grad_selected_weights,
        const float* grad_final_scores,
        const float* next_raw_grad,
        float* prev_raw_grad,
        float* grad_log_probs,
        int t
    ) const {
        const int K = fwd.beam_size;

        const float* weights =
            fwd.weights.data() + static_cast<size_t>(t) * K;

        const float* gw =
            grad_selected_weights
                ? grad_selected_weights + static_cast<size_t>(t) * K
                : nullptr;

        const float weighted_dot =
            gw ? dot_simd_or_scalar(weights, gw, K) : 0.0f;

        for (int k = 0; k < K; ++k) {
            const size_t idx = static_cast<size_t>(t) * K + k;

            const int parent = fwd.parents[idx];
            const int token = fwd.tokens[idx];

            if (parent < 0) continue;

            float drank = 0.0f;

            if (gw) {
                const float w = weights[k];
                drank +=
                    (w * (gw[k] - weighted_dot)) /
                    fwd.selected_temperature;
            }

            if (grad_final_scores && t == fwd.steps - 1) {
                drank += grad_final_scores[k];
            }

            const int len = std::max(1, static_cast<int>(fwd.lengths[idx]));
            const float inv_penalty =
                1.0f / gnmt_length_penalty(len, fwd.length_penalty_alpha);

            const float draw = next_raw_grad[k] + drank * inv_penalty;

            scatter_raw_candidate_grad(
                fwd,
                grad_log_probs,
                prev_raw_grad,
                t,
                parent,
                token,
                fwd.from_logprob[idx],
                draw
            );
        }
    }

    void backward_relaxed_topk(
        const DecodeResult& fwd,
        const float* grad_relaxed_weights,
        float* prev_raw_grad,
        float* grad_log_probs,
        int t
    ) const {
        if (!grad_relaxed_weights) return;

        constexpr float EPS = 1.0e-12f;

        const int P = fwd.relaxed_pool_size;

        const float* w =
            fwd.relaxed_weights.data() + static_cast<size_t>(t) * P;

        const float* gw =
            grad_relaxed_weights + static_cast<size_t>(t) * P;

        float denom = 0.0f;
        float numer = 0.0f;

        for (int p = 0; p < P; ++p) {
            const float a = w[p] * (1.0f - w[p]);
            denom += a;
            numer += gw[p] * a;
        }

        if (denom <= EPS) return;

        const float center = numer / denom;
        const float inv_temp = 1.0f / fwd.soft_topk_temperature;

        for (int p = 0; p < P; ++p) {
            const size_t idx = static_cast<size_t>(t) * P + p;

            const int parent = fwd.pool_parents[idx];
            const int token = fwd.pool_tokens[idx];

            if (parent < 0) continue;

            const float a = w[p] * (1.0f - w[p]);
            const float drank = a * inv_temp * (gw[p] - center);

            if (drank == 0.0f) continue;

            const int len = std::max(1, static_cast<int>(fwd.pool_lengths[idx]));
            const float inv_penalty =
                1.0f / gnmt_length_penalty(len, fwd.length_penalty_alpha);

            scatter_raw_candidate_grad(
                fwd,
                grad_log_probs,
                prev_raw_grad,
                t,
                parent,
                token,
                fwd.pool_from_logprob[idx],
                drank * inv_penalty
            );
        }
    }

    void backward_selected_sparse(
        const DecodeResult& fwd,
        const float* grad_selected_weights,
        const float* grad_final_scores,
        const float* next_raw_grad,
        float* prev_raw_grad,
        std::vector<SparseGradEntry>& entries,
        int t
    ) const {
        const int K = fwd.beam_size;

        const float* weights =
            fwd.weights.data() + static_cast<size_t>(t) * K;

        const float* gw =
            grad_selected_weights
                ? grad_selected_weights + static_cast<size_t>(t) * K
                : nullptr;

        const float weighted_dot = gw ? dot_simd_or_scalar(weights, gw, K) : 0.0f;

        for (int k = 0; k < K; ++k) {
            const size_t idx = static_cast<size_t>(t) * K + k;

            const int parent = fwd.parents[idx];
            const int token = fwd.tokens[idx];

            if (parent < 0) continue;

            float drank = 0.0f;

            if (gw) {
                const float w = weights[k];
                drank += (w * (gw[k] - weighted_dot)) / fwd.selected_temperature;
            }

            if (grad_final_scores && t == fwd.steps - 1) {
                drank += grad_final_scores[k];
            }

            const int len = std::max(1, static_cast<int>(fwd.lengths[idx]));
            const float inv_penalty = 1.0f / gnmt_length_penalty(len, fwd.length_penalty_alpha);

            const float draw = next_raw_grad[k] + drank * inv_penalty;

            scatter_raw_candidate_grad_sparse(
                fwd,
                entries,
                prev_raw_grad,
                t,
                parent,
                token,
                fwd.from_logprob[idx],
                draw
            );
        }
    }

    void backward_relaxed_topk_sparse(
        const DecodeResult& fwd,
        const float* grad_relaxed_weights,
        float* prev_raw_grad,
        std::vector<SparseGradEntry>& entries,
        int t
    ) const {
        if (!grad_relaxed_weights) return;

        constexpr float EPS = 1.0e-12f;

        const int P = fwd.relaxed_pool_size;

        const float* w =
            fwd.relaxed_weights.data() + static_cast<size_t>(t) * P;

        const float* gw =
            grad_relaxed_weights + static_cast<size_t>(t) * P;

        float denom = 0.0f;
        float numer = 0.0f;

        for (int p = 0; p < P; ++p) {
            const float a = w[p] * (1.0f - w[p]);
            denom += a;
            numer += gw[p] * a;
        }

        if (denom <= EPS) return;

        const float center = numer / denom;
        const float inv_temp = 1.0f / fwd.soft_topk_temperature;

        for (int p = 0; p < P; ++p) {
            const size_t idx = static_cast<size_t>(t) * P + p;

            const int parent = fwd.pool_parents[idx];
            const int token = fwd.pool_tokens[idx];

            if (parent < 0) continue;

            const float a = w[p] * (1.0f - w[p]);
            const float drank = a * inv_temp * (gw[p] - center);

            if (drank == 0.0f) continue;

            const int len = std::max(1, static_cast<int>(fwd.pool_lengths[idx]));
            const float inv_penalty = 1.0f / gnmt_length_penalty(len, fwd.length_penalty_alpha);

            scatter_raw_candidate_grad_sparse(
                fwd,
                entries,
                prev_raw_grad,
                t,
                parent,
                token,
                fwd.pool_from_logprob[idx],
                drank * inv_penalty
            );
        }
    }

    static std::vector<std::vector<int32_t>> build_sequences(const DecodeResult& r) {
        const int T = r.steps;
        const int K = r.beam_size;

        std::vector<std::vector<int32_t>> seqs(K);

        for (int k = 0; k < K; ++k) {
            std::vector<int32_t> seq(T, -1);

            int cur = k;

            for (int t = T - 1; t >= 0; --t) {
                if (cur < 0) break;

                const size_t idx = static_cast<size_t>(t) * K + cur;

                seq[t] = r.tokens[idx];
                cur = r.parents[idx];
            }

            if (r.eos_token >= 0) {
                auto it = std::find(seq.begin(), seq.end(), r.eos_token);

                if (it != seq.end()) {
                    seq.erase(it, seq.end());
                }
            }
            seq.erase(
                seq.begin(),
                std::find_if(seq.begin(), seq.end(), [](int32_t token) { return token >= 0; })
            );

            seqs[k] = std::move(seq);
        }

        return seqs;
    }
};

namespace internal_test {

namespace {

static void reset_report(ParityReport* report) {
    if (!report) return;
    report->cases_run = 0;
    report->simd_paths_run = 0;
    report->failures = 0;
    report->message[0] = '\0';
}

static void report_failure(ParityReport* report, const char* detail) {
    if (!report) return;
    ++report->failures;
    if (report->message[0] == '\0') {
        std::snprintf(report->message, sizeof(report->message), "%s", detail);
    }
}

static bool close_enough(float a, float b, float atol, float rtol) {
    if (a == b) return true;
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    const float scale = std::max(1.0f, std::fabs(b));
    return std::fabs(a - b) <= atol + rtol * scale;
}

template <class ActualVec, class ExpectedVec>
static bool compare_float_vectors(
    const ActualVec& actual,
    const ExpectedVec& expected,
    float atol,
    float rtol,
    const char* label,
    ParityReport* report
) {
    if (actual.size() != expected.size()) {
        char msg[256];
        std::snprintf(msg, sizeof(msg), "%s size mismatch: %zu != %zu", label, actual.size(), expected.size());
        report_failure(report, msg);
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!close_enough(actual[i], expected[i], atol, rtol)) {
            char msg[256];
            std::snprintf(
                msg,
                sizeof(msg),
                "%s[%zu] mismatch: actual=%g expected=%g",
                label,
                i,
                static_cast<double>(actual[i]),
                static_cast<double>(expected[i]));
            report_failure(report, msg);
            return false;
        }
    }
    return true;
}

template <class ActualVec, class ExpectedVec>
static bool compare_int_vectors(
    const ActualVec& actual,
    const ExpectedVec& expected,
    const char* label,
    ParityReport* report
) {
    if (actual.size() != expected.size()) {
        char msg[256];
        std::snprintf(msg, sizeof(msg), "%s size mismatch: %zu != %zu", label, actual.size(), expected.size());
        report_failure(report, msg);
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            char msg[256];
            std::snprintf(
                msg,
                sizeof(msg),
                "%s[%zu] mismatch: actual=%lld expected=%lld",
                label,
                i,
                static_cast<long long>(actual[i]),
                static_cast<long long>(expected[i]));
            report_failure(report, msg);
            return false;
        }
    }
    return true;
}

struct KernelOverrideScope {
    bool previous_enabled;
    KernelPath previous_path;

    explicit KernelOverrideScope(KernelPath path)
        : previous_enabled(g_kernel_override_enabled),
          previous_path(g_kernel_override) {
        g_kernel_override_enabled = true;
        g_kernel_override = path;
    }

    ~KernelOverrideScope() {
        g_kernel_override_enabled = previous_enabled;
        g_kernel_override = previous_path;
    }
};

static size_t lp_index(int t, int k, int v, int K, int V) {
    return (static_cast<size_t>(t) * static_cast<size_t>(K) + static_cast<size_t>(k)) * static_cast<size_t>(V) + static_cast<size_t>(v);
}

struct InternalParityCase {
    const char* name;
    BeamOptions options;
    int steps;
    int vocab_size;
    std::vector<float> log_probs;
    std::vector<uint8_t> banned_tokens;
    std::vector<int32_t> forced_tokens;
    int constraint_min_length;
    bool expect_invalid;
};

static BeamOptions base_options(int beam_size) {
    BeamOptions opt;
    opt.beam_size = beam_size;
    opt.selected_temperature = 0.85f;
    opt.soft_topk_temperature = 0.4f;
    opt.relaxed_pool_multiplier = 3;
    opt.vocab_block = 3;
    opt.soft_topk_tolerance = 1.0e-5f;
    opt.soft_topk_max_iters = 72;
    opt.validate_inputs = 1;
    return opt;
}

static std::vector<float> filled_log_probs(int steps, int beam_size, int vocab_size, float value) {
    return std::vector<float>(
        static_cast<size_t>(steps) * static_cast<size_t>(beam_size) * static_cast<size_t>(vocab_size),
        value);
}

static std::vector<InternalParityCase> build_internal_parity_cases() {
    std::vector<InternalParityCase> cases;

    {
        const int T = 3, K = 3, V = 6;
        BeamOptions opt = base_options(K);
        std::vector<float> x = filled_log_probs(T, K, V, -8.0f);
        for (int t = 0; t < T; ++t) {
            for (int k = 0; k < K; ++k) {
                x[lp_index(t, k, 0, K, V)] = -0.25f;
                x[lp_index(t, k, 1, K, V)] = -0.25f;
                x[lp_index(t, k, 2, K, V)] = -0.5f;
            }
        }
        x[lp_index(1, 1, 3, K, V)] = -0.25f;
        x[lp_index(2, 2, 4, K, V)] = -0.25f;
        cases.push_back(InternalParityCase{"ties", opt, T, V, x, {}, {}, -1, false});
    }

    {
        const int T = 4, K = 3, V = 5;
        BeamOptions opt = base_options(K);
        opt.eos_token = 2;
        std::vector<float> x = filled_log_probs(T, K, V, -7.0f);
        x[lp_index(0, 0, 2, K, V)] = 0.0f;
        x[lp_index(0, 0, 3, K, V)] = -0.1f;
        x[lp_index(0, 0, 1, K, V)] = -0.2f;
        for (int t = 1; t < T; ++t) {
            for (int k = 0; k < K; ++k) {
                x[lp_index(t, k, 4, K, V)] = -0.05f;
                x[lp_index(t, k, 0, K, V)] = -0.2f;
            }
        }
        cases.push_back(InternalParityCase{"eos_carry_forward", opt, T, V, x, {}, {}, -1, false});
    }

    {
        const int T = 3, K = 4, V = 7;
        BeamOptions opt = base_options(K);
        std::vector<float> x = filled_log_probs(T, K, V, -std::numeric_limits<float>::infinity());
        for (int t = 0; t < T; ++t) {
            for (int k = 0; k < K; ++k) {
                x[lp_index(t, k, (k + t) % V, K, V)] = -0.1f * static_cast<float>(k + 1);
                x[lp_index(t, k, (k + t + 2) % V, K, V)] = -0.4f;
            }
        }
        cases.push_back(InternalParityCase{"negative_infinity", opt, T, V, x, {}, {}, -1, false});
    }

    {
        const int T = 2, K = 3, V = 8;
        BeamOptions opt = base_options(K);
        opt.validate_inputs = 0;
        std::vector<float> x = filled_log_probs(T, K, V, -4.0f);
        for (int t = 0; t < T; ++t) {
            for (int k = 0; k < K; ++k) {
                x[lp_index(t, k, 1, K, V)] = -0.05f;
                x[lp_index(t, k, 3, K, V)] = -0.10f;
                x[lp_index(t, k, 6, K, V)] = -0.25f;
            }
        }
        x[lp_index(0, 0, 2, K, V)] = std::numeric_limits<float>::infinity();
        x[lp_index(1, 1, 5, K, V)] = std::numeric_limits<float>::infinity();
        cases.push_back(InternalParityCase{"positive_infinity_validation_disabled", opt, T, V, x, {}, {}, -1, false});
    }

    {
        const int T = 3, K = 3, V = 7;
        BeamOptions opt = base_options(K);
        opt.eos_token = 5;
        opt.min_length = 2;
        opt.length_penalty_alpha = 0.35f;
        std::vector<float> x = filled_log_probs(T, K, V, -6.0f);
        for (int t = 0; t < T; ++t) {
            for (int k = 0; k < K; ++k) {
                x[lp_index(t, k, 0, K, V)] = -0.05f;
                x[lp_index(t, k, 3, K, V)] = -0.04f;
                x[lp_index(t, k, 5, K, V)] = -0.01f;
            }
        }
        std::vector<uint8_t> banned(static_cast<size_t>(V), 0);
        banned[0] = 1;
        banned[4] = 1;
        std::vector<int32_t> forced = {-1, 3, -1};
        cases.push_back(InternalParityCase{"constraints_min_length_forced_banned", opt, T, V, x, banned, forced, 2, false});
    }

    {
        const int T = 5, K = 4, V = 8;
        BeamOptions opt = base_options(K);
        opt.eos_token = 6;
        opt.length_penalty_alpha = 0.75f;
        std::vector<float> x = filled_log_probs(T, K, V, -5.0f);
        for (int t = 0; t < T; ++t) {
            for (int k = 0; k < K; ++k) {
                x[lp_index(t, k, (t + k) % V, K, V)] = -0.08f;
                x[lp_index(t, k, 6, K, V)] = (t < 2) ? -0.6f : -0.04f;
                x[lp_index(t, k, 7, K, V)] = -0.12f;
            }
        }
        cases.push_back(InternalParityCase{"length_penalty", opt, T, V, x, {}, {}, -1, false});
    }

    {
        const int T = 2, K = 2, V = 4;
        BeamOptions opt = base_options(K);
        std::vector<float> x = filled_log_probs(T, K, V, -1.0f);
        x[lp_index(1, 1, 2, K, V)] = std::numeric_limits<float>::quiet_NaN();
        cases.push_back(InternalParityCase{"nan_rejection", opt, T, V, x, {}, {}, -1, true});
    }

    return cases;
}

struct DecodeBackwardBundle {
    DecodeResult forward;
    BackwardResult dense_backward;
    BackwardResult sparse_backward;
};

static DecodeBackwardBundle run_decode_backward_case(const InternalParityCase& tc, KernelPath path) {
    KernelOverrideScope override(path);
    DifferentiableBeamSearchAVX512 decoder(tc.options);

    DecodeConstraints constraints;
    DecodeConstraints* constraint_ptr = nullptr;
    if (!tc.banned_tokens.empty() || !tc.forced_tokens.empty() || tc.constraint_min_length >= 0) {
        constraints.banned_tokens = tc.banned_tokens.empty() ? nullptr : tc.banned_tokens.data();
        constraints.forced_tokens = tc.forced_tokens.empty() ? nullptr : tc.forced_tokens.data();
        constraints.min_length = tc.constraint_min_length;
        constraint_ptr = &constraints;
    }

    DecodeBackwardBundle bundle;
    bundle.forward = decoder.decode_constrained(tc.log_probs.data(), tc.steps, tc.vocab_size, constraint_ptr);

    const size_t selected_count = static_cast<size_t>(bundle.forward.steps) * static_cast<size_t>(bundle.forward.beam_size);
    const size_t pool_count = static_cast<size_t>(bundle.forward.steps) * static_cast<size_t>(bundle.forward.relaxed_pool_size);

    std::vector<float> grad_selected(selected_count, 0.0f);
    std::vector<float> grad_relaxed(pool_count, 0.0f);
    std::vector<float> grad_final(static_cast<size_t>(bundle.forward.beam_size), 0.0f);
    for (size_t i = 0; i < grad_selected.size(); ++i) {
        grad_selected[i] = static_cast<float>((static_cast<int>(i % 7) - 3)) * 0.125f;
    }
    for (size_t i = 0; i < grad_relaxed.size(); ++i) {
        grad_relaxed[i] = static_cast<float>((static_cast<int>(i % 11) - 5)) * 0.03125f;
    }
    for (size_t i = 0; i < grad_final.size(); ++i) {
        grad_final[i] = static_cast<float>(static_cast<int>(i) + 1) * 0.2f;
    }

    bundle.dense_backward = decoder.backward(bundle.forward, grad_selected.data(), grad_relaxed.data(), grad_final.data());
    bundle.sparse_backward = decoder.backward_sparse(bundle.forward, grad_selected.data(), grad_relaxed.data(), grad_final.data());
    return bundle;
}

static bool compare_decode_backward_bundles(
    const DecodeBackwardBundle& actual,
    const DecodeBackwardBundle& expected,
    const char* case_name,
    const char* path_name,
    ParityReport* report
) {
    char label[128];

    std::snprintf(label, sizeof(label), "%s/%s parents", path_name, case_name);
    if (!compare_int_vectors(actual.forward.parents, expected.forward.parents, label, report)) return false;
    std::snprintf(label, sizeof(label), "%s/%s tokens", path_name, case_name);
    if (!compare_int_vectors(actual.forward.tokens, expected.forward.tokens, label, report)) return false;
    std::snprintf(label, sizeof(label), "%s/%s lengths", path_name, case_name);
    if (!compare_int_vectors(actual.forward.lengths, expected.forward.lengths, label, report)) return false;

    std::snprintf(label, sizeof(label), "%s/%s final_scores", path_name, case_name);
    if (!compare_float_vectors(actual.forward.final_scores, expected.forward.final_scores, 2.0e-4f, 2.0e-4f, label, report)) return false;
    std::snprintf(label, sizeof(label), "%s/%s weights", path_name, case_name);
    if (!compare_float_vectors(actual.forward.weights, expected.forward.weights, 3.0e-3f, 3.0e-3f, label, report)) return false;
    std::snprintf(label, sizeof(label), "%s/%s relaxed_weights", path_name, case_name);
    if (!compare_float_vectors(actual.forward.relaxed_weights, expected.forward.relaxed_weights, 5.0e-3f, 5.0e-3f, label, report)) return false;

    std::snprintf(label, sizeof(label), "%s/%s dense_grad_log_probs", path_name, case_name);
    if (!compare_float_vectors(actual.dense_backward.grad_log_probs, expected.dense_backward.grad_log_probs, 8.0e-3f, 8.0e-3f, label, report)) return false;
    std::snprintf(label, sizeof(label), "%s/%s dense_grad_initial", path_name, case_name);
    if (!compare_float_vectors(actual.dense_backward.grad_initial_scores, expected.dense_backward.grad_initial_scores, 8.0e-3f, 8.0e-3f, label, report)) return false;

    std::snprintf(label, sizeof(label), "%s/%s sparse_indices", path_name, case_name);
    if (!compare_int_vectors(actual.sparse_backward.sparse_logprob_indices, expected.sparse_backward.sparse_logprob_indices, label, report)) return false;
    std::snprintf(label, sizeof(label), "%s/%s sparse_values", path_name, case_name);
    if (!compare_float_vectors(actual.sparse_backward.sparse_logprob_values, expected.sparse_backward.sparse_logprob_values, 8.0e-3f, 8.0e-3f, label, report)) return false;
    std::snprintf(label, sizeof(label), "%s/%s sparse_grad_initial", path_name, case_name);
    return compare_float_vectors(actual.sparse_backward.grad_initial_scores, expected.sparse_backward.grad_initial_scores, 8.0e-3f, 8.0e-3f, label, report);
}

static std::vector<KernelPath> available_simd_paths() {
    std::vector<KernelPath> paths;
#if DBS_CAN_COMPILE_AVX512
    if (runtime_has_avx512()) paths.push_back(KernelPath::AVX512);
#endif
#if DBS_CAN_COMPILE_AVX2
    if (runtime_has_avx2()) paths.push_back(KernelPath::AVX2);
#endif
#if DBS_CAN_COMPILE_SSE42
    if (runtime_has_sse42()) paths.push_back(KernelPath::SSE42);
#endif
#if DBS_ARM_NEON
    if (runtime_has_neon()) paths.push_back(KernelPath::NEON);
#endif
    return paths;
}

#if DBS_CAN_COMPILE_AVX512
DBS_AVX512_TARGET static int run_avx512_vector_math_parity_impl(ParityReport* report) {
    {
        alignas(64) float input[16] = {
            -12.0f, -8.0f, -4.0f, -2.0f, -1.0f, -0.5f, -0.125f, 0.0f,
             0.125f, 0.5f, 1.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f,
        };
        alignas(64) float output[16] = {};
        const __m512 y = avx512::exp512_ps(_mm512_load_ps(input));
        _mm512_store_ps(output, y);
        for (int i = 0; i < 16; ++i) {
            if (!close_enough(output[i], safe_exp_scalar(input[i]), 2.5e-4f, 2.5e-4f)) {
                report_failure(report, "avx512 exp512_ps diverged from scalar exp");
                return 1;
            }
        }
        if (report) ++report->cases_run;
    }

    {
        std::vector<float> scores = {
            -4.0f, -1.5f, -0.25f, 0.0f, 0.125f, 0.5f, 1.0f, 2.0f,
            -1.0e31f, -3.0f, 3.5f, -2.25f, 0.75f, -0.75f, 1.5f, -5.0f,
            2.75f, -1.25f, 0.25f,
        };
        std::vector<float> scalar(scores.size(), 0.0f);
        std::vector<float> simd(scores.size(), 0.0f);
        softmax_selected_scalar(scores.data(), scalar.data(), static_cast<int>(scores.size()), 0.7f);
        avx512::softmax_selected(scores.data(), simd.data(), static_cast<int>(scores.size()), 0.7f);
        if (!compare_float_vectors(simd, scalar, 8.0e-4f, 8.0e-4f, "avx512 selected softmax", report)) return 1;
        if (report) ++report->cases_run;
    }

    {
        std::vector<float> scores = {
            3.0f, 2.75f, 2.0f, 1.25f, 0.5f, 0.25f, -0.25f, -0.75f,
            -1.0f, -1.5f, -2.0f, -3.5f, -1.0e31f, 1.0f, 1.5f, -4.0f,
            0.0f, 2.25f, -2.5f, 0.875f,
        };
        std::vector<float> scalar(scores.size(), 0.0f);
        std::vector<float> simd(scores.size(), 0.0f);
        soft_topk_inclusion(scores.data(), scalar.data(), static_cast<int>(scores.size()), 5, 0.35f, 1.0e-5f, 80);

        KernelOverrideScope override(KernelPath::AVX512);
        soft_topk_inclusion(scores.data(), simd.data(), static_cast<int>(scores.size()), 5, 0.35f, 1.0e-5f, 80);
        if (!compare_float_vectors(simd, scalar, 1.5e-3f, 1.5e-3f, "avx512 relaxed topk", report)) return 1;
        if (report) ++report->cases_run;
    }

    if (report) report->simd_paths_run = 1;
    return report && report->failures ? 1 : 0;
}
#endif

} // namespace

SimdCapabilities simd_capabilities() {
    SimdCapabilities caps{};
    caps.can_compile_avx512 = DBS_CAN_COMPILE_AVX512;
    caps.can_compile_avx2 = DBS_CAN_COMPILE_AVX2;
    caps.can_compile_sse42 = DBS_CAN_COMPILE_SSE42;
    caps.can_compile_neon = DBS_ARM_NEON;
    caps.runtime_avx512 = runtime_has_avx512() ? 1 : 0;
    caps.runtime_avx2 = runtime_has_avx2() ? 1 : 0;
    caps.runtime_sse42 = runtime_has_sse42() ? 1 : 0;
    caps.runtime_neon = runtime_has_neon() ? 1 : 0;
    return caps;
}

int run_avx512_vector_math_parity(ParityReport* report) {
    reset_report(report);
#if DBS_CAN_COMPILE_AVX512
    if (!runtime_has_avx512()) {
        if (report) std::snprintf(report->message, sizeof(report->message), "AVX-512 runtime support unavailable");
        return 0;
    }
    return run_avx512_vector_math_parity_impl(report);
#else
    if (report) std::snprintf(report->message, sizeof(report->message), "AVX-512 compile support unavailable");
    return 0;
#endif
}

int run_scalar_vs_simd_decode_backward_parity(ParityReport* report) {
    reset_report(report);
    const std::vector<KernelPath> simd_paths = available_simd_paths();
    if (report) report->simd_paths_run = static_cast<int>(simd_paths.size());
    if (simd_paths.empty()) {
        if (report) std::snprintf(report->message, sizeof(report->message), "SIMD runtime support unavailable");
        return 0;
    }

    const std::vector<InternalParityCase> cases = build_internal_parity_cases();
    for (const InternalParityCase& tc : cases) {
        if (tc.expect_invalid) {
            bool scalar_rejected = false;
            try {
                (void)run_decode_backward_case(tc, KernelPath::Scalar);
            } catch (const std::invalid_argument&) {
                scalar_rejected = true;
            }
            if (!scalar_rejected) {
                report_failure(report, "scalar path accepted invalid NaN input");
                return 1;
            }

            for (KernelPath path : simd_paths) {
                bool simd_rejected = false;
                try {
                    (void)run_decode_backward_case(tc, path);
                } catch (const std::invalid_argument&) {
                    simd_rejected = true;
                }
                if (!simd_rejected) {
                    char msg[256];
                    std::snprintf(msg, sizeof(msg), "%s path accepted invalid NaN input", kernel_path_name(path));
                    report_failure(report, msg);
                    return 1;
                }
                if (report) ++report->cases_run;
            }
            continue;
        }

        DecodeBackwardBundle scalar;
        try {
            scalar = run_decode_backward_case(tc, KernelPath::Scalar);
        } catch (const std::exception& exc) {
            char msg[256];
            std::snprintf(msg, sizeof(msg), "scalar path failed case %s: %s", tc.name, exc.what());
            report_failure(report, msg);
            return 1;
        }

        for (KernelPath path : simd_paths) {
            DecodeBackwardBundle simd;
            try {
                simd = run_decode_backward_case(tc, path);
            } catch (const std::exception& exc) {
                char msg[256];
                std::snprintf(msg, sizeof(msg), "%s path failed case %s: %s", kernel_path_name(path), tc.name, exc.what());
                report_failure(report, msg);
                return 1;
            }

            if (!compare_decode_backward_bundles(simd, scalar, tc.name, kernel_path_name(path), report)) {
                return 1;
            }
            if (report) ++report->cases_run;
        }
    }

    return report && report->failures ? 1 : 0;
}

} // namespace internal_test

} // namespace dbs

extern "C" {

struct DBSOptionsC {
    int beam_size;
    int eos_token;
    float selected_temperature;
    float soft_topk_temperature;
    int relaxed_pool_multiplier;
    int vocab_block;
    float length_penalty_alpha;
    float soft_topk_tolerance;
    int soft_topk_max_iters;
    int min_length;
    int validate_inputs;
    int max_dense_gradient_elements;
    int reserved0;
    int reserved1;
};

struct DBSDecoderHandle;
struct DBSResultHandle;
struct DBSBackwardHandle;
struct DBSBatchResultHandle;
struct DBSWorkspaceHandle;

typedef int (*DBSModelStepFn)(
    void* user_data,
    int batch_index,
    int step,
    const int32_t* prev_tokens,
    const float* prev_scores,
    int beam_size,
    int vocab_size,
    float* out_log_probs
);

typedef int (*DBSTokenFilterFn)(
    void* user_data,
    int batch_index,
    int step,
    int parent_beam,
    const int32_t* prefix_tokens,
    int prefix_len,
    int token
);

struct DBSAdvancedConstraintsC {
    const uint8_t* banned_tokens;
    const int32_t* forced_tokens;
    int min_length;
    float repetition_penalty;
    int no_repeat_ngram_size;
    DBSTokenFilterFn token_filter;
    void* token_filter_user_data;
    int batch_index;
};

struct DBSStatsC {
    int abi_version;
    int last_kernel;
    int used_sparse_backward;
    int used_dense_backward;
    int used_batch_threads;
    int used_model_step_callback;
    int last_error_category;
    int64_t last_decode_ns;
    int64_t last_backward_ns;
    int64_t last_allocation_bytes;
    int64_t last_selected_count;
    int64_t last_pool_count;
    int64_t last_logprob_count;
    int64_t last_sparse_grad_count;
    int64_t total_allocator_calls;
    int64_t total_allocator_bytes;
    int64_t reserved_stats0;
    int64_t reserved_stats1;
};

enum DBSDataTypeC {
    DBS_DTYPE_F32 = 0,
    DBS_DTYPE_F16 = 1,
    DBS_DTYPE_BF16 = 2
};

DBS_EXPORT int dbs_abi_version();
DBS_EXPORT const char* dbs_version_string();
DBS_EXPORT const char* dbs_last_global_error();
DBS_EXPORT int dbs_workspace_create(DBSWorkspaceHandle** out_workspace);
DBS_EXPORT void dbs_workspace_destroy(DBSWorkspaceHandle* workspace);
DBS_EXPORT int dbs_workspace_reserve(DBSWorkspaceHandle* workspace, int64_t float_count, int64_t int_count);
DBS_EXPORT int64_t dbs_workspace_allocated_bytes(DBSWorkspaceHandle* workspace);
DBS_EXPORT int dbs_create_ex(DBSOptionsC options, DBSDecoderHandle** out_handle);
DBS_EXPORT DBSDecoderHandle* dbs_create(DBSOptionsC options);
DBS_EXPORT void dbs_destroy(DBSDecoderHandle* handle);
DBS_EXPORT const char* dbs_last_error(DBSDecoderHandle* handle);

DBS_EXPORT int dbs_decode(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int steps,
    int vocab_size,
    DBSResultHandle** out_result
);

DBS_EXPORT int dbs_decode_typed(
    DBSDecoderHandle* handle,
    const void* log_probs,
    int data_type,
    int steps,
    int vocab_size,
    DBSResultHandle** out_result
);

DBS_EXPORT int dbs_decode_batch_typed(
    DBSDecoderHandle* handle,
    const void* log_probs,
    int data_type,
    int batch_size,
    int steps,
    int vocab_size,
    int num_threads,
    DBSBatchResultHandle** out_result
);

DBS_EXPORT int dbs_decode_constrained(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int steps,
    int vocab_size,
    const uint8_t* banned_tokens,
    const int32_t* forced_tokens,
    int min_length,
    DBSResultHandle** out_result
);

DBS_EXPORT int dbs_decode_constrained_ex(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int steps,
    int vocab_size,
    const DBSAdvancedConstraintsC* constraints,
    DBSResultHandle** out_result
);

DBS_EXPORT int dbs_decode_model_steps(
    DBSDecoderHandle* handle,
    DBSModelStepFn step_fn,
    void* user_data,
    int batch_index,
    int steps,
    int vocab_size,
    DBSResultHandle** out_result
);

DBS_EXPORT int dbs_decode_model_steps_with_workspace(
    DBSDecoderHandle* handle,
    DBSWorkspaceHandle* workspace,
    DBSModelStepFn step_fn,
    void* user_data,
    int batch_index,
    int steps,
    int vocab_size,
    DBSResultHandle** out_result
);

DBS_EXPORT int dbs_decode_batch(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int batch_size,
    int steps,
    int vocab_size,
    int num_threads,
    DBSBatchResultHandle** out_result
);

DBS_EXPORT int dbs_decode_batch_variable(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int batch_size,
    int max_steps,
    int max_beam_size,
    int vocab_size,
    const int32_t* steps_per_example,
    const int32_t* beam_sizes_per_example,
    const int32_t* eos_tokens_per_example,
    const int32_t* min_lengths_per_example,
    const uint8_t* banned_tokens_per_example,
    const int32_t* forced_tokens_per_example,
    int num_threads,
    DBSBatchResultHandle** out_result
);

DBS_EXPORT int dbs_backward(
    DBSDecoderHandle* handle,
    const DBSResultHandle* result,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores,
    DBSBackwardHandle** out_backward
);

DBS_EXPORT int dbs_backward_dense(
    DBSDecoderHandle* handle,
    const DBSResultHandle* result,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores,
    DBSBackwardHandle** out_backward
);

DBS_EXPORT int dbs_backward_sparse(
    DBSDecoderHandle* handle,
    const DBSResultHandle* result,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores,
    DBSBackwardHandle** out_backward
);

DBS_EXPORT int dbs_backward_default(
    DBSDecoderHandle* handle,
    const DBSResultHandle* result,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores,
    DBSBackwardHandle** out_backward
);

DBS_EXPORT void dbs_free_result(DBSResultHandle* result);
DBS_EXPORT void dbs_free_batch_result(DBSBatchResultHandle* result);
DBS_EXPORT int dbs_batch_result_size(const DBSBatchResultHandle* result);
DBS_EXPORT const DBSResultHandle* dbs_batch_result_at(const DBSBatchResultHandle* result, int batch_index);
DBS_EXPORT void dbs_free_backward(DBSBackwardHandle* result);

DBS_EXPORT int dbs_result_steps(const DBSResultHandle* result);
DBS_EXPORT int dbs_result_beam_size(const DBSResultHandle* result);
DBS_EXPORT int dbs_result_vocab_size(const DBSResultHandle* result);
DBS_EXPORT int dbs_result_pool_size(const DBSResultHandle* result);

DBS_EXPORT const int32_t* dbs_result_tokens(const DBSResultHandle* result);
DBS_EXPORT const int32_t* dbs_result_parents(const DBSResultHandle* result);

DBS_EXPORT const float* dbs_result_scores(const DBSResultHandle* result);
DBS_EXPORT const float* dbs_result_raw_scores(const DBSResultHandle* result);
DBS_EXPORT const float* dbs_result_relaxed_weights(const DBSResultHandle* result);

DBS_EXPORT const int32_t* dbs_result_pool_tokens(const DBSResultHandle* result);
DBS_EXPORT const int32_t* dbs_result_pool_parents(const DBSResultHandle* result);
DBS_EXPORT const float* dbs_result_pool_scores(const DBSResultHandle* result);

DBS_EXPORT const float* dbs_backward_grad_log_probs(const DBSBackwardHandle* result);
DBS_EXPORT const float* dbs_backward_grad_initial_scores(const DBSBackwardHandle* result);
DBS_EXPORT const int64_t* dbs_backward_sparse_logprob_indices(const DBSBackwardHandle* result);
DBS_EXPORT const float* dbs_backward_sparse_logprob_values(const DBSBackwardHandle* result);
DBS_EXPORT int64_t dbs_backward_sparse_logprob_count(const DBSBackwardHandle* result);

DBS_EXPORT const int32_t* dbs_result_lengths(const DBSResultHandle* result);
DBS_EXPORT const int32_t* dbs_result_pool_lengths(const DBSResultHandle* result);

DBS_EXPORT const float* dbs_result_weights(const DBSResultHandle* result);
DBS_EXPORT const float* dbs_result_final_scores(const DBSResultHandle* result);
DBS_EXPORT const float* dbs_result_final_raw_scores(const DBSResultHandle* result);
DBS_EXPORT const float* dbs_result_pool_raw_scores(const DBSResultHandle* result);

DBS_EXPORT int64_t dbs_result_selected_count(const DBSResultHandle* result);
DBS_EXPORT int64_t dbs_result_pool_count(const DBSResultHandle* result);
DBS_EXPORT int64_t dbs_result_logprob_count(const DBSResultHandle* result);
DBS_EXPORT int64_t dbs_backward_grad_log_probs_count(const DBSResultHandle* result);
DBS_EXPORT int64_t dbs_backward_grad_initial_scores_count(const DBSResultHandle* result);
DBS_EXPORT int64_t dbs_result_eos_count(const DBSResultHandle* result, int eos_token);
DBS_EXPORT int dbs_result_validate_deterministic_order(const DBSResultHandle* result);
DBS_EXPORT int dbs_result_summary_json(const DBSResultHandle* result, int eos_token, char* out_json, int64_t out_json_capacity);
DBS_EXPORT int dbs_validate_production_gate_manifest(const char* manifest_json, char* out_error, int64_t out_error_capacity);

DBS_EXPORT int dbs_has_avx512();
DBS_EXPORT int dbs_has_avx2();
DBS_EXPORT int dbs_has_sse42();
DBS_EXPORT int dbs_has_neon();
DBS_EXPORT const char* dbs_selected_kernel_name();
DBS_EXPORT int dbs_get_stats(DBSDecoderHandle* handle, DBSStatsC* out_stats);
DBS_EXPORT int dbs_get_stats_json(DBSDecoderHandle* handle, char* out_json, int64_t out_json_capacity);
DBS_EXPORT int dbs_is_deterministic();
DBS_EXPORT void dbs_reset_stats(DBSDecoderHandle* handle);
}

struct DBSDecoderHandle {
    std::unique_ptr<dbs::DifferentiableBeamSearchAVX512> decoder;
    dbs::BeamOptions options;
    int beam_size = 0;
    mutable std::mutex error_mutex;
    std::string last_error;
    mutable std::mutex stats_mutex;
    DBSStatsC stats{};
    uint64_t deterministic_seed = 0;
};

struct DBSResultHandle {
    dbs::DecodeResult result;
};

struct DBSBackwardHandle {
    dbs::BackwardResult result;
};

struct DBSBatchResultHandle {
    std::vector<DBSResultHandle> results;
};

struct DBSWorkspaceHandle {
    std::mutex mutex;
    std::vector<float> f32;
    std::vector<int32_t> i32;
};

static thread_local std::string g_dbs_last_error;

static void dbs_set_error(DBSDecoderHandle* handle, const std::string& message) {
    g_dbs_last_error = message;
    if (handle) {
        std::lock_guard<std::mutex> lock(handle->error_mutex);
        handle->last_error = message;
    }
}

static void dbs_clear_error(DBSDecoderHandle* handle) {
    g_dbs_last_error.clear();
    if (handle) {
        std::lock_guard<std::mutex> lock(handle->error_mutex);
        handle->last_error.clear();
    }
}

static int classify_exception(const std::exception& e) {
    if (dynamic_cast<const std::invalid_argument*>(&e)) return 1;
    if (dynamic_cast<const std::bad_alloc*>(&e)) return 3;
    if (dynamic_cast<const std::overflow_error*>(&e)) return 4;
    if (dynamic_cast<const std::length_error*>(&e)) return 4;
    return 2;
}

static int64_t now_ns_since(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
}

static void dbs_mark_error(DBSDecoderHandle* handle, int category) {
    if (!handle) return;
    std::lock_guard<std::mutex> lock(handle->stats_mutex);
    handle->stats.last_error_category = category;
}

static void dbs_record_decode_stats(DBSDecoderHandle* handle, const dbs::DecodeResult& r, int64_t elapsed_ns, int threads, bool model_step) {
    if (!handle) return;
    std::lock_guard<std::mutex> lock(handle->stats_mutex);
    handle->stats.abi_version = DBS_ABI_VERSION;
    handle->stats.last_kernel = static_cast<int>(dbs::selected_kernel_path());
    handle->stats.used_batch_threads = threads;
    handle->stats.used_model_step_callback = model_step ? 1 : 0;
    handle->stats.last_decode_ns = elapsed_ns;
    handle->stats.last_selected_count = static_cast<int64_t>(r.steps) * r.beam_size;
    handle->stats.last_pool_count = static_cast<int64_t>(r.steps) * r.relaxed_pool_size;
    handle->stats.last_logprob_count = static_cast<int64_t>(r.steps) * r.beam_size * r.vocab_size;
    handle->stats.last_allocation_bytes =
        static_cast<int64_t>(r.tokens.size() * sizeof(int32_t) + r.parents.size() * sizeof(int32_t) +
                             r.scores.size() * sizeof(float) + r.raw_scores.size() * sizeof(float) +
                             r.pool_tokens.size() * sizeof(int32_t) + r.pool_scores.size() * sizeof(float) +
                             r.relaxed_weights.size() * sizeof(float));
    handle->stats.last_error_category = 0;
}

static void dbs_record_backward_stats(DBSDecoderHandle* handle, const dbs::BackwardResult& r, int64_t elapsed_ns) {
    if (!handle) return;
    std::lock_guard<std::mutex> lock(handle->stats_mutex);
    handle->stats.abi_version = DBS_ABI_VERSION;
    handle->stats.used_sparse_backward = r.sparse ? 1 : 0;
    handle->stats.used_dense_backward = r.sparse ? 0 : 1;
    handle->stats.last_backward_ns = elapsed_ns;
    handle->stats.last_sparse_grad_count = static_cast<int64_t>(r.sparse_logprob_values.size());
    handle->stats.last_error_category = 0;
}

static dbs::BeamOptions from_c_options(DBSOptionsC c) {
    dbs::BeamOptions o;

    o.beam_size = c.beam_size > 0 ? c.beam_size : 8;
    o.eos_token = c.eos_token;

    o.selected_temperature =
        c.selected_temperature > 0.0f ? c.selected_temperature : 1.0f;

    o.soft_topk_temperature =
        c.soft_topk_temperature > 0.0f ? c.soft_topk_temperature : 0.25f;

    o.relaxed_pool_multiplier =
        c.relaxed_pool_multiplier > 0 ? c.relaxed_pool_multiplier : 8;

    o.vocab_block = c.vocab_block > 0 ? c.vocab_block : 4096;

    o.length_penalty_alpha =
        c.length_penalty_alpha >= 0.0f ? c.length_penalty_alpha : 0.0f;

    o.soft_topk_tolerance =
        c.soft_topk_tolerance > 0.0f ? c.soft_topk_tolerance : 1.0e-4f;

    o.soft_topk_max_iters =
        c.soft_topk_max_iters > 0 ? c.soft_topk_max_iters : 48;

    o.min_length = c.min_length >= 0 ? c.min_length : 0;
    o.validate_inputs = c.validate_inputs == 0 ? 0 : 1;
    o.max_dense_gradient_elements =
        c.max_dense_gradient_elements > 0 ? c.max_dense_gradient_elements : 100000000;

    return o;
}

extern "C" DBS_EXPORT int dbs_abi_version() {
    return DBS_ABI_VERSION;
}

extern "C" DBS_EXPORT const char* dbs_version_string() {
    return "1.0.0rc9";
}

extern "C" DBS_EXPORT const char* dbs_last_global_error() {
    return g_dbs_last_error.empty() ? "" : g_dbs_last_error.c_str();
}


extern "C" DBS_EXPORT int dbs_workspace_create(DBSWorkspaceHandle** out_workspace) {
    if (out_workspace) *out_workspace = nullptr;
    if (!out_workspace) {
        dbs_set_error(nullptr, "out_workspace cannot be null");
        return -1;
    }
    try {
        *out_workspace = new DBSWorkspaceHandle;
        return 0;
    } catch (const std::exception& e) {
        dbs_set_error(nullptr, e.what());
        return -2;
    }
}

extern "C" DBS_EXPORT void dbs_workspace_destroy(DBSWorkspaceHandle* workspace) {
    delete workspace;
}

extern "C" DBS_EXPORT int dbs_workspace_reserve(DBSWorkspaceHandle* workspace, int64_t float_count, int64_t int_count) {
    if (!workspace || float_count < 0 || int_count < 0) return -1;
    try {
        std::lock_guard<std::mutex> lock(workspace->mutex);
        workspace->f32.reserve(static_cast<size_t>(float_count));
        workspace->i32.reserve(static_cast<size_t>(int_count));
        return 0;
    } catch (...) {
        return -2;
    }
}

extern "C" DBS_EXPORT int64_t dbs_workspace_allocated_bytes(DBSWorkspaceHandle* workspace) {
    if (!workspace) return 0;
    std::lock_guard<std::mutex> lock(workspace->mutex);
    return static_cast<int64_t>(workspace->f32.capacity() * sizeof(float) + workspace->i32.capacity() * sizeof(int32_t));
}

extern "C" DBS_EXPORT int dbs_create_ex(DBSOptionsC options, DBSDecoderHandle** out_handle) {
    if (out_handle) *out_handle = nullptr;
    if (!out_handle) {
        dbs_set_error(nullptr, "out_handle cannot be null");
        return -1;
    }

    try {
        auto h = std::make_unique<DBSDecoderHandle>();
        dbs::BeamOptions parsed = from_c_options(options);
        h->options = parsed;
        h->beam_size = parsed.beam_size;
        h->stats.abi_version = DBS_ABI_VERSION;
        h->stats.last_kernel = static_cast<int>(dbs::selected_kernel_path());
        h->decoder =
            std::make_unique<dbs::DifferentiableBeamSearchAVX512>(parsed);

        DBSDecoderHandle* raw = h.get();
        *out_handle = h.release();
        dbs_clear_error(raw);
        return 0;
    } catch (const std::exception& e) {
        dbs_set_error(nullptr, e.what());
        return -2;
    } catch (...) {
        dbs_set_error(nullptr, "unknown exception");
        return -3;
    }
}

extern "C" DBS_EXPORT DBSDecoderHandle* dbs_create(DBSOptionsC options) {
    DBSDecoderHandle* h = nullptr;
    return dbs_create_ex(options, &h) == 0 ? h : nullptr;
}

extern "C" DBS_EXPORT void dbs_destroy(DBSDecoderHandle* handle) {
    delete handle;
}

extern "C" DBS_EXPORT const char* dbs_last_error(DBSDecoderHandle* handle) {
    if (!handle) return dbs_last_global_error();
    std::lock_guard<std::mutex> lock(handle->error_mutex);
    return handle->last_error.empty() ? "" : handle->last_error.c_str();
}


static float dbs_fp16_to_float(uint16_t h) noexcept {
    const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x03ffu;
    uint32_t out = 0;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            uint32_t m = mant;
            uint32_t e = 113u;
            while ((m & 0x0400u) == 0) { m <<= 1; --e; }
            m &= 0x03ffu;
            out = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

static float dbs_bf16_to_float(uint16_t h) noexcept {
    const uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static std::vector<float> dbs_convert_to_f32_checked(const void* data, int data_type, int64_t count) {
    if (!data) throw std::invalid_argument("typed log_probs cannot be null");
    if (count < 0) throw std::overflow_error("negative element count");
    std::vector<float> out(static_cast<size_t>(count));
    if (data_type == DBS_DTYPE_F32) {
        const float* p = static_cast<const float*>(data);
        std::copy(p, p + count, out.begin());
    } else if (data_type == DBS_DTYPE_F16) {
        const uint16_t* p = static_cast<const uint16_t*>(data);
        for (int64_t i = 0; i < count; ++i) out[static_cast<size_t>(i)] = dbs_fp16_to_float(p[i]);
    } else if (data_type == DBS_DTYPE_BF16) {
        const uint16_t* p = static_cast<const uint16_t*>(data);
        for (int64_t i = 0; i < count; ++i) out[static_cast<size_t>(i)] = dbs_bf16_to_float(p[i]);
    } else {
        throw std::invalid_argument("unsupported DBS data type");
    }
    return out;
}

extern "C" DBS_EXPORT int dbs_decode(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int steps,
    int vocab_size,
    DBSResultHandle** out_result
) {
    if (out_result) *out_result = nullptr;
    if (!handle || !handle->decoder || !out_result) return -1;

    try {
        const auto start = std::chrono::steady_clock::now();
        auto r = std::make_unique<DBSResultHandle>();

        r->result = handle->decoder->decode(log_probs, steps, vocab_size);
        dbs_record_decode_stats(handle, r->result, now_ns_since(start), 1, false);

        *out_result = r.release();
        dbs_clear_error(handle);

        return 0;
    } catch (const std::exception& e) {
        dbs_set_error(handle, e.what());
        dbs_mark_error(handle, classify_exception(e));
        return -2;
    } catch (...) {
        dbs_set_error(handle, "unknown exception");
        dbs_mark_error(handle, 2);
        return -3;
    }
}


extern "C" DBS_EXPORT int dbs_decode_typed(
    DBSDecoderHandle* handle,
    const void* log_probs,
    int data_type,
    int steps,
    int vocab_size,
    DBSResultHandle** out_result
) {
    if (out_result) *out_result = nullptr;
    if (!handle || !handle->decoder || !out_result) return -1;
    try {
        const int64_t count = static_cast<int64_t>(steps) * static_cast<int64_t>(handle->beam_size) * static_cast<int64_t>(vocab_size);
        if (count <= 0) throw std::invalid_argument("typed decode dimensions must be positive");
        if (data_type == DBS_DTYPE_F32) {
            return dbs_decode(handle, static_cast<const float*>(log_probs), steps, vocab_size, out_result);
        }
        std::vector<float> f32 = dbs_convert_to_f32_checked(log_probs, data_type, count);
        return dbs_decode(handle, f32.data(), steps, vocab_size, out_result);
    } catch (const std::exception& e) {
        dbs_set_error(handle, e.what());
        dbs_mark_error(handle, classify_exception(e));
        return -2;
    } catch (...) {
        dbs_set_error(handle, "unknown exception");
        dbs_mark_error(handle, 2);
        return -3;
    }
}

extern "C" DBS_EXPORT int dbs_decode_batch_typed(
    DBSDecoderHandle* handle,
    const void* log_probs,
    int data_type,
    int batch_size,
    int steps,
    int vocab_size,
    int num_threads,
    DBSBatchResultHandle** out_result
) {
    if (out_result) *out_result = nullptr;
    if (!handle || !handle->decoder || !out_result) return -1;
    try {
        const int64_t count = static_cast<int64_t>(batch_size) * static_cast<int64_t>(steps) * static_cast<int64_t>(handle->beam_size) * static_cast<int64_t>(vocab_size);
        if (count <= 0) throw std::invalid_argument("typed batch decode dimensions must be positive");
        if (data_type == DBS_DTYPE_F32) {
            return dbs_decode_batch(handle, static_cast<const float*>(log_probs), batch_size, steps, vocab_size, num_threads, out_result);
        }
        std::vector<float> f32 = dbs_convert_to_f32_checked(log_probs, data_type, count);
        return dbs_decode_batch(handle, f32.data(), batch_size, steps, vocab_size, num_threads, out_result);
    } catch (const std::exception& e) {
        dbs_set_error(handle, e.what());
        dbs_mark_error(handle, classify_exception(e));
        return -2;
    } catch (...) {
        dbs_set_error(handle, "unknown exception");
        dbs_mark_error(handle, 2);
        return -3;
    }
}

extern "C" DBS_EXPORT int dbs_decode_constrained(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int steps,
    int vocab_size,
    const uint8_t* banned_tokens,
    const int32_t* forced_tokens,
    int min_length,
    DBSResultHandle** out_result
) {
    DBSAdvancedConstraintsC c{};
    c.banned_tokens = banned_tokens;
    c.forced_tokens = forced_tokens;
    c.min_length = min_length;
    c.repetition_penalty = 1.0f;
    c.no_repeat_ngram_size = 0;
    return dbs_decode_constrained_ex(handle, log_probs, steps, vocab_size, &c, out_result);
}

extern "C" DBS_EXPORT int dbs_decode_constrained_ex(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int steps,
    int vocab_size,
    const DBSAdvancedConstraintsC* constraints_c,
    DBSResultHandle** out_result
) {
    if (out_result) *out_result = nullptr;
    if (!handle || !handle->decoder || !out_result) {
        dbs_set_error(handle, "invalid decoder, result, or output pointer");
        dbs_mark_error(handle, 1);
        return -1;
    }

    try {
        const auto start_time = std::chrono::steady_clock::now();
        dbs::DecodeConstraints constraints;
        if (constraints_c) {
            constraints.banned_tokens = constraints_c->banned_tokens;
            constraints.forced_tokens = constraints_c->forced_tokens;
            constraints.min_length = constraints_c->min_length;
            constraints.repetition_penalty = constraints_c->repetition_penalty > 0.0f ? constraints_c->repetition_penalty : 1.0f;
            constraints.no_repeat_ngram_size = constraints_c->no_repeat_ngram_size;
            constraints.token_filter = constraints_c->token_filter;
            constraints.token_filter_user_data = constraints_c->token_filter_user_data;
            constraints.batch_index = constraints_c->batch_index;
        }

        auto r = std::make_unique<DBSResultHandle>();
        r->result = handle->decoder->decode_constrained(log_probs, steps, vocab_size, &constraints);
        dbs_record_decode_stats(handle, r->result, now_ns_since(start_time), 1, false);
        *out_result = r.release();
        dbs_clear_error(handle);
        return 0;
    } catch (const std::exception& e) {
        dbs_set_error(handle, e.what());
        dbs_mark_error(handle, classify_exception(e));
        return -2;
    } catch (...) {
        dbs_set_error(handle, "unknown exception");
        dbs_mark_error(handle, 2);
        return -3;
    }
}


extern "C" DBS_EXPORT int dbs_decode_model_steps(
    DBSDecoderHandle* handle,
    DBSModelStepFn step_fn,
    void* user_data,
    int batch_index,
    int steps,
    int vocab_size,
    DBSResultHandle** out_result
) {
    DBSWorkspaceHandle local_workspace;
    return dbs_decode_model_steps_with_workspace(
        handle,
        &local_workspace,
        step_fn,
        user_data,
        batch_index,
        steps,
        vocab_size,
        out_result);
}

extern "C" DBS_EXPORT int dbs_decode_model_steps_with_workspace(
    DBSDecoderHandle* handle,
    DBSWorkspaceHandle* workspace,
    DBSModelStepFn step_fn,
    void* user_data,
    int batch_index,
    int steps,
    int vocab_size,
    DBSResultHandle** out_result
) {
    if (out_result) *out_result = nullptr;
    if (!handle || !handle->decoder || !workspace || !step_fn || !out_result) {
        dbs_set_error(handle, "invalid decoder, workspace, model step callback, or output pointer");
        dbs_mark_error(handle, 1);
        return -1;
    }
    if (steps <= 0 || vocab_size <= 0) {
        dbs_set_error(handle, "steps and vocab_size must be positive");
        dbs_mark_error(handle, 1);
        return -1;
    }

    try {
        const auto start_time = std::chrono::steady_clock::now();
        const int K = handle->beam_size;
        const size_t row_count = static_cast<size_t>(K) * static_cast<size_t>(vocab_size);
        const size_t total = static_cast<size_t>(steps) * row_count;

        std::lock_guard<std::mutex> workspace_lock(workspace->mutex);
        workspace->f32.assign(total, -std::numeric_limits<float>::infinity());
        workspace->i32.assign(static_cast<size_t>(K), -1);
        std::vector<float> prev_scores(static_cast<size_t>(K), 0.0f);

        for (int t = 0; t < steps; ++t) {
            float* out_row = workspace->f32.data() + static_cast<size_t>(t) * row_count;
            const int rc = step_fn(
                user_data,
                batch_index,
                t,
                workspace->i32.data(),
                prev_scores.data(),
                K,
                vocab_size,
                out_row);
            if (rc != 0) {
                throw std::runtime_error("model step callback returned non-zero status");
            }

            if (t + 1 < steps) {
                dbs::DecodeResult partial = handle->decoder->decode(workspace->f32.data(), t + 1, vocab_size);
                const int32_t* tok = partial.tokens.data() + static_cast<size_t>(t) * K;
                const float* scores = partial.final_scores.data();
                for (int k = 0; k < K; ++k) {
                    workspace->i32[static_cast<size_t>(k)] = tok[k];
                    prev_scores[static_cast<size_t>(k)] = scores[k];
                }
            }
        }

        auto r = std::make_unique<DBSResultHandle>();
        r->result = handle->decoder->decode(workspace->f32.data(), steps, vocab_size);
        dbs_record_decode_stats(handle, r->result, now_ns_since(start_time), 1, true);
        {
            const int64_t workspace_bytes = static_cast<int64_t>(
                workspace->f32.capacity() * sizeof(float) + workspace->i32.capacity() * sizeof(int32_t));
            std::lock_guard<std::mutex> lock(handle->stats_mutex);
            handle->stats.last_allocation_bytes += workspace_bytes;
        }
        *out_result = r.release();
        dbs_clear_error(handle);
        return 0;
    } catch (const std::exception& e) {
        dbs_set_error(handle, e.what());
        dbs_mark_error(handle, classify_exception(e));
        return -2;
    } catch (...) {
        dbs_set_error(handle, "unknown exception");
        dbs_mark_error(handle, 2);
        return -3;
    }
}

extern "C" DBS_EXPORT int dbs_decode_batch(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int batch_size,
    int steps,
    int vocab_size,
    int num_threads,
    DBSBatchResultHandle** out_result
) {
    if (out_result) *out_result = nullptr;
    if (!handle || !handle->decoder || !out_result) {
        dbs_set_error(handle, "invalid decoder, result, or output pointer");
        return -1;
    }
    if (!log_probs || batch_size <= 0 || steps <= 0 || vocab_size <= 0) {
        dbs_set_error(handle, "invalid batch decode arguments");
        return -1;
    }

    try {
        auto br = std::make_unique<DBSBatchResultHandle>();
        br->results.resize(static_cast<size_t>(batch_size));

        const size_t stride =
            static_cast<size_t>(steps) * static_cast<size_t>(handle->beam_size) * static_cast<size_t>(vocab_size);
        if (stride == 0 || stride / static_cast<size_t>(vocab_size) != static_cast<size_t>(steps) * static_cast<size_t>(handle->beam_size)) {
            dbs_set_error(handle, "batch stride overflow");
            return -1;
        }

        const int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
        const int threads = std::max(1, std::min(batch_size, num_threads > 0 ? num_threads : std::max(1, hw_threads)));
        std::atomic<int> index{0};
        std::mutex error_mutex;
        std::exception_ptr first_exception = nullptr;

        auto worker = [&]() {
            for (;;) {
                const int b = index.fetch_add(1);
                if (b >= batch_size) break;
                try {
                    br->results[static_cast<size_t>(b)].result =
                        handle->decoder->decode(log_probs + static_cast<size_t>(b) * stride, steps, vocab_size);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!first_exception) first_exception = std::current_exception();
                }
            }
        };

        std::vector<std::thread> pool;
        pool.reserve(static_cast<size_t>(threads));
        for (int i = 0; i < threads; ++i) pool.emplace_back(worker);
        for (auto& th : pool) th.join();
        if (first_exception) {
            std::rethrow_exception(first_exception);
        }

        if (!br->results.empty()) {
            dbs_record_decode_stats(handle, br->results.front().result, 0, threads, false);
        }
        *out_result = br.release();
        dbs_clear_error(handle);
        return 0;
    } catch (const std::exception& e) {
        dbs_set_error(handle, e.what());
        dbs_mark_error(handle, classify_exception(e));
        return -2;
    } catch (...) {
        dbs_set_error(handle, "unknown exception");
        dbs_mark_error(handle, 2);
        return -3;
    }
}


extern "C" DBS_EXPORT int dbs_decode_batch_variable(
    DBSDecoderHandle* handle,
    const float* log_probs,
    int batch_size,
    int max_steps,
    int max_beam_size,
    int vocab_size,
    const int32_t* steps_per_example,
    const int32_t* beam_sizes_per_example,
    const int32_t* eos_tokens_per_example,
    const int32_t* min_lengths_per_example,
    const uint8_t* banned_tokens_per_example,
    const int32_t* forced_tokens_per_example,
    int num_threads,
    DBSBatchResultHandle** out_result
) {
    if (out_result) *out_result = nullptr;
    if (!handle || !log_probs || !out_result || batch_size <= 0 || max_steps <= 0 || max_beam_size <= 0 || vocab_size <= 0) {
        dbs_set_error(handle, "invalid variable batch decode arguments");
        dbs_mark_error(handle, 1);
        return -1;
    }

    try {
        const auto start_time = std::chrono::steady_clock::now();
        auto br = std::make_unique<DBSBatchResultHandle>();
        br->results.resize(static_cast<size_t>(batch_size));
        const size_t input_stride = static_cast<size_t>(max_steps) * static_cast<size_t>(max_beam_size) * static_cast<size_t>(vocab_size);

        const int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
        const int threads = std::max(1, std::min(batch_size, num_threads > 0 ? num_threads : std::max(1, hw_threads)));
        std::atomic<int> index{0};
        std::mutex error_mutex;
        std::exception_ptr first_exception = nullptr;

        auto worker = [&]() {
            for (;;) {
                const int b = index.fetch_add(1);
                if (b >= batch_size) break;
                try {
                    const int steps = steps_per_example ? steps_per_example[b] : max_steps;
                    const int beam = beam_sizes_per_example ? beam_sizes_per_example[b] : handle->beam_size;
                    if (steps <= 0 || steps > max_steps) throw std::invalid_argument("invalid per-example steps");
                    if (beam <= 0 || beam > max_beam_size) throw std::invalid_argument("invalid per-example beam size");

                    dbs::BeamOptions opt = handle->options;
                    opt.beam_size = beam;
                    if (eos_tokens_per_example) opt.eos_token = eos_tokens_per_example[b];
                    if (min_lengths_per_example) opt.min_length = std::max(0, min_lengths_per_example[b]);
                    dbs::DifferentiableBeamSearchAVX512 local_decoder(opt);

                    std::vector<float> local(static_cast<size_t>(steps) * static_cast<size_t>(beam) * static_cast<size_t>(vocab_size));
                    const float* base = log_probs + static_cast<size_t>(b) * input_stride;
                    for (int t = 0; t < steps; ++t) {
                        for (int k = 0; k < beam; ++k) {
                            const float* src = base + (static_cast<size_t>(t) * max_beam_size + static_cast<size_t>(k)) * static_cast<size_t>(vocab_size);
                            float* dst = local.data() + (static_cast<size_t>(t) * beam + static_cast<size_t>(k)) * static_cast<size_t>(vocab_size);
                            std::copy(src, src + vocab_size, dst);
                        }
                    }

                    dbs::DecodeConstraints constraints;
                    if (banned_tokens_per_example) constraints.banned_tokens = banned_tokens_per_example + static_cast<size_t>(b) * static_cast<size_t>(vocab_size);
                    if (forced_tokens_per_example) constraints.forced_tokens = forced_tokens_per_example + static_cast<size_t>(b) * static_cast<size_t>(max_steps);
                    constraints.min_length = min_lengths_per_example ? min_lengths_per_example[b] : -1;
                    constraints.batch_index = b;

                    br->results[static_cast<size_t>(b)].result =
                        local_decoder.decode_constrained(local.data(), steps, vocab_size,
                            (banned_tokens_per_example || forced_tokens_per_example || min_lengths_per_example) ? &constraints : nullptr);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!first_exception) first_exception = std::current_exception();
                }
            }
        };

        std::vector<std::thread> pool;
        pool.reserve(static_cast<size_t>(threads));
        for (int i = 0; i < threads; ++i) pool.emplace_back(worker);
        for (auto& th : pool) th.join();
        if (first_exception) {
            std::rethrow_exception(first_exception);
        }

        if (!br->results.empty()) dbs_record_decode_stats(handle, br->results.front().result, now_ns_since(start_time), threads, false);
        *out_result = br.release();
        dbs_clear_error(handle);
        return 0;
    } catch (const std::exception& e) {
        dbs_set_error(handle, e.what());
        dbs_mark_error(handle, classify_exception(e));
        return -2;
    } catch (...) {
        dbs_set_error(handle, "unknown exception");
        dbs_mark_error(handle, 2);
        return -3;
    }
}

extern "C" DBS_EXPORT int dbs_backward(
    DBSDecoderHandle* handle,
    const DBSResultHandle* result,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores,
    DBSBackwardHandle** out_backward
) {
    return dbs_backward_sparse(
        handle,
        result,
        grad_selected_weights,
        grad_relaxed_weights,
        grad_final_scores,
        out_backward
    );
}

extern "C" DBS_EXPORT int dbs_backward_dense(
    DBSDecoderHandle* handle,
    const DBSResultHandle* result,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores,
    DBSBackwardHandle** out_backward
) {
    if (out_backward) *out_backward = nullptr;
    if (!handle || !handle->decoder || !result || !out_backward) {
        dbs_set_error(handle, "invalid decoder, result, or output pointer");
        dbs_mark_error(handle, 1);
        return -1;
    }

    try {
        const auto start_time = std::chrono::steady_clock::now();
        auto b = std::make_unique<DBSBackwardHandle>();
        b->result = handle->decoder->backward(
            result->result,
            grad_selected_weights,
            grad_relaxed_weights,
            grad_final_scores);
        dbs_record_backward_stats(handle, b->result, now_ns_since(start_time));
        *out_backward = b.release();
        dbs_clear_error(handle);
        return 0;
    } catch (const std::exception& e) {
        dbs_set_error(handle, e.what());
        dbs_mark_error(handle, classify_exception(e));
        return -2;
    } catch (...) {
        dbs_set_error(handle, "unknown exception");
        dbs_mark_error(handle, 2);
        return -3;
    }
}

extern "C" DBS_EXPORT int dbs_backward_sparse(
    DBSDecoderHandle* handle,
    const DBSResultHandle* result,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores,
    DBSBackwardHandle** out_backward
) {
    if (out_backward) *out_backward = nullptr;
    if (!handle || !handle->decoder || !result || !out_backward) {
        dbs_set_error(handle, "invalid decoder, result, or output pointer");
        dbs_mark_error(handle, 1);
        return -1;
    }

    try {
        const auto start_time = std::chrono::steady_clock::now();
        auto b = std::make_unique<DBSBackwardHandle>();
        b->result = handle->decoder->backward_sparse(
            result->result,
            grad_selected_weights,
            grad_relaxed_weights,
            grad_final_scores);
        dbs_record_backward_stats(handle, b->result, now_ns_since(start_time));
        *out_backward = b.release();
        dbs_clear_error(handle);
        return 0;
    } catch (const std::exception& e) {
        dbs_set_error(handle, e.what());
        dbs_mark_error(handle, classify_exception(e));
        return -2;
    } catch (...) {
        dbs_set_error(handle, "unknown exception");
        dbs_mark_error(handle, 2);
        return -3;
    }
}

extern "C" DBS_EXPORT int dbs_backward_default(
    DBSDecoderHandle* handle,
    const DBSResultHandle* result,
    const float* grad_selected_weights,
    const float* grad_relaxed_weights,
    const float* grad_final_scores,
    DBSBackwardHandle** out_backward
) {
    return dbs_backward_sparse(
        handle,
        result,
        grad_selected_weights,
        grad_relaxed_weights,
        grad_final_scores,
        out_backward
    );
}

extern "C" DBS_EXPORT void dbs_free_result(DBSResultHandle* result) {
    delete result;
}

extern "C" DBS_EXPORT void dbs_free_batch_result(DBSBatchResultHandle* result) {
    delete result;
}

extern "C" DBS_EXPORT int dbs_batch_result_size(const DBSBatchResultHandle* result) {
    return result ? static_cast<int>(result->results.size()) : 0;
}

extern "C" DBS_EXPORT const DBSResultHandle* dbs_batch_result_at(const DBSBatchResultHandle* result, int batch_index) {
    if (!result || batch_index < 0 || batch_index >= static_cast<int>(result->results.size())) return nullptr;
    return &result->results[static_cast<size_t>(batch_index)];
}

extern "C" DBS_EXPORT void dbs_free_backward(DBSBackwardHandle* result) {
    delete result;
}

extern "C" DBS_EXPORT int dbs_result_steps(const DBSResultHandle* result) {
    return result ? result->result.steps : 0;
}

extern "C" DBS_EXPORT int dbs_result_beam_size(const DBSResultHandle* result) {
    return result ? result->result.beam_size : 0;
}

extern "C" DBS_EXPORT int dbs_result_vocab_size(const DBSResultHandle* result) {
    return result ? result->result.vocab_size : 0;
}

extern "C" DBS_EXPORT int dbs_result_pool_size(const DBSResultHandle* result) {
    return result ? result->result.relaxed_pool_size : 0;
}

extern "C" DBS_EXPORT const int32_t* dbs_result_tokens(const DBSResultHandle* result) {
    return result ? result->result.tokens.data() : nullptr;
}

extern "C" DBS_EXPORT const int32_t* dbs_result_parents(const DBSResultHandle* result) {
    return result ? result->result.parents.data() : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_scores(const DBSResultHandle* result) {
    return result ? result->result.scores.data() : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_raw_scores(const DBSResultHandle* result) {
    return result ? result->result.raw_scores.data() : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_relaxed_weights(const DBSResultHandle* result) {
    return result ? result->result.relaxed_weights.data() : nullptr;
}

extern "C" DBS_EXPORT const int32_t* dbs_result_pool_tokens(const DBSResultHandle* result) {
    return result ? result->result.pool_tokens.data() : nullptr;
}

extern "C" DBS_EXPORT const int32_t* dbs_result_pool_parents(const DBSResultHandle* result) {
    return result ? result->result.pool_parents.data() : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_pool_scores(const DBSResultHandle* result) {
    return result ? result->result.pool_scores.data() : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_backward_grad_log_probs(const DBSBackwardHandle* result) {
    return result ? result->result.grad_log_probs.data() : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_backward_grad_initial_scores(const DBSBackwardHandle* result) {
    return result ? result->result.grad_initial_scores.data() : nullptr;
}

extern "C" DBS_EXPORT const int64_t* dbs_backward_sparse_logprob_indices(const DBSBackwardHandle* result) {
    return result ? result->result.sparse_logprob_indices.data() : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_backward_sparse_logprob_values(const DBSBackwardHandle* result) {
    return result ? result->result.sparse_logprob_values.data() : nullptr;
}

extern "C" DBS_EXPORT int64_t dbs_backward_sparse_logprob_count(const DBSBackwardHandle* result) {
    return result ? static_cast<int64_t>(result->result.sparse_logprob_values.size()) : 0;
}

extern "C" DBS_EXPORT const int32_t* dbs_result_lengths(const DBSResultHandle* result) {
    return result ? result->result.lengths.data() : nullptr;
}

extern "C" DBS_EXPORT const int32_t* dbs_result_pool_lengths(const DBSResultHandle* result) {
    return result ? result->result.pool_lengths.data() : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_weights(const DBSResultHandle* result) {
    return result ? result->result.weights.data() : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_final_scores(const DBSResultHandle* result) {
    return result ? result->result.final_scores.data() : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_final_raw_scores(const DBSResultHandle* result) {
    return result ? result->result.final_raw_scores.data() : nullptr;
}

extern "C" DBS_EXPORT const float* dbs_result_pool_raw_scores(const DBSResultHandle* result) {
    return result ? result->result.pool_raw_scores.data() : nullptr;
}

extern "C" DBS_EXPORT int64_t dbs_result_selected_count(const DBSResultHandle* result) {
    return result ? static_cast<int64_t>(result->result.steps) * result->result.beam_size : 0;
}

extern "C" DBS_EXPORT int64_t dbs_result_pool_count(const DBSResultHandle* result) {
    return result ? static_cast<int64_t>(result->result.steps) * result->result.relaxed_pool_size : 0;
}

extern "C" DBS_EXPORT int64_t dbs_result_logprob_count(const DBSResultHandle* result) {
    return result
        ? static_cast<int64_t>(result->result.steps) * result->result.beam_size * result->result.vocab_size
        : 0;
}

extern "C" DBS_EXPORT int64_t dbs_backward_grad_log_probs_count(const DBSResultHandle* result) {
    return dbs_result_logprob_count(result);
}

extern "C" DBS_EXPORT int64_t dbs_backward_grad_initial_scores_count(const DBSResultHandle* result) {
    return result ? result->result.beam_size : 0;
}

extern "C" DBS_EXPORT int dbs_backward_is_sparse(const DBSBackwardHandle* result) {
    return result && result->result.sparse ? 1 : 0;
}


extern "C" DBS_EXPORT int64_t dbs_result_eos_count(const DBSResultHandle* result, int eos_token) {
    if (!result || eos_token < 0) return 0;
    int64_t count = 0;
    for (int32_t token : result->result.tokens) {
        if (token == eos_token) ++count;
    }
    return count;
}

extern "C" DBS_EXPORT int dbs_result_validate_deterministic_order(const DBSResultHandle* result) {
    if (!result) return -1;
    const int T = result->result.steps;
    const int K = result->result.beam_size;
    if (T < 0 || K < 0) return -2;
    for (int t = 0; t < T; ++t) {
        for (int k = 1; k < K; ++k) {
            const size_t prev = static_cast<size_t>(t) * K + (k - 1);
            const size_t cur = static_cast<size_t>(t) * K + k;
            const float a = result->result.scores[prev];
            const float b = result->result.scores[cur];
            if (std::isfinite(a) && std::isfinite(b) && b > a) return -3;
            if (a == b) {
                const int32_t pa = result->result.parents[prev];
                const int32_t pb = result->result.parents[cur];
                const int32_t ta = result->result.tokens[prev];
                const int32_t tb = result->result.tokens[cur];
                if (pb < pa || (pb == pa && tb < ta)) return -4;
            }
        }
    }
    return 0;
}

extern "C" DBS_EXPORT int dbs_result_summary_json(const DBSResultHandle* result, int eos_token, char* out_json, int64_t out_json_capacity) {
    if (!result || !out_json || out_json_capacity <= 0) return -1;
    const auto& r = result->result;
    int min_len = r.lengths.empty() ? 0 : std::numeric_limits<int>::max();
    int max_len = 0;
    for (int32_t len : r.lengths) {
        if (len < min_len) min_len = len;
        if (len > max_len) max_len = len;
    }
    if (r.lengths.empty()) min_len = 0;
    float min_final = r.final_scores.empty() ? 0.0f : r.final_scores[0];
    float max_final = r.final_scores.empty() ? 0.0f : r.final_scores[0];
    for (float score : r.final_scores) {
        if (score < min_final) min_final = score;
        if (score > max_final) max_final = score;
    }
    const int64_t eos_count = dbs_result_eos_count(result, eos_token);
    const int order_ok = dbs_result_validate_deterministic_order(result) == 0 ? 1 : 0;
    const int n = std::snprintf(out_json, static_cast<size_t>(out_json_capacity),
        "{\"abi_version\":%d,\"steps\":%d,\"beam_size\":%d,\"vocab_size\":%d,\"selected_count\":%lld,\"pool_count\":%lld,\"eos_token\":%d,\"eos_count\":%lld,\"min_length\":%d,\"max_length\":%d,\"min_final_score\":%.9g,\"max_final_score\":%.9g,\"deterministic_order\":%d}",
        DBS_ABI_VERSION,
        r.steps,
        r.beam_size,
        r.vocab_size,
        static_cast<long long>(static_cast<int64_t>(r.steps) * r.beam_size),
        static_cast<long long>(static_cast<int64_t>(r.steps) * r.relaxed_pool_size),
        eos_token,
        static_cast<long long>(eos_count),
        min_len,
        max_len,
        static_cast<double>(min_final),
        static_cast<double>(max_final),
        order_ok);
    if (n < 0) return -2;
    return n < out_json_capacity ? 0 : 1;
}

extern "C" DBS_EXPORT int dbs_validate_production_gate_manifest(const char* manifest_json, char* out_error, int64_t out_error_capacity) {
    auto write_error = [&](const char* msg) -> int {
        if (out_error && out_error_capacity > 0) {
            std::snprintf(out_error, static_cast<size_t>(out_error_capacity), "%s", msg);
        }
        return -1;
    };
    if (!manifest_json) return write_error("manifest_json is null");
    const std::string m(manifest_json);
    const char* required[] = {
        "\"cuda_parity\":true",
        "\"torch_wheel_cpu\":true",
        "\"torch_wheel_cuda\":true",
        "\"large_vocab_benchmarks\":true",
        "\"sanitizers\":true",
        "\"fuzzing\":true",
        "\"abi_compatibility\":true",
        "\"zero_allocation_hot_path\":true",
        "\"mixed_precision_parity\":true",
        "\"hardware_matrix\":true"
    };
    for (const char* needle : required) {
        if (m.find(needle) == std::string::npos) return write_error(needle);
    }
    if (out_error && out_error_capacity > 0) out_error[0] = '\0';
    return 0;
}

extern "C" DBS_EXPORT int dbs_has_avx512() {
    return dbs::runtime_has_avx512() ? 1 : 0;
}

extern "C" DBS_EXPORT int dbs_has_avx2() {
    return dbs::runtime_has_avx2() ? 1 : 0;
}

extern "C" DBS_EXPORT int dbs_has_sse42() {
    return dbs::runtime_has_sse42() ? 1 : 0;
}

extern "C" DBS_EXPORT int dbs_has_neon() {
    return dbs::runtime_has_neon() ? 1 : 0;
}

extern "C" DBS_EXPORT const char* dbs_selected_kernel_name() {
    return dbs::kernel_path_name(dbs::selected_kernel_path());
}

extern "C" DBS_EXPORT int dbs_get_stats(DBSDecoderHandle* handle, DBSStatsC* out_stats) {
    if (!handle || !out_stats) return -1;
    std::lock_guard<std::mutex> lock(handle->stats_mutex);
    *out_stats = handle->stats;
    out_stats->total_allocator_calls = dbs::g_allocator_calls.load(std::memory_order_relaxed);
    out_stats->total_allocator_bytes = dbs::g_allocator_bytes.load(std::memory_order_relaxed);
    return 0;
}


extern "C" DBS_EXPORT int dbs_get_stats_json(DBSDecoderHandle* handle, char* out_json, int64_t out_json_capacity) {
    if (!handle || !out_json || out_json_capacity <= 0) return -1;
    DBSStatsC s{};
    if (dbs_get_stats(handle, &s) != 0) return -1;
    const int n = std::snprintf(
        out_json,
        static_cast<size_t>(out_json_capacity),
        "{\"abi_version\":%d,\"kernel\":\"%s\",\"used_sparse_backward\":%d,\"used_dense_backward\":%d,\"batch_threads\":%d,\"model_step\":%d,\"last_error_category\":%d,\"decode_ns\":%lld,\"backward_ns\":%lld,\"allocation_bytes\":%lld,\"selected_count\":%lld,\"pool_count\":%lld,\"logprob_count\":%lld,\"sparse_grad_count\":%lld,\"allocator_calls\":%lld,\"allocator_bytes\":%lld}",
        s.abi_version,
        dbs::kernel_path_name(static_cast<dbs::KernelPath>(s.last_kernel)),
        s.used_sparse_backward,
        s.used_dense_backward,
        s.used_batch_threads,
        s.used_model_step_callback,
        s.last_error_category,
        static_cast<long long>(s.last_decode_ns),
        static_cast<long long>(s.last_backward_ns),
        static_cast<long long>(s.last_allocation_bytes),
        static_cast<long long>(s.last_selected_count),
        static_cast<long long>(s.last_pool_count),
        static_cast<long long>(s.last_logprob_count),
        static_cast<long long>(s.last_sparse_grad_count),
        static_cast<long long>(s.total_allocator_calls),
        static_cast<long long>(s.total_allocator_bytes));
    if (n < 0) return -2;
    return n < out_json_capacity ? 0 : 1;
}

extern "C" DBS_EXPORT int dbs_is_deterministic() {
    return 1;
}

extern "C" DBS_EXPORT int dbs_set_deterministic_seed(DBSDecoderHandle* handle, uint64_t seed) {
    if (!handle) return -1;
    handle->deterministic_seed = seed;
    dbs::g_deterministic_seed_tls = seed;
    return 0;
}

extern "C" DBS_EXPORT uint64_t dbs_get_deterministic_seed(DBSDecoderHandle* handle) {
    return handle ? handle->deterministic_seed : 0;
}

extern "C" DBS_EXPORT void dbs_allocator_counters_reset() {
    dbs::g_allocator_calls.store(0, std::memory_order_relaxed);
    dbs::g_allocator_bytes.store(0, std::memory_order_relaxed);
}

extern "C" DBS_EXPORT int64_t dbs_allocator_call_count() {
    return dbs::g_allocator_calls.load(std::memory_order_relaxed);
}

extern "C" DBS_EXPORT int64_t dbs_allocator_byte_count() {
    return dbs::g_allocator_bytes.load(std::memory_order_relaxed);
}

extern "C" DBS_EXPORT void dbs_reset_stats(DBSDecoderHandle* handle) {
    if (!handle) return;
    std::lock_guard<std::mutex> lock(handle->stats_mutex);
    handle->stats = DBSStatsC{};
    handle->stats.abi_version = DBS_ABI_VERSION;
    handle->stats.last_kernel = static_cast<int>(dbs::selected_kernel_path());
}

#ifdef DIFFERENTIABLE_BEAM_DEMO
int main() {
    constexpr int T = 4;
    constexpr int K = 4;
    constexpr int V = 256;

    dbs::AlignedFloatVector log_probs(static_cast<size_t>(T) * K * V, -20.0f);

    for (int t = 0; t < T; ++t) {
        for (int b = 0; b < K; ++b) {
            for (int v = 0; v < V; ++v) {
                log_probs[(static_cast<size_t>(t) * K + b) * V + v] =
                    -0.01f * static_cast<float>((v + 7 * t + 13 * b) % V);
            }
        }
    }

    dbs::BeamOptions opt;
    opt.beam_size = K;
    opt.eos_token = 255;
    opt.selected_temperature = 0.7f;
    opt.soft_topk_temperature = 0.25f;
    opt.relaxed_pool_multiplier = 8;
    opt.length_penalty_alpha = 1.0f;

    dbs::DifferentiableBeamSearchAVX512 decoder(opt);

    const auto t0 = std::chrono::high_resolution_clock::now();
    dbs::DecodeResult result = decoder.decode(log_probs.data(), T, V);
    const auto t1 = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < K; ++k) {
        std::cout
            << "beam " << k
            << " score=" << result.final_scores[k]
            << " raw=" << result.final_raw_scores[k]
            << " seq:";

        for (int tok : result.sequences[k]) std::cout << ' ' << tok;
        std::cout << '\n';
    }

    const int P = result.relaxed_pool_size;

    dbs::AlignedFloatVector grad_relaxed(static_cast<size_t>(T) * P, 0.0f);
    dbs::AlignedFloatVector grad_final(K, 0.0f);

    for (int t = 0; t < T; ++t) {
        grad_relaxed[static_cast<size_t>(t) * P] = -1.0f;
    }

    grad_final[0] = 1.0f;

    dbs::BackwardResult grad =
        decoder.backward(result, nullptr, grad_relaxed.data(), grad_final.data());

    const auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "grad_initial_score[0]="
              << grad.grad_initial_scores[0]
              << '\n';

    std::cout << "avx512=" << (dbs::runtime_has_avx512() ? 1 : 0) << '\n';
    std::cout << "forward_us="
              << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
              << '\n';
    std::cout << "backward_us="
              << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
              << '\n';
}
#endif
