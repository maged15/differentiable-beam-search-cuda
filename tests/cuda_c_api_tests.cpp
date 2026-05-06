// SPDX-License-Identifier: MIT
#include "dbs_cuda.h"

#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t err, const char* expr) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << expr << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr)

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void expect_status(int got, int expected, const char* context) {
    if (got != expected) {
        std::ostringstream oss;
        oss << context << " returned " << got << " (" << dbs_cuda_status_string(got)
            << "), expected " << expected << " (" << dbs_cuda_status_string(expected) << ")";
        throw std::runtime_error(oss.str());
    }
}

struct CudaStream {
    cudaStream_t stream = nullptr;

    CudaStream() { CUDA_CHECK(cudaStreamCreate(&stream)); }
    ~CudaStream() {
        if (stream) cudaStreamDestroy(stream);
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
};

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr_; }
    std::size_t count() const { return count_; }

    void fill_zero(cudaStream_t stream) {
        CUDA_CHECK(cudaMemsetAsync(ptr_, 0, count_ * sizeof(T), stream));
    }

    void copy_from(const std::vector<T>& host, cudaStream_t stream) {
        check(host.size() == count_, "host vector size must match device buffer");
        CUDA_CHECK(cudaMemcpyAsync(ptr_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    std::vector<T> copy_to(cudaStream_t stream) const {
        std::vector<T> host(count_);
        CUDA_CHECK(cudaMemcpyAsync(host.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return host;
    }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

void expect_close(float actual, float expected, const char* context) {
    if (std::fabs(actual - expected) > 1e-6f) {
        std::ostringstream oss;
        oss << context << " expected " << expected << " got " << actual;
        throw std::runtime_error(oss.str());
    }
}

void test_sparse_scatter(cudaStream_t stream) {
    DeviceBuffer<int64_t> indices(4);
    DeviceBuffer<float> values(4);
    DeviceBuffer<float> grad(5);
    indices.copy_from({0, 1, 1, 4}, stream);
    values.copy_from({1.0f, 2.0f, 3.0f, 5.0f}, stream);
    grad.fill_zero(stream);

    expect_status(
        dbs_cuda_sparse_backward_scatter(indices.get(), values.get(), 4, grad.get(), 5, stream),
        DBS_CUDA_STATUS_OK,
        "valid duplicate sparse scatter");
    const auto out = grad.copy_to(stream);
    expect_close(out[0], 1.0f, "grad[0]");
    expect_close(out[1], 5.0f, "grad[1]");
    expect_close(out[2], 0.0f, "grad[2]");
    expect_close(out[3], 0.0f, "grad[3]");
    expect_close(out[4], 5.0f, "grad[4]");

    DeviceBuffer<int64_t> bad_indices(2);
    DeviceBuffer<float> bad_values(2);
    DeviceBuffer<float> bad_grad(3);
    bad_values.copy_from({1.0f, 2.0f}, stream);
    bad_grad.fill_zero(stream);

    bad_indices.copy_from({0, -1}, stream);
    expect_status(
        dbs_cuda_sparse_backward_scatter(bad_indices.get(), bad_values.get(), 2, bad_grad.get(), 3, stream),
        DBS_CUDA_STATUS_INVALID_ARGUMENT,
        "negative sparse index");

    bad_indices.copy_from({0, 3}, stream);
    expect_status(
        dbs_cuda_sparse_backward_scatter(bad_indices.get(), bad_values.get(), 2, bad_grad.get(), 3, stream),
        DBS_CUDA_STATUS_INVALID_ARGUMENT,
        "too-large sparse index");
}

void test_direct_decode_eos_validation(cudaStream_t stream) {
    constexpr int B = 1;
    constexpr int T = 2;
    constexpr int K = 2;
    constexpr int V = 8;
    DeviceBuffer<float> log_probs(B * T * K * V);
    DeviceBuffer<int32_t> tokens(B * T * K);
    DeviceBuffer<float> scores(B * K);
    log_probs.copy_from(std::vector<float>(B * T * K * V, -1.0f), stream);

    expect_status(
        dbs_cuda_decode_forward(log_probs.get(), B, T, K, V, -2, tokens.get(), scores.get(), stream),
        DBS_CUDA_STATUS_INVALID_ARGUMENT,
        "serial decode negative eos");
    expect_status(
        dbs_cuda_decode_forward(log_probs.get(), B, T, K, V, V, tokens.get(), scores.get(), stream),
        DBS_CUDA_STATUS_INVALID_ARGUMENT,
        "serial decode too-large eos");
    expect_status(
        dbs_cuda_decode_forward_fast_ex(log_probs.get(), B, T, K, V, -2, 0, tokens.get(), scores.get(), stream),
        DBS_CUDA_STATUS_INVALID_ARGUMENT,
        "fast decode negative eos");
    expect_status(
        dbs_cuda_decode_forward_fast_ex(log_probs.get(), B, T, K, V, V, 0, tokens.get(), scores.get(), stream),
        DBS_CUDA_STATUS_INVALID_ARGUMENT,
        "fast decode too-large eos");
    expect_status(
        dbs_cuda_decode_forward(log_probs.get(), B, T, DBS_CUDA_MAX_BEAM + 1, V, -1, tokens.get(), scores.get(), stream),
        DBS_CUDA_STATUS_INVALID_ARGUMENT,
        "serial decode beam above CUDA maximum");
    expect_status(
        dbs_cuda_decode_forward_fast_ex(log_probs.get(), B, T, DBS_CUDA_MAX_BEAM + 1, V, -1, 0, tokens.get(), scores.get(), stream),
        DBS_CUDA_STATUS_INVALID_ARGUMENT,
        "fast decode beam above CUDA maximum");
    expect_status(
        dbs_cuda_decode_forward_fast_ex(log_probs.get(), B, T, K, V, -1, -1, tokens.get(), scores.get(), stream),
        DBS_CUDA_STATUS_INVALID_ARGUMENT,
        "fast decode negative min_length");
    expect_status(
        dbs_cuda_decode_forward_fast_ex(log_probs.get(), B, T, K, INT_MAX, -1, 0, tokens.get(), scores.get(), stream),
        DBS_CUDA_STATUS_INVALID_ARGUMENT,
        "fast decode reserved vocab sentinel");
}

void test_fast_decode_eos_carry_matches_serial(cudaStream_t stream) {
    constexpr int B = 1;
    constexpr int T = 4;
    constexpr int K = 4;
    constexpr int V = 160;
    constexpr int eos = 0;

    std::vector<float> host_log_probs(B * T * K * V, -9.0f);
    auto set_lp = [&](int t, int k, int v, float value) {
        host_log_probs[((static_cast<std::size_t>(t) * K + k) * V) + v] = value;
    };

    set_lp(0, 0, eos, 0.0f);
    set_lp(0, 0, 1, -0.01f);
    set_lp(0, 0, 2, -0.02f);
    set_lp(0, 0, 3, -0.03f);

    for (int t = 1; t < T; ++t) {
        for (int k = 0; k < K; ++k) {
            set_lp(t, k, eos, -9.0f);
            for (int v = 1; v < V; ++v) {
                const float score = -0.04f - 0.0001f * static_cast<float>((v + k * 11 + t * 17) % 97);
                set_lp(t, k, v, score);
            }
        }
    }

    DeviceBuffer<float> log_probs(host_log_probs.size());
    DeviceBuffer<int32_t> serial_tokens(B * T * K);
    DeviceBuffer<int32_t> fast_tokens(B * T * K);
    DeviceBuffer<float> serial_scores(B * K);
    DeviceBuffer<float> fast_scores(B * K);
    log_probs.copy_from(host_log_probs, stream);

    expect_status(
        dbs_cuda_decode_forward(log_probs.get(), B, T, K, V, eos, serial_tokens.get(), serial_scores.get(), stream),
        DBS_CUDA_STATUS_OK,
        "serial decode eos carry parity");
    expect_status(
        dbs_cuda_decode_forward_fast_ex(log_probs.get(), B, T, K, V, eos, 0, fast_tokens.get(), fast_scores.get(), stream),
        DBS_CUDA_STATUS_OK,
        "fast decode eos carry parity");

    const auto serial_token_host = serial_tokens.copy_to(stream);
    const auto fast_token_host = fast_tokens.copy_to(stream);
    check(serial_token_host == fast_token_host, "fast decode tokens must match serial decode with EOS carry-forward");

    const auto serial_score_host = serial_scores.copy_to(stream);
    const auto fast_score_host = fast_scores.copy_to(stream);
    for (std::size_t i = 0; i < serial_score_host.size(); ++i) {
        expect_close(fast_score_host[i], serial_score_host[i], "fast decode score must match serial decode with EOS carry-forward");
    }
}

void test_synchronization_policy_api() {
    check(dbs_cuda_get_synchronization() == 1, "CUDA C API must synchronize by default");
    expect_status(dbs_cuda_set_synchronization(0), DBS_CUDA_STATUS_OK, "disable CUDA synchronization");
    check(dbs_cuda_get_synchronization() == 0, "CUDA synchronization should be disabled explicitly");
    expect_status(dbs_cuda_set_synchronization(2), DBS_CUDA_STATUS_INVALID_ARGUMENT, "invalid synchronization setting");
    expect_status(dbs_cuda_set_synchronization(1), DBS_CUDA_STATUS_OK, "enable CUDA synchronization");
    check(dbs_cuda_get_synchronization() == 1, "CUDA synchronization should be enabled explicitly");
}

std::size_t lp_index(int b, int t, int k, int v, int max_steps, int max_beam, int vocab) {
    return (((static_cast<std::size_t>(b) * max_steps + t) * max_beam + k) * vocab + v);
}

void test_variable_decode_metadata_and_padding(cudaStream_t stream) {
    constexpr int B = 2;
    constexpr int T = 3;
    constexpr int K = 2;
    constexpr int V = 8;
    std::vector<float> host_log_probs(B * T * K * V, -3.0f);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int k = 0; k < K; ++k) {
                host_log_probs[lp_index(b, t, k, 0, T, K, V)] = -0.1f;
                host_log_probs[lp_index(b, t, k, 1, T, K, V)] = -0.2f;
            }
        }
    }

    DeviceBuffer<float> log_probs(host_log_probs.size());
    DeviceBuffer<int32_t> steps(B);
    DeviceBuffer<int32_t> beams(B);
    DeviceBuffer<int32_t> eos(B);
    DeviceBuffer<int32_t> min_lengths(B);
    DeviceBuffer<int32_t> tokens(B * T * K);
    DeviceBuffer<float> scores(B * K);

    log_probs.copy_from(host_log_probs, stream);
    steps.copy_from({2, 3}, stream);
    beams.copy_from({1, 2}, stream);
    eos.copy_from({-1, -1}, stream);
    min_lengths.copy_from({0, 0}, stream);

    expect_status(
        dbs_cuda_decode_forward_variable(
            log_probs.get(), B, T, K, V,
            steps.get(), beams.get(), eos.get(), min_lengths.get(),
            tokens.get(), scores.get(), stream),
        DBS_CUDA_STATUS_OK,
        "valid variable decode");

    const auto host_tokens = tokens.copy_to(stream);
    const auto host_scores = scores.copy_to(stream);
    check(host_tokens[(0 * T + 0) * K + 1] == -1, "trailing beam token at t0 must be initialized");
    check(host_tokens[(0 * T + 1) * K + 1] == -1, "trailing beam token at t1 must be initialized");
    check(host_tokens[(0 * T + 2) * K + 0] == -1, "trailing step token k0 must be initialized");
    check(host_tokens[(0 * T + 2) * K + 1] == -1, "trailing step token k1 must be initialized");
    check(std::isfinite(host_scores[0]), "valid score must be finite");
    check(std::isinf(host_scores[1]) && host_scores[1] < 0.0f, "trailing score must be -Inf");
    check(std::isfinite(host_scores[2]) && std::isfinite(host_scores[3]), "second example scores must be finite");

    steps.copy_from({0, 3}, stream);
    expect_status(
        dbs_cuda_decode_forward_variable(
            log_probs.get(), B, T, K, V,
            steps.get(), beams.get(), eos.get(), min_lengths.get(),
            tokens.get(), scores.get(), stream),
        DBS_CUDA_STATUS_INVALID_ARGUMENT,
        "invalid per-example steps");

    steps.copy_from({2, 3}, stream);
    eos.copy_from({-2, -1}, stream);
    expect_status(
        dbs_cuda_decode_forward_variable(
            log_probs.get(), B, T, K, V,
            steps.get(), beams.get(), eos.get(), min_lengths.get(),
            tokens.get(), scores.get(), stream),
        DBS_CUDA_STATUS_INVALID_ARGUMENT,
        "invalid per-example eos");

    eos.copy_from({-1, -1}, stream);
    min_lengths.copy_from({-1, 0}, stream);
    expect_status(
        dbs_cuda_decode_forward_variable(
            log_probs.get(), B, T, K, V,
            steps.get(), beams.get(), eos.get(), min_lengths.get(),
            tokens.get(), scores.get(), stream),
        DBS_CUDA_STATUS_INVALID_ARGUMENT,
        "invalid per-example min_length");
}

}  // namespace

int main() {
    int device_count = 0;
    cudaError_t count_err = cudaGetDeviceCount(&device_count);
    if (count_err != cudaSuccess || device_count <= 0) {
        cudaGetLastError();
        std::cout << "CUDA device unavailable; skipping dbs_cuda_c_api_tests\n";
        return 0;
    }

    try {
        CUDA_CHECK(cudaSetDevice(0));
        CudaStream stream;
        check(dbs_cuda_available() == 1, "dbs_cuda_available must report an available CUDA device");
        test_synchronization_policy_api();
        test_sparse_scatter(stream.stream);
        test_direct_decode_eos_validation(stream.stream);
        test_fast_decode_eos_carry_matches_serial(stream.stream);
        test_variable_decode_metadata_and_padding(stream.stream);
        CUDA_CHECK(cudaStreamSynchronize(stream.stream));
    } catch (const std::exception& ex) {
        std::cerr << "dbs_cuda_c_api_tests failed: " << ex.what() << '\n';
        return 1;
    }

    std::cout << "dbs_cuda_c_api_tests passed\n";
    return 0;
}
