// SPDX-License-Identifier: MIT
#include "dbs_cuda.h"

#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <cstdlib>

#ifndef DBS_CUDA_MAX_BEAM
#define DBS_CUDA_MAX_BEAM 64
#endif

static __device__ __forceinline__ bool better_candidate(float score, int parent, int token, float best_score, int best_parent, int best_token) {
    return score > best_score ||
           (score == best_score && (parent < best_parent || (parent == best_parent && token < best_token)));
}

__global__ void dbs_forward_kernel(
    const float* __restrict__ log_probs,
    int batch_size,
    int steps,
    int beam_size,
    int vocab_size,
    int eos_token,
    const int32_t* __restrict__ steps_per_example,
    const int32_t* __restrict__ beam_sizes_per_example,
    const int32_t* __restrict__ eos_tokens_per_example,
    const int32_t* __restrict__ min_lengths_per_example,
    int32_t* __restrict__ tokens,
    float* __restrict__ final_scores) {

    const int b = blockIdx.x;
    if (b >= batch_size) return;

    const int T = steps_per_example ? steps_per_example[b] : steps;
    const int K = beam_sizes_per_example ? beam_sizes_per_example[b] : beam_size;
    const int eos = eos_tokens_per_example ? eos_tokens_per_example[b] : eos_token;
    const int min_len = min_lengths_per_example ? min_lengths_per_example[b] : 0;
    if (K <= 0 || K > DBS_CUDA_MAX_BEAM || T <= 0) return;

    float prev_scores[DBS_CUDA_MAX_BEAM];
    float next_scores[DBS_CUDA_MAX_BEAM];
    int prev_lengths[DBS_CUDA_MAX_BEAM];
    int next_lengths[DBS_CUDA_MAX_BEAM];
    unsigned char ended[DBS_CUDA_MAX_BEAM];
    unsigned char ended_next[DBS_CUDA_MAX_BEAM];

    for (int k = 0; k < K; ++k) {
        prev_scores[k] = (k == 0) ? 0.0f : -CUDART_INF_F;
        prev_lengths[k] = 0;
        ended[k] = 0;
    }

    for (int t = 0; t < T; ++t) {
        float best_scores[DBS_CUDA_MAX_BEAM];
        int best_parent[DBS_CUDA_MAX_BEAM];
        int best_token[DBS_CUDA_MAX_BEAM];
        int best_length[DBS_CUDA_MAX_BEAM];
        unsigned char best_ended[DBS_CUDA_MAX_BEAM];
        for (int i = 0; i < K; ++i) {
            best_scores[i] = -CUDART_INF_F;
            best_parent[i] = DBS_CUDA_MAX_BEAM;
            best_token[i] = 2147483647;
            best_length[i] = 0;
            best_ended[i] = 0;
        }

        for (int parent = 0; parent < K; ++parent) {
            if (!isfinite(prev_scores[parent])) continue;
            if (eos >= 0 && ended[parent]) {
                const float score = prev_scores[parent];
                int pos = -1;
                for (int j = 0; j < K; ++j) {
                    if (better_candidate(score, parent, eos, best_scores[j], best_parent[j], best_token[j])) { pos = j; break; }
                }
                if (pos >= 0) {
                    for (int j = K - 1; j > pos; --j) {
                        best_scores[j] = best_scores[j - 1]; best_parent[j] = best_parent[j - 1]; best_token[j] = best_token[j - 1]; best_length[j] = best_length[j - 1]; best_ended[j] = best_ended[j - 1];
                    }
                    best_scores[pos] = score; best_parent[pos] = parent; best_token[pos] = eos; best_length[pos] = prev_lengths[parent]; best_ended[pos] = 1;
                }
                continue;
            }
            const int64_t base = ((static_cast<int64_t>(b) * steps + t) * beam_size + parent) * vocab_size;
            for (int v = 0; v < vocab_size; ++v) {
                if (eos >= 0 && v == eos && prev_lengths[parent] + 1 < min_len) continue;
                const float lp = log_probs[base + v];
                if (!isfinite(lp)) continue;
                const float score = prev_scores[parent] + lp;
                int pos = -1;
                for (int j = 0; j < K; ++j) {
                    if (better_candidate(score, parent, v, best_scores[j], best_parent[j], best_token[j])) { pos = j; break; }
                }
                if (pos >= 0) {
                    for (int j = K - 1; j > pos; --j) {
                        best_scores[j] = best_scores[j - 1]; best_parent[j] = best_parent[j - 1]; best_token[j] = best_token[j - 1]; best_length[j] = best_length[j - 1]; best_ended[j] = best_ended[j - 1];
                    }
                    best_scores[pos] = score; best_parent[pos] = parent; best_token[pos] = v; best_length[pos] = prev_lengths[parent] + 1; best_ended[pos] = (eos >= 0 && v == eos) ? 1 : 0;
                }
            }
        }

        for (int k = 0; k < K; ++k) {
            next_scores[k] = best_scores[k];
            next_lengths[k] = best_length[k];
            ended_next[k] = best_ended[k];
            tokens[(b * steps + t) * beam_size + k] = (best_token[k] == 2147483647) ? -1 : best_token[k];
        }
        for (int k = 0; k < K; ++k) { prev_scores[k] = next_scores[k]; prev_lengths[k] = next_lengths[k]; ended[k] = ended_next[k]; }
    }

    for (int k = 0; k < K; ++k) final_scores[b * beam_size + k] = prev_scores[k];
}

__global__ void dbs_sparse_scatter_kernel(const int64_t* idx, const float* val, int64_t nnz, float* grad, int64_t grad_count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < nnz; i += stride) {
        const int64_t j = idx[i];
        if (j >= 0 && j < grad_count) atomicAdd(grad + j, val[i]);
    }
}

static bool dbs_cuda_sync_checks_enabled() {
    const char* sync_check = std::getenv("DBS_CUDA_SYNC_CHECK");
    const char* debug_sync = std::getenv("DBS_CUDA_DEBUG_SYNC");
    return (sync_check && sync_check[0] == '1') || (debug_sync && debug_sync[0] == '1');
}

static int finish_cuda(cudaError_t err, cudaStream_t stream) {
    if (err != cudaSuccess) return DBS_CUDA_STATUS_LAUNCH_FAILED;
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return DBS_CUDA_STATUS_LAUNCH_FAILED;
    if (dbs_cuda_sync_checks_enabled()) {
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) return DBS_CUDA_STATUS_LAUNCH_FAILED;
    }
    err = cudaGetLastError();
    return err == cudaSuccess ? DBS_CUDA_STATUS_OK : DBS_CUDA_STATUS_LAUNCH_FAILED;
}

extern "C" int dbs_cuda_available(void) {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0 ? 1 : 0;
}

extern "C" const char* dbs_cuda_status_string(int status) {
    switch (status) {
        case DBS_CUDA_STATUS_OK: return "ok";
        case DBS_CUDA_STATUS_UNAVAILABLE: return "cuda backend unavailable";
        case DBS_CUDA_STATUS_INVALID_ARGUMENT: return "invalid argument";
        case DBS_CUDA_STATUS_LAUNCH_FAILED: return "cuda launch failed";
        default: return "unknown cuda status";
    }
}

extern "C" int dbs_cuda_decode_forward(
    const float* device_log_probs,
    int batch_size,
    int steps,
    int beam_size,
    int vocab_size,
    int eos_token,
    int32_t* device_tokens,
    float* device_final_scores,
    void* cuda_stream) {
    if (!device_log_probs || !device_tokens || !device_final_scores || batch_size <= 0 || steps <= 0 || beam_size <= 0 || beam_size > DBS_CUDA_MAX_BEAM || vocab_size <= 0) return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    if (eos_token >= vocab_size) return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    dbs_forward_kernel<<<batch_size, 1, 0, stream>>>(device_log_probs, batch_size, steps, beam_size, vocab_size, eos_token, nullptr, nullptr, nullptr, nullptr, device_tokens, device_final_scores);
    return finish_cuda(cudaPeekAtLastError(), stream);
}

extern "C" int dbs_cuda_decode_forward_variable(
    const float* device_log_probs,
    int batch_size,
    int max_steps,
    int max_beam_size,
    int vocab_size,
    const int32_t* device_steps_per_example,
    const int32_t* device_beam_sizes_per_example,
    const int32_t* device_eos_tokens_per_example,
    const int32_t* device_min_lengths_per_example,
    int32_t* device_tokens,
    float* device_final_scores,
    void* cuda_stream) {
    if (!device_log_probs || !device_tokens || !device_final_scores || batch_size <= 0 || max_steps <= 0 || max_beam_size <= 0 || max_beam_size > DBS_CUDA_MAX_BEAM || vocab_size <= 0) return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    dbs_forward_kernel<<<batch_size, 1, 0, stream>>>(device_log_probs, batch_size, max_steps, max_beam_size, vocab_size, -1,
        device_steps_per_example, device_beam_sizes_per_example, device_eos_tokens_per_example, device_min_lengths_per_example,
        device_tokens, device_final_scores);
    return finish_cuda(cudaPeekAtLastError(), stream);
}

extern "C" int dbs_cuda_sparse_backward_scatter(
    const int64_t* device_indices,
    const float* device_values,
    int64_t nnz,
    float* device_grad_out,
    int64_t grad_out_count,
    void* cuda_stream) {
    if (!device_grad_out || nnz < 0 || grad_out_count <= 0) return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    if (nnz == 0) return DBS_CUDA_STATUS_OK;
    if (!device_indices || !device_values) return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    const int threads = 256;
    constexpr int max_portable_grid_x = 65535;
    const int64_t needed_blocks = (nnz + threads - 1) / threads;
    const int blocks = static_cast<int>(std::min<int64_t>(needed_blocks, max_portable_grid_x));
    if (blocks <= 0) return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    dbs_sparse_scatter_kernel<<<blocks, threads, 0, stream>>>(device_indices, device_values, nnz, device_grad_out, grad_out_count);
    return finish_cuda(cudaPeekAtLastError(), stream);
}

#ifndef DBS_CUDA_FAST_THREADS
#define DBS_CUDA_FAST_THREADS 64
#endif

static __device__ __forceinline__ void insert_local_candidate(
    float* scores,
    int* parents,
    int* tokens,
    int* lengths,
    unsigned char* ended,
    int K,
    float score,
    int parent,
    int token,
    int length,
    unsigned char end_flag) {
    if (!better_candidate(score, parent, token, scores[K - 1], parents[K - 1], tokens[K - 1])) return;
    int pos = K - 1;
    while (pos > 0 && better_candidate(score, parent, token, scores[pos - 1], parents[pos - 1], tokens[pos - 1])) --pos;
    for (int j = K - 1; j > pos; --j) {
        scores[j] = scores[j - 1];
        parents[j] = parents[j - 1];
        tokens[j] = tokens[j - 1];
        lengths[j] = lengths[j - 1];
        ended[j] = ended[j - 1];
    }
    scores[pos] = score;
    parents[pos] = parent;
    tokens[pos] = token;
    lengths[pos] = length;
    ended[pos] = end_flag;
}

__global__ void dbs_forward_fast_kernel(
    const float* __restrict__ log_probs,
    int batch_size,
    int steps,
    int beam_size,
    int vocab_size,
    int eos_token,
    int32_t* __restrict__ tokens_out,
    float* __restrict__ final_scores) {

    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int tid = threadIdx.x;
    const int K = beam_size;
    if (K <= 0 || K > DBS_CUDA_FAST_MAX_BEAM || steps <= 0 || vocab_size <= 0) return;

    __shared__ float prev_scores[DBS_CUDA_FAST_MAX_BEAM];
    __shared__ int prev_lengths[DBS_CUDA_FAST_MAX_BEAM];
    __shared__ unsigned char prev_ended[DBS_CUDA_FAST_MAX_BEAM];
    __shared__ float cand_scores[DBS_CUDA_FAST_THREADS * DBS_CUDA_FAST_MAX_BEAM];
    __shared__ int cand_parents[DBS_CUDA_FAST_THREADS * DBS_CUDA_FAST_MAX_BEAM];
    __shared__ int cand_tokens[DBS_CUDA_FAST_THREADS * DBS_CUDA_FAST_MAX_BEAM];
    __shared__ int cand_lengths[DBS_CUDA_FAST_THREADS * DBS_CUDA_FAST_MAX_BEAM];
    __shared__ unsigned char cand_ended[DBS_CUDA_FAST_THREADS * DBS_CUDA_FAST_MAX_BEAM];

    if (tid < K) {
        prev_scores[tid] = (tid == 0) ? 0.0f : -CUDART_INF_F;
        prev_lengths[tid] = 0;
        prev_ended[tid] = 0;
    }
    __syncthreads();

    float local_scores[DBS_CUDA_FAST_MAX_BEAM];
    int local_parents[DBS_CUDA_FAST_MAX_BEAM];
    int local_tokens[DBS_CUDA_FAST_MAX_BEAM];
    int local_lengths[DBS_CUDA_FAST_MAX_BEAM];
    unsigned char local_ended[DBS_CUDA_FAST_MAX_BEAM];

    for (int t = 0; t < steps; ++t) {
        for (int i = 0; i < K; ++i) {
            local_scores[i] = -CUDART_INF_F;
            local_parents[i] = 2147483647;
            local_tokens[i] = 2147483647;
            local_lengths[i] = 0;
            local_ended[i] = 0;
        }

        for (int parent = 0; parent < K; ++parent) {
            const float ps = prev_scores[parent];
            if (!isfinite(ps)) continue;
            if (eos_token >= 0 && prev_ended[parent]) {
                // Only thread 0 inserts the EOS carry-forward candidate. This is safe because
                // thread 0's local list is merged into the global best by the __syncthreads
                // reduction below before any thread reads it, preventing double-counting.
                if (tid == 0) insert_local_candidate(local_scores, local_parents, local_tokens, local_lengths, local_ended, K, ps, parent, eos_token, prev_lengths[parent], 1);
                continue;
            }
            const int64_t base = ((static_cast<int64_t>(b) * steps + t) * K + parent) * vocab_size;
            for (int v = tid; v < vocab_size; v += blockDim.x) {
                const float lp = log_probs[base + v];
                if (!isfinite(lp)) continue;
                const float score = ps + lp;
                insert_local_candidate(local_scores, local_parents, local_tokens, local_lengths, local_ended, K, score, parent, v, prev_lengths[parent] + 1, (eos_token >= 0 && v == eos_token) ? 1 : 0);
            }
        }

        const int off = tid * DBS_CUDA_FAST_MAX_BEAM;
        for (int i = 0; i < K; ++i) {
            cand_scores[off + i] = local_scores[i];
            cand_parents[off + i] = local_parents[i];
            cand_tokens[off + i] = local_tokens[i];
            cand_lengths[off + i] = local_lengths[i];
            cand_ended[off + i] = local_ended[i];
        }
        __syncthreads();

        if (tid == 0) {
            float best_scores[DBS_CUDA_FAST_MAX_BEAM];
            int best_parents[DBS_CUDA_FAST_MAX_BEAM];
            int best_tokens[DBS_CUDA_FAST_MAX_BEAM];
            int best_lengths[DBS_CUDA_FAST_MAX_BEAM];
            unsigned char best_ended[DBS_CUDA_FAST_MAX_BEAM];
            for (int i = 0; i < K; ++i) {
                best_scores[i] = -CUDART_INF_F;
                best_parents[i] = 2147483647;
                best_tokens[i] = 2147483647;
                best_lengths[i] = 0;
                best_ended[i] = 0;
            }
            for (int lane = 0; lane < DBS_CUDA_FAST_THREADS; ++lane) {
                const int lane_off = lane * DBS_CUDA_FAST_MAX_BEAM;
                for (int i = 0; i < K; ++i) {
                    insert_local_candidate(best_scores, best_parents, best_tokens, best_lengths, best_ended, K,
                        cand_scores[lane_off + i], cand_parents[lane_off + i], cand_tokens[lane_off + i], cand_lengths[lane_off + i], cand_ended[lane_off + i]);
                }
            }
            for (int i = 0; i < K; ++i) {
                prev_scores[i] = best_scores[i];
                prev_lengths[i] = best_lengths[i];
                prev_ended[i] = best_ended[i];
                tokens_out[(b * steps + t) * K + i] = (best_tokens[i] == 2147483647) ? -1 : best_tokens[i];
            }
        }
        __syncthreads();
    }

    if (tid < K) final_scores[b * K + tid] = prev_scores[tid];
}

extern "C" int dbs_cuda_decode_forward_fast(
    const float* device_log_probs,
    int batch_size,
    int steps,
    int beam_size,
    int vocab_size,
    int eos_token,
    int32_t* device_tokens,
    float* device_final_scores,
    void* cuda_stream) {
    if (!device_log_probs || !device_tokens || !device_final_scores || batch_size <= 0 || steps <= 0 || beam_size <= 0 || vocab_size <= 0) return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    if (eos_token >= vocab_size) return DBS_CUDA_STATUS_INVALID_ARGUMENT;
    if (beam_size > DBS_CUDA_FAST_MAX_BEAM) {
        return dbs_cuda_decode_forward(device_log_probs, batch_size, steps, beam_size, vocab_size, eos_token, device_tokens, device_final_scores, cuda_stream);
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    dbs_forward_fast_kernel<<<batch_size, DBS_CUDA_FAST_THREADS, 0, stream>>>(device_log_probs, batch_size, steps, beam_size, vocab_size, eos_token, device_tokens, device_final_scores);
    return finish_cuda(cudaPeekAtLastError(), stream);
}
