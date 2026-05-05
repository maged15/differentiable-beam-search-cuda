// SPDX-License-Identifier: MIT
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DBS_CUDA_STATUS_OK 0
#define DBS_CUDA_STATUS_UNAVAILABLE 1
#define DBS_CUDA_STATUS_INVALID_ARGUMENT 2
#define DBS_CUDA_STATUS_LAUNCH_FAILED 3

#ifndef DBS_CUDA_MAX_BEAM
#define DBS_CUDA_MAX_BEAM 64
#endif

#ifndef DBS_CUDA_FAST_MAX_BEAM
#define DBS_CUDA_FAST_MAX_BEAM 32
#endif

int dbs_cuda_available(void);
const char* dbs_cuda_status_string(int status);

/* CUDA calls are asynchronous by default. Set DBS_CUDA_SYNC_CHECK=1 or
 * DBS_CUDA_DEBUG_SYNC=1 in validation builds to synchronize the supplied stream
 * after launches and report device-side execution failures through the status. */

/* Real CUDA entry point. Inputs/outputs are device pointers. Tokens shape is
 * [B, T, K]; final_scores shape is [B, K]. The implementation performs hard
 * deterministic beam expansion on device and carries EOS beams forward.
 *
 * CUDA C entry points intentionally implement only hard final-score decoding:
 * no surrogate backward, selected/soft-top-k temperatures, relaxed pool,
 * GNMT length penalty, token constraints, or non-default CPU decoder shaping
 * options are accepted here. eos_token must be -1 or within [0, vocab_size). */
int dbs_cuda_decode_forward(
    const float* device_log_probs,
    int batch_size,
    int steps,
    int beam_size,
    int vocab_size,
    int eos_token,
    int32_t* device_tokens,
    float* device_final_scores,
    void* cuda_stream);

/* Variable-length batched variant. Optional per-example arrays are device
 * pointers; null arrays use the scalar arguments/defaults. Per-example
 * metadata is validated on device before decode. Dense output slots outside a
 * per-example beam/step range are initialized to -1 tokens and -Inf scores. */

/* Cooperative CUDA backend. One block decodes one batch example; threads cooperatively
 * scan vocabulary blocks and reduce deterministic top-k candidates in shared memory.
 * Falls back to the serial device kernel internally when beam_size exceeds the fast path. */
int dbs_cuda_decode_forward_fast(
    const float* device_log_probs,
    int batch_size,
    int steps,
    int beam_size,
    int vocab_size,
    int eos_token,
    int32_t* device_tokens,
    float* device_final_scores,
    void* cuda_stream);

int dbs_cuda_decode_forward_fast_ex(
    const float* device_log_probs,
    int batch_size,
    int steps,
    int beam_size,
    int vocab_size,
    int eos_token,
    int min_length,
    int32_t* device_tokens,
    float* device_final_scores,
    void* cuda_stream);

int dbs_cuda_decode_forward_variable(
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
    void* cuda_stream);

/* Sparse backward scatter: grad_out[index[i]] += value[i]. grad_out is a device
 * pointer with grad_out_count float elements. */
int dbs_cuda_sparse_backward_scatter(
    const int64_t* device_indices,
    const float* device_values,
    int64_t nnz,
    float* device_grad_out,
    int64_t grad_out_count,
    void* cuda_stream);

#ifdef __cplusplus
}
#endif
