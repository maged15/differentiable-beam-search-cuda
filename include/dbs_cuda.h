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

/* CUDA calls synchronize the supplied stream by default so correctness-oriented
 * callers receive device-side failures through the returned status. Use
 * dbs_cuda_set_synchronization(0) only for callers that explicitly want normal
 * asynchronous launch semantics. Validation flags DBS_CUDA_SYNC_CHECK=1 or
 * DBS_CUDA_DEBUG_SYNC=1 still force a sync. */
int dbs_cuda_set_synchronization(int synchronize);
int dbs_cuda_get_synchronization(void);

/* Real CUDA entry point. Inputs/outputs are device pointers. Tokens shape is
 * [B, T, K]; final_scores shape is [B, K]. The implementation performs hard
 * deterministic beam expansion on device and carries EOS beams forward.
 *
 * CUDA decode entry points intentionally implement hard final-score decoding.
 * Sparse surrogate backward is exposed through the separate trace/backward
 * helpers below. selected/soft-top-k temperatures, relaxed pool, GNMT length
 * penalty, token constraints, and non-default CPU decoder shaping options are
 * not accepted here. eos_token must be -1 or within [0, vocab_size). */
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
 * metadata is validated on device before decode. device_log_probs must use dense
 * [B, max_steps, max_beam_size, vocab_size] layout with max_beam_size as the
 * beam stride, not a tightly packed per-example beam_size stride. Dense output
 * slots outside a per-example beam/step range are initialized to -1 tokens and
 * -Inf scores. */

/* Cooperative CUDA backend. One block decodes one batch example; threads cooperatively
 * scan vocabulary blocks and reduce deterministic top-k candidates in shared memory.
 * beam_size <= DBS_CUDA_FAST_MAX_BEAM uses the cooperative path. Larger beams
 * fall back to a serial correctness kernel that launches one device thread per
 * batch item; do not treat that fallback as a throughput CUDA implementation. */
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

/* Full forward: identical to decode_forward_fast_ex but also writes
 * parents[B,T,K] and from_logprob[B,T,K] needed for sparse backward.
 * beam_size must be <= DBS_CUDA_FAST_MAX_BEAM because the serial fallback does
 * not emit parent/from_logprob traces. Public PyTorch final_scores() uses CPU
 * semantic fallback for larger CUDA beam sizes. */
int dbs_cuda_decode_forward_full(
    const float* device_log_probs,
    int batch_size,
    int steps,
    int beam_size,
    int vocab_size,
    int eos_token,
    int min_length,
    int32_t* device_tokens,
    float* device_final_scores,
    int32_t* device_parents,
    uint8_t* device_from_logprob,
    void* cuda_stream);

/* Limited sparse backward helper: build (flat_index, value) pairs for valid
 * selected-beam slots using a score-softmax estimator over final_scores. This
 * helper is intentionally narrower than the CPU C ABI backward, which also
 * supports selected-weight and relaxed-pool surrogate gradients. Public
 * PyTorch CUDA autograd uses CPU semantic fallback for full final_scores()
 * parity. Pass the result to dbs_cuda_sparse_backward_scatter to accumulate
 * into grad_log_probs. out_indices and out_values must have B*T*K elements. */
int dbs_cuda_backward_build_sparse(
    const int32_t* device_tokens,
    const int32_t* device_parents,
    const uint8_t* device_from_logprob,
    const float*   device_final_scores,
    const float*   device_grad_output,
    int batch_size,
    int steps,
    int beam_size,
    int vocab_size,
    int64_t* device_out_indices,
    float*   device_out_values,
    void* cuda_stream);

/* Sparse backward scatter: grad_out[index[i]] += value[i]. grad_out is a device
 * pointer with grad_out_count float elements. Duplicate valid indices
 * accumulate. Out-of-range indices, including -1 sentinels emitted by the
 * sparse backward builder, are skipped by the scatter kernel. */
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
