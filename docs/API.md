# API reference

## Handles

`DBSDecoderHandle`, `DBSResultHandle`, `DBSBackwardHandle`, `DBSBatchResultHandle`, and `DBSWorkspaceHandle` are opaque. Objects returned by the C API must be released with the matching `dbs_free_*` or `dbs_destroy` function.

## Forward decode

`dbs_decode()` consumes a contiguous float32 tensor shaped `[T, K, V]` and returns selected parents, tokens, lengths, scores, raw scores, relaxed-pool candidates, relaxed weights, and final scores.

`dbs_decode_typed()` accepts float32, IEEE fp16, or bf16 input through `DBSDataTypeC`; fp16/bf16 are converted to float32 for accumulation.

`dbs_decode_batch()` consumes `[B, T, K, V]`. `dbs_decode_batch_variable()` consumes `[B, max_T, max_K, V]` plus per-example lengths, beam sizes, EOS, min lengths, banned masks, and forced-token schedules.

`dbs_decode_model_steps()` and `dbs_decode_model_steps_with_workspace()` call a user callback once per step. The callback can own KV cache/recurrent state through `user_data`.

## Backward

`dbs_backward()` and `dbs_backward_sparse()` return sparse flattened `[T*K*V]` gradient indices and values. `dbs_backward_dense()` is opt-in and protected by `max_dense_gradient_elements`.

## Constraints

`dbs_decode_constrained_ex()` supports banned tokens, forced tokens, min length, repetition penalty, no-repeat n-gram, and a token filter callback. Grammar, regex, lexicon, and prefix-automaton constraints should be implemented behind the token filter callback in production integrations.

## Observability

`dbs_get_stats()` returns structured counters. `dbs_get_stats_json()` returns a compact JSON object suitable for logs and tracing.

## v1.0 observability and reproducibility additions

`dbs_set_deterministic_seed()` and `dbs_get_deterministic_seed()` record the seed associated with a decoder handle. The current hard beam path is deterministic and does not consume randomness; this API exists to preserve reproducibility metadata if stochastic decoding modes are added later.

`dbs_allocator_counters_reset()`, `dbs_allocator_call_count()`, and `dbs_allocator_byte_count()` expose process-level allocator instrumentation for DBS aligned allocations. `dbs_get_stats()` and `dbs_get_stats_json()` include these totals as `total_allocator_calls`, `total_allocator_bytes`, `allocator_calls`, and `allocator_bytes`.

## CUDA backend API

`dbs_cuda_decode_forward_fast()` is the cooperative CUDA forward path. It uses one block per batch example, threads cooperatively scan vocabulary entries, and a deterministic shared-memory reduction selects the top beams. For `beam_size > 32`, it falls back to the serial CUDA correctness kernel. Device tensors are expected to be contiguous FP32.

CUDA autograd is limited to the selected-path sparse surrogate backward for `beam_size <= 32`; larger beams are forward-only on CUDA. Run `python/tests/test_cuda_parity.py` on NVIDIA hardware before enabling CUDA in any training or inference path.

## v1.0 observability and gate APIs

`dbs_result_eos_count(result, eos_token)` counts selected EOS tokens in a decode result.

`dbs_result_validate_deterministic_order(result)` verifies that each selected beam row remains sorted by score with deterministic parent/token tie ordering.

`dbs_result_summary_json(result, eos_token, buffer, capacity)` writes a compact JSON summary with selected count, relaxed pool count, EOS count, min/max lengths, final-score range, and deterministic-order status.

`dbs_validate_production_gate_manifest(manifest_json, error_buffer, capacity)` validates that a release manifest has all required production gates set to true. This is a fail-closed API: missing or false gates return non-zero.


## PyTorch public tensor API

The compiled wrapper `torch_dbs_extension.final_scores()` accepts `[T,K,V]` and `[B,T,K,V]` on both CPU and CUDA. CPU supports surrogate autograd. CUDA supports hard forward and limited selected-path sparse surrogate backward for `beam_size <= 32`. See `docs/PYTORCH_API.md`.
