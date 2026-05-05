# PyTorch API contract

`torch_dbs_extension.final_scores(log_probs, options)` accepts the same public shapes on CPU and CUDA:

- `[T, K, V] -> [K]`
- `[B, T, K, V] -> [B, K]`

`K` must equal `options.beam_size`. All dimensions must be non-empty. Inputs are promoted to FP32 internally; gradients are cast back to the original input dtype on CPU.

## Autograd support

CPU supports sparse surrogate autograd for both public shapes.

CUDA is forward-only. Calling `.backward()` through a CUDA input raises a `RuntimeError`. This is intentional until a CUDA sparse surrogate backward is implemented and validated against CPU gradients on adversarial cases.

## CUDA fast path

The exact custom CUDA path is the default. The score-only ATen top-k fast path is opt-in with `DBS_ENABLE_SCORE_ONLY_FAST_PATH=1`; it is intended for no-EOS final-score workloads and should not be used when parent/token trace semantics or model-state reordering are required unless separately validated.


## CUDA option support

CUDA forward supports the public tensor shapes `[T,K,V] -> [K]` and `[B,T,K,V] -> [B,K]`.
The CUDA extension currently implements hard final-score forward decoding for `beam_size`, `eos_token`, `min_length`, and `validate_inputs`. With validation enabled it rejects NaN and `+Inf` while allowing `-Inf` masked logits. Options that require CPU surrogate or shaping semantics, such as `length_penalty_alpha`, temperature fields, relaxed-pool sizing, `vocab_block`, and soft-top-k iteration settings, are rejected on CUDA instead of being silently ignored. CPU tensors continue to accept the full `DBSOptions` tuple and support surrogate autograd.

| Option | CPU | CUDA |
| --- | --- | --- |
| `beam_size` | supported | supported |
| `eos_token` | supported | supported |
| `min_length` | supported | supported |
| `validate_inputs` | supported | supported |
| `selected_temperature`, `soft_topk_temperature` | supported for surrogate gradients | rejected |
| `relaxed_pool_multiplier`, `soft_topk_*` iterations/tolerance | supported for surrogate gradients | rejected |
| `length_penalty_alpha`, `vocab_block` | supported by CPU decoder/shaping paths | rejected |
