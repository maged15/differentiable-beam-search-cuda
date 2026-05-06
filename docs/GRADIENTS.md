# Gradient semantics

The decoder returns hard beam-search results. Hard top-k selection is discontinuous, so no implementation can provide exact gradients through the selected indices.

`dbs` uses surrogate gradients:

- selected beam weights: softmax over selected beam scores
- relaxed pool weights: sigmoid-bisection k-hot relaxation over candidate scores
- sparse log-prob gradients: flattened `[T * K * V]` index/value pairs

Use these gradients for surrogate-gradient training and validation experiments. Do not interpret them as mathematical gradients of the hard top-k arg-selection itself.

## CUDA gradient status

CPU backward defaults to sparse surrogate gradients. Public PyTorch CUDA `final_scores()` backward uses the CPU semantic implementation internally and returns CUDA gradients, so CPU/CUDA tensor gradients are expected to match for this public API. The direct CUDA C sparse-backward helper is a limited selected-path estimator used for low-level validation; it does not implement the full lower-level CPU selected-weight plus relaxed-pool surrogate contract. Hard beam selection remains discontinuous; gradients are surrogate gradients, not exact derivatives of top-k selection.
