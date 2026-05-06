# Gradient semantics

The decoder returns hard beam-search results. Hard top-k selection is discontinuous, so no implementation can provide exact gradients through the selected indices.

`dbs` uses surrogate gradients:

- selected beam weights: softmax over selected beam scores
- relaxed pool weights: sigmoid-bisection k-hot relaxation over candidate scores
- sparse log-prob gradients: flattened `[T * K * V]` index/value pairs

Use these gradients for surrogate-gradient training and validation experiments. Do not interpret them as mathematical gradients of the hard top-k arg-selection itself.

## CUDA gradient status

CPU backward defaults to sparse surrogate gradients. CUDA PyTorch autograd supports a limited selected-path sparse surrogate backward for `beam_size <= 32`, using the trace emitted by the cooperative CUDA forward kernel. Larger CUDA beams are forward-only until a trace-emitting large-beam kernel is added. Hard beam selection remains discontinuous; gradients are surrogate gradients, not exact derivatives of top-k selection.
