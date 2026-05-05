# Gradient semantics

The decoder returns hard beam-search results. Hard top-k selection is discontinuous, so no implementation can provide exact gradients through the selected indices.

`dbs` uses surrogate gradients:

- selected beam weights: softmax over selected beam scores
- relaxed pool weights: sigmoid-bisection k-hot relaxation over candidate scores
- sparse log-prob gradients: flattened `[T * K * V]` index/value pairs

Use these gradients for surrogate-gradient training and validation experiments. Do not interpret them as mathematical gradients of the hard top-k arg-selection itself.

## CUDA gradient status

CPU backward defaults to sparse surrogate gradients. CUDA sparse scatter source is present, but fused CUDA surrogate backward through PyTorch autograd is disabled until CPU/GPU gradient parity is demonstrated on hardware. Hard beam selection remains discontinuous; gradients are surrogate gradients, not exact derivatives of top-k selection.
