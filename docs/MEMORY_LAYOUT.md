# Memory layout

Float32 forward input: `[T, K, V]` in row-major order, indexed as `(t*K + k)*V + v`.

Uniform batch input: `[B, T, K, V]`, indexed as `((b*T + t)*K + k)*V + v`.

Variable batch input: `[B, max_T, max_K, V]`, indexed as `((b*max_T + t)*max_K + k)*V + v`.

Sparse backward indices flatten `[T, K, V]` using the same `(t*K + k)*V + v` layout.
