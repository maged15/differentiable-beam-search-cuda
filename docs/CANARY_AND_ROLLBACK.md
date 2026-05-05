# Canary and Rollback

Production enablement should use a feature flag with an immediate fallback to scalar CPU decoding. Canary traffic should start below 1%, then increase only if p95/p99 latency, fallback rate, CUDA error rate, memory pressure, allocation count, and output parity diagnostics stay within the release thresholds.

Rollback is mandatory if CUDA parity checks fail, unsupported ISA fallback exceeds threshold, allocator counters increase in hot paths, SLOs regress, diagnostics are incomplete, or any crash/sanitizer signal appears.
