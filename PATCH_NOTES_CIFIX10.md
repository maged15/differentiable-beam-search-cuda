# cifix10

- Fixed `test_rejects_zero_dimensions_and_bad_beam` so it rejects an actual beam-size mismatch instead of valid `[B,T,K,V]` input with `B=1`.
- Added `scripts/check_version_metadata.py` and `make version-check`.
- Bumped package version metadata to `1.0.0rc8` using `VERSION` as the source of truth.
- Reaffirmed CUDA-forward-only package positioning in README.
