#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${1:-$ROOT/validation/v10_production_manifest.json}"
if [[ ! -f "$MANIFEST" ]]; then
  echo "missing production evidence manifest: $MANIFEST" >&2
  echo "copy validation/v10_production_manifest.template.json to this path only after CI has produced every artifact" >&2
  exit 1
fi
python3 "$ROOT/scripts/validate_v10_production_manifest.py" "$MANIFEST"
if [[ ! -f "$ROOT/release/RELEASE_REPORT_v1.0.md" ]]; then
  echo "missing attached release report: release/RELEASE_REPORT_v1.0.md" >&2
  exit 1
fi
for f in \
  release/checksums.txt \
  release/sbom.spdx.json \
  release/provenance.intoto.jsonl \
  release/vulnerability_scan.json \
  release/license_scan.json \
  release/cuda_parity_raw_logs.tar.zst \
  release/fuzz_sanitizer_artifacts.tar.zst \
  release/soak_slo_artifacts.tar.zst; do
  if [[ ! -f "$ROOT/$f" ]]; then
    echo "missing required release artifact: $f" >&2
    exit 1
  fi
done
echo "v1.0 production gate passed"
