#!/usr/bin/env bash
set -euo pipefail
LIB="${1:-build/libdbs.so}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="${2:-$ROOT/docs/abi/v0.5.symbols}"
if [[ ! -f "$LIB" || ! -f "$BASELINE" ]]; then
  echo "usage: $0 path/to/libdbs.so [baseline-symbols]" >&2
  exit 2
fi
nm -D --defined-only "$LIB" | awk '{print $3}' | sed 's/@@.*//' | grep '^dbs_' | sort -u > /tmp/dbs_symbols.current
missing=0
while IFS= read -r sym; do
  [[ -z "$sym" ]] && continue
  if ! grep -qx "$sym" /tmp/dbs_symbols.current; then
    echo "missing backwards-compatible symbol: $sym" >&2
    missing=1
  fi
done < <(sort -u "$BASELINE")
exit "$missing"
