#!/usr/bin/env bash
set -euo pipefail
LIB="${1:-build/libdbs.so}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED="${2:-$ROOT/docs/abi/v1.0.symbols}"
if [[ ! -f "$LIB" ]]; then
  echo "usage: $0 path/to/libdbs.so [expected-symbols]" >&2
  exit 2
fi
if [[ ! -f "$EXPECTED" ]]; then
  echo "expected symbols file not found: $EXPECTED" >&2
  exit 2
fi
nm -D --defined-only "$LIB" | awk '{print $3}' | sed 's/@@.*//' | grep '^dbs_' | sort -u > /tmp/dbs_symbols.current
sort -u "$EXPECTED" > /tmp/dbs_symbols.expected
diff -u /tmp/dbs_symbols.expected /tmp/dbs_symbols.current
