#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CC:=clang}"
: "${CXX:=clang++}"
: "${DBS_FUZZ_SECONDS:=300}"
: "${DBS_RELEASE_ARTIFACT_DIR:=$ROOT/release}"
ART="${DBS_FUZZ_ARTIFACT_DIR:-$DBS_RELEASE_ARTIFACT_DIR/fuzz}"
mkdir -p "$ART" "$ART/corpus" "$ART/crashes"

{
  echo "fuzz_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cc=$CC"
  echo "cxx=$CXX"
  echo "seconds=$DBS_FUZZ_SECONDS"
  "$CC" --version | head -1 || true
  "$CXX" --version | head -1 || true
} | tee "$ART/environment.txt"

cmake -S "$ROOT" -B "$ROOT/build-fuzz-asan" \
  -DDBS_BUILD_FUZZER=ON \
  -DDBS_ENABLE_SANITIZERS=ON \
  -DDBS_BUILD_TESTS=OFF \
  -DDBS_BUILD_BENCHMARKS=OFF \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  2>&1 | tee "$ART/configure-fuzz.log"
cmake --build "$ROOT/build-fuzz-asan" --parallel 2>&1 | tee "$ART/build-fuzz.log"
"$ROOT/build-fuzz-asan/dbs_fuzz" "$ART/corpus" \
  -max_total_time="$DBS_FUZZ_SECONDS" \
  -artifact_prefix="$ART/crashes/" \
  -rss_limit_mb="${DBS_FUZZ_RSS_MB:-4096}" \
  2>&1 | tee "$ART/fuzzer.log"

cmake -S "$ROOT" -B "$ROOT/build-tsan" \
  -DDBS_ENABLE_TSAN=ON \
  -DDBS_BUILD_TESTS=ON \
  -DDBS_BUILD_BENCHMARKS=OFF \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  2>&1 | tee "$ART/configure-tsan.log"
cmake --build "$ROOT/build-tsan" --parallel 2>&1 | tee "$ART/build-tsan.log"
ctest --test-dir "$ROOT/build-tsan" --output-on-failure 2>&1 | tee "$ART/ctest-tsan.log"

cat > "$ART/summary.json" <<EOF
{
  "fuzz_seconds": ${DBS_FUZZ_SECONDS},
  "asan_ubsan_fuzzer": true,
  "tsan_tests": true,
  "logs": ["environment.txt", "configure-fuzz.log", "build-fuzz.log", "fuzzer.log", "configure-tsan.log", "build-tsan.log", "ctest-tsan.log"]
}
EOF

echo "fuzz artifacts written to $ART"
