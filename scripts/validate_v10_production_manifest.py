#!/usr/bin/env python3
import json, sys
from pathlib import Path

REQUIRED_TRUE = [
    "ci_produced_artifacts_only",
    "no_skipped_targets",
    "supported_matrix_frozen",
    "cpu_wheel_clean_install_passed",
    "cuda_wheel_clean_install_passed",
    "cuda_parity_passed",
    "cuda_fixture_hashes_verified",
    "fuzz_sanitizer_artifacts_archived",
    "soak_passed",
    "slo_load_test_passed",
    "performance_thresholds_passed",
    "simd_correctness_passed",
    "simd_speedup_passed",
    "zero_allocation_hot_path_passed",
    "artifacts_signed",
    "checksums_attached",
    "sbom_attached",
    "provenance_attached",
    "vulnerability_scan_passed",
    "license_scan_passed",
    "canary_failure_injection_passed",
    "abi_chain_passed",
    "release_report_attached",
]
REQUIRED_REVIEWS = ["api_abi", "security", "numerical_correctness", "performance", "ml_integration"]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_v10_production_manifest.py <manifest.json>", file=sys.stderr)
        return 2
    manifest = Path(sys.argv[1])
    data = json.loads(manifest.read_text())
    failures = []
    if data.get("version") not in {"1.0.0", "1.0.0rc7"}:
        failures.append("version must be 1.0.0 or 1.0.0rc7")
    for key in REQUIRED_TRUE:
        if data.get(key) is not True:
            failures.append(f"{key} must be true")
    if float(data.get("fuzz_sanitizer_hours", 0)) < 24:
        failures.append("fuzz_sanitizer_hours must be >= 24")
    if float(data.get("soak_hours", 0)) < 4:
        failures.append("soak_hours must be >= 4")
    reviews = data.get("independent_reviews", {})
    for key in REQUIRED_REVIEWS:
        if reviews.get(key) is not True:
            failures.append(f"independent_reviews.{key} must be true")
    if failures:
        print("PRODUCTION GATE FAILED")
        for f in failures:
            print(f"- {f}")
        return 1
    print("PRODUCTION GATE PASSED")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
