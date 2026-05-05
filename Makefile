# Convenience Makefile wrapper around CMake for DBS.
# Usage examples:
#   make
#   make test
#   make bench
#   make install PREFIX=$PWD/install
#   make clean
#   make cuda
#   make asan test

BUILD_DIR ?= build
BUILD_TYPE ?= Release
PREFIX ?= $(CURDIR)/install
JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
CMAKE ?= cmake
CTEST ?= ctest

DBS_BUILD_SHARED ?= ON
DBS_BUILD_TESTS ?= ON
DBS_BUILD_BENCHMARKS ?= ON
DBS_BUILD_FUZZER ?= OFF
DBS_ENABLE_CUDA ?= OFF
DBS_ENABLE_SANITIZERS ?= OFF
DBS_ENABLE_TSAN ?= OFF
DBS_ENABLE_MSAN ?= OFF
DBS_BUILD_HARDWARE_GATES ?= ON

.PHONY: help all configure build test bench install clean distclean rebuild \
        debug release shared static cuda cuda-test asan tsan msan fuzzer \
        abi abi-compat hardware-gate release-gate wheels cuda-wheels wheel-test cuda-wheel-test python-test package \
        cuda-validate perf-gate soak long-fuzz release-artifacts

help:
	@printf '%s\n' \
	  'DBS Makefile targets:' \
	  '  make                 Configure and build Release shared library, tests, and benchmark' \
	  '  make test            Run CTest' \
	  '  make bench           Build and run benchmark smoke test' \
	  '  make install         Install to PREFIX, default ./install' \
	  '  make clean           Remove the active build directory' \
	  '  make distclean       Remove common build/install/artifact directories' \
	  '  make debug           Configure/build Debug' \
	  '  make static          Configure/build static library' \
	  '  make cuda            Configure/build CUDA backend; requires CUDA toolkit' \
	  '  make asan            Configure/build Address+UB sanitizer build' \
	  '  make tsan            Configure/build ThreadSanitizer build' \
	  '  make msan            Configure/build MemorySanitizer build; Clang recommended' \
	  '  make fuzzer          Build libFuzzer harness; Clang recommended' \
	  '  make abi             Run exact ABI symbol check' \
	  '  make abi-compat      Run ABI compatibility chain check' \
	  '  make wheels          Build Python wheels via scripts/build_wheels.sh' \
	  '  make release-gate    Run fail-closed production gate; requires full CI evidence' \
	  '  make cuda-validate  Build CUDA wheel, run parity, benchmark, and perf thresholds' \
	  '  make soak           Run repeated decode/backward soak test' \
	  '  make long-fuzz      Run longer sanitizer/fuzzer campaign with artifacts' \
	  '  make release-artifacts Generate checksums and release metadata artifacts' \
	  '' \
	  'Useful variables:' \
	  '  BUILD_DIR=build-cuda BUILD_TYPE=RelWithDebInfo PREFIX=/opt/dbs JOBS=8'

all: configure build

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) \
	  -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	  -DDBS_BUILD_SHARED=$(DBS_BUILD_SHARED) \
	  -DDBS_BUILD_TESTS=$(DBS_BUILD_TESTS) \
	  -DDBS_BUILD_BENCHMARKS=$(DBS_BUILD_BENCHMARKS) \
	  -DDBS_BUILD_FUZZER=$(DBS_BUILD_FUZZER) \
	  -DDBS_ENABLE_CUDA=$(DBS_ENABLE_CUDA) \
	  -DDBS_ENABLE_SANITIZERS=$(DBS_ENABLE_SANITIZERS) \
	  -DDBS_ENABLE_TSAN=$(DBS_ENABLE_TSAN) \
	  -DDBS_ENABLE_MSAN=$(DBS_ENABLE_MSAN) \
	  -DDBS_BUILD_HARDWARE_GATES=$(DBS_BUILD_HARDWARE_GATES)

build: configure
	$(CMAKE) --build $(BUILD_DIR) -j$(JOBS)

release:
	$(MAKE) BUILD_TYPE=Release all

debug:
	$(MAKE) BUILD_TYPE=Debug BUILD_DIR=build-debug all

shared:
	$(MAKE) DBS_BUILD_SHARED=ON all

static:
	$(MAKE) DBS_BUILD_SHARED=OFF BUILD_DIR=build-static all

cuda:
	$(MAKE) DBS_ENABLE_CUDA=ON BUILD_DIR=build-cuda all

cuda-test: cuda
	$(CTEST) --test-dir build-cuda --output-on-failure

asan:
	$(MAKE) BUILD_TYPE=Debug DBS_ENABLE_SANITIZERS=ON BUILD_DIR=build-asan all

tsan:
	$(MAKE) BUILD_TYPE=Debug DBS_ENABLE_TSAN=ON BUILD_DIR=build-tsan all

msan:
	$(MAKE) BUILD_TYPE=Debug DBS_ENABLE_MSAN=ON BUILD_DIR=build-msan all

fuzzer:
	$(MAKE) BUILD_TYPE=Debug DBS_BUILD_FUZZER=ON BUILD_DIR=build-fuzzer all

test: build
	$(CTEST) --test-dir $(BUILD_DIR) --output-on-failure

bench: build
	./$(BUILD_DIR)/dbs_bench

install: build
	$(CMAKE) --install $(BUILD_DIR) --prefix $(PREFIX)

abi: build
	$(CMAKE) --build $(BUILD_DIR) --target dbs_abi_check

abi-compat: build
	$(CMAKE) --build $(BUILD_DIR) --target dbs_abi_compat_check

hardware-gate: build
	$(CMAKE) --build $(BUILD_DIR) --target dbs_hardware_validation

release-gate: build
	$(CMAKE) --build $(BUILD_DIR) --target dbs_release_gate

wheels:
	bash scripts/build_wheels.sh --no-isolation

cuda-wheels:
	bash scripts/build_wheels.sh --cuda --no-isolation

wheel-test:
	bash scripts/test_wheels_clean.sh

cuda-wheel-test:
	bash scripts/test_wheels_clean.sh --cuda

python-test:
	python -m pytest python/tests

package:
	$(CMAKE) --build $(BUILD_DIR) --target package


cuda-validate:
	mkdir -p logs benchmarks/results validation/release_artifacts
	bash scripts/build_wheels.sh --cuda --no-isolation
	bash scripts/test_wheels_clean.sh --cuda
	python3 python/tests/test_cuda_parity.py
	python3 benchmarks/bench_dbs_cuda_direct.py
	python3 scripts/check_perf_thresholds.py benchmarks/results/bench-dbs-cuda-direct.csv

perf-gate:
	python3 scripts/check_perf_thresholds.py benchmarks/results/bench-dbs-cuda-direct.csv

soak:
	python3 scripts/run_soak_test.py --device cpu --iterations $${DBS_SOAK_ITERATIONS:-1000}
	@if command -v nvidia-smi >/dev/null 2>&1; then python3 scripts/run_soak_test.py --device cuda --iterations $${DBS_SOAK_ITERATIONS:-1000}; else echo "CUDA not detected; skipping CUDA soak"; fi

long-fuzz:
	DBS_FUZZ_SECONDS=$${DBS_FUZZ_SECONDS:-86400} bash scripts/run_fuzz_campaign.sh

release-artifacts:
	bash scripts/generate_release_artifacts.sh

rebuild: clean all

clean:
	rm -rf $(BUILD_DIR)

distclean:
	rm -rf build build-* install dist wheelhouse *.egg-info .pytest_cache
	find . -type f \( -name '*.so' -o -name '*.a' -o -name '*.dll' -o -name '*.dylib' -o -name '*.o' -o -name '*.obj' -o -name '*.exe' \) -delete

version-check:
	python3 scripts/check_version_metadata.py
