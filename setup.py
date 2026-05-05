from pathlib import Path
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

ROOT = Path(__file__).resolve().parent
version = (ROOT / "VERSION").read_text().strip()

# setuptools rejects absolute source paths, but absolute include paths are valid and
# needed because PEP 517 builds compile from temporary directories.
include_dirs = [str(ROOT / "include")]

# Build the C ABI directly into the torch extension so wheels are self-contained.
cpu_sources = [
    "python/torch_extension.cpp",
    "src/differentiable_beam.cpp",
]

ext_modules = [
    CppExtension(
        "dbs_torch_ext",
        cpu_sources,
        include_dirs=include_dirs,
        extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
    )
]

build_cuda = os.environ.get("DBS_BUILD_TORCH_CUDA", "0") == "1"
if build_cuda:
    if CUDA_HOME is None:
        raise RuntimeError("DBS_BUILD_TORCH_CUDA=1 was set, but CUDA_HOME was not found. Install a CUDA toolkit with nvcc and set CUDA_HOME.")
    ext_modules.append(
        CUDAExtension(
            "dbs_torch_cuda_ext",
            ["python/torch_cuda_extension.cpp", "cuda/dbs_cuda.cu"],
            include_dirs=include_dirs,
            extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": ["-O3", "--use_fast_math"]},
        )
    )

setup(
    name="dbs-torch",
    version=version,
    py_modules=["torch_dbs", "torch_dbs_extension", "jax_dbs"],
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
