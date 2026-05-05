# CUDA wheel fix 3

- Uses absolute include paths in `setup.py` while keeping extension source paths relative, fixing `fatal error: dbs.h: No such file or directory` under PEP 517 builds.
- Requires `packaging>=24.2` for cleaner setuptools metadata validation.
- Makes `scripts/test_wheels_clean.sh` test the wheel in the current Python environment by default to avoid PyTorch ABI/CUDA-version drift; pass `--isolated` for a temporary environment.
