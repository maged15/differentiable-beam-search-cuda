#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[1]
version = (root / "VERSION").read_text().strip()
pyproject = (root / "pyproject.toml").read_text()
readme = (root / "README.md").read_text()

errors = []
if 'dynamic = ["version"]' not in pyproject:
    errors.append('pyproject.toml must use dynamic = ["version"]')
if 'version =' in re.sub(r'\[tool\.setuptools\.dynamic\][\s\S]*', '', pyproject):
    errors.append('pyproject.toml must not hardcode project.version')
if version not in readme:
    errors.append(f'README.md does not mention VERSION {version}')
for stale in re.findall(r'1\.0\.0rc\d+', readme):
    if stale != version:
        errors.append(f'README.md contains stale version {stale}, expected {version}')
if errors:
    for e in errors:
        print(e, file=sys.stderr)
    raise SystemExit(1)
print(f'version metadata ok: {version}')
