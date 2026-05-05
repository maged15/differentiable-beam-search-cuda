#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[1]
version = (root / "VERSION").read_text().strip()
pyproject = (root / "pyproject.toml").read_text()
readme = (root / "README.md").read_text()
changelog = (root / "CHANGELOG.md").read_text()
cmake = (root / "CMakeLists.txt").read_text()
header = (root / "include" / "dbs.h").read_text()
source = (root / "src" / "differentiable_beam.cpp").read_text()

errors = []
if 'dynamic = ["version"]' not in pyproject:
    errors.append('pyproject.toml must use dynamic = ["version"]')
if 'version =' in re.sub(r'\[tool\.setuptools\.dynamic\][\s\S]*', '', pyproject):
    errors.append('pyproject.toml must not hardcode project.version')
if version not in readme:
    errors.append(f'README.md does not mention VERSION {version}')
changelog_current = re.search(r'^##\s+([^\s]+)', changelog, re.MULTILINE)
if not changelog_current:
    errors.append('CHANGELOG.md must start with a version heading')
elif changelog_current.group(1) != version:
    errors.append(f'CHANGELOG.md current heading is {changelog_current.group(1)}, expected {version}')
for stale in re.findall(r'1\.0\.0rc\d+', readme):
    if stale != version:
        errors.append(f'README.md contains stale version {stale}, expected {version}')
if f'return "{version}"' not in source:
    errors.append(f'dbs_version_string() must return VERSION {version}')
for rel in ["validation/v10_production_manifest.template.json", "validation/fixtures/cuda/golden_fixture_manifest.json"]:
    text = (root / rel).read_text()
    for stale in re.findall(r'1\.0\.0rc\d+', text):
        if stale != version:
            errors.append(f'{rel} contains stale version {stale}, expected {version}')
abi_values = set(re.findall(r'#define\s+DBS_ABI_VERSION\s+(\d+)', header + "\n" + source))
if len(abi_values) != 1:
    errors.append(f'DBS_ABI_VERSION definitions disagree: {sorted(abi_values)}')
else:
    abi = next(iter(abi_values))
    if f'set(DBS_ABI_VERSION {abi}' not in cmake:
        errors.append(f'CMakeLists.txt does not set DBS_ABI_VERSION {abi}')
    if 'SOVERSION ${DBS_ABI_VERSION}' not in cmake:
        errors.append('CMake shared-library SOVERSION must follow DBS_ABI_VERSION')
    if f'C ABI version: `{abi}`' not in readme:
        errors.append(f'README.md does not mention C ABI version {abi}')
if 'All rights reserved' in readme:
    errors.append('README.md contains license text that conflicts with MIT')
if 'license = {file = "LICENSE"}' not in pyproject:
    errors.append('pyproject.toml must point package license metadata at LICENSE')
if errors:
    for e in errors:
        print(e, file=sys.stderr)
    raise SystemExit(1)
print(f'version metadata ok: {version}')
