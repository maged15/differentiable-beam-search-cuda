#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DBS_RELEASE_ARTIFACT_DIR:=$ROOT/release}"
OUT="$DBS_RELEASE_ARTIFACT_DIR"
mkdir -p "$OUT"

VERSION="$(cat "$ROOT/VERSION" 2>/dev/null || echo 1.0.0)"
DATE_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Build source archive excluding transient build outputs and git metadata.
tar --exclude='.git' --exclude='build*' --exclude='dist' --exclude='*.egg-info' --exclude='logs' --exclude='profiles' \
  -czf "$OUT/dbs-${VERSION}-source.tar.gz" -C "$ROOT" .

(
  cd "$ROOT"
  find dist release -maxdepth 1 -type f \( -name '*.whl' -o -name '*.so' -o -name '*.tar.gz' \) -print0 2>/dev/null \
    | xargs -0 -r sha256sum
) > "$OUT/SHA256SUMS"

cat > "$OUT/SBOM.spdx.json" <<EOF
{
  "spdxVersion": "SPDX-2.3",
  "dataLicense": "CC0-1.0",
  "SPDXID": "SPDXRef-DOCUMENT",
  "name": "dbs-${VERSION}",
  "documentNamespace": "https://github.com/maged15/differentiable-beam-search-cuda/releases/${VERSION}/sbom",
  "creationInfo": {
    "created": "${DATE_UTC}",
    "creators": ["Tool: scripts/generate_release_artifacts.sh"]
  },
  "packages": [
    {"name": "dbs", "SPDXID": "SPDXRef-Package-dbs", "versionInfo": "${VERSION}", "downloadLocation": "NOASSERTION", "filesAnalyzed": false, "licenseConcluded": "MIT", "licenseDeclared": "MIT"}
  ]
}
EOF

cat > "$OUT/provenance.intoto.json" <<EOF
{
  "_type": "https://in-toto.io/Statement/v1",
  "subject": [{"name": "dbs", "digest": {"sha256": "$(sha256sum "$OUT/dbs-${VERSION}-source.tar.gz" | awk '{print $1}')"}}],
  "predicateType": "https://slsa.dev/provenance/v1",
  "predicate": {
    "buildType": "manual-or-ci-release",
    "builder": {"id": "github-actions-or-local"},
    "metadata": {"buildStartedOn": "${DATE_UTC}", "completeness": {"parameters": true, "environment": true, "materials": true}, "reproducible": false}
  }
}
EOF

echo "release artifacts written to $OUT"
