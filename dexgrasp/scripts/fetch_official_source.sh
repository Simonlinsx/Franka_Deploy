#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/third_party/AnyDexGrasp"
COMMIT="c9c4a43df33e40860417c7e2dd02f5122d3b2da2"

if [[ ! -d "$TARGET/.git" ]]; then
  mkdir -p "$ROOT/third_party"
  git clone https://github.com/graspnet/AnyDexGrasp.git "$TARGET"
fi

git -C "$TARGET" fetch --depth 1 origin "$COMMIT"
git -C "$TARGET" checkout --detach "$COMMIT"
echo "[ok] AnyDexGrasp source: $TARGET @ $COMMIT"
