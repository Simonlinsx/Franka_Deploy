#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$WORKSPACE_ROOT/.venv/bin/python" -m sim2real.tasks.launcher thrown_object "$@"
