#!/usr/bin/env bash

# Shared path and immutable-asset helpers for workspace launchers.
# This file is sourced by scripts; it intentionally performs no command or
# hardware action on import.

if [[ -n "${FRANKA_WORKSPACE_LIB_LOADED:-}" ]]; then
  return 0
fi
readonly FRANKA_WORKSPACE_LIB_LOADED=1

readonly FRANKA_WORKSPACE_ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd
)"
readonly FRANKA_PYTHON_BIN="${FRANKA_PYTHON_BIN:-${FRANKA_WORKSPACE_ROOT}/.venv/bin/python}"

franka_require_file() {
  local path="$1"
  local label="${2:-file}"
  if [[ ! -f "${path}" ]]; then
    echo "[REFUSED] missing ${label}: ${path}" >&2
    return 2
  fi
}

franka_verify_sha256() {
  local path="$1"
  local expected="$2"
  local label="${3:-asset}"
  franka_require_file "${path}" "${label}" || return

  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "[REFUSED] ${label} SHA-256 changed" >&2
    echo "expected=${expected}" >&2
    echo "actual=${actual}" >&2
    return 2
  fi
  printf '%s\n' "${actual}"
}

franka_timestamped_run_id() {
  local prefix="$1"
  printf '%s-%s\n' "${prefix}" "$(date +%Y%m%d-%H%M%S)"
}

franka_default_video_path() {
  local run_id="$1"
  printf '%s/dexgrasp/runs/%s.mp4\n' "${FRANKA_WORKSPACE_ROOT}" "${run_id}"
}
