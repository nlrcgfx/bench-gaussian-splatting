#!/usr/bin/env bash
set -euo pipefail

export GS_DATASETS="${GS_DATASETS:-/workspace/datasets}"
export GS_RUNS="${GS_RUNS:-/workspace/runs}"
export GS_ARTIFACTS="${GS_ARTIFACTS:-/workspace/artifacts}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/cache}"
export TORCH_HOME="${TORCH_HOME:-${XDG_CACHE_HOME}/torch}"

mkdir -p \
  "${GS_DATASETS}" \
  "${GS_RUNS}" \
  "${GS_ARTIFACTS}" \
  "${XDG_CACHE_HOME}" \
  "${TORCH_HOME}"

exec "$@"
