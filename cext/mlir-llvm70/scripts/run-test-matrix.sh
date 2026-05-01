#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Run the llvm70 test suite across a matrix of CUDA versions × SM architectures.
#
# Usage:
#   ./scripts/run-matrix.sh                                  # full default matrix
#   ./scripts/run-matrix.sh --filter atomics                 # only matching tests
#   ./scripts/run-matrix.sh --cuda 12.4,12.9 --sm sm_75,sm_90
#   ./scripts/run-matrix.sh --cuda 12.9 --sm sm_80           # single cell

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${LLVM70_BUILD_DIR:-$PROJECT_DIR/build}"
BUILD_TEST_DIR="$BUILD_DIR/test"

CUDA_BASE="/proj/cuda"
DEFAULT_CUDAS="12.2,12.3,12.4,12.5,12.6,12.7,12.8,12.9,13.0,13.1,13.2,13.3"
DEFAULT_SMS="sm_75,sm_80,sm_90"

# Discover lit.
if [[ -n "${LLVM_EXTERNAL_LIT:-}" && -x "$LLVM_EXTERNAL_LIT" ]]; then
  LIT="$LLVM_EXTERNAL_LIT"
elif [[ -x /usr/bin/lit ]]; then
  LIT=/usr/bin/lit
elif command -v lit &>/dev/null; then
  LIT="$(command -v lit)"
else
  echo "error: cannot find lit (set LLVM_EXTERNAL_LIT or install lit)" >&2
  exit 1
fi

# Parse args.
FILTER=""
CUDA_LIST=""
SM_LIST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --filter)   FILTER="$2"; shift 2 ;;
    --filter=*) FILTER="${1#--filter=}"; shift ;;
    --cuda)     CUDA_LIST="$2"; shift 2 ;;
    --cuda=*)   CUDA_LIST="${1#--cuda=}"; shift ;;
    --sm)       SM_LIST="$2"; shift 2 ;;
    --sm=*)     SM_LIST="${1#--sm=}"; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

IFS=',' read -ra CUDAS <<< "${CUDA_LIST:-$DEFAULT_CUDAS}"
IFS=',' read -ra SMS <<< "${SM_LIST:-$DEFAULT_SMS}"

PASS=0
FAIL=0
SKIP=0
FAILED_CELLS=()

for cuda_ver in "${CUDAS[@]}"; do
  LIBNVVM="$CUDA_BASE/$cuda_ver/Linux_x86_64/nvvm/lib64/libnvvm.so"
  if [[ ! -f "$LIBNVVM" ]]; then
    echo "=== CUDA $cuda_ver: SKIP (libnvvm not found) ==="
    ((SKIP += ${#SMS[@]})) || true
    continue
  fi

  for sm in "${SMS[@]}"; do
    LABEL="CUDA $cuda_ver × $sm"
    echo "=== $LABEL ==="

    LIT_ARGS=(-v)
    if [[ -n "$FILTER" ]]; then
      LIT_ARGS+=(--filter "$FILTER")
    fi

    if LLVM70_LIBNVVM="$LIBNVVM" LLVM70_CHIP="$sm" \
       "$LIT" "${LIT_ARGS[@]}" "$BUILD_TEST_DIR"; then
      echo "=== $LABEL: PASS ==="
      ((PASS++)) || true
    else
      echo "=== $LABEL: FAIL ==="
      ((FAIL++)) || true
      FAILED_CELLS+=("$LABEL")
    fi
    echo
  done
done

TOTAL=$((PASS + FAIL + SKIP))
echo "========================================"
echo "Matrix: ${#CUDAS[@]} CUDA × ${#SMS[@]} SM = $TOTAL cells"
echo "  PASS: $PASS   FAIL: $FAIL   SKIP: $SKIP"
echo "  CUDA: ${CUDAS[*]}"
echo "  SM:   ${SMS[*]}"
if [[ ${#FAILED_CELLS[@]} -gt 0 ]]; then
  echo "  Failed:"
  for cell in "${FAILED_CELLS[@]}"; do
    echo "    - $cell"
  done
fi
echo "========================================"
[[ $FAIL -eq 0 ]]
