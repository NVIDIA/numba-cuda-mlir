#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build modern LLVM + MLIR from source for numba-cuda-mlir.
# Produces: $LLVM_INSTALL with cmake configs usable via MLIR_DIR.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/llvm-version.env"

LLVM_COMMIT="${LLVM_MODERN_COMMIT}"
BUILD_ROOT="${BUILD_ROOT:-${PWD}/_build}"
LLVM_SRC="${LLVM_MODERN_SRC:-${BUILD_ROOT}/llvm-project}"
LLVM_BUILD="${LLVM_MODERN_BUILD:-${BUILD_ROOT}/llvm-build}"
LLVM_INSTALL="${LLVM_MODERN_INSTALL:-${PWD}/llvm-modern-install}"
PARALLEL="${PARALLEL:-$(nproc)}"
PYTHON="${PYTHON:-python3}"

echo "=== Building modern LLVM+MLIR (${LLVM_COMMIT}) ==="
echo "  Source:  ${LLVM_SRC}"
echo "  Build:   ${LLVM_BUILD}"
echo "  Install: ${LLVM_INSTALL}"

mkdir -p "${BUILD_ROOT}"

# Download source if not present
if [ ! -d "${LLVM_SRC}/llvm" ]; then
    echo ">>> Cloning LLVM (commit ${LLVM_COMMIT})"
    git clone --depth 1 https://github.com/llvm/llvm-project.git "${LLVM_SRC}"
    cd "${LLVM_SRC}"
    git fetch --depth 1 origin "${LLVM_COMMIT}"
    git checkout "${LLVM_COMMIT}"
    cd -
fi

# Require sccache for compiler caching
command -v sccache &>/dev/null || { echo "ERROR: sccache not found"; exit 1; }
export CMAKE_C_COMPILER_LAUNCHER="$(which sccache)"
export CMAKE_CXX_COMPILER_LAUNCHER="$(which sccache)"

# Configure
# MLIR_PYTHON_PACKAGE_PREFIX bakes "numba_cuda_mlir._mlir." into all compiled .so
#   files, so bindings expect numba_cuda_mlir._mlir.ir (not mlir.ir) at runtime.
# MLIR_BINDINGS_PYTHON_INSTALL_PREFIX controls on-disk install path.
# MLIR_BINDINGS_PYTHON_NB_DOMAIN isolates nanobind typeids from other
#   MLIR-based projects that may coexist in the same process.
# See: https://github.com/llvm/llvm-project/pull/171775
cmake -G Ninja -S "${LLVM_SRC}/llvm" -B "${LLVM_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL}" \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="NVPTX" \
    -DLLVM_BUILD_TOOLS=OFF \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_DOCS=OFF \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DCMAKE_CXX_FLAGS="-DMLIR_PYTHON_PACKAGE_PREFIX=numba_cuda_mlir._mlir." \
    -DMLIR_BINDINGS_PYTHON_INSTALL_PREFIX="python_packages/numba_cuda_mlir_mlir/numba_cuda_mlir/_mlir" \
    -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=numba_cuda_mlir \
    -DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON \
    -DPython3_EXECUTABLE="$($PYTHON -c 'import sys; print(sys.executable)')"

# Build & install
cmake --build "${LLVM_BUILD}" -j "${PARALLEL}"
cmake --install "${LLVM_BUILD}"

echo "=== Modern LLVM installed to ${LLVM_INSTALL} ==="
echo "  MLIR_DIR=${LLVM_INSTALL}/lib/cmake/mlir"
ls -lh "${LLVM_INSTALL}/python_packages/numba_cuda_mlir_mlir/numba_cuda_mlir/_mlir/_mlir_libs/libMLIRPythonCAPI.so" 2>/dev/null || true

echo "=== sccache stats ==="
sccache --show-stats
