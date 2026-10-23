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
# The bindings install as `cudamlir`, a name shared across this ecosystem
# rather than owned by this repository. They were previously built as
# `numba_cuda_mlir._mlir` with a private nanobind domain, which isolated them
# from any other MLIR in the same interpreter. A dialect built elsewhere has to
# register into a context created here, which it can only do if both sides
# resolve to the same MLIR, so that isolation is what we are trading away.
# Not upstream's `mlir`: this is one particular build, not general MLIR, and it
# should not be mistakable for an official package.
#
# MLIR_BINDINGS_PYTHON_NB_DOMAIN is deliberately left at its default, because
# the nanobind domain is the part that was purely isolation. The prefixes below
# are naming -- though the package prefix is also baked into PyCapsule names,
# so it has to agree across everyone exchanging capsules.
python_executable="$($PYTHON -c 'import sys; print(sys.executable)')"
python_root="$(dirname "${python_executable}")"

cmake_args=(
    -G Ninja
    -S "${LLVM_SRC}/llvm"
    -B "${LLVM_BUILD}"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL}"
    -DLLVM_ENABLE_PROJECTS="mlir"
    -DLLVM_TARGETS_TO_BUILD="NVPTX"
    -DLLVM_BUILD_TOOLS=OFF
    -DLLVM_BUILD_EXAMPLES=OFF
    -DLLVM_INCLUDE_TESTS=OFF
    -DLLVM_INCLUDE_BENCHMARKS=OFF
    -DLLVM_INCLUDE_DOCS=OFF
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
    -DMLIR_PYTHON_PACKAGE_PREFIX="cudamlir"
    -DCMAKE_CXX_FLAGS="-DMLIR_PYTHON_PACKAGE_PREFIX=cudamlir."
    -DMLIR_BINDINGS_PYTHON_INSTALL_PREFIX="python_packages/cuda_mlir_core/cudamlir"
    -DMLIR_PYTHON_STUBGEN_ENABLED=OFF
    -DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON
    -DPython_ROOT_DIR="${python_root}"
    -DPython_EXECUTABLE="${python_executable}"
    -DPython_FIND_REGISTRY=NEVER
    -DPython3_ROOT_DIR="${python_root}"
    -DPython3_EXECUTABLE="${python_executable}"
    -DPython3_FIND_REGISTRY=NEVER
)

cmake "${cmake_args[@]}"

# Build & install. --strip removes the static symbol table (.symtab +
# .strtab); LLVM is built without -g so there's no DWARF to strip, but
# the symbol table itself accounts for ~70 MB on libMLIRPythonCAPI.so
# alone -- ~25 MB compressed in the wheel.
cmake --build "${LLVM_BUILD}" -j "${PARALLEL}"
cmake --install "${LLVM_BUILD}" --strip

echo "=== Modern LLVM installed to ${LLVM_INSTALL} ==="
echo "  MLIR_DIR=${LLVM_INSTALL}/lib/cmake/mlir"
ls -lh "${LLVM_INSTALL}/python_packages/cuda_mlir_core/cudamlir/_mlir_libs/libMLIRPythonCAPI.so" 2>/dev/null || true

echo "=== sccache stats ==="
sccache --show-stats
