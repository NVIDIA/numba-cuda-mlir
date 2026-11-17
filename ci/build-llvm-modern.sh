#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build modern LLVM + MLIR from source for numba-cuda-mlir, on Linux or on
# Windows (Git Bash with MSVC on PATH).
# Produces: $LLVM_INSTALL with cmake configs usable via MLIR_DIR.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/llvm-version.env"

case "$(uname -s)" in
    MINGW*|MSYS*) IS_WINDOWS=1 ;;
    *) IS_WINDOWS=0 ;;
esac

LLVM_COMMIT="${LLVM_MODERN_COMMIT}"
BUILD_ROOT="${BUILD_ROOT:-${PWD}/_build}"
LLVM_SRC="${LLVM_MODERN_SRC:-${BUILD_ROOT}/llvm-project}"
LLVM_BUILD="${LLVM_MODERN_BUILD:-${BUILD_ROOT}/llvm-build}"
LLVM_INSTALL="${LLVM_MODERN_INSTALL:-${PWD}/llvm-modern-install}"
PARALLEL="${PARALLEL:-$(nproc)}"
if [[ "${IS_WINDOWS}" == 1 ]]; then
    # Windows venvs have no python3.exe.
    PYTHON="${PYTHON:-python}"
else
    PYTHON="${PYTHON:-python3}"
fi

# CMake on Windows needs native paths (C:/...).
cmake_path() {
    if [[ "${IS_WINDOWS}" == 1 ]]; then
        cygpath -m "$1"
    else
        printf '%s\n' "$1"
    fi
}

echo "=== Building modern LLVM+MLIR (${LLVM_COMMIT}) ==="
echo "  Source:  ${LLVM_SRC}"
echo "  Build:   ${LLVM_BUILD}"
echo "  Install: ${LLVM_INSTALL}"

mkdir -p "${BUILD_ROOT}"

# Download source if not present: only the pinned commit, and only the
# directories the build reads.
if [ ! -d "${LLVM_SRC}/llvm" ]; then
    echo ">>> Fetching LLVM (commit ${LLVM_COMMIT})"
    git init -q "${LLVM_SRC}"
    git -C "${LLVM_SRC}" remote add origin https://github.com/llvm/llvm-project.git
    git -C "${LLVM_SRC}" sparse-checkout set llvm mlir cmake third-party
    git -C "${LLVM_SRC}" fetch --depth 1 --filter=blob:none origin "${LLVM_COMMIT}"
    git -C "${LLVM_SRC}" checkout -q FETCH_HEAD
fi

# Require sccache for compiler caching
command -v sccache &>/dev/null || { echo "ERROR: sccache not found"; exit 1; }
CMAKE_C_COMPILER_LAUNCHER="$(cmake_path "$(command -v sccache)")"
CMAKE_CXX_COMPILER_LAUNCHER="${CMAKE_C_COMPILER_LAUNCHER}"
export CMAKE_C_COMPILER_LAUNCHER CMAKE_CXX_COMPILER_LAUNCHER

# Configure
# MLIR_PYTHON_PACKAGE_PREFIX bakes "numba_cuda_mlir._mlir." into all compiled
#   extension modules, so bindings expect numba_cuda_mlir._mlir.ir (not mlir.ir)
#   at runtime.
# MLIR_BINDINGS_PYTHON_INSTALL_PREFIX controls on-disk install path.
# MLIR_BINDINGS_PYTHON_NB_DOMAIN isolates nanobind typeids from other
#   MLIR-based projects that may coexist in the same process.
# See: https://github.com/llvm/llvm-project/pull/171775
python_executable="$(cmake_path "$("${PYTHON}" -c 'import sys; print(sys.executable)')")"
python_root="$(dirname "${python_executable}")"

cxx_flags="-DMLIR_PYTHON_PACKAGE_PREFIX=numba_cuda_mlir._mlir."
if [[ "${IS_WINDOWS}" == 1 ]]; then
    # TypeIDs cross DLL boundaries on Windows, so resolve them by name.
    cxx_flags+=" -DMLIR_USE_FALLBACK_TYPE_IDS=1"
fi

cmake_args=(
    -G Ninja
    -S "$(cmake_path "${LLVM_SRC}/llvm")"
    -B "$(cmake_path "${LLVM_BUILD}")"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="$(cmake_path "${LLVM_INSTALL}")"
    -DLLVM_ENABLE_PROJECTS="mlir"
    -DLLVM_TARGETS_TO_BUILD="NVPTX"
    -DLLVM_BUILD_TOOLS=OFF
    -DLLVM_BUILD_EXAMPLES=OFF
    -DLLVM_INCLUDE_TESTS=OFF
    -DLLVM_INCLUDE_BENCHMARKS=OFF
    -DLLVM_INCLUDE_DOCS=OFF
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
    -DMLIR_PYTHON_PACKAGE_PREFIX="numba_cuda_mlir._mlir"
    -DCMAKE_CXX_FLAGS="${cxx_flags}"
    -DMLIR_BINDINGS_PYTHON_INSTALL_PREFIX="python_packages/numba_cuda_mlir_mlir/numba_cuda_mlir/_mlir"
    -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=numba_cuda_mlir
    -DMLIR_PYTHON_STUBGEN_ENABLED=OFF
    -DPython_ROOT_DIR="${python_root}"
    -DPython_EXECUTABLE="${python_executable}"
    -DPython_FIND_REGISTRY=NEVER
    -DPython3_ROOT_DIR="${python_root}"
    -DPython3_EXECUTABLE="${python_executable}"
    -DPython3_FIND_REGISTRY=NEVER
    -DLLVM_PARALLEL_LINK_JOBS="${LLVM_PARALLEL_LINK_JOBS:-${PARALLEL}}"
    # Adjusts the MLIR Python CAPI library; see cext/mlir-python/CMakeLists.txt.
    -DLLVM_EXTERNAL_PROJECTS=numba_cuda_mlir_python
    -DLLVM_EXTERNAL_NUMBA_CUDA_MLIR_PYTHON_SOURCE_DIR="$(cmake_path "${REPO_ROOT}/cext/mlir-python")"
)
if [[ "${IS_WINDOWS}" == 1 ]]; then
    # Static CRT (/MT), so the wheel needs no MSVC runtime DLLs.
    cmake_args+=(
        -DCMAKE_C_COMPILER=cl
        -DCMAKE_CXX_COMPILER=cl
        -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
        -DLLVM_ENABLE_ZLIB=OFF
        -DLLVM_ENABLE_ZSTD=OFF
    )
else
    cmake_args+=(-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON)
fi

cmake "${cmake_args[@]}"

# Build & install. --strip removes the static symbol table (.symtab +
# .strtab); LLVM is built without -g so there's no DWARF to strip, but
# the symbol table itself accounts for ~70 MB on the Linux CAPI library
# alone -- ~25 MB compressed in the wheel. MSVC builds have nothing to strip.
cmake --build "$(cmake_path "${LLVM_BUILD}")" -j "${PARALLEL}"
cmake --install "$(cmake_path "${LLVM_BUILD}")" --strip

echo "=== Modern LLVM installed to ${LLVM_INSTALL} ==="
echo "  MLIR_DIR=${LLVM_INSTALL}/lib/cmake/mlir"
ls -lh "${LLVM_INSTALL}/python_packages/numba_cuda_mlir_mlir/numba_cuda_mlir/_mlir/_mlir_libs/"*MLIRPythonCAPI*

echo "=== sccache stats ==="
sccache --show-stats
