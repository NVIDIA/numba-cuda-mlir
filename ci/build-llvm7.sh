#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build LLVM 7 (shared libLLVM-7.so) for the LLVM70 path.
#
# Based on Vijay/Brandon's setup_cusimt_wheel.sh.
# Note: does NOT cmake --install; the .so is used directly from the build dir.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/llvm-version.env"

LLVM_TAG="${LLVM7_TAG}"
BUILD_ROOT="${BUILD_ROOT:-${PWD}/_build}"
LLVM7_SRC="${LLVM7_SRC:-${BUILD_ROOT}/llvm7-src}"
LLVM7_BUILD="${LLVM7_BUILD:-${BUILD_ROOT}/llvm7-build}"
# Output dir — the .so is copied here for artifact passing
LLVM7_INSTALL="${LLVM7_INSTALL:-${PWD}/llvm7-install}"
PARALLEL="${PARALLEL:-$(nproc)}"

echo "=== Building LLVM 7 (${LLVM_TAG}) ==="
echo "  Source: ${LLVM7_SRC}"
echo "  Build:  ${LLVM7_BUILD}"

mkdir -p "${BUILD_ROOT}" "${LLVM7_INSTALL}/lib"

# Download source if not present
if [ ! -d "${LLVM7_SRC}/llvm" ]; then
    echo ">>> Cloning LLVM 7 (${LLVM_TAG})"
    git clone --depth 1 --branch "${LLVM_TAG}" \
        https://github.com/llvm/llvm-project.git "${LLVM7_SRC}"
fi

# Patch CMP0051 OLD → NEW for modern CMake compatibility
sed -i 's/cmake_policy(SET CMP0051 OLD)/cmake_policy(SET CMP0051 NEW)/' \
    "${LLVM7_SRC}/llvm/CMakeLists.txt"

# Require sccache for compiler caching
command -v sccache &>/dev/null || { echo "ERROR: sccache not found"; exit 1; }
export CMAKE_C_COMPILER_LAUNCHER="$(which sccache)"
export CMAKE_CXX_COMPILER_LAUNCHER="$(which sccache)"

# Configure
cmake -G Ninja -S "${LLVM7_SRC}/llvm" -B "${LLVM7_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DLLVM_TARGETS_TO_BUILD="NVPTX" \
    -DLLVM_BUILD_LLVM_DYLIB=ON \
    -DLLVM_BUILD_TOOLS=OFF \
    -DLLVM_BUILD_UTILS=OFF \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_DOCS=OFF \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_ENABLE_ZLIB=ON

# Build only the LLVM shared lib (not install)
cmake --build "${LLVM7_BUILD}" -j "${PARALLEL}" --target LLVM

# Copy the .so to the install dir with a stable name for artifact passing
LLVM7_SO="$(ls "${LLVM7_BUILD}"/lib/libLLVM-7*.so | head -1)"
cp "${LLVM7_SO}" "${LLVM7_INSTALL}/lib/libLLVM-7.so"

echo "=== LLVM 7 built ==="
ls -lh "${LLVM7_INSTALL}/lib/libLLVM-7.so"

echo "=== sccache stats ==="
sccache --show-stats
