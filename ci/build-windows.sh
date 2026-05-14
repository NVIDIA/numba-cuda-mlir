#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/llvm-version.env"

MODE="${1:-all}"
BUILD_ROOT="${BUILD_ROOT:-${REPO_ROOT}/_build}"
LLVM7_SRC="${LLVM7_SRC:-${BUILD_ROOT}/llvm7-src}"
LLVM7_BUILD="${LLVM7_BUILD:-${BUILD_ROOT}/llvm7-build}"
LLVM7_INSTALL="${LLVM7_INSTALL:-${REPO_ROOT}/llvm7-install}"
LLVM_MODERN_SRC="${LLVM_MODERN_SRC:-${BUILD_ROOT}/llvm-project}"
LLVM_MODERN_BUILD="${LLVM_MODERN_BUILD:-${BUILD_ROOT}/llvm-build}"
LLVM_MODERN_INSTALL="${LLVM_MODERN_INSTALL:-${REPO_ROOT}/llvm-modern-install}"
PYTHON="${PYTHON:-python}"
PARALLEL="${PARALLEL:-${NUMBER_OF_PROCESSORS:-2}}"

case "${MODE}" in
  all|llvm7|modern) ;;
  *)
    echo "Usage: $0 [all|llvm7|modern]" >&2
    exit 2
    ;;
esac

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

step() {
  local name="$1"
  shift
  local start end
  start="$(date +%s)"
  echo "[$(timestamp)] >>> ${name}"
  "$@"
  end="$(date +%s)"
  echo "[$(timestamp)] <<< ${name} completed in $((end - start))s"
}

require_tool() {
  local tool="$1"
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "ERROR: required tool not found in PATH: ${tool}" >&2
    exit 1
  fi
}

cmake_path() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -m "$1"
  else
    printf '%s\n' "$1"
  fi
}

check_prereqs() {
  require_tool cmake
  require_tool ninja
  require_tool git
  require_tool cl
  "${PYTHON}" -c "import sys; print(sys.executable)"
}

clone_llvm7() {
  mkdir -p "${BUILD_ROOT}"
  if [[ ! -d "${LLVM7_SRC}/llvm" ]]; then
    git clone --depth 1 --branch "${LLVM7_TAG}" \
      https://github.com/llvm/llvm-project.git "${LLVM7_SRC}"
  fi
  "${PYTHON}" - <<'PY' "${LLVM7_SRC}/llvm/CMakeLists.txt"
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
text = text.replace("cmake_policy(SET CMP0051 OLD)", "cmake_policy(SET CMP0051 NEW)")
path.write_text(text)
PY
}

build_llvm7() {
  mkdir -p "${LLVM7_BUILD}" "${LLVM7_INSTALL}"
  cmake -G Ninja \
    -S "$(cmake_path "${LLVM7_SRC}/llvm")" \
    -B "$(cmake_path "${LLVM7_BUILD}")" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$(cmake_path "${LLVM7_INSTALL}")" \
    -DCMAKE_C_COMPILER=cl \
    -DCMAKE_CXX_COMPILER=cl \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DLLVM_TARGETS_TO_BUILD=NVPTX \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLVM_ENABLE_PIC=ON \
    -DLLVM_BUILD_TOOLS=OFF \
    -DLLVM_BUILD_UTILS=OFF \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_DOCS=OFF \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_ENABLE_ZSTD=OFF
  cmake --build "$(cmake_path "${LLVM7_BUILD}")" --target install -j "${PARALLEL}"
}

clone_modern_llvm() {
  mkdir -p "${BUILD_ROOT}"
  if [[ ! -d "${LLVM_MODERN_SRC}/llvm" ]]; then
    git clone --depth 1 https://github.com/llvm/llvm-project.git "${LLVM_MODERN_SRC}"
  fi
  git -C "${LLVM_MODERN_SRC}" fetch --depth 1 origin "${LLVM_MODERN_COMMIT}"
  git -C "${LLVM_MODERN_SRC}" checkout "${LLVM_MODERN_COMMIT}"
}

build_modern_llvm() {
  mkdir -p "${LLVM_MODERN_BUILD}" "${LLVM_MODERN_INSTALL}"
  cmake -G Ninja \
    -S "$(cmake_path "${LLVM_MODERN_SRC}/llvm")" \
    -B "$(cmake_path "${LLVM_MODERN_BUILD}")" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$(cmake_path "${LLVM_MODERN_INSTALL}")" \
    -DCMAKE_C_COMPILER=cl \
    -DCMAKE_CXX_COMPILER=cl \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD=NVPTX \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLVM_ENABLE_PIC=ON \
    -DLLVM_BUILD_TOOLS=OFF \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_DOCS=OFF \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_ENABLE_ZSTD=OFF \
    -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DMLIR_PYTHON_PACKAGE_PREFIX="numba_cuda_mlir._mlir" \
    -DMLIR_BINDINGS_PYTHON_INSTALL_PREFIX="python_packages/numba_cuda_mlir_mlir/numba_cuda_mlir/_mlir" \
    -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=numba_cuda_mlir \
    -DMLIR_PYTHON_STUBGEN_ENABLED=OFF \
    -DPython3_EXECUTABLE="$("${PYTHON}" -c 'import sys; print(sys.executable)')"
  cmake --build "$(cmake_path "${LLVM_MODERN_BUILD}")" -j "${PARALLEL}"
  cmake --install "$(cmake_path "${LLVM_MODERN_BUILD}")"
  local capi_import_lib="${LLVM_MODERN_BUILD}/tools/mlir/python_packages/numba_cuda_mlir_mlir/numba_cuda_mlir/_mlir/_mlir_libs/MLIRPythonCAPI.lib"
  if [[ -f "${capi_import_lib}" ]]; then
    cp "${capi_import_lib}" "${LLVM_MODERN_INSTALL}/lib/MLIRPythonCAPI.lib"
  fi
}

step "Validate Windows build prerequisites" check_prereqs

if [[ "${MODE}" == "all" || "${MODE}" == "llvm7" ]]; then
  step "Clone LLVM 7 (${LLVM7_TAG})" clone_llvm7
  step "Build LLVM 7 static install" build_llvm7
fi

if [[ "${MODE}" == "all" || "${MODE}" == "modern" ]]; then
  step "Clone modern LLVM (${LLVM_MODERN_COMMIT})" clone_modern_llvm
  step "Build modern LLVM+MLIR static install" build_modern_llvm
fi

echo "[$(timestamp)] === Build complete ==="
