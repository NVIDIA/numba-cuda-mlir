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
)
if [[ "${IS_WINDOWS}" == 1 ]]; then
    # Static CRT (/MT), so the wheel needs no MSVC runtime DLLs.
    cmake_args+=(
        -DCMAKE_C_COMPILER=cl
        -DCMAKE_CXX_COMPILER=cl
        -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
        -DLLVM_PARALLEL_LINK_JOBS="${LLVM_PARALLEL_LINK_JOBS:-${PARALLEL}}"
        -DLLVM_ENABLE_ZLIB=OFF
        -DLLVM_ENABLE_ZSTD=OFF
    )
else
    cmake_args+=(-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON)
fi

cmake "${cmake_args[@]}"

# Windows-only: stage the MLIR Python runtime DLLs/.pyd that `cmake --install`
# leaves in the build tree, then build the modern-to-NVVM bridge and smoke-import
# the result.
stage_windows_mlir_python() {
    local mlir_pkg_rel="python_packages/numba_cuda_mlir_mlir/numba_cuda_mlir/_mlir"
    local build_mlir_pkg="${LLVM_BUILD}/tools/mlir/${mlir_pkg_rel}"
    local install_mlir_pkg="${LLVM_INSTALL}/${mlir_pkg_rel}"
    local install_mlir_libs="${install_mlir_pkg}/_mlir_libs"
    if [[ -d "${build_mlir_pkg}" ]]; then
        mkdir -p "${install_mlir_pkg}"
        cp -a "${build_mlir_pkg}/." "${install_mlir_pkg}/"
    else
        echo "ERROR: MLIR Python build package not found at ${build_mlir_pkg}" >&2
        exit 1
    fi

    mkdir -p "${install_mlir_libs}"
    while IFS= read -r -d '' mlir_runtime; do
        cp "${mlir_runtime}" "${install_mlir_libs}/"
    done < <(
        find "${LLVM_BUILD}" -type f \( \
            -name '_mlir*.pyd' -o \
            -name 'MLIRPython*.dll' -o \
            -name 'nanobind*.dll' \
        \) -print0
    )

    local python_ext_suffix
    python_ext_suffix="$(
        "${PYTHON}" - <<'PY'
import sysconfig

print(sysconfig.get_config_var("EXT_SUFFIX") or ".pyd")
PY
    )"
    local core_mlir_extension="${install_mlir_libs}/_mlir${python_ext_suffix}"

    if [[ ! -f "${core_mlir_extension}" ]]; then
        echo "ERROR: MLIR Python native extension for ${python_ext_suffix} was not staged" >&2
        echo "Expected: ${core_mlir_extension}" >&2
        echo "Contents of ${install_mlir_libs}:" >&2
        ls -la "${install_mlir_libs}" >&2 || true
        echo "MLIR native artifacts found under ${LLVM_BUILD}:" >&2
        find "${LLVM_BUILD}" -type f \( \
            -name '_mlir*.pyd' -o \
            -name 'MLIRPython*.dll' -o \
            -name 'nanobind*.dll' \
        \) -print >&2 || true
        exit 1
    fi

    local capi_import_lib="${build_mlir_pkg}/_mlir_libs/MLIRPythonCAPI.lib"
    if [[ -f "${capi_import_lib}" ]]; then
        cp "${capi_import_lib}" "${LLVM_INSTALL}/lib/MLIRPythonCAPI.lib"
    fi

    build_modern_to_nvvm_bridge "${install_mlir_libs}"

    "${PYTHON}" - "$(cmake_path "${LLVM_INSTALL}")" <<'PY'
import ctypes
import os
import pathlib
import sys
import traceback

debug = os.environ.get("NUMBA_CUDA_MLIR_CI_DEBUG") == "1"


def log(message):
    if debug:
        print(message)


def describe_failure():
    print(f"  python={sys.executable}", file=sys.stderr)
    print(f"  install_root={install_root}", file=sys.stderr)
    print(f"  mlir_libs={mlir_libs} exists={mlir_libs.is_dir()}", file=sys.stderr)
    if mlir_libs.is_dir():
        for path in sorted(mlir_libs.iterdir()):
            if path.is_file() and path.suffix.lower() in {".dll", ".pyd", ".lib", ".exp"}:
                print(f"    {path.name} size={path.stat().st_size}", file=sys.stderr)


install_root = pathlib.Path(sys.argv[1])
pkg_root = install_root / "python_packages" / "numba_cuda_mlir_mlir"
mlir_pkg = pkg_root / "numba_cuda_mlir" / "_mlir"
mlir_libs = mlir_pkg / "_mlir_libs"

handles = []
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    for directory in (
        mlir_libs,
        install_root / "lib",
        install_root / "bin",
    ):
        if directory.is_dir():
            log(f"  add_dll_directory={directory}")
            handles.append(os.add_dll_directory(str(directory)))

for name in (
    "nanobind-numba_cuda_mlir.dll",
    "MLIRPythonSupport-numba_cuda_mlir.dll",
    "MLIRModernToNVVM.dll",
    "MLIRPythonCAPI.dll",
):
    path = mlir_libs / name
    if path.exists():
        log(f"  loading {path}")
        ctypes.WinDLL(str(path))

sys.path.insert(0, str(pkg_root))
try:
    from numba_cuda_mlir._mlir import ir  # noqa: F401
except BaseException:
    print("Modern MLIR Python artifact smoke import failed:", file=sys.stderr)
    describe_failure()
    traceback.print_exc()
    raise

log("Modern MLIR Python artifact smoke import passed")
PY
}

build_modern_to_nvvm_bridge() {
    local install_mlir_libs="$1"
    local bridge_build="${LLVM_BUILD}/mlir-modern-to-nvvm-build"
    local bridge_source="${REPO_ROOT}/cext/mlir-modern"

    cmake -G Ninja \
        -S "$(cmake_path "${bridge_source}")" \
        -B "$(cmake_path "${bridge_build}")" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$(cmake_path "${LLVM_INSTALL}")" \
        -DMLIR_DIR="$(cmake_path "${LLVM_INSTALL}/lib/cmake/mlir")" \
        -DLLVM_DIR="$(cmake_path "${LLVM_INSTALL}/lib/cmake/llvm")" \
        -DCMAKE_C_COMPILER=cl \
        -DCMAKE_CXX_COMPILER=cl \
        -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
    cmake --build "$(cmake_path "${bridge_build}")" \
        --target MLIRModernToNVVM MLIRModernToNVVMSmoke -j "${PARALLEL}"
    ctest --test-dir "$(cmake_path "${bridge_build}")" --output-on-failure

    local bridge_dll
    bridge_dll="$(find "${bridge_build}" -type f -name 'MLIRModernToNVVM.dll' -print -quit)"
    if [[ -z "${bridge_dll}" || ! -f "${bridge_dll}" ]]; then
        echo "ERROR: failed to produce MLIRModernToNVVM.dll under ${bridge_build}" >&2
        exit 1
    fi

    cp "${bridge_dll}" "${install_mlir_libs}/"

    local bridge_import_lib
    bridge_import_lib="$(find "${bridge_build}" -type f -name 'MLIRModernToNVVM.lib' -print -quit)"
    if [[ -n "${bridge_import_lib}" && -f "${bridge_import_lib}" ]]; then
        cp "${bridge_import_lib}" "${install_mlir_libs}/"
    fi
}

# Build & install. --strip removes the static symbol table (.symtab +
# .strtab); LLVM is built without -g so there's no DWARF to strip, but
# the symbol table itself accounts for ~70 MB on libMLIRPythonCAPI.so
# alone -- ~25 MB compressed in the wheel.
cmake --build "$(cmake_path "${LLVM_BUILD}")" -j "${PARALLEL}"
if [[ "${IS_WINDOWS}" == 1 ]]; then
    cmake --install "$(cmake_path "${LLVM_BUILD}")"
    stage_windows_mlir_python
else
    cmake --install "${LLVM_BUILD}" --strip
fi

echo "=== Modern LLVM installed to ${LLVM_INSTALL} ==="
echo "  MLIR_DIR=${LLVM_INSTALL}/lib/cmake/mlir"
ls -lh "${LLVM_INSTALL}/python_packages/numba_cuda_mlir_mlir/numba_cuda_mlir/_mlir/_mlir_libs/libMLIRPythonCAPI.so" 2>/dev/null || true

echo "=== sccache stats ==="
sccache --show-stats
