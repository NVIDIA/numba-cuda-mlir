#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build LLVM 7 for the LLVM70 path: shared libLLVM-7.so on Linux, LLVM-C.dll on
# Windows (Git Bash with MSVC on PATH).
#
# Note: does NOT cmake --install; the shared library is copied (Linux, lib/) or
# linked (Windows, bin/) into $LLVM7_INSTALL from the build dir.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/llvm-version.env"

case "$(uname -s)" in
    MINGW*|MSYS*) IS_WINDOWS=1 ;;
    *) IS_WINDOWS=0 ;;
esac

LLVM_TAG="${LLVM7_TAG}"
BUILD_ROOT="${BUILD_ROOT:-${PWD}/_build}"
LLVM7_SRC="${LLVM7_SRC:-${BUILD_ROOT}/llvm7-src}"
LLVM7_BUILD="${LLVM7_BUILD:-${BUILD_ROOT}/llvm7-build}"
# Output dir — the shared library lands here for artifact passing
LLVM7_INSTALL="${LLVM7_INSTALL:-${PWD}/llvm7-install}"
PARALLEL="${PARALLEL:-$(nproc)}"
# Windows venvs have no python3.exe.
PYTHON="${PYTHON:-python}"

if [[ "${IS_WINDOWS}" == 1 ]] && [[ "$(uname -m)" == "arm64" || "$(uname -m)" == "aarch64" ]]; then
    echo "LLVM 7 is not supported on Windows ARM64" >&2
    exit 2
fi

# CMake and MSVC tools on Windows need native paths (C:/...).
cmake_path() {
    if [[ "${IS_WINDOWS}" == 1 ]]; then
        cygpath -m "$1"
    else
        printf '%s\n' "$1"
    fi
}

echo "=== Building LLVM 7 (${LLVM_TAG}) ==="
echo "  Source: ${LLVM7_SRC}"
echo "  Build:  ${LLVM7_BUILD}"

mkdir -p "${BUILD_ROOT}" "${LLVM7_INSTALL}/lib"

# Download source if not present: only the tagged commit, and only llvm/.
if [ ! -d "${LLVM7_SRC}/llvm" ]; then
    echo ">>> Fetching LLVM 7 (${LLVM_TAG})"
    git init -q "${LLVM7_SRC}"
    git -C "${LLVM7_SRC}" remote add origin https://github.com/llvm/llvm-project.git
    git -C "${LLVM7_SRC}" sparse-checkout set llvm
    git -C "${LLVM7_SRC}" fetch --depth 1 --filter=blob:none origin "refs/tags/${LLVM_TAG}"
    git -C "${LLVM7_SRC}" checkout -q FETCH_HEAD
fi

# Patch CMP0051 OLD → NEW for modern CMake compatibility
sed -i 's/cmake_policy(SET CMP0051 OLD)/cmake_policy(SET CMP0051 NEW)/' \
    "${LLVM7_SRC}/llvm/CMakeLists.txt"

# Require sccache for compiler caching
command -v sccache &>/dev/null || { echo "ERROR: sccache not found"; exit 1; }
CMAKE_C_COMPILER_LAUNCHER="$(cmake_path "$(command -v sccache)")"
CMAKE_CXX_COMPILER_LAUNCHER="${CMAKE_C_COMPILER_LAUNCHER}"
export CMAKE_C_COMPILER_LAUNCHER CMAKE_CXX_COMPILER_LAUNCHER

# Configure
cmake_args=(
    -G Ninja
    -S "$(cmake_path "${LLVM7_SRC}/llvm")"
    -B "$(cmake_path "${LLVM7_BUILD}")"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    # No backends: this LLVM only builds IR and writes bitcode for libNVVM.
    -DLLVM_TARGETS_TO_BUILD=""
    -DLLVM_BUILD_TOOLS=OFF
    -DLLVM_BUILD_UTILS=OFF
    -DLLVM_BUILD_EXAMPLES=OFF
    -DLLVM_INCLUDE_TESTS=OFF
    -DLLVM_INCLUDE_BENCHMARKS=OFF
    -DLLVM_INCLUDE_DOCS=OFF
    -DLLVM_ENABLE_TERMINFO=OFF
)
if [[ "${IS_WINDOWS}" == 1 ]]; then
    # MSVC cannot build libLLVM as a DLL: build static libs (static CRT) and
    # link LLVM-C.dll from them below. LLVM 7 predates CMAKE_MSVC_RUNTIME_LIBRARY
    # support, so LLVM_USE_CRT_RELEASE is what selects /MT.
    cmake_args+=(
        -DCMAKE_C_COMPILER=cl
        -DCMAKE_CXX_COMPILER=cl
        -DLLVM_USE_CRT_RELEASE=MT
        -DLLVM_ENABLE_ZLIB=OFF
        -DLLVM_ENABLE_DIA_SDK=OFF
    )
else
    cmake_args+=(
        -DLLVM_BUILD_LLVM_DYLIB=ON
        -DLLVM_ENABLE_ZLIB=ON
    )
fi
cmake "${cmake_args[@]}"

# Windows-only: link the static LLVM libs into LLVM-C.dll, exporting the
# LLVM C API.
link_llvm_c_dll() {
    local lib_dir="${LLVM7_BUILD}/lib"
    local def_file="${LLVM7_BUILD}/LLVM-C.def"
    local link_rsp="${LLVM7_BUILD}/llvm-c-link.rsp"
    local llvm_libs=("${lib_dir}"/LLVM*.lib)
    # Git Bash's /usr/bin/link.exe (coreutils) shadows MSVC's; use CMake's.
    local link_tool
    link_tool="$(sed -n 's#^CMAKE_LINKER:FILEPATH=##p' "${LLVM7_BUILD}/CMakeCache.txt")"

    local lib_args=()
    for lib_path in "${llvm_libs[@]}"; do
        lib_args+=("$(cmake_path "${lib_path}")")
    done
    "${PYTHON}" "$(cmake_path "${SCRIPT_DIR}/tools/gen-llvm-c-exports.py")" \
        --def "$(cmake_path "${def_file}")" "${lib_args[@]}"

    pushd "${LLVM7_BUILD}" > /dev/null
    printf 'int _fltused = 0;\n' > llvm-c-stub.c
    # /MT here pairs with the /MT-built LLVM static libs linked below to
    # produce a single LLVM-C.dll. Within one DLL/binary every .obj and
    # .lib must agree on /MT vs /MD or the linker errors.
    cl -nologo -c -O2 -MT -Follvm-c-stub.obj llvm-c-stub.c
    popd > /dev/null

    {
        printf '/NOLOGO\n'
        printf '/DLL\n'
        printf '/MACHINE:X64\n'
        printf '/OUT:"%s"\n' "$(cmake_path "${LLVM7_INSTALL}/bin/LLVM-C.dll")"
        printf '/IMPLIB:"%s"\n' "$(cmake_path "${LLVM7_BUILD}/LLVM-C.lib")"
        printf '/DEF:"%s"\n' "$(cmake_path "${def_file}")"
        printf '"%s"\n' "$(cmake_path "${LLVM7_BUILD}/llvm-c-stub.obj")"
        for lib_path in "${lib_args[@]}"; do
            printf '/WHOLEARCHIVE:"%s"\n' "${lib_path}"
        done
    } > "${link_rsp}"
    mkdir -p "${LLVM7_INSTALL}/bin"
    "$(cygpath -u "${link_tool}")" @"$(cmake_path "${link_rsp}")"
}

if [[ "${IS_WINDOWS}" == 1 ]]; then
    # Build the static libs (not install).
    cmake --build "$(cmake_path "${LLVM7_BUILD}")" -j "${PARALLEL}"
    link_llvm_c_dll

    echo "=== LLVM 7 built ==="
    ls -lh "${LLVM7_INSTALL}/bin/LLVM-C.dll"
else
    # Build only the LLVM shared lib (not install).
    cmake --build "${LLVM7_BUILD}" -j "${PARALLEL}" --target LLVM

    # Asymmetric with ci/build-llvm-modern.sh which uses
    # `cmake --install --strip`: LLVM 7's tools/llvm-shlib creates extra
    # compatibility symlinks (libLLVM-7.so, libLLVM.so -> libLLVM-7.1.so)
    # regardless of CMAKE_PLATFORM_NO_VERSIONED_SONAME, and
    # `actions/upload-artifact`'s zip format materializes each symlink into
    # a full byte-identical copy -- 3x-bloating the artifact. The narrow
    # `cp` + manual `strip` avoids that entirely.
    LLVM7_SO="$(ls "${LLVM7_BUILD}"/lib/libLLVM-7*.so | head -1)"
    strip --strip-unneeded "${LLVM7_SO}"
    cp "${LLVM7_SO}" "${LLVM7_INSTALL}/lib/libLLVM-7.so"

    echo "=== LLVM 7 built ==="
    ls -lh "${LLVM7_INSTALL}/lib/libLLVM-7.so"
fi

echo "=== sccache stats ==="
sccache --show-stats
