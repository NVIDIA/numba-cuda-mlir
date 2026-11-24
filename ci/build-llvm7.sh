#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build LLVM 7 for the LLVM70 path: libLLVM-7.so on Linux, LLVM-C.dll on
# Windows (Git Bash with MSVC on PATH), exporting only the LLVM-C functions
# listed in cext/mlir-llvm70/llvm7-exports.txt.
#
# Note: does NOT cmake --install; the shared library is linked into
# $LLVM7_INSTALL (Linux, lib/; Windows, bin/) from the build dir's static libs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
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

# Libraries defining the exported functions (Core, BitWriter) and what they
# link (llvm/lib/*/LLVMBuild.txt). The link fails if one is missing.
LLVM7_LIBS=(
    LLVMCore LLVMBitWriter LLVMAnalysis LLVMObject LLVMProfileData LLVMMC
    LLVMMCParser LLVMBitReader LLVMBinaryFormat LLVMSupport LLVMDemangle
)
mapfile -t LLVM7_EXPORTS < <(sed -e 's/\r$//' -e '/^#/d' -e '/^$/d' \
    "${REPO_ROOT}/cext/mlir-llvm70/llvm7-exports.txt")

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
    -DLLVM_ENABLE_ZLIB=OFF
)
if [[ "${IS_WINDOWS}" == 1 ]]; then
    # MSVC cannot build libLLVM as a DLL: build static libs (static CRT) and
    # link LLVM-C.dll from them below. LLVM 7 predates CMAKE_MSVC_RUNTIME_LIBRARY
    # support, so LLVM_USE_CRT_RELEASE is what selects /MT.
    cmake_args+=(
        -DCMAKE_C_COMPILER=cl
        -DCMAKE_CXX_COMPILER=cl
        -DLLVM_USE_CRT_RELEASE=MT
        -DLLVM_ENABLE_DIA_SDK=OFF
    )
fi
cmake "${cmake_args[@]}"

# Windows: link the static LLVM libs into LLVM-C.dll.
link_llvm_c_dll() {
    local def_file="${LLVM7_BUILD}/LLVM-C.def"
    local link_rsp="${LLVM7_BUILD}/llvm-c-link.rsp"
    # Git Bash's /usr/bin/link.exe (coreutils) shadows MSVC's; use CMake's.
    local link_tool
    link_tool="$(sed -n 's#^CMAKE_LINKER:FILEPATH=##p' "${LLVM7_BUILD}/CMakeCache.txt")"

    { printf 'LIBRARY LLVM-C\nEXPORTS\n'; printf '%s\n' "${LLVM7_EXPORTS[@]}"; } > "${def_file}"

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
        for lib in "${LLVM7_LIBS[@]}"; do
            printf '"%s"\n' "$(cmake_path "${LLVM7_BUILD}/lib/${lib}.lib")"
        done
    } > "${link_rsp}"
    mkdir -p "${LLVM7_INSTALL}/bin"
    "$(cygpath -u "${link_tool}")" @"$(cmake_path "${link_rsp}")"
}

# Linux: link the static LLVM libs into libLLVM-7.so, dropping unused code.
link_libllvm7_so() {
    local version_script="${LLVM7_BUILD}/libLLVM-7.map"
    local cxx
    cxx="$(sed -n 's#^CMAKE_CXX_COMPILER:[A-Z]*=##p' "${LLVM7_BUILD}/CMakeCache.txt")"

    { printf '{\n  global:\n'; printf '    %s;\n' "${LLVM7_EXPORTS[@]}"; printf '  local: *;\n};\n'; } \
        > "${version_script}"

    local link_args=()
    for sym in "${LLVM7_EXPORTS[@]}"; do
        link_args+=("-Wl,--undefined=${sym}")
    done
    link_args+=(-Wl,--start-group)
    for lib in "${LLVM7_LIBS[@]}"; do
        link_args+=("${LLVM7_BUILD}/lib/lib${lib}.a")
    done
    link_args+=(-Wl,--end-group)

    "${cxx}" -shared -o "${LLVM7_INSTALL}/lib/libLLVM-7.so" \
        -Wl,-soname,libLLVM-7.1.so \
        -Wl,--version-script="${version_script}" -Wl,--no-undefined-version \
        -Wl,--no-undefined -Wl,--gc-sections \
        "${link_args[@]}" -pthread -ldl
    strip --strip-unneeded "${LLVM7_INSTALL}/lib/libLLVM-7.so"
}

# Build the static libs (not install).
cmake --build "$(cmake_path "${LLVM7_BUILD}")" -j "${PARALLEL}" --target "${LLVM7_LIBS[@]}"
if [[ "${IS_WINDOWS}" == 1 ]]; then
    link_llvm_c_dll
    echo "=== LLVM 7 built ==="
    ls -lh "${LLVM7_INSTALL}/bin/LLVM-C.dll"
else
    link_libllvm7_so
    echo "=== LLVM 7 built ==="
    ls -lh "${LLVM7_INSTALL}/lib/libLLVM-7.so"
fi

echo "=== sccache stats ==="
sccache --show-stats
