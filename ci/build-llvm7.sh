#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build LLVM 7 for the LLVM70 path: shared libLLVM-7.so on Linux, LLVM-C.dll on
# Windows (Git Bash with MSVC on PATH).
#
# Note: on Linux, does NOT cmake --install; the .so is used directly from the
# build dir.
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
# Output dir — the .so is copied here for artifact passing
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
    -DLLVM_TARGETS_TO_BUILD="NVPTX"
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
        -DCMAKE_INSTALL_PREFIX="$(cmake_path "${LLVM7_INSTALL}")"
        -DCMAKE_C_COMPILER=cl
        -DCMAKE_CXX_COMPILER=cl
        -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
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

resolve_msvc_linker() {
    local cmake_cache="$1"
    local link_tool=""
    if [[ -f "${cmake_cache}" ]]; then
        link_tool="$(sed -n 's#^CMAKE_LINKER:FILEPATH=##p' "${cmake_cache}" | head -n 1)"
    fi
    if [[ -n "${link_tool}" ]]; then
        link_tool="$(cygpath -u "${link_tool}")"
    fi
    if [[ -z "${link_tool}" || ! -f "${link_tool}" ]]; then
        if command -v link.exe >/dev/null 2>&1; then
            link_tool="$(command -v link.exe)"
        elif command -v where.exe >/dev/null 2>&1; then
            link_tool="$(where.exe link.exe 2>/dev/null | tr -d '\r' | head -n 1)"
            if [[ -n "${link_tool}" ]]; then
                link_tool="$(cygpath -u "${link_tool}")"
            fi
        fi
    fi
    if [[ -z "${link_tool}" || ! -f "${link_tool}" ]]; then
        echo "ERROR: unable to resolve MSVC link.exe (CMAKE_LINKER/PATH/where.exe)" >&2
        exit 1
    fi
    if ! ("${link_tool}" /? 2>&1 || true) | grep -iq 'Incremental Linker'; then
        echo "ERROR: resolved linker is not MSVC link.exe: ${link_tool}" >&2
        exit 1
    fi
    printf '%s\n' "${link_tool}"
}

resolve_dumpbin_tool() {
    local link_tool="$1"
    local dumpbin_tool=""
    local link_dir
    link_dir="$(dirname "${link_tool}")"
    if [[ -f "${link_dir}/dumpbin.exe" ]]; then
        dumpbin_tool="${link_dir}/dumpbin.exe"
    elif command -v dumpbin.exe >/dev/null 2>&1; then
        dumpbin_tool="$(command -v dumpbin.exe)"
    elif command -v where.exe >/dev/null 2>&1; then
        dumpbin_tool="$(where.exe dumpbin.exe 2>/dev/null | tr -d '\r' | head -n 1)"
        if [[ -n "${dumpbin_tool}" ]]; then
            dumpbin_tool="$(cygpath -u "${dumpbin_tool}")"
        fi
    fi
    if [[ -z "${dumpbin_tool}" || ! -f "${dumpbin_tool}" ]]; then
        echo "ERROR: unable to resolve dumpbin.exe (alongside link.exe/PATH/where.exe)" >&2
        exit 1
    fi
    if ! ("${dumpbin_tool}" /? 2>&1 || true) | grep -iq 'COFF/PE Dumper'; then
        echo "ERROR: resolved dumpbin tool is not MSVC dumpbin.exe: ${dumpbin_tool}" >&2
        exit 1
    fi
    printf '%s\n' "${dumpbin_tool}"
}

# Windows-only: link the installed static LLVM libs into LLVM-C.dll, exporting
# the LLVM C API.
link_llvm_c_dll() {
    local llvm7_lib_dir="${LLVM7_INSTALL}/lib"
    local llvm7_bin_dir="${LLVM7_INSTALL}/bin"
    local exports_script="${SCRIPT_DIR}/tools/gen-llvm-c-exports.py"
    local libs_rsp="${LLVM7_BUILD}/llvm-c-libs.rsp"
    local exports_file="${LLVM7_BUILD}/LLVM-C.exports"
    local def_file="${LLVM7_BUILD}/LLVM-C.def"
    local link_rsp="${LLVM7_BUILD}/llvm-c-link.rsp"
    local llvm_c_dll="${llvm7_bin_dir}/LLVM-C.dll"
    local llvm_c_import_lib="${llvm7_lib_dir}/LLVM-C.lib"
    local cmake_cache="${LLVM7_BUILD}/CMakeCache.txt"
    local link_tool
    local dumpbin_tool
    link_tool="$(resolve_msvc_linker "${cmake_cache}")"
    dumpbin_tool="$(resolve_dumpbin_tool "${link_tool}")"

    local llvm_libs=()
    mapfile -t llvm_libs < <(
        find "${llvm7_lib_dir}" -maxdepth 1 -type f -name 'LLVM*.lib' ! -name 'LLVM-C.lib' ! -name 'LLVM.lib' | sort
    )
    if [[ "${#llvm_libs[@]}" -eq 0 ]]; then
        echo "ERROR: no LLVM static/import libs found under ${llvm7_lib_dir}" >&2
        exit 1
    fi
    : > "${libs_rsp}"
    for lib_path in "${llvm_libs[@]}"; do
        printf '%s\n' "$(cmake_path "${lib_path}")" >> "${libs_rsp}"
    done

    "${PYTHON}" "${exports_script}" \
        --dumpbin "${dumpbin_tool}" \
        --libsfile "${libs_rsp}" \
        --output "${exports_file}" \
        --deffile "${def_file}" \
        --dll-name LLVM-C
    if ! grep -qx 'LLVMContextCreate' "${exports_file}"; then
        echo "ERROR: generated export list is missing LLVMContextCreate: ${exports_file}" >&2
        exit 1
    fi

    mkdir -p "${llvm7_bin_dir}"

    local stub_obj="${LLVM7_BUILD}/llvm-c-stub.obj"
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
        printf '/OUT:%s\n' "$(cmake_path "${llvm_c_dll}")"
        printf '/IMPLIB:%s\n' "$(cmake_path "${llvm_c_import_lib}")"
        printf '/DEF:%s\n' "$(cmake_path "${def_file}")"
        printf '/INCLUDE:LLVMContextCreate\n'
        printf '%s\n' "$(cmake_path "${stub_obj}")"
        while IFS= read -r lib_path; do
            [[ -z "${lib_path}" ]] && continue
            printf '/WHOLEARCHIVE:%s\n' "${lib_path}"
        done < "${libs_rsp}"
    } > "${link_rsp}"

    "${link_tool}" @"$(cmake_path "${link_rsp}")"

    if [[ ! -f "${llvm_c_dll}" ]]; then
        echo "ERROR: failed to produce ${llvm_c_dll}" >&2
        exit 1
    fi

    local dumpbin_rsp="${LLVM7_BUILD}/dumpbin-exports.rsp"
    printf '/NOLOGO\n/EXPORTS\n%s\n' "$(cmake_path "${llvm_c_dll}")" > "${dumpbin_rsp}"
    if ! "${dumpbin_tool}" @"$(cmake_path "${dumpbin_rsp}")" | grep -q 'LLVMContextCreate'; then
        echo "ERROR: ${llvm_c_dll} does not export LLVMContextCreate" >&2
        echo "DLL size: $(ls -la "${llvm_c_dll}" 2>&1)"
        "${dumpbin_tool}" @"$(cmake_path "${dumpbin_rsp}")" 2>&1 | head -40
        exit 1
    fi
}

if [[ "${IS_WINDOWS}" == 1 ]]; then
    cmake --build "$(cmake_path "${LLVM7_BUILD}")" --target install -j "${PARALLEL}"
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
