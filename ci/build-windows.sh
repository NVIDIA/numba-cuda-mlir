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

shell_path() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -u "$1"
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
    -DLLVM_ENABLE_ZSTD=OFF \
    -DLLVM_ENABLE_DIA_SDK=OFF
  cmake --build "$(cmake_path "${LLVM7_BUILD}")" --target install -j "${PARALLEL}"

  local llvm7_lib_dir="${LLVM7_INSTALL}/lib"
  local llvm7_bin_dir="${LLVM7_INSTALL}/bin"
  local exports_script="${REPO_ROOT}/ci/tools/gen-llvm-c-exports.py"
  local libs_rsp="${LLVM7_BUILD}/llvm-c-libs.rsp"
  local exports_file="${LLVM7_BUILD}/LLVM-C.exports"
  local def_file="${LLVM7_BUILD}/LLVM-C.def"
  local link_rsp="${LLVM7_BUILD}/llvm-c-link.rsp"
  local llvm_c_dll="${llvm7_bin_dir}/LLVM-C.dll"
  local llvm_c_import_lib="${llvm7_lib_dir}/LLVM-C.lib"
  local cmake_cache="${LLVM7_BUILD}/CMakeCache.txt"
  local link_tool=""
  if [[ -f "${cmake_cache}" ]]; then
    link_tool="$(sed -n 's#^CMAKE_LINKER:FILEPATH=##p' "${cmake_cache}" | head -n 1)"
  fi
  if [[ -n "${link_tool}" ]]; then
    link_tool="$(shell_path "${link_tool}")"
  fi
  if [[ -z "${link_tool}" || ! -f "${link_tool}" ]]; then
    if command -v link.exe >/dev/null 2>&1; then
      link_tool="$(command -v link.exe)"
    elif command -v where.exe >/dev/null 2>&1; then
      link_tool="$(where.exe link.exe 2>/dev/null | tr -d '\r' | head -n 1)"
      if [[ -n "${link_tool}" ]]; then
        link_tool="$(shell_path "${link_tool}")"
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
      dumpbin_tool="$(shell_path "${dumpbin_tool}")"
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
  cl -nologo -c -O2 -MD -Follvm-c-stub.obj llvm-c-stub.c
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

clone_modern_llvm() {
  mkdir -p "${BUILD_ROOT}"
  if [[ ! -d "${LLVM_MODERN_SRC}/llvm" ]]; then
    git clone --depth 1 https://github.com/llvm/llvm-project.git "${LLVM_MODERN_SRC}"
    git -C "${LLVM_MODERN_SRC}" fetch --depth 1 origin "${LLVM_MODERN_COMMIT}"
    git -C "${LLVM_MODERN_SRC}" checkout "${LLVM_MODERN_COMMIT}"
  fi
}

build_modern_llvm() {
  mkdir -p "${LLVM_MODERN_BUILD}" "${LLVM_MODERN_INSTALL}"
  local python_executable
  local python_root
  python_executable="$("${PYTHON}" -c 'import sys; print(sys.executable)')"
  python_root="$(dirname "${python_executable}")"
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
    -DCMAKE_CXX_FLAGS="-DMLIR_PYTHON_PACKAGE_PREFIX=numba_cuda_mlir._mlir. -DMLIR_USE_FALLBACK_TYPE_IDS=1" \
    -DMLIR_BINDINGS_PYTHON_INSTALL_PREFIX="python_packages/numba_cuda_mlir_mlir/numba_cuda_mlir/_mlir" \
    -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=numba_cuda_mlir \
    -DMLIR_PYTHON_STUBGEN_ENABLED=OFF \
    -DPython_ROOT_DIR="${python_root}" \
    -DPython_EXECUTABLE="${python_executable}" \
    -DPython_FIND_REGISTRY=NEVER \
    -DPython3_ROOT_DIR="${python_root}" \
    -DPython3_EXECUTABLE="${python_executable}" \
    -DPython3_FIND_REGISTRY=NEVER
  cmake --build "$(cmake_path "${LLVM_MODERN_BUILD}")" -j "${PARALLEL}"
  cmake --install "$(cmake_path "${LLVM_MODERN_BUILD}")"

  local mlir_pkg_rel="python_packages/numba_cuda_mlir_mlir/numba_cuda_mlir/_mlir"
  local build_mlir_pkg="${LLVM_MODERN_BUILD}/tools/mlir/${mlir_pkg_rel}"
  local install_mlir_pkg="${LLVM_MODERN_INSTALL}/${mlir_pkg_rel}"
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
    find "${LLVM_MODERN_BUILD}" -type f \( \
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
    echo "MLIR native artifacts found under ${LLVM_MODERN_BUILD}:" >&2
    find "${LLVM_MODERN_BUILD}" -type f \( \
      -name '_mlir*.pyd' -o \
      -name 'MLIRPython*.dll' -o \
      -name 'nanobind*.dll' \
    \) -print >&2 || true
    exit 1
  fi

  local capi_import_lib="${build_mlir_pkg}/_mlir_libs/MLIRPythonCAPI.lib"
  if [[ -f "${capi_import_lib}" ]]; then
    cp "${capi_import_lib}" "${LLVM_MODERN_INSTALL}/lib/MLIRPythonCAPI.lib"
  fi

  build_modern_llvm_c_dll "${install_mlir_libs}"

  local smoke_install_root
  smoke_install_root="$(cmake_path "${LLVM_MODERN_INSTALL}")"
  "${PYTHON}" - "${smoke_install_root}" <<'PY'
import ctypes
import os
import pathlib
import sys
import traceback

install_root = pathlib.Path(sys.argv[1])
pkg_root = install_root / "python_packages" / "numba_cuda_mlir_mlir"
mlir_pkg = pkg_root / "numba_cuda_mlir" / "_mlir"
mlir_libs = mlir_pkg / "_mlir_libs"

print("Smoke testing modern MLIR Python artifact")
print(f"  python={sys.executable}")
print(f"  install_root={install_root}")
print(f"  mlir_libs={mlir_libs} exists={mlir_libs.is_dir()}")

handles = []
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    for directory in (
        mlir_libs,
        install_root / "lib",
        install_root / "bin",
    ):
        if directory.is_dir():
            print(f"  add_dll_directory={directory}")
            handles.append(os.add_dll_directory(str(directory)))

for name in (
    "nanobind-numba_cuda_mlir.dll",
    "MLIRPythonSupport-numba_cuda_mlir.dll",
    "LLVM-C-modern.dll",
    "MLIRPythonCAPI.dll",
):
    path = mlir_libs / name
    if path.exists():
        print(f"  loading {path}")
        ctypes.WinDLL(str(path))

sys.path.insert(0, str(pkg_root))
try:
    from numba_cuda_mlir._mlir import ir  # noqa: F401
except BaseException:
    print("Modern MLIR Python artifact smoke import failed:", file=sys.stderr)
    traceback.print_exc()
    raise

print("Modern MLIR Python artifact smoke import passed")
PY
}

build_modern_llvm_c_dll() {
  local install_mlir_libs="$1"
  local modern_lib_dir="${LLVM_MODERN_INSTALL}/lib"
  local exports_script="${REPO_ROOT}/ci/tools/gen-llvm-c-exports.py"
  local libs_rsp="${LLVM_MODERN_BUILD}/llvm-c-modern-libs.rsp"
  local exports_file="${LLVM_MODERN_BUILD}/LLVM-C-modern.exports"
  local def_file="${LLVM_MODERN_BUILD}/LLVM-C-modern.def"
  local link_rsp="${LLVM_MODERN_BUILD}/llvm-c-modern-link.rsp"
  local llvm_c_dll="${install_mlir_libs}/LLVM-C-modern.dll"
  local llvm_c_import_lib="${install_mlir_libs}/LLVM-C-modern.lib"
  local windows_system_libs=(
    winhttp.lib
    crypt32.lib
    psapi.lib
    shell32.lib
    ole32.lib
    uuid.lib
    advapi32.lib
    ws2_32.lib
    ntdll.lib
  )
  local cmake_cache="${LLVM_MODERN_BUILD}/CMakeCache.txt"
  local link_tool=""
  if [[ -f "${cmake_cache}" ]]; then
    link_tool="$(sed -n 's#^CMAKE_LINKER:FILEPATH=##p' "${cmake_cache}" | head -n 1)"
  fi
  if [[ -n "${link_tool}" ]]; then
    link_tool="$(shell_path "${link_tool}")"
  fi
  if [[ -z "${link_tool}" || ! -f "${link_tool}" ]]; then
    if command -v link.exe >/dev/null 2>&1; then
      link_tool="$(command -v link.exe)"
    elif command -v where.exe >/dev/null 2>&1; then
      link_tool="$(where.exe link.exe 2>/dev/null | tr -d '\r' | head -n 1)"
      if [[ -n "${link_tool}" ]]; then
        link_tool="$(shell_path "${link_tool}")"
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
      dumpbin_tool="$(shell_path "${dumpbin_tool}")"
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

  local llvm_libs=()
  mapfile -t llvm_libs < <(
    find "${modern_lib_dir}" -maxdepth 1 -type f -name 'LLVM*.lib' ! -name 'LLVM-C.lib' ! -name 'LLVM-C-modern.lib' ! -name 'LLVM.lib' | sort
  )
  if [[ "${#llvm_libs[@]}" -eq 0 ]]; then
    echo "ERROR: no LLVM static/import libs found under ${modern_lib_dir}" >&2
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
    --dll-name LLVM-C-modern
  if ! grep -qx 'LLVMContextCreate' "${exports_file}"; then
    echo "ERROR: generated export list is missing LLVMContextCreate: ${exports_file}" >&2
    exit 1
  fi

  mkdir -p "${install_mlir_libs}"

  local stub_obj="${LLVM_MODERN_BUILD}/llvm-c-modern-stub.obj"
  pushd "${LLVM_MODERN_BUILD}" > /dev/null
  printf 'int _fltused = 0;\n' > llvm-c-modern-stub.c
  cl -nologo -c -O2 -MD -Follvm-c-modern-stub.obj llvm-c-modern-stub.c
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
    for system_lib in "${windows_system_libs[@]}"; do
      printf '%s\n' "${system_lib}"
    done
  } > "${link_rsp}"

  echo "=== LLVM-C-modern link response file (last 20 lines) ==="
  tail -n 20 "${link_rsp}"

  "${link_tool}" @"$(cmake_path "${link_rsp}")"

  if [[ ! -f "${llvm_c_dll}" ]]; then
    echo "ERROR: failed to produce ${llvm_c_dll}" >&2
    exit 1
  fi

  local dumpbin_rsp="${LLVM_MODERN_BUILD}/dumpbin-modern-exports.rsp"
  printf '/NOLOGO\n/EXPORTS\n%s\n' "$(cmake_path "${llvm_c_dll}")" > "${dumpbin_rsp}"
  if ! "${dumpbin_tool}" @"$(cmake_path "${dumpbin_rsp}")" | grep -q 'LLVMContextCreate'; then
    echo "ERROR: ${llvm_c_dll} does not export LLVMContextCreate" >&2
    echo "DLL size: $(ls -la "${llvm_c_dll}" 2>&1)"
    "${dumpbin_tool}" @"$(cmake_path "${dumpbin_rsp}")" 2>&1 | head -40
    exit 1
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
