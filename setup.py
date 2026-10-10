# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import platform
import shutil
import sys
import sysconfig
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension

ROOT = Path(__file__).resolve().parent
IS_WINDOWS = os.name == "nt"
IS_WINDOWS_ARM64 = IS_WINDOWS and platform.machine() == "ARM64"


def _shared_lib_name(name: str) -> str:
    return f"{name}.dll" if IS_WINDOWS else f"lib{name}.so"


def _find_mlir_python_capi() -> str | None:
    """Find the MLIRPythonCAPI link library, preferring the MLIR_DIR install.

    When MLIR_DIR is set (pointing to <install>/lib/cmake/mlir), look for
    the library under <install>/python_packages/cuda_mlir_core/cudamlir/_mlir_libs/
    (our custom install prefix). This ensures libMLIRToLLVM70.so links against
    the same libMLIRPythonCAPI.so that will be loaded at runtime.
    """
    capi_names = ["MLIRPythonCAPI.lib"] if IS_WINDOWS else ["libMLIRPythonCAPI.so"]
    mlir_dir = os.environ.get("MLIR_DIR")
    if mlir_dir:
        install_root = Path(mlir_dir).resolve().parent.parent.parent
        if IS_WINDOWS:
            capi = install_root / "lib" / "MLIRPythonCAPI.lib"
            if capi.exists():
                return str(capi)
        mlir_libs = install_root / "python_packages" / "cuda_mlir_core" / "cudamlir" / "_mlir_libs"
        for capi_name in capi_names:
            capi = mlir_libs / capi_name
            if capi.exists():
                return str(capi)

    sp = Path(sysconfig.get_path("platlib"))
    mlir_libs = sp / "cudamlir" / "_mlir_libs"
    for capi_name in capi_names:
        capi = mlir_libs / capi_name
        if capi.exists():
            return str(capi)
    return None


class BuildExtWithCmake(build_ext):
    def run(self):
        build_dir = os.getenv("NUMBA_CUDA_MLIR_BUILD_DIR")
        if build_dir is None or build_dir == "":
            if self.editable_mode:
                build_dir = ROOT / "build"
            else:
                build_dir = Path(self.build_temp)
        build_dir = Path(build_dir)
        if not self.editable_mode and build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)
        print(f"Build directory: {build_dir}")
        build_type = "Debug" if self.debug else "Release"
        cmake_cmd = ["cmake"]
        if IS_WINDOWS:
            cmake_cmd += ["-G", "Ninja"]
        cmake_cmd += ["-B", build_dir, ROOT, f"-DCMAKE_BUILD_TYPE={build_type}"]
        py_gil_disabled = sysconfig.get_config_var("Py_GIL_DISABLED") in (1, "1")
        cmake_cmd.append(f"-DNUMBA_CUDA_MLIR_PY_GIL_DISABLED={'ON' if py_gil_disabled else 'OFF'}")
        if IS_WINDOWS:
            # Static-link the MSVC C runtime (/MT) so each resulting DLL
            # embeds its own CRT copy and has no external msvcp140 /
            # vcruntime140 runtime dependency. Across DLLs the embedded
            # CRTs are independent (different heaps, etc.); that's fine
            # here because the cross-DLL ABI is C only -- MLIRPythonCAPI
            # exposes a C API, and our cext modules link against its
            # import lib and call C functions only. The per-binary
            # constraint that does apply is that every .obj and statically-
            # linked .lib feeding *this* cmake build must also use /MT;
            # ci/build-windows.sh sets the same flag for the LLVM static
            # libs and MLIRPythonCAPI, so MLIRToLLVM70 here can link them.
            cmake_cmd.append("-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded")
        python_root = Path(sys.executable).resolve().parent
        cmake_cmd += [
            f"-DPython_ROOT_DIR={python_root}",
            f"-DPython_EXECUTABLE={sys.executable}",
            "-DPython_FIND_REGISTRY=NEVER",
            f"-DPython3_ROOT_DIR={python_root}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DPython3_FIND_REGISTRY=NEVER",
        ]
        for launcher_var in ("CMAKE_C_COMPILER_LAUNCHER", "CMAKE_CXX_COMPILER_LAUNCHER"):
            launcher = os.environ.get(launcher_var)
            if launcher:
                cmake_cmd.append(f"-D{launcher_var}={launcher}")
        mlir_dir = os.environ.get("MLIR_DIR")
        if mlir_dir:
            cmake_cmd.append(f"-DMLIR_DIR={mlir_dir}")
            if not IS_WINDOWS_ARM64:
                cmake_cmd.append("-DBUILD_LLVM70=ON")
                capi = _find_mlir_python_capi()
                if capi:
                    cmake_cmd.append(f"-DLLVM70_MLIR_PYTHON_CAPI={capi}")
                else:
                    print(
                        "WARNING: MLIR_DIR is set but the MLIRPythonCAPI link library "
                        "was not found; MLIRToLLVM70 will not link against MLIRPythonCAPI."
                    )
        self.spawn(cmake_cmd)
        parallel = 1 if self.parallel is None else self.parallel
        self.spawn(["cmake", "--build", build_dir, "-j", str(parallel)])

        # TODO: ideally, we should "make install" the library somewhere, so that CMake removes
        #   any build RPATHs etc. But I'll leave that for another day.

        for ext in self.extensions:
            src_dir = _get_csrc_dir(ext.name)
            ext_build_path = build_dir / src_dir / _get_build_lib_filename(ext.name)
            ext_path = Path(self.get_ext_fullpath(ext.name))
            # Create a symlink to the build directory if in editable mode, otherwise copy
            if not self.dry_run:
                ext_path.parent.mkdir(parents=True, exist_ok=True)
                if ext_path.exists() or ext_path.is_symlink():
                    ext_path.unlink()
                if self.editable_mode:
                    ext_path.symlink_to(ext_build_path)
                else:
                    shutil.copy2(ext_build_path, ext_path)

        llvm70_capi = build_dir / "cext" / "mlir-llvm70" / "lib" / _shared_lib_name("MLIRToLLVM70")

        if llvm70_capi.exists():
            # The bridge used to be staged into the vendored MLIR package, next
            # to the MLIRPythonCAPI it links. That directory belongs to a
            # separately installed package and is not ours to write into, so it
            # lives here instead; it still resolves the CAPI by soname against
            # the copy numba_cuda_mlir/__init__.py has already loaded.
            pkg = (
                Path(self.get_ext_fullpath("numba_cuda_mlir._cext")).parent
                if self.editable_mode
                else Path(self.build_lib) / "numba_cuda_mlir"
            )
            mlir_libs_dir = pkg / "lib"
            mlir_libs_dir.mkdir(parents=True, exist_ok=True)
            if not self.dry_run:
                mlir_dest = mlir_libs_dir / llvm70_capi.name
                if mlir_dest.exists() or mlir_dest.is_symlink():
                    mlir_dest.unlink()
                print(f"Staging {llvm70_capi.name}: {llvm70_capi} -> {mlir_dest}")
                if self.editable_mode:
                    mlir_dest.symlink_to(llvm70_capi)
                else:
                    shutil.copy2(llvm70_capi, mlir_dest)
        if not IS_WINDOWS_ARM64:
            self._stage_libllvm7()

    def _stage_libllvm7(self):
        """Copy the optional LLVM 7 runtime library into the wheel."""
        libllvm7 = os.environ.get("LIBLLVM7")
        if not libllvm7:
            return
        libllvm7 = Path(libllvm7)
        if not libllvm7.exists():
            print(f"WARNING: LIBLLVM7 not found at {libllvm7}")
            return
        pkg = (
            Path(self.get_ext_fullpath("numba_cuda_mlir._cext")).parent
            if self.editable_mode
            else Path(self.build_lib) / "numba_cuda_mlir"
        )
        dest_dir = pkg / "lib"
        dest_dir.mkdir(parents=True, exist_ok=True)
        if IS_WINDOWS:
            # Preserve the incoming DLL basename (e.g. LLVM-C.dll).
            dest_name = libllvm7.name
        else:
            dest_name = "libLLVM-7.so"
        dest = dest_dir / dest_name
        print(f"Staging {dest_name}: {libllvm7} -> {dest}")
        if self.editable_mode:
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(libllvm7.resolve())
        else:
            shutil.copy2(str(libllvm7), str(dest))


def _get_csrc_dir(ext_name: str):
    prefix = "numba_cuda_mlir._"
    assert ext_name.startswith(prefix)
    name = ext_name[len(prefix) :]
    # The `_cext` module lives in cext/launcher/; other modules use their own name.
    subdir = "launcher" if name == "cext" else name
    return f"cext/{subdir}"


def _get_build_lib_filename(ext_name: str):
    name = ext_name.split(".")[-1]
    return _shared_lib_name(name)


VERSION = os.getenv("NUMBA_CUDA_MLIR_VERSION")
if VERSION is None:
    version_file = ROOT / "src" / "numba_cuda_mlir" / "VERSION"
    if not version_file.exists():
        raise RuntimeError(
            f"Version file {version_file} does not exist and NUMBA_CUDA_MLIR_VERSION is not set in environment"
        )
    VERSION = version_file.read_text().strip()

setup(
    version=VERSION,
    ext_modules=[
        Extension("numba_cuda_mlir._cext", []),
        Extension("numba_cuda_mlir._typeconv", []),
        Extension("numba_cuda_mlir._mviewbuf", []),
        Extension("numba_cuda_mlir._helperlib", []),
    ],
    cmdclass=dict(
        build_ext=BuildExtWithCmake,
    ),
)
