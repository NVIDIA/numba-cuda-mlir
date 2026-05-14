# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import shutil

from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IS_WINDOWS = os.name == "nt"


def _shared_lib_name(name: str) -> str:
    return f"{name}.dll" if IS_WINDOWS else f"lib{name}.so"


def _find_mlir_python_capi_link_library() -> str | None:
    """Find the MLIRPythonCAPI link library, preferring the MLIR_DIR install.

    When MLIR_DIR is set (pointing to <install>/lib/cmake/mlir), look for
    the Python bindings under our custom install prefix. On Windows the linker
    needs MLIRPythonCAPI.lib; at runtime MLIRPythonCAPI.dll is loaded from the
    staged _mlir_libs directory.
    """
    capi_names = ["MLIRPythonCAPI.lib"] if IS_WINDOWS else ["libMLIRPythonCAPI.so"]
    mlir_dir = os.environ.get("MLIR_DIR")
    if mlir_dir:
        install_root = Path(mlir_dir).resolve().parent.parent.parent
        for capi_name in capi_names:
            capi = install_root / "lib" / capi_name
            if capi.exists():
                return str(capi)
        mlir_libs = (
            install_root
            / "python_packages"
            / "numba_cuda_mlir_mlir"
            / "numba_cuda_mlir"
            / "_mlir"
            / "_mlir_libs"
        )
        for capi_name in capi_names:
            capi = mlir_libs / capi_name
            if capi.exists():
                return str(capi)

    import sysconfig

    sp = Path(sysconfig.get_path("platlib"))
    mlir_libs = sp / "numba_cuda_mlir" / "_mlir" / "_mlir_libs"
    for capi_name in capi_names:
        capi = mlir_libs / capi_name
        if capi.exists():
            return str(capi)
    return None


def _env_switch(name: str, default: str = "auto") -> str:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default

    value = raw.strip().lower()
    if value == "auto":
        return "auto"
    if value in ("1", "true", "yes", "on"):
        return "on"
    if value in ("0", "false", "no", "off"):
        return "off"
    raise RuntimeError(f"{name} must be one of auto, on, off, 1, 0, true, or false")


def _should_build_llvm70(mlir_dir: str | None) -> bool:
    mode = _env_switch("NUMBA_CUDA_MLIR_BUILD_LLVM70")
    if mode == "auto":
        return bool(mlir_dir)
    if mode == "on":
        if not mlir_dir:
            raise RuntimeError("NUMBA_CUDA_MLIR_BUILD_LLVM70=1 requires MLIR_DIR to be set")
        return True
    return False


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
        for launcher_var in ("CMAKE_C_COMPILER_LAUNCHER", "CMAKE_CXX_COMPILER_LAUNCHER"):
            launcher = os.environ.get(launcher_var)
            if launcher:
                cmake_cmd.append(f"-D{launcher_var}={launcher}")
        mlir_dir = os.environ.get("MLIR_DIR")
        build_llvm70 = _should_build_llvm70(mlir_dir)
        if build_llvm70:
            cmake_cmd += ["-DBUILD_LLVM70=ON", f"-DMLIR_DIR={mlir_dir}"]
            llvm70_root = os.environ.get("LLVM70_ROOT")
            if llvm70_root:
                cmake_cmd.append(f"-DLLVM70_ROOT={llvm70_root}")
            capi = _find_mlir_python_capi_link_library()
            if capi:
                cmake_cmd.append(f"-DLLVM70_MLIR_PYTHON_CAPI={capi}")
            else:
                raise RuntimeError(
                    "BUILD_LLVM70 requires MLIRPythonCAPI under MLIR_DIR; set MLIR_DIR "
                    "to a modern MLIR install that includes MLIRPythonCAPI.dll/.lib "
                    "or libMLIRPythonCAPI.so"
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

        self._stage_mlir_bindings()

        if build_llvm70:
            self._stage_llvm70_bridge(build_dir)
            self._stage_libllvm7()
        else:
            self._remove_llvm70_outputs()

    def _package_root(self):
        return (
            Path(self.get_ext_fullpath("numba_cuda_mlir._cext")).parent
            if self.editable_mode
            else Path(self.build_lib) / "numba_cuda_mlir"
        )

    def _stage_llvm70_bridge(self, build_dir):
        """Copy the LLVM70 translator into the package if it was built."""
        llvm70_capi = build_dir / "cext" / "mlir-llvm70" / "lib" / _shared_lib_name("MLIRToLLVM70")
        if not llvm70_capi.exists():
            raise RuntimeError(f"BUILD_LLVM70 did not produce {llvm70_capi}")

        pkg = self._package_root()
        dest = pkg / llvm70_capi.name
        if not self.dry_run:
            pkg.mkdir(parents=True, exist_ok=True)
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            if self.editable_mode:
                dest.symlink_to(llvm70_capi)
            else:
                shutil.copy2(llvm70_capi, dest)

        # Also place alongside MLIRPythonCAPI so the runtime loader can
        # resolve it from the wheel on DLL platforms.
        mlir_libs_dir = pkg / "_mlir" / "_mlir_libs"
        if mlir_libs_dir.exists() and not self.dry_run:
            mlir_dest = mlir_libs_dir / llvm70_capi.name
            if mlir_dest.exists() or mlir_dest.is_symlink():
                mlir_dest.unlink()
            print(f"Staging {llvm70_capi.name}: {llvm70_capi} -> {mlir_dest}")
            shutil.copy2(llvm70_capi, mlir_dest)

    def _remove_llvm70_outputs(self):
        """Remove stale LLVM 7 artifacts from reused build directories."""
        pkg = self._package_root()
        candidates = [
            pkg / _shared_lib_name("MLIRToLLVM70"),
            pkg / "_mlir" / "_mlir_libs" / _shared_lib_name("MLIRToLLVM70"),
            pkg / "lib" / "libLLVM-7.so",
        ]
        for candidate in candidates:
            if candidate.exists() or candidate.is_symlink():
                candidate.unlink()

    def _stage_mlir_bindings(self):
        """Copy MLIR Python bindings from the LLVM install into the wheel."""
        mlir_dir = os.environ.get("MLIR_DIR")
        if not mlir_dir:
            return
        install_root = Path(mlir_dir).resolve().parent.parent.parent
        mlir_pkg = (
            install_root / "python_packages" / "numba_cuda_mlir_mlir" / "numba_cuda_mlir" / "_mlir"
        )
        if not mlir_pkg.exists():
            print(f"WARNING: MLIR Python bindings not found at {mlir_pkg}")
            return
        pkg = self._package_root()
        dest = pkg / "_mlir"
        if self.editable_mode:
            if dest.exists() or dest.is_symlink():
                if dest.is_symlink():
                    dest.unlink()
                else:
                    shutil.rmtree(dest)
            dest.symlink_to(mlir_pkg)
            print(f"Symlinking MLIR Python bindings: {mlir_pkg} -> {dest}")
        else:
            if dest.exists():
                shutil.rmtree(dest)
            print(f"Staging MLIR Python bindings: {mlir_pkg} -> {dest}")
            shutil.copytree(str(mlir_pkg), str(dest))

    def _stage_libllvm7(self):
        """Copy the optional LLVM 7 runtime library into the wheel."""
        libllvm7 = os.environ.get("LIBLLVM7")
        if not libllvm7:
            return
        libllvm7 = Path(libllvm7)
        if not libllvm7.exists():
            print(f"WARNING: LIBLLVM7 not found at {libllvm7}")
            return
        pkg = self._package_root()
        dest_dir = pkg / "lib"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / libllvm7.name
        print(f"Staging {libllvm7.name}: {libllvm7} -> {dest}")
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
