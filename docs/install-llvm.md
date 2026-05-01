# Building LLVM from source

Most developers should use the cached LLVM builds from Artifactory (see
README.md, Option 2). Build from source only if you need to modify LLVM/MLIR
itself or the cache doesn't cover your platform.

## Quick start

The CI build scripts handle cloning, configuring, and building:

```shell
# Install build prerequisites
pip install pybind11 nanobind numpy ninja cmake

# Both scripts require sccache for compiler caching
# (install via conda, or from https://github.com/rapidsai/sccache/releases)

# Build modern LLVM + MLIR with numba-cuda-mlir-namespaced Python bindings
ci/build-llvm-modern.sh    # -> llvm-modern-install/

# Build LLVM 7 shared library (for the LLVM70 path)
ci/build-llvm7.sh           # -> llvm7-install/
```

The pinned LLVM commits are in `ci/llvm-version.env`. The scripts
automatically clone the correct version if the source tree isn't present.

## What the build produces

**Modern LLVM** (`llvm-modern-install/`):
- `lib/cmake/mlir/` — CMake configs used by `MLIR_DIR`
- `python_packages/numba_cuda_mlir_mlir/numba_cuda_mlir/_mlir/` — MLIR Python bindings
  namespaced under `numba_cuda_mlir._mlir` (not top-level `mlir`)
- `lib/libMLIRPythonCAPI.so` — MLIR C API shared library (unversioned SONAME)

**LLVM 7** (`llvm7-install/`):
- `lib/libLLVM-7.so` — shared library for the LLVM70 translator

## Key cmake flags

The modern LLVM build uses several flags to produce numba-cuda-mlir-compatible
bindings. See `ci/build-llvm-modern.sh` for the full invocation. The
important ones:

| Flag | Purpose |
|------|---------|
| `-DMLIR_ENABLE_BINDINGS_PYTHON=ON` | Build the MLIR Python bindings |
| `-DCMAKE_CXX_FLAGS="-DMLIR_PYTHON_PACKAGE_PREFIX=numba_cuda_mlir._mlir."` | Namespace bindings under `numba_cuda_mlir._mlir` |
| `-DMLIR_BINDINGS_PYTHON_INSTALL_PREFIX=python_packages/numba_cuda_mlir_mlir/numba_cuda_mlir/_mlir` | On-disk install path |
| `-DMLIR_BINDINGS_PYTHON_NB_DOMAIN=numba_cuda_mlir` | Isolate nanobind typeids |
| `-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON` | Ship `lib*.so` without versioned symlinks (required for wheels) |

## Using the build

After building, install numba-cuda-mlir pointing at the artifacts:

```shell
MLIR_DIR=$PWD/llvm-modern-install/lib/cmake/mlir \
LIBLLVM7=$PWD/llvm7-install/lib/libLLVM-7.so \
  pip install -e '.[cu13,dev]'
```

`setup.py` detects these environment variables and stages the shared libraries
and Python bindings into the package automatically. No manual `PYTHONPATH` or
`.pth` file setup is needed.
