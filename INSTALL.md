# Installing Numba-CUDA-MLIR

Numba-CUDA-MLIR can be installed:

- [With the package managers, pip and conda](#option-1-pre-built-packages):
  - Recommended for most users.
- [From source, with pre-built LLVM
  binaries](#option-2-editable-install-with-pre-built-llvm):
  - Recommended for most contributors.
- [From source, with LLVM builds from source](#option-3-build-llvm-from-source):
  - Recommended only for debugging interactions between Numba-CUDA-MLIR and
    contributions that involve modifications to the layers below
    Numba-CUDA-MLIR.

## Prerequisites

- Python >= 3.11, with:
  - The `cuda.core` and `cuda-bindings` packages
  - NumPy >= 1.22
- CUDA Toolkit components (CUDA Runtime, NVCC, NVRTC, nvJitLink, and CCCL)
  installed via pip or conda, or a system package manager (Linux).
- NVIDIA GPU with Compute Capability 7.0 or greater and a compatible driver:
  - &gt;= r525 for CUDA 12.x
  - &gt;= r580 for CUDA 13.x

If building from source, specific versions of LLVM are required. Version
information is maintained in the file
[`ci/llvm-version.env`](ci/llvm-version.env).


## Option 1: Pre-built packages

### Released packages (suitable for most users)

Install with pip:

```python
pip install numba-cuda-mlir[<options>]
```

where `<options>` can include:

- `cu12`: Install CUDA toolkit dependencies matching CUDA 12 versions.
- `cu13`: Install CUDA toolkit dependencies matching CUDA 13 versions.

Development and test dependencies are defined as
[dependency groups](https://peps.python.org/pep-0735/) (requires pip >= 25.1):

```shell
pip install --group dev   # pre-commit tools and linters
pip install --group test  # pytest and required plugins
```

Or, install with conda:

```
conda install numba-cuda-mlir
```

Appropriate CUDA toolkit dependencies will be installed for the `cuda-version`
package in the conda environment.

### Top-of-tree builds

These are useful for users looking to test new features or bug fixes prior to
their inclusion in a release.

CI publishes wheels as GitHub Actions artifacts on every push to `main`. To
obtain the most recent build, use the following commands:

```shell
# Find the latest successful CI run on main:
RUN_ID=$(gh run list -R NVIDIA/numba-cuda-mlir -w ci.yaml -b main -s success -L1 --json databaseId -q '.[0].databaseId')

# Download the wheel (pick your Python version and platform):
gh run download "$RUN_ID" -R NVIDIA/numba-cuda-mlir -p "numba-cuda-mlir-python312-linux-64-*"

# Install the downloaded wheel:
pip install numba-cuda-mlir-python312-linux-64-*/numba_cuda_mlir*.whl[cu13]
```

Replace `python312` with your Python version (e.g. `python313`, `python314`, `python314t`).
For aarch64, replace `linux-64` with `linux-aarch64`.
Replace `cu13` with `cu12` for CUDA 12.x environments.

### Option 2: Editable install with pre-built LLVM

CI publishes pre-built LLVM artifacts on every push to `main`. You can download
them instead of building LLVM from scratch (which takes at least 1 hour,
depending on the machine).

First, download the LLVM artifacts from the latest CI run using the following
commands:

```shell
# Find the latest successful CI run on main:
RUN_ID=$(gh run list -R NVIDIA/numba-cuda-mlir -w ci.yaml -b main -s success -L1 --json databaseId -q '.[0].databaseId')

# Download LLVM Modern (pick your Python version: cp312, cp313, cp314, cp314t):
gh run download "$RUN_ID" -R NVIDIA/numba-cuda-mlir -n llvm-modern-install-cp312-linux-64 -D llvm-modern-install

# Download LLVM 7:
gh run download "$RUN_ID" -R NVIDIA/numba-cuda-mlir -n llvm7-install-linux-64 -D llvm7-install
```
For aarch64, replace `linux-64` with `linux-aarch64` in the artifact names.
This produces `llvm-modern-install/` and `llvm7-install/` directories.

Then, create a virtualenv and install Numba-CUDA-MLIR in editable mode:

> **Note for VS Code / Pyright users:**
> By default, `pip install -e .` on modern `setuptools` uses dynamic import hooks that static analysis tools like Pyright cannot resolve. To ensure jump-to-definition works correctly in your IDE, use `--config-settings editable_mode=compat`. See [Pyright Import Resolution: Editable Installs](https://microsoft.github.io/pyright/#/import-resolution?id=editable-installs) for more details.

```shell
python3 -m venv numba-cuda-mlir-env && source numba-cuda-mlir-env/bin/activate

MLIR_DIR=$PWD/llvm-modern-install/lib/cmake/mlir \
LIBLLVM7=$PWD/llvm7-install/lib/libLLVM-7.so \
  pip install -e '.[cu13]' --group dev --config-settings editable_mode=compat
```

### Option 3: Build LLVM from source

If you need to modify LLVM/MLIR or want to build without cached artifacts:

```shell
# Install build prerequisites for the LLVM build scripts
pip install pybind11 nanobind numpy ninja cmake sccache

# Build modern LLVM + MLIR (uses ci/llvm-version.env for the commit)
ci/build-llvm-modern.sh    # produces llvm-modern-install/

# Build LLVM 7 shared library
ci/build-llvm7.sh          # produces llvm7-install/

# Then install numba-cuda-mlir as in Option 2:
MLIR_DIR=$PWD/llvm-modern-install/lib/cmake/mlir \
LIBLLVM7=$PWD/llvm7-install/lib/libLLVM-7.so \
  pip install -e '.[cu13]' --group dev --config-settings editable_mode=compat
```

## Testing

The Numba-CUDA-MLIR testsuite is in the `tests` directory. Use `pytest` from
the project's root directory to run tests. Install the `test` dependency group
first (`pip install --group test`), which includes `pytest-xdist` for parallel
execution:

```
pytest -n 4
```

Some linker and linkable-code tests require pre-built CUDA test fixtures. Build them
before running the full suite:

```
make -C tests/numba_cuda_tests/testing/
export NUMBA_CUDA_MLIR_TEST_BIN_DIR=$PWD/tests/numba_cuda_tests/testing
```

Note that tests can fail when many threads are used due to the GPU running out
of memory; the test suite re-runs tests that fail due to GPU out-of-memory
errors by default. All other errors are reported as test failures.
