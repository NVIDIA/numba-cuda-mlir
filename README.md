# Numba-CUDA-MLIR — CUDA-like Programming Model for Python

numba-cuda-mlir provides a low-level programming model similar to CUDA C++ in Python.
The main goals of the project are:

1. Do not inhibit experts
2. Interoperate well with existing programming models

numba-cuda-mlir is built on MLIR. We don't use any downstream dialects.
This is not a wrapper around the MLIR Python bindings, however;
user programs look like regular Python, and users do not need
to be compiler experts to use it.

## Quick Start

```python
import numpy as np
from numba_cuda_mlir import cuda

@cuda.jit
def vector_add(a, b, out):
    i = cuda.grid(1)
    if i < out.shape[0]:
        out[i] = a[i] + b[i]

n = 1_000_000
a = np.ones(n, dtype=np.float32)
b = np.ones(n, dtype=np.float32)
out = np.zeros(n, dtype=np.float32)

threads_per_block = 256
blocks = (n + threads_per_block - 1) // threads_per_block
vector_add[blocks, threads_per_block](a, b, out)
```

## Prerequisites

- Python >= 3.12
- NVIDIA GPU with a compatible driver (CUDA 12.2+ or 13.x)
    - CUDA Toolkit is **not** required at build time (numba-cuda-mlir uses a driver API
      shim header).
    - CUDA Toolkit components (ex: nvJitLink, libNVVM) can be installed via pip (Linux/Windows), conda (Linux/Windows), or any system package manager (Linux).
    - At runtime, CUDA driver and Toolkit components are dynamically loaded.
    - Set `CUDA_HOME` if you need libdevice linking for older paths.

The pinned LLVM commit is in [`ci/llvm-version.env`](ci/llvm-version.env).

## Installation

### Option 1: Pre-built wheel (fastest)

CI publishes wheels as GitHub Actions artifacts on every push to `main`.
No LLVM, cmake, or CUDA Toolkit needed:

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

### Option 2: Editable install with cached LLVM (recommended for development)

CI publishes pre-built LLVM artifacts on every push to `main`.
You can download them instead of building LLVM from scratch
(~1 hour depending on the machine):

1. Download the LLVM artifacts from the latest CI run using the
   GitHub CLI (`gh`):
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

2. Create a venv and install numba-cuda-mlir in editable mode:

```shell
python3 -m venv numba-cuda-mlir-env && source numba-cuda-mlir-env/bin/activate

MLIR_DIR=$PWD/llvm-modern-install/lib/cmake/mlir \
LIBLLVM7=$PWD/llvm7-install/lib/libLLVM-7.so \
  pip install -e '.[cu13,dev]'
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
  pip install -e '.[cu13,dev]'
```

## Testing

Our tests are placed in the `tests` directory.
Using `pytest` from the project's root directory after installing numba-cuda-mlir will
run our tests. `pytest-xdist` is installed with our testing dependencies, so
they may be run in parallel with:

```
pytest -n 4
```

Some linker/linkable-code tests require pre-built CUDA test fixtures. Build them
before running the full suite:

```
make -C tests/numba_cuda_tests/testing/
export NUMBA_CUDA_MLIR_TEST_BIN_DIR=$PWD/tests/numba_cuda_tests/testing
```

Note that tests can fail when many threads are used due to the GPU running out of
memory; we re-run tests that fail due to GPU out-of-memory errors by default.
All other errors are reported as test failures.

## Benchmarks

```
pytest tests/benchmarks/ --benchmark -s
```

## Pre-commit hooks

We use [pre-commit hooks](https://pre-commit.com/) for formatting and basic linting that should
be applied to every commit.
They can be installed with:

```
pip install -e '.[dev]'
pre-commit install
```

Then, every commit will be formatted and linted automatically.

## Debugging

To dump Numba IR and MLIR to stderr before the MLIR-to-NVVM pipeline, enable `dump` in the `@cuda.jit()` decorator options, e.g. `@cuda.jit(dump=True)`.
To print the full list of available debug options, enable `help` in the `@cuda.jit()` decorator options, e.g. `@cuda.jit(help=True)`.

## Licensing

numba-cuda-mlir is distributed under the [Apache License 2.0](LICENSE).

It incorporates the following third-party projects, each retained under its
original license:

1. [numba-cuda](https://github.com/NVIDIA/numba-cuda) — [BSD 2-Clause License](THIRD-PARTY-LICENSES)
2. [cloudpickle](https://github.com/cloudpipe/cloudpickle) — [BSD 3-Clause License](THIRD-PARTY-LICENSES)
3. [appdirs](https://github.com/ActiveState/appdirs) — [MIT License](THIRD-PARTY-LICENSES)
4. [LLVM Project / EUDSL](https://github.com/llvm/llvm-project) — [Apache License 2.0 WITH LLVM-exception](THIRD-PARTY-LICENSES)

See [`NOTICE`](NOTICE) for the full attribution map and per-component locations
in this repository, and [`THIRD-PARTY-LICENSES`](THIRD-PARTY-LICENSES) for the
verbatim upstream license texts.

Contributions are accepted under the terms described in
[`CONTRIBUTING.md`](CONTRIBUTING.md).
