# Numba-CUDA-MLIR

Numba-CUDA-MLIR provides a programming model similar to CUDA C++ in Python. It
is evolved from [Numba-CUDA](https://github.com/NVIDIA/numba-cuda), and is
intended to be compatible with Numba-CUDA kernels.

Numba-CUDA-MLIR aims to interoperate well with existing programming models
whilst also allowing experts sufficient control over code generation.


## Quick Start

Install with pip:

```
pip install numba-cuda-mlir[cu13]  # or [cu12] if using CUDA 12
```

Writing and executing a simple vector add kernel:

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


## Migration from Numba / Numba-CUDA

Change imports to use the `numba_cuda_mlir.cuda` package instead of
`numba.cuda`. For example:

```python
from numba import cuda
```

becomes:

```python
from numba_cuda_mlir import cuda
```

For the majority of code using Numba-CUDA, this should be a sufficient change to
enable the use of Numba-CUDA-MLIR. For code using the extension APIs,
modifications will be required as Numba-CUDA-MLIR uses MLIR in its code
generation process instead of LLVM IR. See the Migration Guidance in the
documentation for further details.


## Installation Requirements

- Python >= 3.11, with:
  - The `cuda.core` and `cuda-bindings` packages
  - NumPy >= 1.22
- CUDA Toolkit components (CUDA Runtime, NVCC, NVRTC, nvJitLink, and CCCL)
  installed via pip or a system package manager (Linux).
- NVIDIA GPU with Compute Capability 7.0 or greater and a compatible driver:
  - &gt;= r525 for CUDA 12.x
  - &gt;= r580 for CUDA 13.x


## Installation guidance

For full details of installation methods including from packages and building
from source and testing, please see
[INSTALL.md](https://github.com/NVIDIA/numba-cuda-mlir/blob/main/INSTALL.md).


## Contributing to Numba-CUDA-MLIR

See the [Contribution
Guidelines](https://github.com/NVIDIA/numba-cuda-mlir/blob/main/CONTRIBUTING.md)
for information on how to set
up a development environment and follow the contribution process.


## Benchmarks

A small suite of benchmarks can be executed from the source repository by
running:

```
pytest tests/benchmarks/ --benchmark -s
```

## Licensing

Numba-CUDA-MLIR is distributed under the [Apache License
2.0](https://github.com/NVIDIA/numba-cuda-mlir/blob/main/LICENSE).

It incorporates the following third-party projects, each retained under its
original license:

1. [numba-cuda](https://github.com/NVIDIA/numba-cuda) — [BSD 2-Clause
   License](https://github.com/NVIDIA/numba-cuda-mlir/blob/main/THIRD-PARTY-LICENSES)
2. [cloudpickle](https://github.com/cloudpipe/cloudpickle) — [BSD 3-Clause
   License](https://github.com/NVIDIA/numba-cuda-mlir/blob/main/THIRD-PARTY-LICENSES)
3. [appdirs](https://github.com/ActiveState/appdirs) — [MIT
   License](https://github.com/NVIDIA/numba-cuda-mlir/blob/main/THIRD-PARTY-LICENSES)
4. [LLVM Project / EUDSL](https://github.com/llvm/llvm-project) — [Apache
   License 2.0 WITH
   LLVM-exception](https://github.com/NVIDIA/numba-cuda-mlir/blob/main/THIRD-PARTY-LICENSES)

See [`NOTICE`](https://github.com/NVIDIA/numba-cuda-mlir/blob/main/NOTICE) for
the full attribution map and per-component locations in this repository, and
[`THIRD-PARTY-LICENSES`](https://github.com/NVIDIA/numba-cuda-mlir/blob/main/THIRD-PARTY-LICENSES)
for the verbatim upstream license texts.

Contributions are accepted under the terms described in
[`CONTRIBUTING.md`](https://github.com/NVIDIA/numba-cuda-mlir/blob/main/CONTRIBUTING.md).
