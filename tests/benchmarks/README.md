# numba-cuda-mlir Benchmarks

Benchmarks to compare JIT compile-time, isolated end-to-end time, and kernel performance between Numba CUDA and numba-cuda-mlir implementations. Uses NVIDIA Nsight Compute (NCU) to profile kernel execution times.

## Quick Start

```bash
# Run correctness tests
pytest tests/benchmarks/

# Run performance benchmarks (requires NCU)
pytest tests/benchmarks/ --benchmark -s
```

## Available Benchmarks

| Benchmark              | Description              | Key Features                            |
|------------------------|--------------------------|------------------------------------------|
| `vector_add/`          | Vector addition          | Scalar & vectorized versions             |
| `softmax/`             | Softmax normalization    | Numerically stable 3-phase reduction     |
| `cholesky/`            | Cholesky factorization   | Blocked & unblocked algorithms           |
| `attention/`           | Self-attention           | Dynamic shared memory                    |
| `blackscholes/`        | Option pricing           | Transcendental functions                 |
| `fft/`                 | Fast Fourier Transform   | Radix-2, bit-reversal                    |
| `test_matmul_smem.py`  | Matrix multiplication    | Shared memory tiling                     |

## Usage

### Three ways to run benchmarks:

1. **Correctness only**: `pytest tests/benchmarks/vector_add/`
2. **With profiling scripts**: `pytest tests/benchmarks/vector_add/ --benchmark -s`
3. **Direct execution**: `python tests/benchmarks/vector_add/test_vector_addition.py scalar --compile-mode warm`

### Compile modes

Standalone benchmark scripts accept `--compile-mode {cold,warm}`:

- `cold` measures compilation in a fresh subprocess without benchmark-side warmup.
- `warm` first compiles a trivial kernel through both backends, then times the benchmark kernel compilation. This removes one-time initialization costs from the measured compile time.

The pytest benchmark runner invokes each script once per compile mode and reports both results in the consolidated table.

### Isolated end-to-end timing

The pytest benchmark runner also invokes each benchmark in fresh child processes with `--backend numba-cuda` and `--backend numba-cuda-mlir`. These runs measure from benchmark script import through synchronized workload completion for only the selected backend, excluding Python interpreter startup.

Standalone benchmark scripts accept `--backend {both,numba-cuda,numba-cuda-mlir}`:

```bash
python tests/benchmarks/vector_add/test_vector_addition.py scalar --compile-mode cold --backend numba-cuda
python tests/benchmarks/vector_add/test_vector_addition.py scalar --compile-mode cold --backend numba-cuda-mlir
```

## Output

Running benchmarks using pytest with `--benchmark -s` produces:

```
====================================================================================================
BENCHMARK RESULTS SUMMARY
====================================================================================================
+--------------------------+--------------------------------+-------------------------------------+------------------------+--------------------------------+-------------------------------------+------------------------+--------------------------+-------------------------------+------------------+----------------------+---------------------------+---------------+
| Benchmark                |   Numba-CUDA Cold Compile (ms) |   numba-cuda-mlir Cold Compile (ms) | Cold Compile Speedup   |   Numba-CUDA Warm Compile (ms) |   numba-cuda-mlir Warm Compile (ms) | Warm Compile Speedup   |   Numba-CUDA Kernel (ms) |   numba-cuda-mlir Kernel (ms) | Kernel Speedup   |   Numba-CUDA E2E (ms) |   numba-cuda-mlir E2E (ms) | E2E Speedup   |
+==========================+================================+=====================================+========================+================================+=====================================+========================+==========================+===============================+==================+======================+===========================+===============+
| Vector Addition (scalar) |                         487.20 |                               39.19 | 12.43x                 |                          29.04 |                               24.15 | 1.20x                  |                   0.0736 |                        0.0812 | 0.91x            |              1342.15 |                    418.37 | 3.21x         |
+--------------------------+--------------------------------+-------------------------------------+------------------------+--------------------------------+-------------------------------------+------------------------+--------------------------+-------------------------------+------------------+----------------------+---------------------------+---------------+
====================================================================================================
```
