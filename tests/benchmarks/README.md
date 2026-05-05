# numba-cuda-mlir Benchmarks

Benchmarks to compare JIT compile-time and kernel performance between Numba CUDA and numba-cuda-mlir implementations. Uses NVIDIA Nsight Compute (NCU) to profile kernel execution times.

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
3. **Direct execution**: `python tests/benchmarks/vector_add/test_vector_addition.py scalar`

## Output

Running benchmarks using pytest with `--benchmark -s` produces:

```
+------------------------------+---------------------------+-----------------------+-------------------+--------------------------+----------------------+------------------+
| Benchmark                    |   Numba-CUDA Compile (ms) |   numba-cuda-mlir Compile (ms) | Compile Speedup   |   Numba-CUDA Kernel (ms) |   numba-cuda-mlir Kernel (ms) | Kernel Speedup   |
+==============================+===========================+=======================+===================+==========================+======================+==================+
| Attention                    |                    607.41 |                405.14 | 1.50x             |                  10.2513 |               8.3995 | 1.22x            |
+------------------------------+---------------------------+-----------------------+-------------------+--------------------------+----------------------+------------------+
| Blackscholes                 |                    574.71 |                422.94 | 1.36x             |                   0.0229 |               0.0249 | 0.92x            |
+------------------------------+---------------------------+-----------------------+-------------------+--------------------------+----------------------+------------------+
| Cholesky                     |                    596.7  |                498.65 | 1.20x             |                  34.2783 |              29.6348 | 1.16x            |
+------------------------------+---------------------------+-----------------------+-------------------+--------------------------+----------------------+------------------+
| Cholesky Blocked             |                    756.21 |                722.45 | 1.05x             |                   4.4048 |               3.3369 | 1.32x            |
+------------------------------+---------------------------+-----------------------+-------------------+--------------------------+----------------------+------------------+
| Fft                          |                    563.77 |                260.67 | 2.16x             |                   0.0628 |               0.0736 | 0.85x            |
+------------------------------+---------------------------+-----------------------+-------------------+--------------------------+----------------------+------------------+
| Matmul Smem                  |                    556.61 |                415.77 | 1.34x             |                   0.1553 |               0.2252 | 0.69x            |
+------------------------------+---------------------------+-----------------------+-------------------+--------------------------+----------------------+------------------+
| Softmax                      |                    567.8  |                416.38 | 1.36x             |                   0.0056 |               0.0045 | 1.24x            |
+------------------------------+---------------------------+-----------------------+-------------------+--------------------------+----------------------+------------------+
| Softmax Large                |                    564.92 |                406.4  | 1.39x             |                   2.6519 |               2.8426 | 0.93x            |
+------------------------------+---------------------------+-----------------------+-------------------+--------------------------+----------------------+------------------+
| Vector Addition (scalar)     |                    473.65 |                339.47 | 1.40x             |                   0.0742 |               0.081  | 0.92x            |
+------------------------------+---------------------------+-----------------------+-------------------+--------------------------+----------------------+------------------+
| Vector Addition (vectorized) |                    510.54 |                370.53 | 1.38x             |                   0.0687 |               0.0692 | 0.99x            |
+------------------------------+---------------------------+-----------------------+-------------------+--------------------------+----------------------+------------------+
```
