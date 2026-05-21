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
3. **Direct execution**: `python tests/benchmarks/vector_add/test_vector_addition.py scalar --compile-mode warm`

### Compile modes

Standalone benchmark scripts accept `--compile-mode {cold,warm}`:

- `cold` measures compilation in a fresh subprocess without benchmark-side warmup.
- `warm` first compiles a trivial kernel through both backends, then times the benchmark kernel compilation. This removes one-time initialization costs from the measured compile time.

The pytest benchmark runner invokes each script once per compile mode and reports both results in the consolidated table.

## Output

Running benchmarks using pytest with `--benchmark -s` produces:

Machine: AMD Ryzen 9 9950X | NVIDIA RTX PRO 6000 Blackwell | CUDA driver 580.95 | Python 3.12 | Ubuntu 24.04 | CUDA Toolkit 13.0

```
====================================================================================================
BENCHMARK RESULTS SUMMARY
====================================================================================================
Benchmark                    | Numba-CUDA Cold Compile (ms) | numba-cuda-mlir Cold Compile (ms) | Cold Compile Speedup | Numba-CUDA Warm Compile (ms) | numba-cuda-mlir Warm Compile (ms) | Warm Compile Speedup | Numba-CUDA Kernel (ms) | numba-cuda-mlir Kernel (ms) | Kernel Speedup
-----------------------------+------------------------------+-----------------------------------+----------------------+------------------------------+-----------------------------------+----------------------+------------------------+-----------------------------+---------------
Attention                    | 615.90                       | 770.61                            | 0.80x                | 145.62                       | 69.90                             | 2.08x                | 10.2656                | 11.5132                     | 0.89x
Blackscholes                 | 541.78                       | 792.44                            | 0.68x                | 87.30                        | 64.55                             | 1.35x                | 0.0222                 | 0.0238                      | 0.93x
Cholesky                     | 605.83                       | 783.69                            | 0.77x                | 151.94                       | 84.95                             | 1.79x                | 34.2040                | 33.1818                     | 1.03x
Cholesky Blocked             | 833.56                       | 960.71                            | 0.87x                | 372.09                       | 175.80                            | 2.12x                | 4.4067                 | 4.9058                      | 0.90x
Fft                          | 621.72                       | 93.63                             | 6.64x                | 133.34                       | 83.05                             | 1.61x                | 0.0634                 | 0.0612                      | 1.04x
Matmul Smem                  | 614.37                       | 897.37                            | 0.68x                | 130.05                       | 91.67                             | 1.42x                | 0.1551                 | 0.1544                      | 1.00x
Softmax                      | 663.82                       | 882.84                            | 0.75x                | 130.97                       | 69.33                             | 1.89x                | 0.0058                 | 0.0047                      | 1.24x
Softmax Large                | 650.46                       | 945.89                            | 0.69x                | 145.28                       | 73.97                             | 1.96x                | 2.6519                 | 4.4584                      | 0.59x
Vector Addition (scalar)     | 521.86                       | 933.35                            | 0.56x                | 35.05                        | 28.26                             | 1.24x                | 0.0744                 | 0.0807                      | 0.92x
Vector Addition (vectorized) | 567.75                       | 886.43                            | 0.64x                | 78.07                        | 46.73                             | 1.67x                | 0.0679                 | 0.0690                      | 0.98x
GEOMEAN                      | 618.82                       | 696.04                            | 0.89x                | 120.56                       | 71.46                             | 1.69x                | 0.3482                 | 0.3706                      | 0.94x
====================================================================================================
```
