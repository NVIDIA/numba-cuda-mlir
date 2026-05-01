# Differences Between MLIR Operations and NumPy/CPython Semantics

This document lists cases where MLIR's built-in operations cannot be used directly
because they produce different results than NumPy or CPython.

## Complex Math

Some cmath functions don't have ops in the complex dialect, like hyperbolic trig functions.
Some are available, but have different semantics.

MLIR provides `complex.exp`, `complex.sin`, `complex.cos`, but the lowering
(`convert-complex-to-standard` pass) does not seem to fully respect IEEE 754,
even with fast-math flags disabled.

#### `complex.exp`

| Input | MLIR Result | CPython Result | Difference |
|-------|-------------|----------------|------------|
| `inf + 0j` | `inf + 0j` | `inf + 0j` | ✅ Same |
| `-inf + 0j` | `0 + 0j` | `0j` | ✅ Same |
| `nan + 0j` | `nan + nan*j` | `nan + 0j` | ❌ MLIR propagates nan to imag |
| `inf + inf*j` | `inf + nan*j` | ValueError / `nan + nan*j` | ❌ MLIR returns inf real |
| `0 + inf*j` | `nan + nan*j` | ValueError | ✅ Acceptable |

MLIR's `complex.exp` lowering only handles overflow prevention (computing
`exp(x/2) * exp(x/2)` for large x) but doesn't preserve the imaginary part
when it's zero and real is nan.

#### `complex.sin`

| Input | MLIR Result | CPython Result | Difference |
|-------|-------------|----------------|------------|
| `0 + inf*j` | `0 + inf*j` | `0 + inf*j` | ✅ Same |
| `inf + 0j` | `nan + nan*j` | ValueError / `nan + nan*j` | ✅ Same |
| `nan + 0j` | `nan + nan*j` | `nan + 0j` | ❌ MLIR propagates nan |
| `1 + 1j` | correct | correct | ✅ Same |

#### `complex.cos`

| Input | MLIR Result | CPython Result | Difference |
|-------|-------------|----------------|------------|
| `0 + inf*j` | `inf + 0j` | `inf - 0j` | ❌ Signed zero |
| `inf + 0j` | `nan + nan*j` | ValueError / `nan + 0j` | ❌ MLIR propagates nan |
| `1 + 1j` | correct | correct | ✅ Same |


The differences listed above occur **regardless of fast_math setting**.
