# mlir-llvm70

MLIR-to-LLVM70 translator for cuSIMT. Walks `gpu.module` ops (LLVM dialect)
and rebuilds LLVM 7 IR via the old LLVM C API (based on LLVM 7.0.1), then
compiles to PTX through `libnvvm.so`. Used when the target GPU's libnvvm
expects the LLVM 7 dialect of NVVM IR.

## Architecture

```
MLIR (gpu.module with LLVM dialect ops)
  │
  ▼  MLIRToLLVM70 translator
LLVM 7 IR (via dlopen'd libLLVM-7.so C API)
  │
  ▼  Bitcode serialization
NVVM bitcode
  │
  ▼  libnvvm.so (dlopen'd)
PTX
```

**Key components:**

- **`CAPILoader`** — generic `dlopen`/`dlsym` RAII wrapper. Uses
  `RTLD_LOCAL | RTLD_DEEPBIND` to isolate old LLVM symbols from modern LLVM.
- **`LLVM70IRBuilder`** — wraps the old `libLLVM.so` C API (~80 function
  pointers). Builds LLVM 7 IR in memory.
- **`LibNVVMCompiler`** — wraps `libnvvm.so` to compile NVVM bitcode to PTX.
- **`MLIRToLLVM70`** — walks a `gpu.module` and translates each MLIR op to the
  corresponding `LLVM70IRBuilder` call.
- **`#nvvm_llvm70.target`** — TableGen-defined target attribute implementing
  `gpu::TargetAttrInterface`, so GPU modules can be compiled via the standard
  `gpu-module-to-binary` pass.
- **`llvm70-translate`** — standalone CLI tool (used by LIT tests).

## Building

Built automatically as part of cuSIMT when `MLIR_DIR` is set:

```bash
MLIR_DIR=/path/to/llvm-install/lib/cmake/mlir pip install -e .
```

This produces `libMLIRToLLVM70.so` and places it next to `_cext.so`.
The MLIR install must match the version used by `mlir-python-bindings`
to ensure TypeID compatibility at runtime.

## Runtime environment variables

| Variable | Purpose |
|---|---|
| `LIBLLVM7` | Path to a pre-built `libLLVM-7.so` (dlopen'd by the translator) |

## Running LIT tests

```bash
# After building cuSIMT with MLIR_DIR:
export LIBLLVM7=/path/to/libLLVM-7.so
export LLVM70_LIBNVVM=/path/to/cuda/nvvm/lib64/libnvvm.so
export CUDA_HOME=/path/to/cuda

# Single-config run:
ninja -C build check-llvm70

# Multi-CUDA × multi-SM matrix:
ninja -C build check-llvm70-matrix
```

## Usage

### Target attribute on gpu.module

```mlir
gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_80">] {
  llvm.func @my_kernel(%arg0: !llvm.ptr<1>) attributes {gpu.kernel} {
    // ...
  }
}
```

The `#nvvm_llvm70.target` attribute supports:
- `O` — optimization level (0–3, default 3)
- `chip` — GPU architecture (default `"sm_80"`)
- `triple` — target triple (default `"nvptx64-nvidia-cuda"`)
- `libllvm` / `libnvvm` — override library paths per-module
- `link` — extra `.bc` files to link (e.g. libdevice)

### Standalone tool

```bash
# Compile to binary (MLIR output with gpu.binary)
llvm70-translate input.mlir -o output.mlir

# Dump PTX to stderr
llvm70-translate input.mlir --dump-ptx

# Dump intermediate LLVM IR to stderr
llvm70-translate input.mlir --dump-llvm

# Produce host LLVM IR with embedded PTX
llvm70-translate input.mlir --mlir-to-llvm-ir -o output.ll
```

Library paths are resolved from (in priority order):
1. `#nvvm_llvm70.target<libllvm = "...", libnvvm = "...">` attribute
2. `LIBLLVM7` / `LLVM70_LIBNVVM` environment variables
3. CUDA toolkit auto-discovery (`CUDA_ROOT`)
