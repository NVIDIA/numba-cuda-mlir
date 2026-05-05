# mlir-llvm70

MLIR-to-LLVM70 translator for numba-cuda-mlir. Translates MLIR GPU modules
(LLVM dialect) into LLVM 7 IR via the old LLVM C API (based on LLVM 7.0.1),
then compiles to PTX through `libnvvm.so`. Used for pre-Blackwell GPUs
whose libnvvm expects the LLVM 7 dialect of NVVM IR.

Built automatically as part of numba-cuda-mlir when `MLIR_DIR` is set.
See the top-level README for installation instructions.
