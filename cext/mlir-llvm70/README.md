# mlir-llvm70

MLIR-to-LLVM70 translator for numba-cuda-mlir. Translates MLIR GPU modules
(LLVM dialect) into LLVM 7 IR via the old LLVM C API (based on LLVM 7.0.1),
then compiles to PTX through `libnvvm.so`. Used for pre-Blackwell GPUs
whose libnvvm expects the LLVM 7 dialect of NVVM IR.

Built by CMake with `-DBUILD_LLVM70=ON`. Source installs default to building
it when `MLIR_DIR` is set, and can disable it with
`NUMBA_CUDA_MLIR_BUILD_LLVM70=0` for latest-LLVM-only builds. See the top-level
installation guide for details.
