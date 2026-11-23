# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ctypes
import os


NVPTX64_DATALAYOUT = (
    "e-p:64:64:64-p6:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"
    "-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32"
    "-v64:64:64-v128:128:128-n16:32:64-a:8:8"
)
NVPTX64_TRIPLE = "nvptx64-nvidia-cuda"


def _find_mlir_capi_lib():
    import numba_cuda_mlir._mlir._mlir_libs as libs

    name = "MLIRPythonCAPI-numba_cuda_mlir"
    name = f"{name}.dll" if os.name == "nt" else f"lib{name}.so"
    return os.path.join(os.path.dirname(libs.__file__), name)


# The CAPI library also exports the LLVM C API (on Windows via /EXPORT, see
# cext/mlir-python/CMakeLists.txt) for _cext.downgrade_for_libnvvm.
MLIR_CAPI_LIB_PATH = _find_mlir_capi_lib()
LLVM_C_LIB_PATH = MLIR_CAPI_LIB_PATH

_capi = None


def _load_capi_library(path: str, label: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} library not found at {path}")
    return ctypes.CDLL(path)


def _get_capi():
    global _capi
    if _capi is not None:
        return _capi
    _capi = _load_capi_library(MLIR_CAPI_LIB_PATH, "MLIR/LLVM Python C API")
    _capi.LLVMContextCreate.restype = ctypes.c_void_p
    _capi.LLVMContextCreate.argtypes = []
    _capi.mlirTranslateModuleToLLVMIR.restype = ctypes.c_void_p
    _capi.mlirTranslateModuleToLLVMIR.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _capi.LLVMPrintModuleToString.restype = ctypes.c_void_p
    _capi.LLVMPrintModuleToString.argtypes = [ctypes.c_void_p]
    _capi.LLVMDisposeMessage.restype = None
    _capi.LLVMDisposeMessage.argtypes = [ctypes.c_void_p]
    _capi.LLVMDisposeModule.restype = None
    _capi.LLVMDisposeModule.argtypes = [ctypes.c_void_p]
    _capi.LLVMContextDispose.restype = None
    _capi.LLVMContextDispose.argtypes = [ctypes.c_void_p]
    return _capi


def _op_to_raw_ptr(op):
    """Extract the raw C pointer from an MLIR operation's PyCapsule."""
    capsule = op._CAPIPtr
    get_ptr = ctypes.pythonapi.PyCapsule_GetPointer
    get_ptr.restype = ctypes.c_void_p
    get_ptr.argtypes = [ctypes.py_object, ctypes.c_char_p]
    ptr = get_ptr(capsule, b"numba_cuda_mlir._mlir.ir.Operation._CAPIPtr")
    if not ptr:
        raise ValueError(f"failed to extract C pointer from {op!r}")
    return ptr


def translate_to_llvmir(op):
    """Translate an MLIR gpu.module operation to an LLVMModuleRef.

    Returns (llvm_mod_ptr, llvm_ctx_ptr) as raw integer pointers.
    """
    capi = _get_capi()
    op_ptr = _op_to_raw_ptr(op)
    llvm_ctx = capi.LLVMContextCreate()
    if not llvm_ctx:
        raise RuntimeError("LLVMContextCreate failed")
    try:
        llvm_mod = capi.mlirTranslateModuleToLLVMIR(op_ptr, llvm_ctx)
    except Exception:
        capi.LLVMContextDispose(llvm_ctx)
        raise
    if not llvm_mod:
        capi.LLVMContextDispose(llvm_ctx)
        raise RuntimeError("mlirTranslateModuleToLLVMIR failed")
    return llvm_mod, llvm_ctx


def dump_llvmir(llvm_mod_ptr):
    """Return the LLVM IR text for an LLVMModuleRef (for debugging)."""
    capi = _get_capi()
    raw = capi.LLVMPrintModuleToString(llvm_mod_ptr)
    if not raw:
        raise RuntimeError("LLVMPrintModuleToString failed")
    result = ctypes.string_at(raw).decode("utf-8")
    capi.LLVMDisposeMessage(raw)
    return result


def translate_to_llvmir_text(op):
    """Translate an MLIR gpu.module operation to readable LLVM IR text."""
    llvm_mod, llvm_ctx = translate_to_llvmir(op)
    capi = _get_capi()
    try:
        return dump_llvmir(llvm_mod)
    finally:
        capi.LLVMDisposeModule(llvm_mod)
        capi.LLVMContextDispose(llvm_ctx)
