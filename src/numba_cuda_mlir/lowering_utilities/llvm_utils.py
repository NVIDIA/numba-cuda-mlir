# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ctypes
import os
from types import SimpleNamespace

NVPTX64_DATALAYOUT = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64-S128"
NVPTX64_TRIPLE = "nvptx64-nvidia-cuda"


def _find_mlir_capi_lib():
    import numba_cuda_mlir._mlir._mlir_libs as libs

    name = "MLIRPythonCAPI.dll" if os.name == "nt" else "libMLIRPythonCAPI.so"
    return os.path.join(os.path.dirname(libs.__file__), name)


def _find_llvm_c_lib():
    import numba_cuda_mlir._mlir._mlir_libs as libs

    d = os.path.dirname(libs.__file__)
    if os.name == "nt":
        return os.path.join(d, "LLVM-C-modern.dll")
    return os.path.join(d, "libMLIRPythonCAPI.so")


MLIR_CAPI_LIB_PATH = _find_mlir_capi_lib()
LLVM_C_LIB_PATH = _find_llvm_c_lib()

_capi = None


def _get_capi():
    global _capi
    if _capi is not None:
        return _capi
    if os.name == "nt":
        mlir = ctypes.CDLL(MLIR_CAPI_LIB_PATH)
        llvm = ctypes.CDLL(LLVM_C_LIB_PATH)
        mlir.mlirTranslateModuleToLLVMIR.restype = ctypes.c_void_p
        mlir.mlirTranslateModuleToLLVMIR.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        llvm.LLVMContextCreate.restype = ctypes.c_void_p
        llvm.LLVMContextCreate.argtypes = []
        llvm.LLVMPrintModuleToString.restype = ctypes.c_void_p
        llvm.LLVMPrintModuleToString.argtypes = [ctypes.c_void_p]
        llvm.LLVMDisposeMessage.restype = None
        llvm.LLVMDisposeMessage.argtypes = [ctypes.c_void_p]
        llvm.LLVMContextDispose.restype = None
        llvm.LLVMContextDispose.argtypes = [ctypes.c_void_p]
        _capi = SimpleNamespace(
            LLVMContextCreate=llvm.LLVMContextCreate,
            LLVMContextDispose=llvm.LLVMContextDispose,
            LLVMPrintModuleToString=llvm.LLVMPrintModuleToString,
            LLVMDisposeMessage=llvm.LLVMDisposeMessage,
            mlirTranslateModuleToLLVMIR=mlir.mlirTranslateModuleToLLVMIR,
        )
    else:
        _capi = ctypes.CDLL(MLIR_CAPI_LIB_PATH)
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
