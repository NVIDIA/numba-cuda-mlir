# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ctypes
import itertools
import os
import platform

if platform.machine() == "ARM64":
    NVPTX64_DATALAYOUT = "e-p:64:64:64-p6:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8"
else:
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
        return None
    return os.path.join(d, "libMLIRPythonCAPI.so")


def _find_modern_to_nvvm_bridge_lib():
    import numba_cuda_mlir._mlir._mlir_libs as libs

    d = os.path.dirname(libs.__file__)
    if os.name == "nt":
        name = "MLIRModernToNVVM.dll"
    elif os.uname().sysname == "Darwin":
        name = "libMLIRModernToNVVM.dylib"
    else:
        name = "libMLIRModernToNVVM.so"
    return os.path.join(d, name)


MLIR_CAPI_LIB_PATH = _find_mlir_capi_lib()
LLVM_C_LIB_PATH = _find_llvm_c_lib()
MODERN_TO_NVVM_BRIDGE_LIB_PATH = _find_modern_to_nvvm_bridge_lib()

_capi = None
_modern_to_nvvm_bridge = None
_modern_bridge_dump_counter = itertools.count()


def _load_capi_library(path: str, label: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} library not found at {path}")
    return ctypes.CDLL(path)


def _get_capi():
    global _capi
    if _capi is not None:
        return _capi
    if os.name == "nt":
        raise RuntimeError(
            "direct MLIR-to-LLVM pointer translation is not supported on Windows; "
            "use the MLIRModernToNVVM bridge"
        )
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
    _capi.LLVMContextDispose.restype = None
    _capi.LLVMContextDispose.argtypes = [ctypes.c_void_p]
    return _capi


def _get_modern_to_nvvm_bridge():
    global _modern_to_nvvm_bridge
    if _modern_to_nvvm_bridge is not None:
        return _modern_to_nvvm_bridge

    lib = _load_capi_library(MODERN_TO_NVVM_BRIDGE_LIB_PATH, "MLIR modern to NVVM bridge")
    lib.mlir_modern_to_nvvm_translate_for_libnvvm.restype = ctypes.c_int
    lib.mlir_modern_to_nvvm_translate_for_libnvvm.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.mlir_modern_to_nvvm_free.restype = None
    lib.mlir_modern_to_nvvm_free.argtypes = [ctypes.c_void_p]
    _modern_to_nvvm_bridge = lib
    return lib


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


def _maybe_dump_modern_bridge_mlir(gpu_module_text):
    target = os.environ.get("NUMBA_CUDA_MLIR_DUMP_MODERN_BRIDGE_MLIR")
    if not target:
        return

    if target.lower() in {"1", "true", "yes", "stderr"}:
        import sys

        print(
            f"=============== Modern bridge MLIR ===============\n\n{gpu_module_text}\n",
            file=sys.stderr,
        )
        return

    from pathlib import Path

    path = Path(target)
    if path.exists() and path.is_dir():
        name = f"modern-bridge-{os.getpid()}-{next(_modern_bridge_dump_counter)}.mlir"
        path = path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(gpu_module_text, encoding="utf-8")


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


def translate_gpu_module_to_libnvvm_ir(
    gpu_module_text,
    ctk_major,
    ctk_minor,
    nvvm_ir_version,
    dump=False,
    emit_text_ir=False,
):
    """Translate gpu.module text to libnvvm-compatible LLVM IR bytes."""
    lib = _get_modern_to_nvvm_bridge()
    _maybe_dump_modern_bridge_mlir(gpu_module_text)
    mlir_bytes = gpu_module_text.encode("utf-8")
    out = ctypes.c_void_p()
    out_len = ctypes.c_size_t()
    err_out = ctypes.c_void_p()

    rc = lib.mlir_modern_to_nvvm_translate_for_libnvvm(
        mlir_bytes,
        len(mlir_bytes),
        ctk_major,
        ctk_minor,
        nvvm_ir_version[0],
        nvvm_ir_version[1],
        nvvm_ir_version[2],
        nvvm_ir_version[3],
        int(bool(dump)),
        int(bool(emit_text_ir)),
        ctypes.byref(out),
        ctypes.byref(out_len),
        ctypes.byref(err_out),
    )

    if rc != 0:
        msg = (
            ctypes.string_at(err_out).decode("utf-8", errors="replace")
            if err_out.value
            else "unknown error"
        )
        if err_out.value:
            lib.mlir_modern_to_nvvm_free(err_out)
        raise RuntimeError(f"MLIR modern to NVVM bridge failed: {msg}")

    try:
        return ctypes.string_at(out, out_len.value)
    finally:
        if out.value:
            lib.mlir_modern_to_nvvm_free(out)


def dump_llvmir(llvm_mod_ptr):
    """Return the LLVM IR text for an LLVMModuleRef (for debugging)."""
    capi = _get_capi()
    raw = capi.LLVMPrintModuleToString(llvm_mod_ptr)
    if not raw:
        raise RuntimeError("LLVMPrintModuleToString failed")
    result = ctypes.string_at(raw).decode("utf-8")
    capi.LLVMDisposeMessage(raw)
    return result
