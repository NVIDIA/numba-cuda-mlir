# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes

import pytest


def _get_llvm70_bridge():
    from numba_cuda_mlir.mlir_optimization import _get_llvm70_capi

    try:
        return _get_llvm70_capi()
    except (FileNotFoundError, ImportError, OSError) as exc:
        pytest.skip(f"LLVM70 C API bridge is not available: {exc}")


def test_llvm70_capi_accepts_python_gpu_module_op():
    from numba_cuda_mlir._mlir import ir
    from numba_cuda_mlir._mlir.dialects import gpu
    from numba_cuda_mlir.mlir_optimization import _get_op_ptr

    lib = _get_llvm70_bridge()

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(
            """
            module {
              gpu.module @kernels {
              }
            }
            """
        )

        gpu_modules = [op for op in module.body if isinstance(op, gpu.GPUModuleOp)]
        assert len(gpu_modules) == 1
        raw_op = _get_op_ptr(gpu_modules[0].operation)

        out = ctypes.c_char_p()
        out_len = ctypes.c_size_t()
        err_out = ctypes.c_char_p()

        missing_path = b"/definitely/not/a/real/llvm70/runtime"
        rc = lib.llvm70_translate_gpu_module_from_op(
            raw_op,
            b"sm_80",
            None,
            missing_path,
            missing_path,
            missing_path,
            0,
            2,
            0,
            ctypes.byref(out),
            ctypes.byref(out_len),
            ctypes.byref(err_out),
        )

        if rc == 0:
            lib.llvm70_free(out)
            pytest.fail("LLVM70 bridge unexpectedly translated with fake runtime paths")

        msg = err_out.value.decode(errors="replace") if err_out.value else ""
        if err_out.value:
            lib.llvm70_free(err_out)

    assert "Operation is not a gpu.module" not in msg
    assert msg
