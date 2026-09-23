# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numba_cuda_mlir import compiler, cuda, types


class _CachedStyleCompileResult:
    def __init__(self, signature):
        self.signature = signature
        self.metadata = {
            "cubin": b"\x7fELF cached cubin",
            "ptx": "// cached ptx",
            "func_name": "cached_kernel",
            "mlir_module_optimized": "module {}",
            "targetoptions": {},
        }


class _FakeDispatcher(compiler.MLIRDispatcher):
    def __init__(self, py_func):
        self.py_func = py_func
        self.targetoptions = {}
        self.compile_calls = []

    def compile(self, sig, abi_info=None, output=None):
        self.compile_calls.append((sig, abi_info, output))
        return compiler.CompileResult(_CachedStyleCompileResult(sig))


def test_compile_cubin_does_not_optimize_dispatcher_result(monkeypatch):
    """Public compile helpers should trust MLIRDispatcher.compile() to optimize."""

    def kernel(x):
        return x

    dispatcher = _FakeDispatcher(kernel)

    def fake_jit(pyfunc, **targetoptions):
        dispatcher.targetoptions.update(targetoptions)
        return dispatcher

    def fail_if_called(cres):
        raise AssertionError("compiler._compile() should not re-run optimize()")

    monkeypatch.setattr(cuda, "jit", fake_jit)
    monkeypatch.setattr(compiler, "optimize", fail_if_called, raising=False)

    sig = types.void(types.int32[:])

    assert compiler.compile_cubin(kernel, sig) == b"\x7fELF cached cubin"
    assert dispatcher.compile_calls == [(sig, None, None)]
