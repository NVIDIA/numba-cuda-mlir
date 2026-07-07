# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for CPython free-threaded execution."""

from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys
import sysconfig
import textwrap
import threading
import time

import pytest


def _is_free_threaded_python():
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def _run_python(code, *, env=None, timeout=60):
    child_env = os.environ.copy()
    for key, value in (env or {}).items():
        if value is None:
            child_env.pop(key, None)
        else:
            child_env[key] = value

    return subprocess.run(
        [
            sys.executable,
            "-W",
            "error::RuntimeWarning",
            "-c",
            textwrap.dedent(code),
        ],
        capture_output=True,
        env=child_env,
        text=True,
        timeout=timeout,
    )


def _require_free_threaded_python():
    if not _is_free_threaded_python():
        pytest.skip("requires a free-threaded CPython build")


def test_imports_do_not_enable_gil_by_default():
    _require_free_threaded_python()

    result = _run_python(
        """
        import sys
        import warnings

        warnings.filterwarnings(
            "ignore",
            message="The CUDA driver version is older than the backend version.*",
            category=RuntimeWarning,
        )

        assert not sys._is_gil_enabled()
        import importlib
        import pkgutil
        import numba_cuda_mlir
        from numba_cuda_mlir import cuda

        for module_info in pkgutil.iter_modules(numba_cuda_mlir.__path__):
            if module_info.name.startswith("_") and module_info.name != "_version":
                importlib.import_module(f"numba_cuda_mlir.{module_info.name}")

        assert not sys._is_gil_enabled()
        print("gil-disabled", cuda)
        """,
        env={"PYTHON_GIL": None},
    )

    assert result.returncode == 0, result.stderr
    assert "gil-disabled" in result.stdout


@pytest.mark.parametrize(
    "args, match",
    [
        (((3,), (4,), -1, 4), "ndim is negative"),
        (((3,), (4,), 1, 0), "itemsize <= 0"),
        (((3,), (4,), 2, 4), "shape length does not match ndim"),
        (((3, 4, 5), (16, 4), 2, 4), "shape length does not match ndim"),
        (((3, 4), (16,), 2, 4), "strides length does not match ndim"),
        (((3, 4), (16, 4, 1), 2, 4), "strides length does not match ndim"),
    ],
)
def test_mviewbuf_rejects_invalid_extents_inputs(args, match):
    from numba_cuda_mlir import _mviewbuf

    with pytest.raises(ValueError, match=match):
        _mviewbuf.memoryview_get_extents_info(*args)


@pytest.mark.parametrize(
    "args, expected",
    [
        (((), (), 0, 8), (0, 8)),
        (((3,), (4,), 1, 4), (0, 12)),
        (((3,), (-4,), 1, 4), (-8, 4)),
        (((3, 2), (8, 4), 2, 4), (0, 24)),
    ],
)
def test_mviewbuf_gets_extents_info(args, expected):
    from numba_cuda_mlir import _mviewbuf

    assert _mviewbuf.memoryview_get_extents_info(*args) == expected


def test_mviewbuf_concurrent_invalid_inputs_smoke():
    from numba_cuda_mlir import _mviewbuf

    cases = [
        (((3,), (4,), -1, 4), "ndim is negative"),
        (((3,), (4,), 1, 0), "itemsize <= 0"),
        (((3,), (4,), 2, 4), "shape length does not match ndim"),
        (((3, 4, 5), (16, 4), 2, 4), "shape length does not match ndim"),
        (((3, 4), (16,), 2, 4), "strides length does not match ndim"),
        (((3, 4), (16, 4, 1), 2, 4), "strides length does not match ndim"),
    ]

    def check_invalid_inputs(rounds):
        for _ in range(rounds):
            for args, match in cases:
                with pytest.raises(ValueError, match=match):
                    _mviewbuf.memoryview_get_extents_info(*args)

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(check_invalid_inputs, 200) for _ in range(16)]
        for future in futures:
            future.result()


def test_typeconv_concurrent_access_smoke():
    _require_free_threaded_python()

    from numba_cuda_mlir import _typeconv
    from numba_cuda_mlir.numba_cuda import types

    tm = _typeconv.new_type_manager()
    pairs = [
        (types.int32._code, types.int64._code, ord("p"), "promote"),
        (types.int32._code, types.float64._code, ord("s"), "safe"),
        (types.float64._code, types.int32._code, ord("u"), "unsafe"),
    ]

    def writer(rounds):
        for _ in range(rounds):
            for from_code, to_code, compat_code, _ in pairs:
                _typeconv.set_compatible(tm, from_code, to_code, compat_code)

    def reader(rounds):
        for _ in range(rounds):
            for from_code, to_code, _, expected in pairs:
                assert _typeconv.check_compatible(tm, from_code, to_code) in (
                    None,
                    expected,
                )

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(writer, 200) for _ in range(4)]
        futures.extend(executor.submit(reader, 400) for _ in range(12))
        for future in futures:
            future.result()

    for from_code, to_code, _, expected in pairs:
        assert _typeconv.check_compatible(tm, from_code, to_code) == expected


def test_arg_marshaller_recreates_missing_launch_lock(monkeypatch):
    _require_free_threaded_python()

    from numba_cuda_mlir.descriptor import _ArgMarshaller

    class Dispatcher:
        def __init__(self):
            self.ensure_count = 0

        def _ensure_dispatcher_state(self):
            self.ensure_count += 1
            if not hasattr(self, "_launch_lock"):
                self._launch_lock = threading.RLock()
            if not hasattr(self, "_launch_config_lock"):
                self._launch_config_lock = threading.RLock()

    dispatcher = Dispatcher()
    marshaller = _ArgMarshaller(lambda: None, dispatcher=dispatcher)

    def call_impl(*args):
        with dispatcher._launch_config_lock:
            assert dispatcher._launch_lock._is_owned()
        return "ok"

    monkeypatch.setattr(marshaller, "_call_impl", call_impl)

    assert marshaller() == "ok"
    assert dispatcher.ensure_count == 1


def test_dispatcher_state_bootstrap_lock_converges(monkeypatch):
    from numba_cuda_mlir import descriptor

    dispatcher = object.__new__(descriptor.MLIRDispatcher)
    real_rlock = threading.RLock
    created_locks = []
    created_locks_guard = threading.Lock()

    def tracking_rlock():
        lock = real_rlock()
        with created_locks_guard:
            created_locks.append(lock)
        time.sleep(0.01)
        return lock

    monkeypatch.setattr(descriptor, "_new_dispatcher_rlock", tracking_rlock)

    start = threading.Barrier(16)

    def bootstrap():
        start.wait()
        descriptor.MLIRDispatcher._ensure_dispatcher_state(dispatcher)
        return dispatcher._launch_config_lock, dispatcher._launch_lock

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(bootstrap) for _ in range(16)]
        states = [future.result() for future in futures]

    assert {id(launch_config_lock) for launch_config_lock, _ in states} == {
        id(dispatcher._launch_config_lock)
    }
    assert {id(launch_lock) for _, launch_lock in states} == {id(dispatcher._launch_lock)}
    expected_lock_count = 2 if descriptor._PY_GIL_DISABLED else 1
    assert len(created_locks) == expected_lock_count


def test_compile_and_recompile_take_launch_lock(monkeypatch):
    from numba_cuda_mlir import descriptor

    dispatcher = object.__new__(descriptor.MLIRDispatcher)
    descriptor.MLIRDispatcher._ensure_dispatcher_state(dispatcher)
    launch_lock = threading.RLock()
    dispatcher._launch_lock = launch_lock
    calls = []

    def compile_public(sig, abi_info=None, output=None):
        assert launch_lock._is_owned()
        calls.append(("compile", sig, abi_info, output))
        return "compiled"

    def recompile_impl():
        assert launch_lock._is_owned()
        calls.append(("recompile",))
        return "recompiled"

    monkeypatch.setattr(dispatcher, "_compile_public", compile_public)
    monkeypatch.setattr(dispatcher, "_recompile_impl", recompile_impl)

    assert dispatcher.compile("sig", abi_info="abi", output="out") == "compiled"
    assert dispatcher.recompile() == "recompiled"
    assert calls == [
        ("compile", "sig", "abi", "out"),
        ("recompile",),
    ]


def test_nvvm_ir_version_adaptation_is_thread_safe():
    _require_free_threaded_python()
    if os.name == "nt":
        pytest.skip("the Windows ModernBridge path has a native smoke test")

    from numba_cuda_mlir import _cext
    from numba_cuda_mlir._mlir import ir
    from numba_cuda_mlir._mlir.dialects import gpu, llvm
    from numba_cuda_mlir.lowering_utilities.llvm_utils import (
        LLVM_C_LIB_PATH,
        translate_to_llvmir,
    )

    # Importing the dialect modules registers the operations used by the parser.
    _ = gpu, llvm

    module_text = """
    module {
      gpu.module @kernels attributes {
        llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64-S128",
        llvm.target_triple = "nvptx64-nvidia-cuda"
      } {
        llvm.func @simple_kernel() attributes {gpu.kernel} {
          llvm.return
        }
      }
    }
    """

    def make_llvm_module():
        with ir.Context():
            module = ir.Module.parse(module_text)
            return translate_to_llvmir(module.body.operations[0])

    worker_count = 32
    versions = [
        (1000 + worker, 1100 + worker, 1200 + worker, 1300 + worker)
        for worker in range(worker_count)
    ]
    baselines = []
    for version in versions:
        llvm_mod, llvm_ctx = make_llvm_module()
        baselines.append(
            _cext.downgrade_for_libnvvm(
                llvm_mod,
                llvm_ctx,
                13,
                0,
                *version,
                LLVM_C_LIB_PATH,
            )
        )

    modules = [make_llvm_module() for _ in range(worker_count)]
    start = threading.Barrier(worker_count)

    def adapt(worker):
        version = versions[worker]
        llvm_mod, llvm_ctx = modules[worker]
        start.wait()
        result = _cext.downgrade_for_libnvvm(
            llvm_mod,
            llvm_ctx,
            13,
            0,
            *version,
            LLVM_C_LIB_PATH,
        )
        assert result == baselines[worker]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(adapt, worker) for worker in range(worker_count)]
        for future in futures:
            future.result()


def test_cuda_dispatch_with_gc_stress():
    _require_free_threaded_python()
    if os.environ.get("NUMBA_CUDA_MLIR_FT_STRESS") != "1":
        pytest.skip("set NUMBA_CUDA_MLIR_FT_STRESS=1 to run CUDA free-threading stress")

    result = _run_python(
        """
        from concurrent.futures import ThreadPoolExecutor
        import gc
        import os
        import warnings

        import numpy as np

        warnings.filterwarnings(
            "ignore",
            message="The CUDA driver version is older than the backend version.*",
            category=RuntimeWarning,
        )

        from numba_cuda_mlir import cuda
        from numba_cuda_mlir.cuda.experimental import consteval, current_target_options

        if not cuda.is_available():
            print("cuda-unavailable")
            raise SystemExit(0)

        launch_workers = int(os.environ.get("NUMBA_CUDA_MLIR_FT_LAUNCH_WORKERS", "24"))
        launch_iters = int(os.environ.get("NUMBA_CUDA_MLIR_FT_LAUNCH_ITERS", "16"))
        gc_workers = int(os.environ.get("NUMBA_CUDA_MLIR_FT_GC_WORKERS", "8"))
        gc_iters = int(os.environ.get("NUMBA_CUDA_MLIR_FT_GC_ITERS", "500"))
        recompile_workers = int(os.environ.get("NUMBA_CUDA_MLIR_FT_RECOMPILE_WORKERS", "1"))
        recompile_iters = int(os.environ.get("NUMBA_CUDA_MLIR_FT_RECOMPILE_ITERS", "4"))

        class LaunchConfigExtension:
            uses_launch_config = True

            def prepare_args(self, ty, val, stream=None, retr=None):
                return ty, val

        @cuda.jit(extensions=[LaunchConfigExtension()])
        def kernel(out, value):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = value + consteval(
                    current_target_options()["__launch_config__"]["block"][0]
                )

        outputs = [
            cuda.to_device(np.zeros(64, dtype=np.int32))
            for _ in range(launch_workers)
        ]
        kernel[1, 32](outputs[0], 1)
        assert outputs[0].copy_to_host()[0] == 33

        def launch(worker):
            out = outputs[worker]
            for i in range(launch_iters):
                block = 32 if i % 2 == 0 else 64
                kernel[1, block](out, worker)
                if i % 4 == 0:
                    host = out.copy_to_host()
                    assert host[0] in (worker + 32, worker + 64)

        def collect():
            for _ in range(gc_iters):
                gc.collect()

        def recompile():
            for _ in range(recompile_iters):
                kernel.recompile()

        with ThreadPoolExecutor(
            max_workers=launch_workers + gc_workers + recompile_workers
        ) as executor:
            futures = [executor.submit(launch, i) for i in range(launch_workers)]
            futures.extend(executor.submit(collect) for _ in range(gc_workers))
            futures.extend(executor.submit(recompile) for _ in range(recompile_workers))
            for future in futures:
                future.result()

        print("dispatch-stress-ok")
        """,
        env={"PYTHON_GIL": "0"},
        timeout=180,
    )

    assert result.returncode == 0, result.stderr
    if "cuda-unavailable" in result.stdout:
        pytest.skip("CUDA GPU required")
    assert "dispatch-stress-ok" in result.stdout
