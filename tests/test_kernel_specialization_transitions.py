# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
import sys
import sysconfig
import threading

import numpy as np
import pytest

from numba_cuda_mlir import cuda, descriptor
from numba_cuda_mlir.cuda.experimental import consteval, current_target_options

pytestmark = pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")


class _LaunchConfigExtension:
    uses_launch_config = True

    def prepare_args(self, ty, val, stream=None, retr=None):
        return ty, val


@pytest.mark.parametrize("transition", ["reenable-compilation", "compile-exact-overload"])
@pytest.mark.parametrize("scalar_only", [False, True], ids=["array-output", "native-fast"])
def test_frozen_compatible_selection_obeys_changed_policy(transition, scalar_only):
    output = cuda.device_array(1, dtype=np.float64)
    if scalar_only:

        @cuda.jit
        def kernel(x, y):
            output[0] = x + y

    else:

        @cuda.jit
        def kernel(output, x, y):
            output[0] = x + y

    launch = kernel[1, 1]
    prefix = () if scalar_only else (output,)
    launch(*prefix, np.float32(1), np.float32(2**-25))
    assert output.copy_to_host()[0] == 1.0
    kernel.disable_compile()

    # The only overload is float32, so frozen float64 calls must round here.
    args = (*prefix, np.float64(1), np.float64(2**-25))
    for _ in range(2):
        launch(*args)
        assert output.copy_to_host()[0] == 1.0

    if transition == "reenable-compilation":
        kernel.disable_compile(False)
    else:
        # Explicit compilation is permitted even while automatic compilation is frozen.
        kernel.compile_for(*args)

    # Neither the retained launcher nor a newly configured one may keep selecting
    # the previously compatible float32 overload after the policy changes.
    for launcher in (launch, kernel[1, 1], launch):
        launcher(*args)
        assert output.copy_to_host()[0] == 1.0 + 2**-25


def test_frozen_active_launch_config_survives_unused_configure_pressure(monkeypatch):
    cache_size = 4
    monkeypatch.setattr(descriptor, "_LAUNCH_CONFIG_CACHE_SIZE", cache_size)

    @cuda.jit(extensions=[_LaunchConfigExtension()])
    def kernel(output, value):
        output[0] = value + consteval(current_target_options()["__launch_config__"]["block"][0])

    output = cuda.device_array(1, dtype=np.int64)
    active = kernel[1, 1]
    active(output, np.int64(0))
    assert output.copy_to_host()[0] == 1
    kernel.disable_compile()

    for block_size in range(2, 3 * cache_size + 2):
        # Configure, but never launch, a distinct cold geometry between hot calls.
        kernel[2, block_size]
        active(output, np.int64(block_size))
        assert output.copy_to_host()[0] == block_size + 1


_recompile_value = 1


@pytest.fixture
def captured_global(monkeypatch):
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_recompile_value", 1)
    return module


@pytest.mark.skipif(
    sysconfig.get_config_var("Py_GIL_DISABLED") in (1, "1"),
    reason="free-threaded dispatch already serializes whole launches with recompile",
)
def test_recompile_cannot_publish_a_stale_cold_specialization(monkeypatch, captured_global):
    @cuda.jit
    def kernel(output):
        output[0] = _recompile_value

    output = cuda.device_array(1, dtype=np.int64)
    launch = kernel[1, 1]
    compiled = threading.Event()
    release_compile = threading.Event()
    recompile_requested = threading.Event()
    recompiling = threading.local()
    thread_timeout = 60
    original_compile = kernel._compile_impl
    original_acquire = descriptor.global_compiler_lock.acquire

    def paused_compile(*args, **kwargs):
        result = original_compile(*args, **kwargs)
        if not compiled.is_set():
            compiled.set()
            assert release_compile.wait(timeout=thread_timeout), "compiler gate was not released"
        return result

    def acquire_compiler_lock():
        if getattr(recompiling, "active", False):
            # Let the in-flight compile publish before recompile waits for its
            # lock. The real acquisition still determines the operation order.
            release_compile.set()
        return original_acquire()

    def cold_launch():
        launch(output)
        # Complete the initial GPU work before the subsequent launch is checked.
        return output.copy_to_host()[0]

    def recompile():
        recompiling.active = True
        recompile_requested.set()
        try:
            kernel.recompile()
        finally:
            # An implementation without compiler-lock serialization finishes
            # recompile first; only then may the stale compile result publish.
            release_compile.set()
            recompiling.active = False

    monkeypatch.setattr(kernel, "_compile_impl", paused_compile)
    monkeypatch.setattr(descriptor.global_compiler_lock, "acquire", acquire_compiler_lock)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(cold_launch)
        try:
            assert compiled.wait(timeout=thread_timeout), "cold compilation did not finish"
            captured_global._recompile_value = 2
            refreshed = executor.submit(recompile)
            assert recompile_requested.wait(timeout=thread_timeout), "recompile was not requested"
            # The gate opens either on the real lock attempt or after an
            # unserialized recompile, never by waiting for a blocked recompile.
            first.result(timeout=thread_timeout)
            refreshed.result(timeout=thread_timeout)
        finally:
            release_compile.set()

    # The first launch may precede recompile. After both operations complete,
    # neither a retained nor a fresh launcher may execute the old captured value.
    for launcher in (launch, kernel[1, 1], launch):
        launcher(output)
        assert output.copy_to_host()[0] == 2
