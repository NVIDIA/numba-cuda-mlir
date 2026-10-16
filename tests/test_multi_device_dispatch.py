# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small runtime regressions for a dispatcher shared across CUDA devices."""

import numpy as np
import pytest

from numba_cuda_mlir import cuda
from numba_cuda_mlir.cuda.experimental import consteval, current_target_options


def test_warm_launch_does_not_resolve_context(monkeypatch):
    from numba_cuda_mlir.numba_cuda.cudadrv import devices

    @cuda.jit
    def write(out):
        out[0] = 42

    configured = write[1, 1]
    out = cuda.device_array(1, np.int32)
    configured(out)
    with monkeypatch.context() as patch:
        patch.setattr(
            devices, "get_context", lambda: pytest.fail("warm launch resolved the context")
        )
        for _ in range(5):
            configured(out)
    cuda.synchronize()
    assert out.copy_to_host()[0] == 42


def test_warm_launch_resolves_only_after_external_context_switch(monkeypatch):
    from cuda.bindings import driver

    from numba_cuda_mlir.numba_cuda.cudadrv import devices

    if len(cuda.gpus) < 2:
        pytest.skip("requires two CUDA devices")

    @cuda.jit
    def write(out, value):
        out[0] = value

    configured = write[1, 1]
    contexts, arrays = [], []
    for index in (0, 1):
        with cuda.gpus[index]:
            contexts.append(cuda.current_context())
            arrays.append(cuda.device_array(1, np.int32))
            configured(arrays[-1], index)
            cuda.synchronize()

    original_get_context = devices.get_context
    resolutions = []

    def get_context(*args, **kwargs):
        resolutions.append(None)
        return original_get_context(*args, **kwargs)

    monkeypatch.setattr(devices, "get_context", get_context)
    ok, original_context = driver.cuCtxGetCurrent()
    assert ok == driver.CUresult.CUDA_SUCCESS
    try:
        for index in (0, 1, 0):
            # Bypass every Numba wrapper, as an external CUDA library can do.
            assert driver.cuCtxSetCurrent(contexts[index].handle)[0] == ok
            before = len(resolutions)
            configured(arrays[index], index + 40)
            assert len(resolutions) == before + 1
            for _ in range(5):
                configured(arrays[index], index + 41)
            assert len(resolutions) == before + 1
            assert driver.cuCtxSynchronize()[0] == ok
            assert arrays[index].copy_to_host()[0] == index + 41
    finally:
        assert driver.cuCtxSetCurrent(original_context)[0] == ok


@pytest.mark.parametrize("use_nrt", [False, True])
def test_retained_configured_launch_across_devices(use_nrt):
    if len(cuda.gpus) < 2:
        pytest.skip("requires two CUDA devices")

    @cuda.jit
    def write(out, value):
        out[0] = value + 1

    @cuda.jit
    def write_nrt(out, value):
        scratch = np.empty(1, np.int32)
        scratch[0] = value
        out[0] = scratch[0] + 1

    kernel = write_nrt if use_nrt else write
    configured = kernel[1, 1]
    results = []
    # Four bytes and one thread per device. Retain the configured callable
    # and return to the first GPU, exercising both native miss and hit paths.
    for index in (0, 1, 0):
        with cuda.gpus[index]:
            out = cuda.device_array(1, np.int32)
            configured(out, np.int32(index + 40))
            cuda.synchronize()
            assert out.copy_to_host()[0] == index + 41
            state = kernel._get_context_dispatcher()
            metadata = next(iter(state.overloads.values())).metadata
            assert (
                metadata["gpu_target"]["host_cc"]
                == cuda.current_context().device.compute_capability
            )
            results.append(state._c)
    assert results[0] is results[2]
    assert results[0] is not results[1]
    assert "chip" not in kernel.targetoptions or kernel.targetoptions["chip"] is None


def test_frozen_retained_launch_config_across_devices():
    if len(cuda.gpus) < 2:
        pytest.skip("requires two CUDA devices")

    class Extension:
        uses_launch_config = True

        def prepare_args(self, ty, val, stream=None, retr=None):
            return ty, val

    @cuda.jit(extensions=[Extension()])
    def write(out):
        out[0] = consteval(current_target_options()["__launch_config__"]["block"][0])

    configured = write[1, 32]
    write.extensions.clear()
    for index in (0, 1, 0):
        with cuda.gpus[index]:
            out = cuda.device_array(1, np.int32)
            configured(out)
            cuda.synchronize()
            assert out.copy_to_host()[0] == 32
            write.disable_compile()


@pytest.mark.parametrize("warm_cache", [False, True])
def test_managed_argument_from_another_device_preserves_selected_context(warm_cache):
    if len(cuda.gpus) < 2:
        pytest.skip("requires two CUDA devices")

    @cuda.jit
    def write(out):
        out[0] = 42

    configured = write[1, 1]
    with cuda.gpus[1]:
        if not cuda.current_context().device.MANAGED_MEMORY:
            pytest.skip("requires managed memory on both devices")
        foreign = cuda.managed_array(1, np.int32, attach_global=True)
        foreign[0] = 0
    with cuda.gpus[0]:
        selected = cuda.current_context()
        if not selected.device.MANAGED_MEMORY:
            pytest.skip("requires managed memory on both devices")
        local = cuda.managed_array(1, np.int32, attach_global=True)
        if warm_cache:
            configured(local)
        configured(foreign)
        cuda.synchronize()
        assert foreign[0] == 42
        assert cuda.current_context() is selected
        state = write._get_context_dispatcher()
        configured(local)
        cuda.synchronize()
        assert local[0] == 42
        assert write._get_context_dispatcher() is state
        assert (
            next(iter(write.overloads.values())).metadata["gpu_target"]["host_cc"]
            == cuda.current_context().device.compute_capability
        )
