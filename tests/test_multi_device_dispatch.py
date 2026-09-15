# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small runtime regressions for a dispatcher shared across CUDA devices."""

import numpy as np
import pytest

from numba_cuda_mlir import cuda


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
