# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from numba_cuda_mlir import cuda

# Caps the spin kernel so a synchronous-launch regression cannot hang
# the test.
_SPIN_CAP = 220_000_000


def _raising_kernel_func(out, idx):
    t = (10, 20, 30)
    out[0] = t[idx]


def _spin_kernel_func(flag, out):
    # The atomic read cannot be hoisted out of the loop, so the kernel
    # keeps re-reading the flag until the host releases it.
    count = 0
    while count < _SPIN_CAP and cuda.atomic.add(flag, 0, 0) == 0:
        count += 1
    out[0] = count


def test_default_kernel_launch_is_asynchronous():
    # The kernel spins until the host writes the release flag, so the
    # event can only read as pending if the launch call returned while
    # the kernel was still running.
    spin_kernel = cuda.jit(_spin_kernel_func)
    launch_stream = cuda.stream()
    release_stream = cuda.stream()
    released = np.ones(1, dtype=np.int32)
    held = np.zeros(1, dtype=np.int32)
    flag = cuda.to_device(released)
    out = cuda.device_array(1, dtype=np.int32)

    spin_kernel[1, 1, launch_stream](flag, out)  # compile + first run
    cuda.synchronize()

    flag.copy_to_device(held)
    cuda.synchronize()
    event = cuda.event()
    spin_kernel[1, 1, launch_stream](flag, out)
    event.record(launch_stream)
    assert not event.query(), (
        "event completed before the release flag was written: either the "
        "launch was synchronous, or the host stalled for the full spin cap "
        "between launch and query; the two cannot be told apart"
    )
    flag.copy_to_device(released, stream=release_stream)
    cuda.synchronize()
    assert event.query()
    assert out.copy_to_host()[0] < _SPIN_CAP, "kernel exited on the spin cap, not the release flag"


def test_debug_kernel_raises_at_launch():
    raising_kernel = cuda.jit(debug=True, opt=False)(_raising_kernel_func)
    out = cuda.device_array(1, dtype=np.int32)

    raising_kernel[1, 1](out, 0)
    assert out.copy_to_host()[0] == 10

    with pytest.raises(IndexError, match="out of bounds"):
        raising_kernel[1, 1](out, 5)


def test_default_kernel_does_not_raise():
    raising_kernel = cuda.jit(_raising_kernel_func)
    out = cuda.device_array(1, dtype=np.int32)

    raising_kernel[1, 1](out, 0)
    assert out.copy_to_host()[0] == 10

    raising_kernel[1, 1](out, 5)
    cuda.synchronize()
