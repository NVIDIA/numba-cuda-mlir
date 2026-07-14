# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from numba_cuda_mlir import cuda
from numba_cuda_mlir.compiler import compile_result
from numba_cuda_mlir.numba_cuda import types

# Bounds the spin kernel at a few seconds of GPU time so a synchronous
# launch cannot hang the test, while leaving orders of magnitude more
# headroom than the host needs to record and query the event.
_SPIN_CAP = 50_000_000


def _busy_kernel_func(out):
    # Serial multiply-add dependency chain seeded from runtime data, so
    # no optimizer can fold the loop away.
    x = out[0]
    for _ in range(20_000_000):
        x = x * 0.9999999 + 1.0
    out[0] = x


def _raising_kernel_func(out, idx):
    t = (10, 20, 30)
    out[0] = t[idx]


def _busy_raising_kernel_func(out, idx):
    x = out[0]
    for _ in range(20_000_000):
        x = x * 0.9999999 + 1.0
    t = (10, 20, 30)
    out[0] = x + t[idx]


def _spin_kernel_func(flag, out):
    # The atomic read cannot be hoisted out of the loop, so the kernel
    # keeps re-reading the flag until the host releases it.
    count = 0
    while count < _SPIN_CAP and cuda.atomic.add(flag, 0, 0) == 0:
        count += 1
    out[0] = count


def test_raise_free_kernel_skips_error_check():
    sig = types.void(types.float32[::1])
    res = compile_result(_busy_kernel_func, sig)
    assert res.cres.metadata["check_error_code"] is False


def test_raising_kernel_keeps_error_check():
    sig = types.void(types.int32[::1], types.int64)
    res = compile_result(_raising_kernel_func, sig)
    assert res.cres.metadata["check_error_code"] is True


def test_raise_free_kernel_launch_is_asynchronous():
    # The kernel spins until the host writes the release flag, so the
    # event recorded after the launch call can only read as pending if
    # the launch returned while the kernel was still running. The final
    # count assertion proves the kernel exited on the flag, not the cap.
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
    assert not event.query()
    flag.copy_to_device(released, stream=release_stream)
    cuda.synchronize()
    assert event.query()
    assert out.copy_to_host()[0] < _SPIN_CAP


def test_raising_kernel_launch_is_synchronous():
    # A kernel that can raise keeps the post-launch readback, so the
    # launch call blocks and the event is already complete when queried.
    # This also proves event.query() reports completed work, which the
    # pending read in the asynchronous test depends on.
    busy_raising_kernel = cuda.jit(_busy_raising_kernel_func)
    stream = cuda.stream()
    out = cuda.device_array(1, dtype=np.float32)
    busy_raising_kernel[1, 1, stream](out, 0)  # compile + first run
    cuda.synchronize()

    event = cuda.event()
    busy_raising_kernel[1, 1, stream](out, 0)
    event.record(stream)
    assert event.query()


def test_raising_kernel_still_raises_at_launch():
    raising_kernel = cuda.jit(_raising_kernel_func)
    out = cuda.device_array(1, dtype=np.int32)

    raising_kernel[1, 1](out, 0)
    assert out.copy_to_host()[0] == 10

    with pytest.raises(IndexError, match="out of bounds"):
        raising_kernel[1, 1](out, 5)
