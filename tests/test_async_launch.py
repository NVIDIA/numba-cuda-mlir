# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from numba_cuda_mlir import cuda
from numba_cuda_mlir.compiler import compile_result
from numba_cuda_mlir.numba_cuda import types


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


def test_raise_free_kernel_skips_error_check():
    sig = types.void(types.float32[::1])
    res = compile_result(_busy_kernel_func, sig)
    assert res.cres.metadata["check_error_code"] is False


def test_raising_kernel_keeps_error_check():
    sig = types.void(types.int32[::1], types.int64)
    res = compile_result(_raising_kernel_func, sig)
    assert res.cres.metadata["check_error_code"] is True


def test_raise_free_kernel_launch_is_asynchronous():
    # An event recorded on the launch stream immediately after the
    # launch call reads as pending; it completes after synchronization.
    busy_kernel = cuda.jit(_busy_kernel_func)
    stream = cuda.stream()
    out = cuda.device_array(1, dtype=np.float32)
    busy_kernel[1, 1, stream](out)  # compile + first run
    cuda.synchronize()

    event = cuda.event()
    busy_kernel[1, 1, stream](out)
    event.record(stream)
    assert not event.query()
    cuda.synchronize()
    assert event.query()


def test_raising_kernel_still_raises_at_launch():
    raising_kernel = cuda.jit(_raising_kernel_func)
    out = cuda.device_array(1, dtype=np.int32)

    raising_kernel[1, 1](out, 0)
    assert out.copy_to_host()[0] == 10

    with pytest.raises(IndexError, match="out of bounds"):
        raising_kernel[1, 1](out, 5)
