# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dispatchers and their native launch objects are freed once dropped."""

import gc
import weakref

import numpy as np

from numba_cuda_mlir import cuda
from numba_cuda_mlir import descriptor as descriptor_mod


def _make_kernel():
    @cuda.jit(device=True)
    def helper(x):
        return x + 1

    @cuda.jit
    def kern(a):
        a[0] = helper(a[0])

    return kern, helper


def test_dropped_kernels_are_collected_after_launch():
    a = cuda.to_device(np.zeros(1, dtype=np.int64))
    refs = []
    for _ in range(3):
        kern, helper = _make_kernel()
        kern[1, 1](a)
        refs += [weakref.ref(kern), weakref.ref(helper)]
        del kern, helper
    cuda.synchronize()
    assert a.copy_to_host()[0] == 3

    # The first collection frees the kernels, the second the device functions they called.
    gc.collect()
    gc.collect()

    assert all(ref() is None for ref in refs)


def test_dropped_kernel_with_user_array_class_is_collected():
    class Wrapped:
        def __init__(self, arr):
            self.arr = arr

        @property
        def __cuda_array_interface__(self):
            return self.arr.__cuda_array_interface__

    kern, helper = _make_kernel()
    Wrapped.kernel = kern
    a = cuda.to_device(np.zeros(1, dtype=np.int64))
    kern[1, 1](Wrapped(a))
    cuda.synchronize()
    assert a.copy_to_host()[0] == 1

    refs = [weakref.ref(kern), weakref.ref(Wrapped)]
    del kern, helper, Wrapped
    gc.collect()
    gc.collect()

    assert all(ref() is None for ref in refs)


def test_dropped_kernel_dispatcher_teardown_after_launch(monkeypatch):
    # Zero retention frees the launched native dispatcher on recompile().
    monkeypatch.setattr(descriptor_mod, "_OLD_DISPATCHER_RETAIN_LIMIT", 0)
    kern, _ = _make_kernel()
    a = cuda.to_device(np.zeros(1, dtype=np.int64))
    kern[1, 1](a)
    kern.recompile()
    gc.collect()
    kern[1, 1](a)
    cuda.synchronize()
    assert a.copy_to_host()[0] == 2
