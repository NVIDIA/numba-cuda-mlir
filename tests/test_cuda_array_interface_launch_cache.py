# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from numba_cuda_mlir import cuda
from numba_cuda_mlir.descriptor import _ArgMarshaller


class FakeCudaArray:
    def __init__(self, *, readonly=False, mask=None):
        data = (0, readonly)
        self.__cuda_array_interface__ = {
            "version": 1,
            "shape": (4,),
            "typestr": "<f4",
            "data": data,
            "strides": None,
            "stream": None,
        }
        if mask is not None:
            self.__cuda_array_interface__["mask"] = mask


def test_cuda_array_interface_launch_rejects_masked_arrays():
    configured = _ArgMarshaller(lambda *args: None)

    with pytest.raises(NotImplementedError, match="Masked arrays are not supported"):
        configured(FakeCudaArray(mask=object()))


def test_cuda_array_interface_signature_cache_tracks_readonly():
    configured = _ArgMarshaller(lambda *args: None)
    mutable = FakeCudaArray(readonly=False)
    readonly = FakeCudaArray(readonly=True)

    configured(mutable)
    type_key = (FakeCudaArray,)
    assert len(configured._array_sig_cache[type_key]) == 1

    configured(readonly)
    assert len(configured._array_sig_cache[type_key]) == 2


def test_cupy_launch_does_not_reconstruct_device_arrays(monkeypatch):
    cp = pytest.importorskip("cupy")
    import numba_cuda_mlir.numba_cuda.api as cuda_api
    import numba_cuda_mlir.type_defs.cupy_types  # noqa: F401

    calls = []

    def forbidden_from_cuda_array_interface(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("launch path should pass CUDA Array Interface objects through")

    monkeypatch.setattr(
        cuda_api, "from_cuda_array_interface", forbidden_from_cuda_array_interface
    )

    @cuda.jit
    def add(a, b, out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = a[i] + b[i]

    a = cp.arange(32, dtype=cp.float32)
    b = cp.ones(32, dtype=cp.float32)
    out = cp.zeros(32, dtype=cp.float32)

    configured = add[1, 32]
    configured(a, b, out)
    configured(a, b, out)

    np.testing.assert_allclose(cp.asnumpy(out), cp.asnumpy(a + b))
    assert calls == []


def test_cupy_array_signature_cache_tracks_dtype_and_ndim():
    cp = pytest.importorskip("cupy")
    import numba_cuda_mlir.type_defs.cupy_types  # noqa: F401

    @cuda.jit
    def add(a, b, out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = a[i] + b[i]

    configured = add[1, 32]

    a32 = cp.arange(32, dtype=cp.float32)
    b32 = cp.ones(32, dtype=cp.float32)
    out32 = cp.zeros(32, dtype=cp.float32)
    configured(a32, b32, out32)

    type_key = (type(a32), type(b32), type(out32))
    assert len(configured._array_sig_cache[type_key]) == 1

    a32_b = cp.arange(64, dtype=cp.float32)
    b32_b = cp.ones(64, dtype=cp.float32)
    out32_b = cp.zeros(64, dtype=cp.float32)
    configured(a32_b, b32_b, out32_b)
    assert len(configured._array_sig_cache[type_key]) == 1

    a64 = cp.arange(32, dtype=cp.float64)
    b64 = cp.ones(32, dtype=cp.float64)
    out64 = cp.zeros(32, dtype=cp.float64)
    configured(a64, b64, out64)

    assert len(configured._array_sig_cache[type_key]) == 2
    np.testing.assert_allclose(cp.asnumpy(out64), cp.asnumpy(a64 + b64))
