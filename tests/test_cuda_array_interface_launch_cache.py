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

    monkeypatch.setattr(cuda_api, "from_cuda_array_interface", forbidden_from_cuda_array_interface)

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


class _StorageView:
    def __init__(self, owner, host_view, offset, *, implicit_strides=False):
        self.owner = owner
        self.__cuda_array_interface__ = {
            "version": 3,
            "shape": host_view.shape,
            "typestr": host_view.dtype.str,
            "data": (owner.device_ctypes_pointer.value + offset, False),
            "strides": None if implicit_strides else host_view.strides,
        }


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_same_wrapper_dispatches_typed_and_byte_storage(dtype):
    @cuda.jit
    def read_write(ary, out):
        i = cuda.grid(1)
        if i < ary.size:
            out[i] = ary[i]
            ary[i] += 1

    dtype = np.dtype(dtype)
    configured = read_write[1, 32]
    # Revisit both signatures after the byte-backed launch. Complex pointers
    # need component alignment, but their strides still need whole elements.
    cases = [
        (0, dtype.itemsize, True),
        (0, dtype.itemsize + 4, False),
        (1, dtype.itemsize, True),
        (0, -dtype.itemsize, False),
        (0, dtype.itemsize, True),
        (0, dtype.itemsize + 4, False),
    ]
    for pointer_offset, stride, implicit in cases:
        offset = pointer_offset + (5 * abs(stride) if stride < 0 else 0)
        raw = np.full(128, 0xA5, dtype=np.uint8)
        host_view = np.ndarray((6,), dtype=dtype, buffer=raw, offset=offset, strides=(stride,))
        values = np.arange(6, dtype=np.float64) + 10
        if dtype.kind == "c":
            values = values + (np.arange(6) + 20) * 1j
        host_view[:] = values
        device_raw = cuda.to_device(raw)
        view = _StorageView(device_raw, host_view, offset, implicit_strides=implicit)
        out = cuda.device_array(6, dtype=dtype)
        configured(view, out)
        np.testing.assert_array_equal(out.copy_to_host(), values)
        host_view[:] = values + 1
        # Exact byte comparison checks writes and untouched inter-element bytes.
        np.testing.assert_array_equal(device_raw.copy_to_host(), raw)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("protocol", ["cai", "dlpack"])
def test_array_protocol_signature_cache_tracks_effective_pointer_alignment(dtype, protocol):
    cp = pytest.importorskip("cupy")
    import numba_cuda_mlir.type_defs.cupy_types  # noqa: F401

    class DlpackView:
        def __init__(self, array):
            self.array = array

        def __dlpack__(self, *, stream=None):
            return self.array.__dlpack__(stream=stream)

        def __dlpack_device__(self):
            return self.array.__dlpack_device__()

    @cuda.jit
    def read(ary, out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = ary[i]

    configured = read[1, 32]
    dtype = np.dtype(dtype)
    for offset in (0, 1, 0, 1):
        raw = np.zeros(128, dtype=np.uint8)
        expected = np.arange(6, dtype=np.float64) + 10
        if dtype.kind == "c":
            expected = expected + (np.arange(6) + 20) * 1j
        host_view = np.ndarray((6,), dtype=dtype, buffer=raw, offset=offset)
        host_view[:] = expected
        device_raw = cp.asarray(raw)
        array = cp.ndarray((6,), dtype=dtype, memptr=device_raw.data + offset)
        out = cuda.device_array(6, dtype=dtype)
        argument = array if protocol == "cai" else DlpackView(array)
        configured(argument, out)
        np.testing.assert_array_equal(out.copy_to_host(), expected)


@pytest.mark.parametrize("dtype", ["S0", "U0", np.dtype([])])
def test_zero_itemsize_interface_uses_byte_storage(dtype):
    @cuda.jit
    def metadata(ary, out):
        out[0] = ary.size
        out[1] = ary.strides[0]

    raw = np.zeros(1, dtype=np.uint8)
    host_view = np.ndarray((6,), dtype=dtype, buffer=raw, strides=(0,))
    device_raw = cuda.to_device(raw)
    view = _StorageView(device_raw, host_view, 0, implicit_strides=True)
    view.dtype = host_view.dtype
    out = cuda.device_array(2, dtype=np.int64)
    metadata[1, 1](view, out)
    np.testing.assert_array_equal(out.copy_to_host(), [6, 0])


@pytest.mark.parametrize("implicit_strides", [True, False])
def test_void_interface_launch_normalizes_dtype_like_owner(implicit_strides):
    @cuda.jit
    def copy(ary, out):
        i = cuda.grid(1)
        if i < ary.size:
            out[i] = ary[i]

    host = np.array([1.5, 2.5], dtype=np.float64)
    device = cuda.to_device(host)
    out = cuda.device_array(2, dtype=np.float64)
    configured = copy[1, 32]
    for owner_dtype in ("float64", np.float64, np.dtype("float64"), "float64"):
        view = _StorageView(device, host, 0, implicit_strides=implicit_strides)
        view.__cuda_array_interface__["typestr"] = "|V8"
        view.dtype = owner_dtype
        configured(view, out)
        np.testing.assert_array_equal(out.copy_to_host(), host)


def test_void_interface_cache_tracks_owner_record_field_offsets():
    @cuda.jit
    def read_field(ary, out):
        out[0] = ary[0].x

    configured = read_field[1, 1]
    out = cuda.device_array(1, dtype=np.float64)
    raw = np.array([1.5, 2.5], dtype=np.float64)
    device = cuda.to_device(raw)
    for offset in (0, 8, 0, 8):
        dtype = np.dtype(
            {"names": ["x"], "formats": [np.float64], "offsets": [offset], "itemsize": 16}
        )
        host = raw.view(dtype)
        view = _StorageView(device, host, 0, implicit_strides=True)
        view.dtype = dtype
        configured(view, out)
        np.testing.assert_array_equal(out.copy_to_host(), [raw[offset // 8]])


@pytest.mark.parametrize("implicit_strides", [True, False])
def test_bfloat_interface_launch_cache_preserves_metadata(implicit_strides):
    import ml_dtypes

    @cuda.jit
    def metadata(ary, out):
        out[0] = ary.size
        out[1] = ary.strides[0]

    host = np.array([1.5, 2.5], dtype=ml_dtypes.bfloat16)
    device = cuda.to_device(host.view(np.uint8))
    out = cuda.device_array(2, dtype=np.int64)
    configured = metadata[1, 1]
    for size in (2, 1, 2):
        view = _StorageView(device, host[:size], 0, implicit_strides=implicit_strides)
        view.dtype = host.dtype
        configured(view, out)
        np.testing.assert_array_equal(out.copy_to_host(), [size, 2])
