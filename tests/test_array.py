# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from numba_cuda_mlir import cuda
import pytest

ARRAY_LIKE_FUNCTIONS = (
    cuda.device_array_like,
    cuda.mapped_array_like,
    cuda.pinned_array_like,
)


def on_array_like(f):
    return pytest.mark.parametrize("like_func", ARRAY_LIKE_FUNCTIONS)(f)


def array_reshape1d(arr, newshape, got):
    y = arr.reshape(newshape)
    for i in range(y.shape[0]):
        got[i] = y[i]


def array_reshape2d(arr, newshape, got):
    y = arr.reshape(newshape)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            got[i, j] = y[i, j]


def array_reshape3d(arr, newshape, got):
    y = arr.reshape(newshape)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            for k in range(y.shape[2]):
                got[i, j, k] = y[i, j, k]


def array_reshape(arr, newshape):
    return arr.reshape(newshape)


def test_gpu_array_zero_length():
    x = np.arange(0)
    dx = cuda.to_device(x)
    hx = dx.copy_to_host()
    assert x.shape == dx.shape
    assert x.size == dx.size
    assert x.shape == hx.shape
    assert x.size == hx.size


def test_null_shape():
    null_shape = ()
    shape1 = cuda.device_array(()).shape
    shape2 = cuda.device_array_like(np.ndarray(())).shape
    assert shape1 == null_shape
    assert shape2 == null_shape


def test_gpu_array_strided():
    @cuda.jit("void(double[:])")
    def kernel(x):
        i = cuda.grid(1)
        if i < x.shape[0]:
            x[i] = i

    x = np.arange(10, dtype=np.double)
    y = np.ndarray(shape=10 * 8, buffer=x, dtype=np.byte)
    z = np.ndarray(9, buffer=y[4:-4], dtype=np.double)
    dev_z = cuda.to_device(z)
    kernel[10, 10](dev_z)
    z = dev_z.copy_to_host()
    assert np.allclose(z, list(range(9)))


def test_gpu_array_interleaved():
    @cuda.jit("void(double[:], double[:])")
    def copykernel(x, y):
        i = cuda.grid(1)
        if i < x.shape[0]:
            x[i] = i
            y[i] = i

    x = np.arange(10, dtype=np.double)
    y = x[:-1:2]
    # z = x[1::2]
    # n = y.size
    try:
        cuda.devicearray.auto_device(y)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Should raise exception complaining the contiguous-ness of the array."
        )
        # Should we handle this use case?
        # assert z.size == y.size
        # copykernel[1, n](y, x)
        # print(y, z)
        # assert np.all(y == z)
        # assert np.all(y == list(range(n)))


def test_auto_device_const():
    d, _ = cuda.devicearray.auto_device(2)
    assert np.all(d.copy_to_host() == np.array(2))


def _test_array_like_same(like_func, array):
    """
    Tests of *_array_like where shape, strides, dtype, and flags should
    all be equal.
    """
    array_like = like_func(array)
    assert array.shape == array_like.shape
    assert array.strides == array_like.strides
    assert array.dtype == array_like.dtype
    assert array.flags["C_CONTIGUOUS"] == array_like.flags["C_CONTIGUOUS"]
    assert array.flags["F_CONTIGUOUS"] == array_like.flags["F_CONTIGUOUS"]


@on_array_like
def test_array_like_1d(like_func):
    d_a = cuda.device_array(10, order="C")
    _test_array_like_same(like_func, d_a)


@on_array_like
def test_array_like_2d(like_func):
    d_a = cuda.device_array((10, 12), order="C")
    _test_array_like_same(like_func, d_a)


@on_array_like
def test_array_like_2d_transpose(like_func):
    d_a = cuda.device_array((10, 12), order="C")
    _test_array_like_same(like_func, d_a)


@on_array_like
def test_array_like_3d(like_func):
    d_a = cuda.device_array((10, 12, 14), order="C")
    _test_array_like_same(like_func, d_a)


@on_array_like
def test_array_like_1d_f(like_func):
    d_a = cuda.device_array(10, order="F")
    _test_array_like_same(like_func, d_a)


@on_array_like
def test_array_like_2d_f(like_func):
    d_a = cuda.device_array((10, 12), order="F")
    _test_array_like_same(like_func, d_a)


@on_array_like
def test_array_like_2d_f_transpose(like_func):
    d_a = cuda.device_array((10, 12), order="F")
    _test_array_like_same(like_func, d_a)


@on_array_like
def test_array_like_3d_f(like_func):
    d_a = cuda.device_array((10, 12, 14), order="F")
    _test_array_like_same(like_func, d_a)


def _test_array_like_view(like_func, view, d_view):
    """
    Tests of device_array_like where the original array is a view - the
    strides should not be equal because a contiguous array is expected.
    """
    nb_like = like_func(d_view)
    assert d_view.shape == nb_like.shape
    assert d_view.dtype == nb_like.dtype

    # Use NumPy as a reference for the expected strides
    np_like = np.zeros_like(view)
    assert nb_like.strides == np_like.strides
    assert nb_like.flags["C_CONTIGUOUS"] == np_like.flags["C_CONTIGUOUS"]
    assert nb_like.flags["F_CONTIGUOUS"] == np_like.flags["F_CONTIGUOUS"]


@on_array_like
def test_array_like_1d_view(like_func):
    shape = 10
    view = np.zeros(shape)[::2]
    d_view = cuda.device_array(shape)[::2]
    _test_array_like_view(like_func, view, d_view)


@on_array_like
def test_array_like_1d_view_f(like_func):
    shape = 10
    view = np.zeros(shape, order="F")[::2]
    d_view = cuda.device_array(shape, order="F")[::2]
    _test_array_like_view(like_func, view, d_view)


@on_array_like
def test_array_like_2d_view(like_func):
    shape = (10, 12)
    view = np.zeros(shape)[::2, ::2]
    d_view = cuda.device_array(shape)[::2, ::2]
    _test_array_like_view(like_func, view, d_view)


@on_array_like
def test_array_like_2d_view_f(like_func):
    shape = (10, 12)
    view = np.zeros(shape, order="F")[::2, ::2]
    d_view = cuda.device_array(shape, order="F")[::2, ::2]
    _test_array_like_view(like_func, view, d_view)


@on_array_like
def test_array_like_2d_view_transpose_device(like_func):
    shape = (10, 12)
    d_view = cuda.device_array(shape)[::2, ::2].T
    # This is a special case (see issue #4974) because creating the
    # transpose creates a new contiguous allocation with different
    # strides.  In this case, rather than comparing against NumPy,
    # we can only compare against expected values.
    like = like_func(d_view)
    assert d_view.shape == like.shape
    assert d_view.dtype == like.dtype
    assert like.strides == (40, 8)
    assert like.flags["C_CONTIGUOUS"]
    assert not like.flags["F_CONTIGUOUS"]


@pytest.mark.skip("simulator not supported")
@on_array_like
def test_array_like_2d_view_transpose_simulator(like_func):
    shape = (10, 12)
    view = np.zeros(shape)[::2, ::2].T
    d_view = cuda.device_array(shape)[::2, ::2].T
    # On the simulator, the transpose has different strides to on a
    # CUDA device (See issue #4974). Here we can compare strides
    # against NumPy as a reference.
    np_like = np.zeros_like(view)
    _test_array_like_view(like_func, view, d_view)


@on_array_like
def test_array_like_2d_view_f_transpose(like_func):
    shape = (10, 12)
    view = np.zeros(shape, order="F")[::2, ::2].T
    d_view = cuda.device_array(shape, order="F")[::2, ::2].T
    _test_array_like_view(like_func, view, d_view)


@pytest.mark.skip("needs investigation")
@on_array_like
def test_issue_4628(like_func):
    # CUDA Device arrays were reported as always being typed with 'A' order
    # so launching the kernel with a host array and then a device array
    # resulted in two overloads being compiled - one for 'C' order from
    # the host array, and one for 'A' order from the device array. With the
    # resolution of this issue, the order of the device array is also 'C',
    # so after the kernel launches there should only be one overload of
    # the function.
    @cuda.jit
    def func(A, out):
        i = cuda.grid(1)
        out[i] = A[i] * 2

    n = 128
    a = np.ones((n,))
    d_a = cuda.to_device(a)
    result = np.zeros((n,))

    func[1, 128](a, result)
    func[1, 128](d_a, result)

    assert len(func.overloads) == 1


@pytest.mark.skip("NYI: dynamic reshape")
def test_array_reshape():
    def check(pyfunc, kernelfunc, arr, shape):
        shape = np.array(shape)
        kernel = cuda.jit(kernelfunc)
        expected = pyfunc(arr, shape)
        got = np.zeros(expected.shape, dtype=arr.dtype)
        dev_arr = cuda.to_device(arr)
        dev_shape = cuda.to_device(shape)
        dev_got = cuda.to_device(got)
        kernel[1, 1](dev_arr, dev_shape, dev_got)
        got = dev_got.copy_to_host()
        assert all(got == expected)

    def check_only_shape(kernelfunc, arr, shape, expected_shape):
        kernel = cuda.jit(kernelfunc)
        got = np.zeros(expected_shape, dtype=arr.dtype)
        kernel[1, 1](arr, shape, got)
        assert got.shape == expected_shape
        assert got.size == arr.size

    # 0-sized arrays
    def check_empty(arr):
        check(array_reshape, array_reshape1d, arr, 0)
        check(array_reshape, array_reshape1d, arr, (0,))
        check(array_reshape, array_reshape3d, arr, (1, 0, 2))

    # C-contiguous
    arr = np.arange(24)
    check(array_reshape, array_reshape1d, arr, (24,))
    check(array_reshape, array_reshape2d, arr, (4, 6))
    check(array_reshape, array_reshape2d, arr, (8, 3))
    check(array_reshape, array_reshape3d, arr, (8, 1, 3))

    arr = np.arange(24).reshape((1, 8, 1, 1, 3, 1))
    check(array_reshape, array_reshape1d, arr, (24,))
    check(array_reshape, array_reshape2d, arr, (4, 6))
    check(array_reshape, array_reshape2d, arr, (8, 3))
    check(array_reshape, array_reshape3d, arr, (8, 1, 3))

    # Test negative shape value
    arr = np.arange(25).reshape(5, 5)
    check(array_reshape, array_reshape1d, arr, -1)
    check(array_reshape, array_reshape1d, arr, (-1,))
    check(array_reshape, array_reshape2d, arr, (-1, 5))
    check(array_reshape, array_reshape3d, arr, (5, -1, 5))
    check(array_reshape, array_reshape3d, arr, (5, 5, -1))

    arr = np.array([])
    check_empty(arr)


class TestSliceSetitem:
    """Tests for slice setitem operations (arr[:] = value)."""

    def test_slice_setitem_1d(self):
        @cuda.jit
        def kernel(arr):
            arr[:] = 42

        arr = cuda.to_device(np.array([1, 2, 3, 4, 5], dtype=np.float64))
        kernel[1, 1](arr)
        result = arr.copy_to_host()
        np.testing.assert_array_equal(result, [42, 42, 42, 42, 42])

    def test_slice_setitem_2d(self):
        @cuda.jit
        def kernel(arr):
            arr[:] = 0

        arr = cuda.to_device(np.array([[1, 2], [3, 4]], dtype=np.float64))
        kernel[1, 1](arr)
        result = arr.copy_to_host()
        np.testing.assert_array_equal(result, [[0, 0], [0, 0]])

    def test_slice_setitem_local_array(self):
        """Test slice setitem on local arrays (as used in many_locals pattern)."""

        @cuda.jit
        def kernel(out):
            arr = cuda.local.array((2, 2), np.float64)
            arr[:] = 5
            out[0] = arr[0, 0]
            out[1] = arr[0, 1]
            out[2] = arr[1, 0]
            out[3] = arr[1, 1]

        out = cuda.to_device(np.zeros(4, dtype=np.float64))
        kernel[1, 1](out)
        result = out.copy_to_host()
        np.testing.assert_array_equal(result, [5, 5, 5, 5])


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    test_array_reshape()
