# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
import pytest
import numpy as np

DeviceNDArray = cuda.DeviceNDArray


@pytest.mark.parametrize("op", [np.sum, np.mean])
@pytest.mark.parametrize(
    "array",
    [
        np.array([1, 2, 3, 4, 5]),
        np.array(range(9)).reshape(3, 3),
        np.array(range(27)).reshape(3, 3, 3),
    ],
)
def test_np_ops(op, array):
    @cuda.jit(dump=True)
    def kernel(x: DeviceNDArray, a: DeviceNDArray):
        x[0] = op(a)

    x = cuda.to_device(np.zeros(1, dtype=np.int32))
    a = cuda.to_device(array)
    kernel[1, 1, 0, 0](x, a)
    x = x.copy_to_host()
    print(x)
    answer = op(array)
    assert x[0] == answer, f"{x[0]=} != {answer=}, {op=}, {array=}"


@pytest.mark.parametrize(
    "fn, input",
    [
        # np.min tests
        (np.min, np.array([5, 2, 8, 1, 9])),
        (np.min, np.array(range(9)).reshape(3, 3)),
        (np.min, np.array([3.5, 2.1, 4.8, 1.2])),
        (np.min, np.array([42])),  # single element
        (np.min, np.array([-5, -2, -8, -1])),  # negative numbers
        (np.min, np.array([7, 7, 7])),  # all identical values
        # np.max tests
        (np.max, np.array([5, 2, 8, 1, 9])),
        (np.max, np.array(range(9)).reshape(3, 3)),
        (np.max, np.array([3.5, 2.1, 4.8, 1.2])),
        (np.max, np.array([42])),  # single element
        (np.max, np.array([-5, -2, -8, -1])),  # negative numbers
        # np.prod tests
        (np.prod, np.array([1, 2, 3, 4])),
        (np.prod, np.array([2, 2, 2, 2])),
        (np.prod, np.array([-1, -2, 3])),  # negative numbers (result positive)
        (np.prod, np.array([-2, -3])),  # negative numbers (result positive)
        (np.prod, np.array([5])),  # single element
        (np.prod, np.array(range(1, 9)).reshape(2, 2, 2)),  # 3D array
    ],
)
def test_np_reduction_ops(fn, input):
    """Test reduction operations: min, max, prod"""

    def create_kernel(fn):
        def kernel(x: DeviceNDArray, a: DeviceNDArray):
            x[0] = fn(a)

        return kernel

    kernel = cuda.jit(create_kernel(fn), dump=False)
    output_dtype = input.dtype
    x = cuda.to_device(np.zeros(1, dtype=output_dtype))
    a = cuda.to_device(input)
    kernel[1, 1, 0, 0](x, a)
    x_result = x.copy_to_host()[0]
    answer = fn(input)
    if np.issubdtype(output_dtype, np.floating):
        np.testing.assert_almost_equal(x_result, answer, decimal=5)
    else:
        assert x_result == answer, f"{x_result=} != {answer=}, {fn=}, {input=}"


def kernel_sum_method(x: DeviceNDArray, a: DeviceNDArray):
    x[0] = a.sum()


def kernel_min_method(x: DeviceNDArray, a: DeviceNDArray):
    x[0] = a.min()


def kernel_max_method(x: DeviceNDArray, a: DeviceNDArray):
    x[0] = a.max()


def kernel_prod_method(x: DeviceNDArray, a: DeviceNDArray):
    x[0] = a.prod()


@pytest.mark.parametrize(
    "kernel, np_fn, input",
    [
        (kernel_sum_method, np.sum, np.array([1, 2, 3, 4, 5])),
        (kernel_min_method, np.min, np.array([5, 2, 8, 1, 9])),
        (kernel_max_method, np.max, np.array([5, 2, 8, 1, 9])),
        (kernel_prod_method, np.prod, np.array([1, 2, 3, 4])),
    ],
)
def test_np_array_methods(kernel, np_fn, input):
    """Test array methods: .sum(), .min(), .max(), .prod()"""

    kernel = cuda.jit(kernel, dump=False)
    output_dtype = input.dtype
    x = cuda.to_device(np.zeros(1, dtype=output_dtype))
    a = cuda.to_device(input)
    kernel[1, 1, 0, 0](x, a)
    x_result = x.copy_to_host()[0]
    answer = np_fn(input)
    if np.issubdtype(output_dtype, np.floating):
        np.testing.assert_almost_equal(x_result, answer, decimal=5)
    else:
        assert x_result == answer


def test_np_elementwise_ops(subtests):
    """Test element-wise ufunc operations: abs, sqrt, exp, log"""

    # Test by extracting a single value from the result array
    @cuda.jit(dump=False)
    def test_abs_scalar(result: DeviceNDArray, a: DeviceNDArray, idx: int):
        tmp = np.abs(a)
        result[0] = tmp[idx]

    @cuda.jit(dump=False)
    def test_sqrt_scalar(result: DeviceNDArray, a: DeviceNDArray, idx: int):
        tmp = np.sqrt(a)
        result[0] = tmp[idx]

    @cuda.jit(dump=False)
    def test_exp_scalar(result: DeviceNDArray, a: DeviceNDArray, idx: int):
        tmp = np.exp(a)
        result[0] = tmp[idx]

    @cuda.jit(dump=False)
    def test_log_scalar(result: DeviceNDArray, a: DeviceNDArray, idx: int):
        tmp = np.log(a)
        result[0] = tmp[idx]

    test_cases = [
        (test_abs_scalar, np.abs, np.array([-1.0, -2.0, 3.0, -4.0], dtype=np.float32)),
        (test_sqrt_scalar, np.sqrt, np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)),
        (test_exp_scalar, np.exp, np.array([0.0, 1.0, 2.0], dtype=np.float32)),
        (
            test_log_scalar,
            np.log,
            np.array([1.0, 2.718281828, 7.389056099], dtype=np.float32),
        ),
    ]

    for kernel, np_fn, input_data in test_cases:
        with subtests.test(name=f"{np_fn.__name__} on {input_data.shape}"):
            expected = np_fn(input_data)
            result_array = np.zeros(1, dtype=input_data.dtype)
            result = cuda.to_device(result_array)
            a = cuda.to_device(input_data)

            # Test each element
            for idx in range(len(input_data)):
                kernel[1, 1, 0, 0](result, a, idx)
                result_value = result.copy_to_host()[0]
                np.testing.assert_allclose(result_value, expected[idx], rtol=1e-5, atol=1e-6)


def kernel_min(result: DeviceNDArray, a: DeviceNDArray):
    result[0] = np.min(a)


def kernel_max(result: DeviceNDArray, a: DeviceNDArray):
    result[0] = np.max(a)


def kernel_prod(result: DeviceNDArray, a: DeviceNDArray):
    result[0] = np.prod(a)


def kernel_sum(result: DeviceNDArray, a: DeviceNDArray):
    result[0] = np.sum(a)


@pytest.mark.parametrize(
    "kernel,np_fn,input_array",
    [
        # int8 (signed, bitwidth=8, range: -128 to 127)
        (
            kernel_min,
            np.min,
            np.array([127, 100, 50], dtype=np.int8),
        ),
        (
            kernel_max,
            np.max,
            np.array([-128, -100, -50], dtype=np.int8),
        ),
        (kernel_prod, np.prod, np.array([2, 3, 4], dtype=np.int8)),
        (kernel_sum, np.sum, np.array([10, 20, 30], dtype=np.int8)),
        # uint8 (unsigned, bitwidth=8, range: 0 to 255)
        (
            kernel_min,
            np.min,
            np.array([255, 200, 100], dtype=np.uint8),
        ),
        (
            kernel_max,
            np.max,
            np.array([0, 50, 100], dtype=np.uint8),
        ),
        (kernel_prod, np.prod, np.array([2, 3, 4], dtype=np.uint8)),
        (kernel_sum, np.sum, np.array([10, 20, 30], dtype=np.uint8)),
        # int16 (signed, bitwidth=16, range: -32768 to 32767)
        (
            kernel_min,
            np.min,
            np.array([32767, 10000, 5000], dtype=np.int16),
        ),
        (
            kernel_max,
            np.max,
            np.array([-32768, -10000, -5000], dtype=np.int16),
        ),
        (kernel_prod, np.prod, np.array([2, 3, 4], dtype=np.int16)),
        (kernel_sum, np.sum, np.array([1000, 2000, 3000], dtype=np.int16)),
        # uint16 (unsigned, bitwidth=16, range: 0 to 65535)
        (
            kernel_min,
            np.min,
            np.array([65535, 30000, 10000], dtype=np.uint16),
        ),
        (
            kernel_max,
            np.max,
            np.array([0, 30000, 50000], dtype=np.uint16),
        ),
        (kernel_prod, np.prod, np.array([2, 3, 4], dtype=np.uint16)),
        (kernel_sum, np.sum, np.array([1000, 2000, 3000], dtype=np.uint16)),
        # int32 with negative numbers
        (
            kernel_prod,
            np.prod,
            np.array([-2, 3, -4], dtype=np.int32),
        ),
        (
            kernel_prod,
            np.prod,
            np.array([-2, -3, -4], dtype=np.int32),
        ),
        (
            kernel_min,
            np.min,
            np.array([-100, 0, 100], dtype=np.int32),
        ),
        (
            kernel_max,
            np.max,
            np.array([-100, 0, 100], dtype=np.int32),
        ),
        # uint32 (large values)
        (
            kernel_min,
            np.min,
            np.array([1000000, 2000000, 3000000], dtype=np.uint32),
        ),
        (
            kernel_max,
            np.max,
            np.array([1000000, 2000000, 3000000], dtype=np.uint32),
        ),
        # int64 with negative numbers
        (
            kernel_min,
            np.min,
            np.array([-9223372036854775807, 0, 100], dtype=np.int64),
        ),
        (
            kernel_max,
            np.max,
            np.array([-100, 0, 9223372036854775806], dtype=np.int64),
        ),
        # uint64 (very large values)
        (
            kernel_min,
            np.min,
            np.array([1000000000000, 2000000000000, 3000000000000], dtype=np.uint64),
        ),
        (
            kernel_max,
            np.max,
            np.array([1000000000000, 2000000000000, 3000000000000], dtype=np.uint64),
        ),
    ],
)
def test_np_mixed_type_operations(kernel, np_fn, input_array):
    kernel = cuda.jit(kernel, dump=False)
    output_dtype = input_array.dtype
    result = cuda.to_device(np.zeros(1, dtype=output_dtype))
    a = cuda.to_device(input_array)
    kernel[1, 1, 0, 0](result, a)
    result_value = result.copy_to_host()[0]
    expected = np_fn(input_array)

    assert result_value == expected


def test_np_elementwise_mixed_types():
    """Test element-wise operations with various integer and float types."""

    @cuda.jit(dump=False)
    def test_abs_scalar(result: DeviceNDArray, a: DeviceNDArray, idx: int):
        tmp = np.abs(a)
        result[0] = tmp[idx]

    test_cases = [
        # Integer types
        (np.array([-127, -100, 50, 127], dtype=np.int8), "int8 abs"),
        (np.array([-32767, -10000, 5000, 32767], dtype=np.int16), "int16 abs"),
        (np.array([-100, -50, 0, 50, 100], dtype=np.int32), "int32 abs"),
        (np.array([-1000000, -500, 0, 500, 1000000], dtype=np.int64), "int64 abs"),
        # Float types
        (np.array([-1.5, -2.5, 3.5, -4.5], dtype=np.float32), "float32 abs"),
        (np.array([-1.5, -2.5, 3.5, -4.5], dtype=np.float64), "float64 abs"),
    ]

    for input_array, description in test_cases:
        expected = np.abs(input_array)
        result = cuda.to_device(np.zeros(1, dtype=input_array.dtype))
        a = cuda.to_device(input_array)

        for idx in range(len(input_array)):
            test_abs_scalar[1, 1, 0, 0](result, a, idx)
            result_value = result.copy_to_host()[0]

            if np.issubdtype(input_array.dtype, np.floating):
                np.testing.assert_allclose(
                    result_value,
                    expected[idx],
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"Failed: {description} at index {idx}",
                )
            else:
                assert result_value == expected[idx], (
                    f"Failed: {description} at index {idx}\n"
                    f"  Input: {input_array[idx]}\n"
                    f"  Result: {result_value}\n"
                    f"  Expected: {expected[idx]}"
                )


@pytest.mark.parametrize(
    "ufunc",
    [np.sinh, np.cosh, np.tanh, np.arcsinh, np.arccosh, np.arctanh],
)
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_hyperbolic_ufuncs(ufunc, dtype):
    """Test complex hyperbolic ufuncs: sinh, cosh, tanh, arcsinh, arccosh, arctanh"""

    @cuda.jit
    def kernel(result: DeviceNDArray, a: DeviceNDArray):
        i = cuda.grid(1)
        if i < a.shape[0]:
            ufunc(a[i], result[i : i + 1])

    # Use values appropriate for each function
    if ufunc == np.arccosh:
        # arccosh domain: |z| >= 1
        input_data = np.array([1.5 + 0.5j, 2.0 - 1.0j, 1.0 + 2.0j], dtype=dtype)
    elif ufunc == np.arctanh:
        # arctanh has singularities at +/- 1
        input_data = np.array([0.5 + 0.5j, -0.3 + 0.2j, 0.1 - 0.4j], dtype=dtype)
    else:
        input_data = np.array([0.5 + 0.5j, -0.5 - 0.5j, 1.0 + 1.0j], dtype=dtype)

    expected = ufunc(input_data)
    result = cuda.to_device(np.zeros_like(input_data))
    a = cuda.to_device(input_data)

    kernel[1, len(input_data)](result, a)
    result_host = result.copy_to_host()

    np.testing.assert_allclose(
        result_host,
        expected,
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"Failed: {ufunc.__name__} with {dtype.__name__}",
    )


@pytest.mark.parametrize("ufunc", [np.log2, np.log10])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_log_ufuncs(ufunc, dtype):
    """Test complex log ufuncs: log2, log10 (excludes edge cases like 0j)"""

    @cuda.jit
    def kernel(result: DeviceNDArray, a: DeviceNDArray):
        i = cuda.grid(1)
        if i < a.shape[0]:
            ufunc(a[i], result[i : i + 1])

    input_data = np.array([1 + 1j, 2 - 1j, 0.5 + 0.5j], dtype=dtype)
    expected = ufunc(input_data)
    result = cuda.to_device(np.zeros_like(input_data))
    a = cuda.to_device(input_data)

    kernel[1, len(input_data)](result, a)
    result_host = result.copy_to_host()

    np.testing.assert_allclose(
        result_host,
        expected,
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"Failed: {ufunc.__name__} with {dtype.__name__}",
    )


@pytest.mark.parametrize(
    "ufunc",
    [np.equal, np.not_equal, np.greater, np.greater_equal, np.less, np.less_equal],
)
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_comparison_ufuncs(ufunc, dtype):
    """Test complex comparison ufuncs"""

    @cuda.jit
    def kernel(result: DeviceNDArray, a: DeviceNDArray, b: DeviceNDArray):
        i = cuda.grid(1)
        if i < a.shape[0]:
            ufunc(a[i], b[i], result[i : i + 1])

    a_data = np.array([1 + 1j, 2 - 1j, 1 + 0j, 0 + 1j], dtype=dtype)
    b_data = np.array([1 + 1j, 1 - 1j, 1 + 1j, 0 + 0j], dtype=dtype)
    expected = ufunc(a_data, b_data)
    result = cuda.to_device(np.zeros_like(a_data))
    a = cuda.to_device(a_data)
    b = cuda.to_device(b_data)

    kernel[1, len(a_data)](result, a, b)
    result_host = result.copy_to_host()

    np.testing.assert_allclose(
        result_host.real,
        expected.astype(dtype).real,
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"Failed: {ufunc.__name__} with {dtype.__name__}",
    )


@pytest.mark.parametrize("ufunc", [np.logical_and, np.logical_or, np.logical_xor])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_logical_binary_ufuncs(ufunc, dtype):
    """Test complex logical binary ufuncs"""

    @cuda.jit
    def kernel(result: DeviceNDArray, a: DeviceNDArray, b: DeviceNDArray):
        i = cuda.grid(1)
        if i < a.shape[0]:
            ufunc(a[i], b[i], result[i : i + 1])

    a_data = np.array([1 + 1j, 0 + 0j, 1 + 0j, 0 + 1j], dtype=dtype)
    b_data = np.array([1 + 0j, 1 + 1j, 0 + 0j, 0 + 0j], dtype=dtype)
    expected = ufunc(a_data, b_data)
    result = cuda.to_device(np.zeros_like(a_data))
    a = cuda.to_device(a_data)
    b = cuda.to_device(b_data)

    kernel[1, len(a_data)](result, a, b)
    result_host = result.copy_to_host()

    np.testing.assert_allclose(
        result_host.real,
        expected.astype(dtype).real,
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"Failed: {ufunc.__name__} with {dtype.__name__}",
    )


@pytest.mark.parametrize("ufunc", [np.maximum, np.minimum, np.fmax, np.fmin])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_minmax_ufuncs(ufunc, dtype):
    """Test complex minmax ufuncs"""

    @cuda.jit
    def kernel(result: DeviceNDArray, a: DeviceNDArray, b: DeviceNDArray):
        i = cuda.grid(1)
        if i < a.shape[0]:
            ufunc(a[i], b[i], result[i : i + 1])

    a_data = np.array([1 + 1j, 2 - 1j, 0 + 2j], dtype=dtype)
    b_data = np.array([2 + 0j, 1 + 3j, 1 + 0j], dtype=dtype)
    expected = ufunc(a_data, b_data)
    result = cuda.to_device(np.zeros_like(a_data))
    a = cuda.to_device(a_data)
    b = cuda.to_device(b_data)

    kernel[1, len(a_data)](result, a, b)
    result_host = result.copy_to_host()

    np.testing.assert_allclose(
        result_host,
        expected,
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"Failed: {ufunc.__name__} with {dtype.__name__}",
    )


def test_packed_complex_components_preserve_aliases():
    @cuda.jit(dump=False)
    def kernel(records, out):
        real = records.value.real
        imag = records.value.imag
        for i in range(real.size):
            out[i, 0] = real[i]
            out[i, 1] = imag[i]
            real[i] = real[i] + 10
            imag[i] = imag[i] - 20

    records = np.zeros(5, dtype=[("tag", np.uint8), ("value", np.complex128)])
    records["tag"] = np.arange(5) + 70
    records["value"] = np.arange(5) + 0.5 + (np.arange(5) + 1.5) * 1j
    device = cuda.to_device(records)
    out = cuda.device_array((5, 2), dtype=np.float64)
    kernel[1, 1](device, out)
    np.testing.assert_array_equal(
        out.copy_to_host(), np.column_stack((records["value"].real, records["value"].imag))
    )
    expected = records.copy()
    expected["value"] += 10 - 20j
    np.testing.assert_array_equal(device.copy_to_host(), expected)


def test_unaligned_reshape_ravel_aliases():
    class UnalignedArray:
        def __init__(self, buffer):
            self.buffer = buffer
            self.__cuda_array_interface__ = dict(buffer.__cuda_array_interface__)
            pointer, readonly = self.__cuda_array_interface__["data"]
            self.__cuda_array_interface__.update(
                data=(pointer + 1, readonly),
                shape=(2, 3),
                strides=None,
                typestr=np.dtype(np.float64).str,
            )

    @cuda.jit(dump=False)
    def kernel(array, out):
        reshaped = array.reshape((3, 2))
        flat = reshaped.ravel()
        for i in range(flat.size):
            out[i] = flat[i]
        reshaped[1, 1] = 41
        flat[5] = 73

    values = np.arange(6, dtype=np.float64) + 0.5
    raw = np.concatenate(
        (np.array([0xA5], np.uint8), values.view(np.uint8), np.array([0x5A], np.uint8))
    )
    device = cuda.to_device(raw)
    out = cuda.device_array(6, dtype=np.float64)
    kernel[1, 1](UnalignedArray(device), out)
    np.testing.assert_array_equal(out.copy_to_host(), values)
    expected = raw.copy()
    expected[1:-1].view(np.float64)[[3, 5]] = [41, 73]
    np.testing.assert_array_equal(device.copy_to_host(), expected)


def test_packed_field_empty_like_math_and_scalar_output():
    @cuda.jit(dump=False)
    def kernel(records, out):
        source = records.value
        temporary = np.empty_like(source)
        temporary[:] = 3.5
        result = np.add(source, temporary)
        for i in range(source.size):
            out[i] = result[i]
            np.sin(source[i], source[i : i + 1])

    records = np.zeros(5, dtype=[("tag", np.int32), ("value", np.float64)])
    records["tag"] = np.arange(5) + 200
    records["value"] = np.arange(5) / 4
    device = cuda.to_device(records)
    out = cuda.device_array(5, dtype=np.float64)
    kernel[1, 1](device, out)
    np.testing.assert_array_equal(out.copy_to_host(), records["value"] + 3.5)
    actual = device.copy_to_host()
    np.testing.assert_array_equal(actual["tag"], records["tag"])
    np.testing.assert_allclose(actual["value"], np.sin(records["value"]), rtol=1e-14)


def test_packed_field_transpose_preserves_alias():
    @cuda.jit(dump=False)
    def kernel(records, out):
        transposed = np.transpose(records.value)
        for i in range(transposed.shape[0]):
            for j in range(transposed.shape[1]):
                out[i, j] = transposed[i, j]
        transposed[2, 1] = 59

    records = np.zeros((2, 3), dtype=[("tag", np.int32), ("value", np.float64)])
    records["tag"] = np.arange(6).reshape(2, 3) + 90
    records["value"] = np.arange(6).reshape(2, 3) + 0.5
    device = cuda.to_device(records)
    out = cuda.device_array((3, 2), dtype=np.float64)
    kernel[1, 1](device, out)
    np.testing.assert_array_equal(out.copy_to_host(), records["value"].T)
    expected = records.copy()
    expected["value"][1, 2] = 59
    np.testing.assert_array_equal(device.copy_to_host(), expected)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    "form,axes",
    [(form, axes) for form in ("tuple", "varargs") for axes in ((0, 1), (1, 0), (-1, -2))]
    + [("none", (1, 0)), ("default", (1, 0))],
)
def test_transpose_method_axes_preserve_payload_and_alias(packed, form, axes):
    @cuda.jit
    def kernel(owner, out, axes):
        source = owner.value if packed else owner
        if form == "tuple":
            view = source.transpose(axes)
        elif form == "varargs":
            view = source.transpose(axes[0], axes[1])
        elif form == "none":
            view = source.transpose(None)
        else:
            view = source.transpose()
        for i in range(view.shape[0]):
            for j in range(view.shape[1]):
                out[i, j] = view[i, j]
        view[0, 1] = 59

    values = np.arange(6, dtype=np.float64).reshape(2, 3) + 0.5
    host = np.zeros(values.shape, [("guard", np.int32), ("value", np.float64)])
    host["guard"] = 71
    host["value"] = values
    owner = cuda.to_device(host if packed else values)
    permutation = axes if form in ("tuple", "varargs") else None
    expected_values = values.transpose(permutation).copy()
    out = cuda.device_array(expected_values.shape, np.float64)
    kernel[1, 1](owner, out, axes)
    np.testing.assert_array_equal(out.copy_to_host(), expected_values)
    expected = host.copy() if packed else values.copy()
    payload = expected["value"] if packed else expected
    payload.transpose(permutation)[0, 1] = 59
    np.testing.assert_array_equal(owner.copy_to_host(), expected)


@pytest.mark.parametrize("axes", [(0, 0), (0, 2), (-3, 1)])
def test_transpose_method_invalid_axes_do_not_write(axes):
    @cuda.jit(debug=True, opt=False)
    def kernel(source, axes):
        view = source.transpose(axes)
        for i in range(view.shape[0]):
            for j in range(view.shape[1]):
                view[i, j] = 99

    host = np.arange(6, dtype=np.float64).reshape(2, 3)
    source = cuda.to_device(host)
    with pytest.raises(ValueError):
        kernel[1, 1](source, axes)
    np.testing.assert_array_equal(source.copy_to_host(), host)


@pytest.mark.parametrize("order", ["C", "F"])
def test_empty_like_preserves_layout_and_dtype_override(order):
    @cuda.jit(dump=False)
    def kernel(source, out, strides):
        temporary = np.empty_like(source, dtype=np.int32)
        strides[0] = temporary.strides[0]
        strides[1] = temporary.strides[1]
        for i in range(temporary.shape[0]):
            for j in range(temporary.shape[1]):
                temporary[i, j] = source[i, j] + 10
        for i in range(temporary.shape[0]):
            for j in range(temporary.shape[1]):
                out[i, j] = temporary[i, j]

    source = np.array(np.arange(6).reshape(2, 3), dtype=np.float64, order=order)
    device = cuda.to_device(source)
    out = cuda.device_array(source.shape, dtype=np.int32)
    strides = cuda.device_array(2, dtype=np.int64)
    kernel[1, 1](device, out, strides)
    np.testing.assert_array_equal(out.copy_to_host(), source.astype(np.int32) + 10)
    np.testing.assert_array_equal(
        strides.copy_to_host(), np.empty_like(source, dtype=np.int32).strides
    )
    np.testing.assert_array_equal(device.copy_to_host(), source)


@pytest.mark.parametrize("shape", [(3, 2), (-1, 2), (3, -1)])
def test_unaligned_reshape_infers_dimensions_and_preserves_values(shape):
    class View:
        def __init__(self, owner):
            self.owner = owner
            self.__cuda_array_interface__ = {
                "version": 3,
                "shape": (2, 3),
                "strides": None,
                "typestr": "<f8",
                "data": (owner.device_ctypes_pointer.value + 1, False),
            }

    @cuda.jit
    def kernel(source, out, dimensions):
        result = source.reshape(shape)
        dimensions[0] = result.shape[0]
        dimensions[1] = result.shape[1]
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                out[i, j] = result[i, j]

    values = np.arange(6, dtype=np.float64) + 0.5
    raw = np.concatenate((np.array([0xA5], np.uint8), values.view(np.uint8)))
    out = cuda.device_array((3, 2), np.float64)
    dimensions = cuda.device_array(2, np.int64)
    kernel[1, 1](View(cuda.to_device(raw)), out, dimensions)
    np.testing.assert_array_equal(dimensions.copy_to_host(), [3, 2])
    np.testing.assert_array_equal(out.copy_to_host(), values.reshape((3, 2)))


@pytest.mark.parametrize("shape", [(4, 2), (-1, -1), (-2, 3), (0, -1)])
def test_byte_reshape_rejects_invalid_shape_without_writes(shape):
    @cuda.jit(debug=True, opt=False)
    def kernel(records):
        result = records.value.reshape(shape)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = 99

    host = np.zeros((2, 3), [("value", np.float64), ("guard", np.int32)])
    host["value"] = np.arange(6).reshape(2, 3) + 0.5
    host["guard"] = 123
    device = cuda.to_device(host)
    with pytest.raises(ValueError):
        kernel[1, 1](device)
    np.testing.assert_array_equal(device.copy_to_host(), host)


def test_byte_reshape_rejects_incompatible_fortran_strides():
    @cuda.jit(debug=True, opt=False)
    def kernel(records):
        result = records.value.reshape((3, 2))
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = 99

    host = np.zeros((2, 3), [("value", np.float64), ("guard", np.int32)], order="F")
    host["value"] = np.arange(6).reshape(2, 3) + 0.5
    host["guard"] = 123
    device = cuda.to_device(host)
    with pytest.raises(ValueError):
        kernel[1, 1](device)
    np.testing.assert_array_equal(device.copy_to_host(), host)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("shape", [(2, 3), (-1, 3)])
def test_byte_reshape_same_shape_preserves_strides_and_alias(order, shape):
    @cuda.jit(debug=True, opt=False)
    def kernel(records, out, strides, shape):
        result = records.value.reshape(shape)
        strides[0] = result.strides[0]
        strides[1] = result.strides[1]
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                out[i, j] = result[i, j]
                result[i, j] += 10

    host = np.zeros((2, 3), [("value", np.float64), ("guard", np.int32)], order=order)
    host["value"] = np.arange(6).reshape(2, 3) + 0.5
    host["guard"] = 123
    expected = host.copy(order="K")
    expected["value"] += 10
    device = cuda.to_device(host)
    out = cuda.device_array(host.shape, np.float64)
    strides = cuda.device_array(2, np.int64)
    kernel[1, 1](device, out, strides, shape)
    np.testing.assert_array_equal(out.copy_to_host(), host["value"])
    np.testing.assert_array_equal(strides.copy_to_host(), host["value"].strides)
    np.testing.assert_array_equal(device.copy_to_host(), expected)


@pytest.mark.parametrize(
    "source_shape,source_strides,target_shape",
    [
        ((2, 3, 1), (24, 8, 256), (1, 2, 3)),
        ((1, 2, 3), (256, 24, 8), (2, 3, 1)),
        ((2, 1, 3), (24, 256, 8), (1, 2, 3)),
        ((2, 3, 1), (48, 16, 256), (1, 2, 3)),
        ((1, 1, 1), (256, 128, 64), (1, 1, 1)),
    ],
)
def test_byte_reshape_ignores_singleton_strides(source_shape, source_strides, target_shape):
    class View:
        def __init__(self, owner):
            self.owner = owner
            self.__cuda_array_interface__ = {
                "version": 3,
                "shape": source_shape,
                "strides": source_strides,
                "typestr": "<f8",
                "data": (owner.device_ctypes_pointer.value + 1, False),
            }

    @cuda.jit(debug=True, opt=False)
    def kernel(source, out):
        result = source.reshape(target_shape)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                for k in range(result.shape[2]):
                    out[i, j, k] = result[i, j, k]
                    result[i, j, k] += 10

    nbytes = 9 + sum((extent - 1) * stride for extent, stride in zip(source_shape, source_strides))
    raw = np.full(nbytes, 0xA5, np.uint8)
    source = np.ndarray(source_shape, np.float64, buffer=raw, offset=1, strides=source_strides)
    source[:] = np.arange(source.size).reshape(source_shape) + 0.5
    expected_values = source.reshape(target_shape).copy()
    expected_raw = raw.copy()
    np.ndarray(source_shape, np.float64, buffer=expected_raw, offset=1, strides=source_strides)[
        :
    ] += 10
    device = cuda.to_device(raw)
    out = cuda.device_array(target_shape, np.float64)
    kernel[1, 1](View(device), out)
    np.testing.assert_array_equal(out.copy_to_host(), expected_values)
    np.testing.assert_array_equal(device.copy_to_host(), expected_raw)


@pytest.mark.parametrize("offset", [0, 1])
def test_nditer_byte_view_preserves_payload_and_alias(offset):
    @cuda.jit
    def kernel(raw, out):
        values = raw[offset : offset + out.size * 8].view(np.float64)
        i = 0
        for element in np.nditer(values):
            out[i] = element.item()
            element[()] = element.item() + 10
            i += 1

    values = np.array([1.5, 2.5, -3.5], np.float64)
    raw = np.full(offset + values.nbytes + 1, 0xA5, np.uint8)
    raw[offset : offset + values.nbytes] = values.view(np.uint8)
    expected = raw.copy()
    expected[offset : offset + values.nbytes] = (values + 10).view(np.uint8)
    device = cuda.to_device(raw)
    out = cuda.device_array(values.shape, np.float64)
    kernel[1, 1](device, out)
    np.testing.assert_array_equal(out.copy_to_host(), values)
    np.testing.assert_array_equal(device.copy_to_host(), expected)


@pytest.mark.parametrize("byte_first", [True, False])
def test_nditer_mixed_storage_operands(byte_first):
    @cuda.jit
    def kernel(records, typed, out):
        if byte_first:
            operands = (records.value, typed)
        else:
            operands = (typed, records.value)
        i = 0
        for first, second in np.nditer(operands):
            out[i, 0] = first.item()
            out[i, 1] = second.item()
            first[()] = first.item() + 10
            second[()] = second.item() + 20
            i += 1

    records = np.zeros(3, dtype=[("guard", np.int32), ("value", np.float64)])
    records["guard"] = 12345
    records["value"] = [1.5, 2.5, -3.5]
    typed = np.array([4.5, -5.5, 6.5], np.float64)
    expected = records.copy()
    expected["value"] += 10 if byte_first else 20
    expected_typed = typed + (20 if byte_first else 10)
    columns = (records["value"], typed) if byte_first else (typed, records["value"])
    device_records = cuda.to_device(records)
    device_typed = cuda.to_device(typed)
    out = cuda.device_array((3, 2), np.float64)
    kernel[1, 1](device_records, device_typed, out)
    np.testing.assert_array_equal(out.copy_to_host(), np.column_stack(columns))
    np.testing.assert_array_equal(device_records.copy_to_host(), expected)
    np.testing.assert_array_equal(device_typed.copy_to_host(), expected_typed)


if __name__ == "__main__":
    test_np_ops()
    test_np_reduction_ops()
    test_np_array_methods()
    test_np_elementwise_ops()
    test_np_mixed_type_operations()
    test_np_elementwise_mixed_types()
