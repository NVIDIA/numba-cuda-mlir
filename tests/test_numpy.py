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


if __name__ == "__main__":
    test_np_ops()
    test_np_reduction_ops()
    test_np_array_methods()
    test_np_elementwise_ops()
    test_np_mixed_type_operations()
    test_np_elementwise_mixed_types()
