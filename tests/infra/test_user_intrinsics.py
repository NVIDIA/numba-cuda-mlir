# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cuda.simt as cuda
from cuda.simt import types, compiler
from cusimt.errors import MultipleIntrinsicFunctionsError
from cusimt._mlir.dialects import llvm
from cusimt.mlir.dialect_exts import arith
from cusimt._mlir.extras import types as T
import numpy as np
import pytest


def test_intrinsic_from_source():
    add_42 = cuda.intrin.define(
        """
        func.func private @add_42(%x: i32) -> i32 attributes {always_inline} {
            %c42 = arith.constant 42 : i32
            %r = arith.addi %x, %c42 : i32
            return %r : i32
        }
    """
    )

    @cuda.jit
    def kernel(arr):
        i = cuda.grid(1)
        if i < arr.size:
            arr[i] = add_42(arr[i])

    a = np.array([0, 1, 2, 3], dtype=np.int32)
    ad = cuda.to_device(a)
    kernel[1, 4](ad)
    result = ad.copy_to_host()
    expected = np.array([42, 43, 44, 45], dtype=np.int32)
    assert np.array_equal(result, expected), f"{result=} != {expected=}"


def test_intrinsic_void_return_with_llvm():
    llvm_kDynamic = -2147483648

    @cuda.intrin.define
    def store_magic(ptr: llvm.PointerType.get, idx: types.int64):
        magic = arith.constant(0xCAFE, T.i64())
        offset_ptr = llvm.getelementptr(
            llvm.PointerType.get(), ptr, [idx], [llvm_kDynamic], T.i64(), None
        )
        llvm.store(magic, offset_ptr)

    @cuda.jit
    def kernel(arr):
        store_magic(types.ptr(arr), 1)

    a = np.array([0, 0, 0], dtype=np.int64)
    ad = cuda.to_device(a)
    kernel[1, 1](ad)
    result = ad.copy_to_host()
    assert result[1] == 0xCAFE, f"{result=}"


def test_intrinsic_multiple_functions_error():
    with pytest.raises(MultipleIntrinsicFunctionsError) as excinfo:
        cuda.intrin.define(
            """
            func.func private @func_a(%x: i32) -> i32 attributes {always_inline} {
                return %x : i32
            }
            func.func private @func_b(%x: i32) -> i32 attributes {always_inline} {
                return %x : i32
            }
        """
        )

    assert "func_a" in str(excinfo.value)
    assert "func_b" in str(excinfo.value)


def test_intrinsic_no_functions_error():
    with pytest.raises(ValueError) as excinfo:
        cuda.intrin.define(
            """
            // Empty module with no functions
        """
        )

    assert "No functions found" in str(excinfo.value)


def test_intrinsic():
    @cuda.intrin.define
    def mul_by_2(x: types.int32) -> types.int32:
        c2 = arith.constant(2, T.i32())
        return arith.muli(x, c2)

    @cuda.jit
    def kernel(arr):
        i = cuda.grid(1)
        if i < arr.size:
            arr[i] = mul_by_2(arr[i])

    a = np.array([1, 2, 3, 4], dtype=np.int32)
    ad = cuda.to_device(a)
    kernel[1, 4](ad)
    result = ad.copy_to_host()
    expected = np.array([2, 4, 6, 8], dtype=np.int32)
    assert np.array_equal(result, expected), f"{result=} != {expected=}"


def test_intrinsic_callable_type():
    @cuda.intrin.define
    def halve(x: T.f32) -> T.f32:
        c2 = arith.constant(2.0, T.f32())
        return arith.divf(x, c2)

    @cuda.jit
    def kernel(arr):
        i = cuda.grid(1)
        if i < arr.size:
            arr[i] = halve(arr[i])

    a = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    ad = cuda.to_device(a)
    kernel[1, 4](ad)
    result = ad.copy_to_host()
    expected = np.array([5.0, 10.0, 15.0, 20.0], dtype=np.float32)
    assert np.allclose(result, expected), f"{result=} != {expected=}"


def test_intrinsic_multiple_ops():
    @cuda.intrin.define
    def fma(a: types.int32, b: types.int32, c: types.int32) -> types.int32:
        prod = arith.muli(a, b)
        return arith.addi(prod, c)

    @cuda.jit
    def kernel(arr, x, y, z):
        i = cuda.grid(1)
        if i < arr.size:
            arr[i] = fma(x, y, z)

    a = np.zeros(4, dtype=np.int32)
    ad = cuda.to_device(a)
    kernel[1, 4](ad, 3, 4, 5)  # 3*4 + 5 = 17
    result = ad.copy_to_host()
    expected = np.full(4, 17, dtype=np.int32)
    assert np.array_equal(result, expected), f"{result=} != {expected=}"


def test_intrinsic_mixed_annotations():
    @cuda.intrin.define
    def scale_and_add(x: types.int32, scale: T.i32, offset: types.int32) -> T.i32:
        scaled = arith.muli(x, scale)
        return arith.addi(scaled, offset)

    @cuda.jit
    def kernel(arr, scale, offset):
        i = cuda.grid(1)
        if i < arr.size:
            arr[i] = scale_and_add(arr[i], scale, offset)

    a = np.array([1, 2, 3, 4], dtype=np.int32)
    ad = cuda.to_device(a)
    kernel[1, 4](ad, 10, 5)  # x*10 + 5
    result = ad.copy_to_host()
    expected = np.array([15, 25, 35, 45], dtype=np.int32)
    assert np.array_equal(result, expected), f"{result=} != {expected=}"


@pytest.mark.skip(reason="TODO: Fails CI, passes locally.")
def test_cross_file_import():
    from tests.infra.my_intrinsics import elect_sync

    @cuda.jit
    def kernel(arr):
        if elect_sync():
            arr[0] = 42

    a = np.array([0], dtype=np.int32)
    ad = cuda.to_device(a)
    kernel[1, 32](ad)
    result = ad.copy_to_host()
    assert result[0] == 42, f"{result=}"


def test_intrinsic_from_source_inlines():
    inline_double = cuda.intrin.define(
        """
        func.func private @inline_double(%x: i32) -> i32 attributes {always_inline} {
            %c2 = arith.constant 2 : i32
            %r = arith.muli %x, %c2 : i32
            return %r : i32
        }
    """
    )

    @cuda.jit
    def kernel(arr):
        i = cuda.grid(1)
        if i < arr.size:
            arr[i] = inline_double(arr[i])

    mlir_str = compiler.compile_mlir(kernel, "void(int32[:])", optimized=True)
    assert "inline_double" not in mlir_str or "call" not in mlir_str


def test_intrinsic_tuple_return():
    @cuda.intrin.define
    def divmod_i32(a: types.int32, b: types.int32) -> tuple[types.int32, types.int32]:
        quot = arith.divsi(a, b)
        rem = arith.remsi(a, b)
        return quot, rem

    @cuda.jit
    def kernel(arr_quot, arr_rem, divisor):
        i = cuda.grid(1)
        if i < arr_quot.size:
            q, r = divmod_i32(arr_quot[i], divisor)
            arr_quot[i] = q
            arr_rem[i] = r

    a = np.array([10, 23, 37, 45], dtype=np.int32)
    b = np.zeros(4, dtype=np.int32)
    ad = cuda.to_device(a)
    bd = cuda.to_device(b)
    kernel[1, 4](ad, bd, 7)
    result_quot = ad.copy_to_host()
    result_rem = bd.copy_to_host()
    expected_quot = np.array([1, 3, 5, 6], dtype=np.int32)
    expected_rem = np.array([3, 2, 2, 3], dtype=np.int32)
    assert np.array_equal(result_quot, expected_quot), f"{result_quot=} != {expected_quot=}"
    assert np.array_equal(result_rem, expected_rem), f"{result_rem=} != {expected_rem=}"


if __name__ == "__main__":
    test_intrinsic_from_source()
    test_intrinsic_void_return_with_llvm()
    test_intrinsic_multiple_functions_error()
    test_intrinsic_no_functions_error()
    test_intrinsic()
    test_intrinsic_callable_type()
    test_intrinsic_multiple_ops()
    test_intrinsic_mixed_annotations()
    test_cross_file_import()
    test_intrinsic_from_source_inlines()
    test_intrinsic_tuple_return()
