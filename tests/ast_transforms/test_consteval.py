# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numba_cuda_mlir
from numba_cuda_mlir.cuda.experimental import consteval
from numba_cuda_mlir.ast_transforms import ConstevalError
from numba_cuda_mlir import cuda
import numpy as np
import pytest


def test_consteval_freevars():
    @cuda.jit
    def loop_body(x, i):
        x[i] += 1

    @cuda.jit
    def k(x):
        for i in consteval(range(5)):
            loop_body(x, i)

    x = np.zeros(10, dtype=np.float32)
    k[1, 1](x)


GLOBAL_CONST = 16


def test_consteval_global_constant():
    """Test consteval with a global constant."""

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        limit = consteval(GLOBAL_CONST)
        if i < limit:
            arr[i] = float(i)

    a = np.zeros(32, dtype=np.float32)
    d_a = cuda.to_device(a)
    kernel[1, 32](d_a)
    result = d_a.copy_to_host()

    assert result[0] == 0.0
    assert result[15] == 15.0
    assert result[16] == 0.0
    assert result[31] == 0.0


def test_consteval_closure_variable():
    """Test consteval with a closure variable."""

    def make_kernel(n):
        @numba_cuda_mlir.cuda.jit
        def kernel(arr):
            i = cuda.threadIdx.x
            limit = consteval(n)
            if i < limit:
                arr[i] = float(i) * 2.0

        return kernel

    kernel = make_kernel(8)
    a = np.zeros(32, dtype=np.float32)
    d_a = cuda.to_device(a)
    kernel[1, 32](d_a)
    result = d_a.copy_to_host()

    assert result[0] == 0.0
    assert result[7] == 14.0
    assert result[8] == 0.0


def test_consteval_expression():
    """Test consteval with an expression."""

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        limit = consteval(GLOBAL_CONST * 2)
        if i < limit:
            arr[i] = 1.0

    a = np.zeros(64, dtype=np.float32)
    d_a = cuda.to_device(a)
    kernel[1, 64](d_a)
    result = d_a.copy_to_host()

    assert result[31] == 1.0
    assert result[32] == 0.0


def test_consteval_undefined_raises():
    """Test that consteval with undefined variable raises ConstevalError."""

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        limit = consteval(UNDEFINED_VAR)  # noqa: F821
        arr[i] = float(limit)

    # Error is raised at compile time (when we have argtypes)
    with pytest.raises(ConstevalError, match="Cannot evaluate consteval argument"):
        kernel.compile("void(float32[:])")


def test_consteval_kernel_param_resolves_to_type():
    """Test that kernel parameters resolve to their Numba types in consteval."""
    from numba_cuda_mlir import types

    @numba_cuda_mlir.cuda.jit
    def kernel(arr, n):
        i = cuda.threadIdx.x
        # n resolves to its Numba type in consteval context
        n_type = consteval(n)
        if consteval(n_type == types.int32):
            arr[i] = 32.0
        elif consteval(n_type == types.int64):
            arr[i] = 64.0

    cres32 = kernel.compile("void(float32[:], int32)")
    source32 = cres32.metadata["transformed_source"]
    assert "arr[i] = 32.0" in source32

    cres64 = kernel.compile("void(float32[:], int64)")
    source64 = cres64.metadata["transformed_source"]
    assert "arr[i] = 64.0" in source64


# Chained consteval tests


def test_chained_consteval_simple():
    """Test that a = consteval(X) followed by consteval(a) works."""
    X = 42

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        a = consteval(X)
        arr[i] = float(consteval(a))

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert source is not None
    assert "a = 42" in source
    assert "float(42)" in source


def test_chained_consteval_expression():
    """Test chained consteval with expressions."""
    BASE = 10

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        x = consteval(BASE)
        if consteval(x + 5 > 10):
            arr[i] = 1.0
        else:
            arr[i] = 2.0

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert source is not None
    assert "arr[i] = 1.0" in source
    assert "arr[i] = 2.0" not in source


def test_chained_consteval_multiple():
    """Test multiple chained constevals."""
    A = 2
    B = 3

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        x = consteval(A)
        y = consteval(B)
        z = consteval(x * y)
        arr[i] = float(consteval(z + 1))

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert source is not None
    assert "x = 2" in source
    assert "y = 3" in source
    assert "z = 6" in source
    assert "float(7)" in source


def test_consteval_chained_runs_correctly():
    """Test that chained consteval produces correct runtime results."""
    THRESHOLD = 16

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        limit = consteval(THRESHOLD)
        if consteval(limit > 10):
            arr[i] = float(consteval(limit * 2))
        else:
            arr[i] = 0.0

    a = np.zeros(32, dtype=np.float32)
    d_a = cuda.to_device(a)
    kernel[1, 32](d_a)
    result = d_a.copy_to_host()

    assert all(result == 32.0)


# Expression type tests


def test_consteval_tuple_indexing():
    """Test consteval with tuple indexing."""
    ITEMS = (10, 20, 30)

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        arr[i] = float(consteval(ITEMS[1]))

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "float(20)" in source


def test_consteval_dict_access():
    """Test consteval with dict access."""
    CONFIG = {"value": 42}

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        arr[i] = float(consteval(CONFIG["value"]))

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "float(42)" in source


def test_consteval_list_operations():
    """Test consteval with list operations."""
    NUMBERS = [1, 2, 3, 4, 5]

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        arr[i] = float(consteval(len(NUMBERS)))

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "float(5)" in source


def test_consteval_arithmetic():
    """Test consteval with various arithmetic operations."""
    X = 10
    Y = 3

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        a = consteval(X + Y)
        b = consteval(X - Y)
        c = consteval(X * Y)
        d = consteval(X // Y)
        e = consteval(X % Y)
        f = consteval(X**Y)
        arr[i] = float(a + b + c + d + e + f)

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "a = 13" in source
    assert "b = 7" in source
    assert "c = 30" in source
    assert "d = 3" in source
    assert "e = 1" in source
    assert "f = 1000" in source


def test_consteval_boolean_ops():
    """Test consteval with boolean operations."""
    A = True
    B = False

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        if consteval(A and not B):
            arr[i] = 1.0
        else:
            arr[i] = 0.0

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "arr[i] = 1.0" in source
    assert "arr[i] = 0.0" not in source


def test_consteval_comparison():
    """Test consteval with comparison operations."""
    X = 5

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        if consteval(X > 3 and X < 10):
            arr[i] = 1.0
        else:
            arr[i] = 0.0

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "arr[i] = 1.0" in source
    assert "arr[i] = 0.0" not in source


def test_consteval_with_builtin_functions():
    """Test consteval with builtin functions."""
    ITEMS = [3, 1, 4, 1, 5]

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        m = consteval(max(ITEMS))
        n = consteval(min(ITEMS))
        s = consteval(sum(ITEMS))
        arr[i] = float(m + n + s)

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "m = 5" in source
    assert "n = 1" in source
    assert "s = 14" in source


# User-defined function call tests


def _compute_value():
    """Helper function defined at module level."""
    return 42


def _compute_with_args(a, b):
    """Helper with arguments."""
    return a * b + 10


class Config:
    """Helper class for testing method calls."""

    BLOCK_SIZE = 128

    @staticmethod
    def get_limit():
        return 32

    @classmethod
    def get_double_limit(cls):
        return cls.get_limit() * 2


def test_consteval_user_function_no_args():
    """Test consteval with user-defined function (no arguments)."""

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        val = consteval(_compute_value())
        arr[i] = float(val)

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "val = 42" in source


def test_consteval_user_function_with_args():
    """Test consteval with user-defined function with arguments."""

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        val = consteval(_compute_with_args(5, 6))
        arr[i] = float(val)

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    # 5 * 6 + 10 = 40
    assert "val = 40" in source


def test_consteval_user_function_runs():
    """Test that consteval with user function produces correct runtime results."""

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        arr[i] = float(consteval(_compute_value()))

    a = np.zeros(32, dtype=np.float32)
    d_a = cuda.to_device(a)
    kernel[1, 32](d_a)
    result = d_a.copy_to_host()

    assert all(result == 42.0)


def test_consteval_closure_function():
    """Test consteval with function defined in closure."""

    def make_kernel():
        def local_compute(x):
            return x * x

        @numba_cuda_mlir.cuda.jit
        def kernel(arr):
            i = cuda.threadIdx.x
            arr[i] = float(consteval(local_compute(7)))

        return kernel

    kernel = make_kernel()
    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "float(49)" in source


def test_consteval_static_method():
    """Test consteval with static method call."""

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        limit = consteval(Config.get_limit())
        if i < limit:
            arr[i] = 1.0

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "limit = 32" in source


def test_consteval_class_method():
    """Test consteval with class method call."""

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        limit = consteval(Config.get_double_limit())
        arr[i] = float(limit)

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "limit = 64" in source


def test_consteval_class_attribute():
    """Test consteval with class attribute access."""

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        bs = consteval(Config.BLOCK_SIZE)
        arr[i] = float(bs)

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "bs = 128" in source


def test_consteval_lambda():
    """Test consteval with lambda expression."""
    square = lambda x: x * x  # noqa: E731

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        arr[i] = float(consteval(square(8)))

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "float(64)" in source


def test_consteval_chained_function_calls():
    """Test consteval with chained/nested function calls."""

    def outer(x):
        return inner(x) + 1

    def inner(x):
        return x * 2

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        arr[i] = float(consteval(outer(5)))

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    # inner(5) = 10, outer(5) = 11
    assert "float(11)" in source


def test_consteval_function_with_consteval_arg():
    """Test consteval with function argument from another consteval."""
    BASE = 7

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        b = consteval(BASE)
        arr[i] = float(consteval(_compute_with_args(b, 3)))

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    # b=7, 7*3+10=31
    assert "b = 7" in source
    assert "float(31)" in source


# Tests for argument type access in consteval


def test_consteval_arg_type_dtype():
    """Test accessing array dtype via consteval."""
    from numba_cuda_mlir import types

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        # arr resolves to its Numba type in consteval context
        dtype = consteval(arr.dtype)
        # Check that dtype is float32
        if consteval(dtype == types.float32):
            arr[i] = 1.0
        else:
            arr[i] = 0.0

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    # The condition should be evaluated to True and folded
    assert "arr[i] = 1.0" in source


def test_consteval_arg_type_ndim():
    """Test accessing array ndim via consteval."""

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        ndim = consteval(arr.ndim)
        arr[i] = float(ndim)

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "ndim = 1" in source

    # Test with 2D array
    cres2 = kernel.compile("void(float32[:,:])")
    source2 = cres2.metadata["transformed_source"]
    assert "ndim = 2" in source2


def test_consteval_arg_type_print():
    """Test that printing an arg in consteval shows its type."""

    @numba_cuda_mlir.cuda.jit
    def kernel(arr, n):
        i = cuda.threadIdx.x
        # This should print the type, not the value
        consteval(print(arr))
        consteval(print(n))
        arr[i] = float(n)

    # Just verify it compiles without error
    cres = kernel.compile("void(float32[:], int32)")
    assert cres is not None


def test_consteval_arg_type_different_specializations():
    """Test that different arg types produce different consteval results."""
    from numba_cuda_mlir import types

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        dtype = consteval(arr.dtype)
        if consteval(dtype == types.float32):
            arr[i] = 32.0
        elif consteval(dtype == types.float64):
            arr[i] = 64.0

    # Compile for float32
    cres32 = kernel.compile("void(float32[:])")
    source32 = cres32.metadata["transformed_source"]
    assert "arr[i] = 32.0" in source32

    # Compile for float64
    cres64 = kernel.compile("void(float64[:])")
    source64 = cres64.metadata["transformed_source"]
    assert "arr[i] = 64.0" in source64


# Tests for target options access in consteval


def test_consteval_target_options_chip():
    """Test accessing chip target option in consteval."""
    from numba_cuda_mlir.cuda.experimental import current_target_options

    @numba_cuda_mlir.cuda.jit(chip="sm_90")
    def kernel(arr):
        i = cuda.threadIdx.x
        chip = consteval(current_target_options()["chip"])
        if consteval(chip == "sm_90"):
            arr[i] = 90.0
        else:
            arr[i] = 0.0

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "arr[i] = 90.0" in source


def test_consteval_target_options_opt_level():
    """Test accessing opt_level target option in consteval."""
    from numba_cuda_mlir.cuda.experimental import current_target_options

    # Use default opt_level (3) to avoid linker issues
    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        opt = consteval(current_target_options()["opt_level"])
        arr[i] = float(opt)

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    # Default opt_level is 3
    assert "opt = 3" in source


def test_consteval_target_options_fast_math():
    """Test accessing fastmath target option in consteval."""
    from numba_cuda_mlir.cuda.experimental import current_target_options

    @numba_cuda_mlir.cuda.jit(fastmath=True)
    def kernel(arr):
        i = cuda.threadIdx.x
        fm = consteval(current_target_options()["fastmath"])
        if consteval(fm):
            arr[i] = 1.0
        else:
            arr[i] = 0.0

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "arr[i] = 1.0" in source


def test_consteval_current_target_options():
    """Test using numba_cuda_mlir.current_target_options() syntax."""
    from numba_cuda_mlir import cuda

    @cuda.jit(chip="sm_80")
    def kernel(arr):
        i = cuda.threadIdx.x
        chip = consteval(cuda.current_target_options()["chip"])
        if consteval(chip.startswith("sm_8")):
            arr[i] = 1.0
        else:
            arr[i] = 0.0

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    # chip.startswith("sm_8") should be True, so the if branch should remain
    assert "arr[i] = 1.0" in source


# Tests for inspect_transformed_source returning a dict


def test_inspect_transformed_source_returns_dict():
    """Test that inspect_transformed_source() returns a dict."""

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        arr[i] = float(consteval(42))

    # Compile for multiple signatures
    kernel.compile("void(float32[:])")
    kernel.compile("void(float64[:])")

    # Should return a dict
    result = kernel.inspect_transformed_source()
    assert isinstance(result, dict)
    assert len(result) == 2


def test_inspect_transformed_source_with_signature():
    """Test that inspect_transformed_source(sig) returns the source for that sig."""
    from numba_cuda_mlir import types

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        ndim = consteval(arr.ndim)
        arr[i] = float(ndim)

    kernel.compile("void(float32[:])")
    kernel.compile("void(float32[:,:])")

    # Get source for specific signature
    source1d = kernel.inspect_transformed_source((types.Array(types.float32, 1, "C"),))
    source2d = kernel.inspect_transformed_source((types.Array(types.float32, 2, "C"),))

    assert "ndim = 1" in source1d
    assert "ndim = 2" in source2d


# Tests for `with consteval():` blocks


def test_consteval_block_basic():
    """Test basic consteval block execution and variable extraction."""
    VALUE = 42

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        with consteval():
            x = VALUE
            y = x * 2
        arr[i] = float(consteval(y))

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert source is not None
    # Block should be removed
    assert "with" not in source
    # Value should be extracted
    assert "float(84)" in source


def test_consteval_block_runs_correctly():
    """Test that consteval block produces correct runtime results."""
    SCALE = 3

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        with consteval():
            factor = SCALE * 2
        arr[i] = float(consteval(factor))

    a = np.zeros(32, dtype=np.float32)
    d_a = cuda.to_device(a)
    kernel[1, 32](d_a)
    result = d_a.copy_to_host()

    assert all(result == 6.0)


def test_consteval_block_multiple_statements():
    """Test consteval block with multiple statements."""

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        with consteval():
            a = 1
            b = 2
            c = a + b
            d = c * 10
        arr[i] = float(consteval(d))

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "float(30)" in source


def test_consteval_block_with_conditionals():
    """Test consteval block with conditional logic."""
    USE_LARGE = True

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        with consteval():
            if USE_LARGE:
                size = 1024
            else:
                size = 64
        arr[i] = float(consteval(size))

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "float(1024)" in source


def test_consteval_block_with_loop():
    """Test consteval block with a loop inside."""
    ITEMS = [1, 2, 3, 4, 5]

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        with consteval():
            total = 0
            for item in ITEMS:
                total += item
        arr[i] = float(consteval(total))

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "float(15)" in source


def test_consteval_block_with_function_call():
    """Test consteval block calling a function."""

    def compute_value(x, y):
        return x * y + 10

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        with consteval():
            result = compute_value(5, 6)
        arr[i] = float(consteval(result))

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    # 5 * 6 + 10 = 40
    assert "float(40)" in source


def test_consteval_block_as_var_raises():
    """Test that 'with consteval() as x:' raises an error."""

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        with consteval() as x:
            pass
        arr[i] = 1.0

    with pytest.raises(ConstevalError, match="as x.*not supported"):
        kernel.compile("void(float32[:])")


def test_consteval_block_error_propagation():
    """Test that errors in consteval blocks are propagated as ConstevalError."""

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        with consteval():
            x = undefined_variable  # noqa: F821
        arr[i] = float(consteval(x))

    with pytest.raises(ConstevalError, match="Error executing consteval block"):
        kernel.compile("void(float32[:])")


def test_consteval_block_with_complex_objects():
    """Test consteval block with complex objects that need storage."""
    CONFIG = {"a": 1, "b": 2}

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        with consteval():
            val = CONFIG["a"] + CONFIG["b"]
        arr[i] = float(consteval(val))

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "float(3)" in source


def test_consteval_block_nested():
    """Test nested consteval blocks."""
    OUTER = 10

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        with consteval():
            x = OUTER
            with consteval():
                y = x * 2
            z = y + 5
        arr[i] = float(consteval(z))

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    # x=10, y=20, z=25
    assert "float(25)" in source


def test_consteval_block_with_target_options():
    """Test consteval block accessing target options."""
    from numba_cuda_mlir.cuda.experimental import current_target_options

    @numba_cuda_mlir.cuda.jit(chip="sm_80")
    def kernel(arr):
        i = cuda.threadIdx.x
        with consteval():
            chip = current_target_options()["chip"]
            is_sm80 = chip == "sm_80"
        if consteval(is_sm80):
            arr[i] = 80.0
        else:
            arr[i] = 0.0

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "arr[i] = 80.0" in source


def test_consteval_runtime_error_without_ast_transforms():
    """Test that consteval() raises an error when AST transforms are disabled."""
    from numba_cuda_mlir.cuda.experimental import consteval as ce

    with pytest.raises(RuntimeError, match="experimental_ast_transforms"):
        ce(42)


def test_consteval_block_runtime_error_without_ast_transforms():
    """Test that 'with consteval():' raises an error when AST transforms are disabled."""
    from numba_cuda_mlir.cuda.experimental import consteval as ce

    with pytest.raises(RuntimeError, match="experimental_ast_transforms"):
        with ce():
            pass


def test_consteval_block_multiple_extractions():
    """Test extracting multiple values from a consteval block."""
    BASE = 100

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        with consteval():
            a = BASE
            b = BASE * 2
            c = BASE * 3
        x = consteval(a)
        y = consteval(b)
        z = consteval(c)
        arr[i] = float(x + y + z)

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "x = 100" in source
    assert "y = 200" in source
    assert "z = 300" in source


# Unit tests for consteval blocks that don't require GPU
# These test the AST transformer directly


def test_consteval_block_unit_basic():
    """Unit test: basic consteval block transformation."""
    import ast
    from numba_cuda_mlir.ast_transforms.consteval import transform_consteval

    VALUE = 42

    def func():
        with consteval():
            x = VALUE
            y = x * 2

    tree = ast.parse("def func():\n    with consteval():\n        x = VALUE\n        y = x * 2\n")
    func.__globals__["VALUE"] = VALUE
    func.__globals__["consteval"] = consteval

    new_tree, modified, stored = transform_consteval(func, tree)
    source = ast.unparse(new_tree)

    assert modified
    assert "with" not in source
    # The block should be completely removed


def test_consteval_block_unit_variable_tracking():
    """Unit test: variables from block are tracked and usable in consteval()."""
    import ast
    from numba_cuda_mlir.ast_transforms.consteval import ConstevalTransformer
    from numba_cuda_mlir.ast_transforms.common import get_function_ast

    VALUE = 10

    def func():
        with consteval():
            x = VALUE * 2
        y = consteval(x + 5)

    func.__globals__["VALUE"] = VALUE
    func.__globals__["consteval"] = consteval

    tree = get_function_ast(func)
    transformer = ConstevalTransformer(func)
    new_tree = transformer.visit(tree)
    source = ast.unparse(new_tree)

    assert transformer.modified
    assert "with" not in source
    # x should be 20, so y should be 25
    assert "y = 25" in source


def test_consteval_block_unit_as_var_error():
    """Unit test: 'with consteval() as x:' raises error."""
    import ast
    from numba_cuda_mlir.ast_transforms.consteval import ConstevalTransformer

    def func():
        with consteval() as x:
            pass

    func.__globals__["consteval"] = consteval

    code = "def func():\n    with consteval() as x:\n        pass\n"
    tree = ast.parse(code)

    transformer = ConstevalTransformer(func)
    with pytest.raises(ConstevalError, match="as x.*not supported"):
        transformer.visit(tree)


def test_consteval_block_unit_error_propagation():
    """Unit test: errors in block propagate as ConstevalError."""
    import ast
    from numba_cuda_mlir.ast_transforms.consteval import ConstevalTransformer

    def func():
        with consteval():
            x = undefined_var  # noqa: F821

    func.__globals__["consteval"] = consteval

    code = "def func():\n    with consteval():\n        x = undefined_var\n"
    tree = ast.parse(code)

    transformer = ConstevalTransformer(func)
    with pytest.raises(ConstevalError, match="Error executing consteval block"):
        transformer.visit(tree)


def test_consteval_block_unit_multiple_statements():
    """Unit test: multiple statements in block are all executed."""
    import ast
    from numba_cuda_mlir.ast_transforms.consteval import ConstevalTransformer
    from numba_cuda_mlir.ast_transforms.common import get_function_ast

    def func():
        with consteval():
            a = 1
            b = a + 1
            c = b + 1
        result = consteval(c)

    func.__globals__["consteval"] = consteval

    tree = get_function_ast(func)
    transformer = ConstevalTransformer(func)
    new_tree = transformer.visit(tree)
    source = ast.unparse(new_tree)

    assert "result = 3" in source


def test_consteval_block_unit_with_loop():
    """Unit test: loops inside consteval block work."""
    import ast
    from numba_cuda_mlir.ast_transforms.consteval import ConstevalTransformer
    from numba_cuda_mlir.ast_transforms.common import get_function_ast

    ITEMS = [1, 2, 3]

    def func():
        with consteval():
            total = 0
            for x in ITEMS:
                total += x
        result = consteval(total)

    func.__globals__["consteval"] = consteval
    func.__globals__["ITEMS"] = ITEMS

    tree = get_function_ast(func)
    transformer = ConstevalTransformer(func)
    new_tree = transformer.visit(tree)
    source = ast.unparse(new_tree)

    assert "result = 6" in source


def test_consteval_block_unit_nested():
    """Unit test: nested consteval blocks work."""
    import ast
    from numba_cuda_mlir.ast_transforms.consteval import ConstevalTransformer
    from numba_cuda_mlir.ast_transforms.common import get_function_ast

    def func():
        with consteval():
            x = 5
            with consteval():
                y = x * 2
            z = y + 1
        result = consteval(z)

    func.__globals__["consteval"] = consteval

    tree = get_function_ast(func)
    transformer = ConstevalTransformer(func)
    new_tree = transformer.visit(tree)
    source = ast.unparse(new_tree)

    # x=5, y=10, z=11
    assert "result = 11" in source


def test_consteval_block_unit_conditional():
    """Unit test: conditionals in consteval block work."""
    import ast
    from numba_cuda_mlir.ast_transforms.consteval import ConstevalTransformer
    from numba_cuda_mlir.ast_transforms.common import get_function_ast

    FLAG = True

    def func():
        with consteval():
            if FLAG:
                val = 100
            else:
                val = 0
        result = consteval(val)

    func.__globals__["consteval"] = consteval
    func.__globals__["FLAG"] = FLAG

    tree = get_function_ast(func)
    transformer = ConstevalTransformer(func)
    new_tree = transformer.visit(tree)
    source = ast.unparse(new_tree)

    assert "result = 100" in source


def test_consteval_block_unit_function_call():
    """Unit test: function calls in consteval block work."""
    import ast
    from numba_cuda_mlir.ast_transforms.consteval import ConstevalTransformer
    from numba_cuda_mlir.ast_transforms.common import get_function_ast

    def compute(a, b):
        return a * b + 1

    def func():
        with consteval():
            val = compute(3, 4)
        result = consteval(val)

    func.__globals__["consteval"] = consteval
    func.__globals__["compute"] = compute

    tree = get_function_ast(func)
    transformer = ConstevalTransformer(func)
    new_tree = transformer.visit(tree)
    source = ast.unparse(new_tree)

    # 3*4+1 = 13
    assert "result = 13" in source
