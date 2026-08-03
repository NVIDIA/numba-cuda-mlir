# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for _is_valid_memref_element_type — ensuring that externally defined
extension types whose MLIR representation is an ``!llvm.struct`` are routed
through ``llvm.alloca`` / ``llvm.load`` / ``llvm.store`` rather than
``memref.alloca``, which rejects non-builtin element types.

When a variable is assigned more than once in a kernel, numba_cuda_mlir stack-allocates
it.  For builtin scalars (``i32``, ``f64``, …) a ``memref<1xT>`` works fine.
For LLVM dialect types (``!llvm.struct<…>``), ``memref`` is not valid and
``llvm.alloca`` must be used instead.  This situation arises with any
extension type whose data model lowers to ``!llvm.struct<…>``.

The integration tests below define a minimal extension type that lowers to a
padded ``!llvm.struct<(i8, i64)>`` and force multi-assignment so the stack
allocation path is exercised end-to-end.
"""

import numpy as np

from numba_cuda_mlir import cuda, extending, types
from numba_cuda_mlir.extending import lower_cast, lowering_registry
from numba_cuda_mlir.lowering_utilities import constant, convert
from numba_cuda_mlir.models import PrimitiveModel, mlir_data_manager, register_model
from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir._mlir.dialects import llvm
from numba_cuda_mlir.numba_cuda.typeconv import Conversion


# ---------------------------------------------------------------------------
# Custom extension type: i8 and i64 fields in a padded LLVM struct.
# Represents a (value, valid_bit) pair — a minimal "masked" scalar.
# ---------------------------------------------------------------------------


class MiniMaskedType(types.Type):
    def __init__(self):
        super().__init__(name="MiniMasked")

    @property
    def key(self):
        return self.__class__

    def can_convert_to(self, typingctx, other):
        if isinstance(other, types.Integer):
            return Conversion.safe
        return None

    def can_convert_from(self, typingctx, other):
        if isinstance(other, types.Integer):
            return Conversion.safe
        return None


mini_masked = MiniMaskedType()


@register_model(MiniMaskedType)
class MiniMaskedModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        i8 = mlir_ir.IntegerType.get_signless(8)
        i64 = mlir_ir.IntegerType.get_signless(64)
        be_type = llvm.StructType.get_literal([i8, i64])
        super().__init__(dmm, fe_type, be_type)


# ---------------------------------------------------------------------------
# Constructor: make_masked(value, valid) -> MiniMasked
# ---------------------------------------------------------------------------


def make_masked(value, valid):
    raise NotImplementedError("only callable inside a numba_cuda_mlir kernel")


@extending.type_callable(make_masked)
def _type_make_masked(context):
    def typer(value, valid):
        if isinstance(value, types.Integer) and isinstance(valid, types.Integer):
            return mini_masked

    return typer


@lowering_registry.lower(make_masked, types.Integer, types.Integer)
def _lower_make_masked(builder, target, args, kwargs):
    value = builder.load_var(args[0])
    valid = builder.load_var(args[1])
    i8 = mlir_ir.IntegerType.get_signless(8)
    i64 = mlir_ir.IntegerType.get_signless(64)
    value = convert(value, i8)
    valid = convert(valid, i64)
    struct_ty = builder.get_mlir_type(mini_masked)
    undef = llvm.UndefOp(struct_ty)
    with_value = llvm.insertvalue(
        container=undef,
        value=value,
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )
    result = llvm.insertvalue(
        container=with_value,
        value=valid,
        position=mlir_ir.DenseI64ArrayAttr.get([1]),
    )
    builder.store_var(target, result)


# ---------------------------------------------------------------------------
# @lower_cast: MiniMasked -> int32 (extract value field)
# ---------------------------------------------------------------------------


@lower_cast(MiniMaskedType, types.Integer)
def _cast_mini_masked_to_int(context, builder, fromty, toty, val):
    result_ty = builder.get_mlir_type(toty)
    stored_ty = mlir_ir.IntegerType.get_signless(8)
    stored = llvm.extractvalue(
        res=stored_ty,
        container=val,
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )
    return convert(stored, result_ty)


@lower_cast(types.Integer, MiniMaskedType)
def _cast_int_to_mini_masked(context, builder, fromty, toty, val):
    i8 = mlir_ir.IntegerType.get_signless(8)
    i64 = mlir_ir.IntegerType.get_signless(64)
    struct_ty = builder.get_mlir_type(toty)
    undef = llvm.UndefOp(struct_ty)
    with_value = llvm.insertvalue(
        container=undef,
        value=convert(val, i8),
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )
    return llvm.insertvalue(
        container=with_value,
        value=constant(1, i64),
        position=mlir_ir.DenseI64ArrayAttr.get([1]),
    )


class _PointerArray:
    """Keep a device pointer array alive while overriding its Numba dtype."""

    def __init__(self, array):
        self._array = array

    @property
    def __cuda_array_interface__(self):
        return self._array.__cuda_array_interface__


@extending.typeof_impl.register(_PointerArray)
def _typeof_pointer_array(value, context):
    return types.Array(types.CPointer(types.int32), 1, "C")


extending.refresh_registries()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_local_array_public_shape_contract_remains_static():
    """LLVM-backed storage does not broaden CUDA's literal-shape API."""

    from numba_cuda_mlir.typing.cuda import LocalArrayTemplate

    assert LocalArrayTemplate.allow_dynamic_shape is False


def test_extension_array_model_uses_byte_backed_memref():
    """Array models must not form memrefs with LLVM-dialect elements."""

    with mlir_ir.Context(), mlir_ir.Location.unknown():
        array_type = types.Array(mini_masked, 1, "C")
        model = mlir_data_manager.lookup(array_type)

        assert str(model.get_value_type()).startswith("memref<?xi8")


def test_extension_type_multi_assign_uses_alloca():
    """A variable of extension type (!llvm.struct) assigned multiple times
    must use llvm.alloca — not memref.alloca which would be invalid.

    ``m`` is assigned twice (both as MiniMasked) so
    allocate_stack_space_for_vars_with_multiple_assigns fires.  Without the
    _is_valid_memref_element_type guard this would crash at MLIR verification
    because memref<!llvm.struct<(i8, i64)>> is not legal.

    Branch unification (``m`` vs ``int32``) forces a cast that reads ``m``
    back from its alloca slot, verifying the full store-load round trip.
    """

    @cuda.jit
    def kernel(flag, out):
        m = make_masked(1, 0)
        if flag[0]:
            m = make_masked(42, 1)
        if flag[0]:
            x = m
        else:
            x = np.int32(99)
        out[0] = x

    flag = np.array([True], dtype=np.bool_)
    out = np.zeros(1, dtype=np.int32)
    kernel[1, 1](flag, out)
    assert out[0] == 42, f"expected 42, got {out[0]}"

    flag[0] = False
    out[0] = 0
    kernel[1, 1](flag, out)
    assert out[0] == 99, f"expected 99, got {out[0]}"


def test_extension_type_loop_reassign():
    """Extension type variable re-assigned inside a loop.

    Each iteration overwrites ``m`` via llvm.store into its alloca slot.
    After the loop, branch unification reads ``m`` back via llvm.load,
    verifying that store/load work correctly across iterations.
    """

    @cuda.jit
    def kernel(n, out):
        m = make_masked(0, 0)
        for i in range(n[0]):
            m = make_masked(i, 1)
        if n[0] > 0:
            x = m
        else:
            x = np.int32(-1)
        out[0] = x

    n = np.array([5], dtype=np.int32)
    out = np.zeros(1, dtype=np.int32)
    kernel[1, 1](n, out)
    assert out[0] == 4, f"expected 4, got {out[0]}"

    n[0] = 1
    out[0] = 0
    kernel[1, 1](n, out)
    assert out[0] == 0, f"expected 0, got {out[0]}"

    n[0] = 0
    out[0] = 0
    kernel[1, 1](n, out)
    assert out[0] == -1, f"expected -1, got {out[0]}"


def test_extension_type_local_array_round_trip():
    """Local arrays preserve logical shape while storing LLVM-backed values."""

    @cuda.jit
    def kernel(out):
        values = cuda.local.array(4, dtype=mini_masked)
        for i in range(len(values)):
            values[i] = make_masked(i + 10, 1)
        for i in range(len(values)):
            out[i] = values[i]

    out = np.zeros(4, dtype=np.int32)
    kernel[1, 1](out)
    np.testing.assert_array_equal(out, np.arange(10, 14, dtype=np.int32))

    mlir = next(iter(kernel.inspect_mlir().values()))
    assert "!llvm.struct<(i8, i64)>" in mlir


def test_extension_type_local_array_crosses_device_function_boundary():
    """Generic pointer reconstruction survives storage-origin erasure."""

    @cuda.jit(device=True, inline="never")
    def read(values, index):
        return values[index]

    @cuda.jit
    def kernel(out):
        values = cuda.local.array(2, dtype=mini_masked)
        values[0] = make_masked(77, 1)
        values[1] = make_masked(88, 1)
        out[0] = read(values, 0)

    out = np.zeros(1, dtype=np.int32)
    kernel[1, 1](out)
    assert out[0] == 77

    mlir = next(iter(kernel.inspect_mlir().values()))
    assert "func.func" in mlir
    assert "memref<?xi8" in mlir


def test_extension_type_local_array_store_casts():
    """Scalar, tuple-indexed, and slice stores cast to the array dtype."""

    @cuda.jit
    def kernel(out):
        scalar = cuda.local.array(1, dtype=mini_masked)
        matrix = cuda.local.array((2, 2), dtype=mini_masked)
        sliced = cuda.local.array(3, dtype=mini_masked)

        scalar[0] = np.int32(31)
        matrix[1, 1] = np.int32(41)
        sliced[1:] = np.int32(51)

        out[0] = scalar[0]
        out[1] = matrix[1, 1]
        out[2] = sliced[1]
        out[3] = sliced[2]

    out = np.zeros(4, dtype=np.int32)
    kernel[1, 1](out)
    np.testing.assert_array_equal(out, np.array([31, 41, 51, 51], dtype=np.int32))


def test_llvm_backed_global_pointer_array_round_trip():
    """The same generic byte-backed ABI can access global pointer arrays."""

    @cuda.jit
    def kernel(pointers, out):
        out[0] = pointers[0][0]

    pointee = cuda.to_device(np.array([123], dtype=np.int32))
    pointer_value = pointee.__cuda_array_interface__["data"][0]
    pointer_bits = cuda.to_device(np.array([pointer_value], dtype=np.uint64))
    pointers = _PointerArray(pointer_bits)
    out = cuda.to_device(np.zeros(1, dtype=np.int32))

    kernel[1, 1](pointers, out)
    assert out.copy_to_host()[0] == 123

    mlir = next(iter(kernel.inspect_mlir().values()))
    assert "memref<?xi8" in mlir
    assert "llvm.inttoptr" in mlir


def test_extension_type_multidimensional_local_array_round_trip():
    """Logical multidimensional strides address adjacent extension values."""

    @cuda.jit
    def kernel(out):
        values = cuda.local.array((2, 3), dtype=mini_masked)
        for row in range(2):
            for column in range(3):
                values[row, column] = make_masked(row * 10 + column, 1)
        for row in range(2):
            for column in range(3):
                out[row, column] = values[row, column]

    out = np.zeros((2, 3), dtype=np.int32)
    kernel[1, 1](out)
    np.testing.assert_array_equal(
        out,
        np.array([[0, 1, 2], [10, 11, 12]], dtype=np.int32),
    )


def test_extension_type_local_array_slice_preserves_offset():
    """Subview offsets remain logical element offsets for LLVM-backed values."""

    @cuda.jit
    def kernel(out):
        values = cuda.local.array(5, dtype=mini_masked, alignment=16)
        for i in range(len(values)):
            values[i] = make_masked(i + 20, 1)
        tail = values[2:]
        for i in range(len(tail)):
            out[i] = tail[i]

    out = np.zeros(3, dtype=np.int32)
    kernel[1, 1](out)
    np.testing.assert_array_equal(out, np.arange(22, 25, dtype=np.int32))

    mlir = next(iter(kernel.inspect_mlir().values()))
    assert "llvm.alloca" in mlir
    assert "alignment = 16" in mlir
