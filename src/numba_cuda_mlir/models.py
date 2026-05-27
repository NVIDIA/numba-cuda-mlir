# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numba_cuda_mlir.errors import InternalCompilerError
import functools

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.ir import (
    NoneType,
    IntegerType,
    MemRefType,
    ShapedType,
    F16Type,
    F32Type,
    F64Type,
    ComplexType,
)
from numba_cuda_mlir._mlir.dialects import llvm
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir import types
from numba_cuda_mlir.numba_cuda.datamodel.models import PrimitiveModel, DataModel
from numba_cuda_mlir.numba_cuda.datamodel.registry import DataModelManager, register
from numba_cuda_mlir.numba_cuda.types.ext_types import GridGroup as GridGroupClass
from numba_cuda_mlir.type_defs import float_types


class ContextAwareDataModelManager(DataModelManager):
    """DataModelManager that invalidates cached models when the MLIR context changes.

    Numba's DataModelManager caches model instances keyed by numba type.
    Each model stores MLIR types (be_type) tied to the ir.Context active at
    creation time.  If a lookup created the model under context A and a later
    lookup happens under context B, the stale MLIR types cause verification
    failures ("type from a different MLIRContext").

    This subclass checks the cached model's context on every lookup and
    evicts + rebuilds when there is a mismatch.
    """

    def lookup(self, fetype):
        if cached := self._cache.get(fetype):
            be_type = cached.be_type
            sample = be_type[0] if isinstance(be_type, tuple) and be_type else be_type
            try:
                current = ir.Context.current
            except ValueError:
                current = None
            if (
                sample is None
                or not hasattr(sample, "context")
                or current is None
                or sample.context is current
            ):
                return cached
            del self._cache[fetype]

        handler = self._handlers[type(fetype)]
        model = self._cache[fetype] = handler(self, fetype)
        return model


mlir_data_manager = ContextAwareDataModelManager()

register_model = functools.partial(register, mlir_data_manager)


class StructModel(DataModel):
    """Vendored struct model backed by MLIR LLVM dialect types."""

    def __init__(self, dmm, fe_type, members):
        super().__init__(dmm, fe_type)
        if members:
            self._fields, self._members = zip(*members)
        else:
            self._fields = self._members = ()
        self._models = tuple([self._dmm.lookup(t) for t in self._members])
        field_types = [model.get_value_type() for model in self._models]
        self.be_type = llvm.StructType.get_literal(field_types)

    def get_member_fe_type(self, name):
        pos = self.get_field_position(name)
        return self._members[pos]

    def get_value_type(self):
        return self.be_type

    def get_data_type(self):
        return self.be_type

    def get_argument_type(self):
        return self.get_value_type()

    def get_return_type(self):
        return self.get_value_type()

    def as_data(self, builder, value):
        return value

    def as_argument(self, builder, value):
        return value

    def as_return(self, builder, value):
        return value

    def from_data(self, builder, value):
        return value

    def from_argument(self, builder, value):
        return value

    def from_return(self, builder, value):
        return value

    def get_field_position(self, field):
        try:
            return self._fields.index(field)
        except ValueError:
            raise KeyError("%s does not have a field named %r" % (self.__class__.__name__, field))

    @property
    def field_count(self):
        return len(self._fields)

    def get_type(self, pos):
        if isinstance(pos, str):
            pos = self.get_field_position(pos)
        return self._members[pos]

    def get_model(self, pos):
        return self._models[pos]

    def inner_models(self):
        return self._models


@register_model(types.DTypeSpec)
class DTypeSpecModel(PrimitiveModel):
    """
    Convert from a Numba DTypeSpec type instance
    into the MLIR type that this DTypeSpec instance represents.
    For example, a NumberClass instance
        class(float64)
    is converted into a F64Type in MLIR.
    """

    def __init__(self, dmm, fe_type):
        be_type = dmm.lookup(fe_type.dtype).get_value_type()
        super().__init__(dmm, fe_type, be_type)


@register_model(GridGroupClass)
class GridGroupModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = IntegerType.get_signless(64)
        super().__init__(dmm, fe_type, be_type)


@register_model(types.Function)
@register_model(types.Module)
@register_model(types.NoneType)
class OpaqueModel(PrimitiveModel):
    """Passed as opaque pointers"""

    def __init__(self, dmm, fe_type):
        be_type = NoneType.get()
        super().__init__(dmm, fe_type, be_type)


@register_model(types.Tuple)
class TupleModel(PrimitiveModel):
    """
    Assumption: Every element of the tuple type can resolve into
    the same MLIR type.
    """

    def __init__(self, dmm, fe_type):
        be_type = tuple(map(lambda t: dmm.lookup(t).get_value_type(), fe_type.types))
        super().__init__(dmm, fe_type, be_type)


@register_model(types.CPointer)
@register_model(types.RawPointer)
class CPointerModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        from numba_cuda_mlir._mlir.dialects import llvm

        be_type = llvm.PointerType.get()
        super().__init__(dmm, fe_type, be_type)


@register_model(types.StringLiteral)
class StringLiteralModel(PrimitiveModel):
    """String literals in kernels: value type is !llvm.ptr (i8*).

    String representation in numba_cuda_mlir follows numba-cuda: pointer-only, null-terminated
    (UTF-8 bytes plus null terminator in constant memory when materialized). We do not use a string_view
    (ptr+len) or memref<?xi8> so we stay aligned with numba-cuda and keep lowering
    simple. For literal-only use (e.g. st == "a") we constant-fold and never emit
    the pointer; when we need a value (e.g. print, or passing to a function), we
    would insert a global constant and use its address as !llvm.ptr.
    """

    def __init__(self, dmm, fe_type):
        be_type = llvm.PointerType.get()
        super().__init__(dmm, fe_type, be_type)


@register_model(types.UnicodeType)
class UnicodeTypeModel(PrimitiveModel):
    """Runtime unicode string type matching numba-cuda's UnicodeModel.

    The struct layout is:
      (data: ptr, length: i64, kind: i32, is_ascii: u32,
       hash: i64, meminfo: ptr, parent: ptr)
    """

    _fields = ("data", "length", "kind", "is_ascii", "hash", "meminfo", "parent")

    def __init__(self, dmm, fe_type):
        import sys

        hash_width = sys.hash_info.width
        be_type = llvm.StructType.get_literal(
            [
                llvm.PointerType.get(),  # data (voidptr)
                ir.IntegerType.get_signless(64),  # length (intp)
                ir.IntegerType.get_signless(32),  # kind (int32)
                ir.IntegerType.get_signless(32),  # is_ascii (uint32)
                ir.IntegerType.get_signless(hash_width),  # hash (_Py_hash_t)
                llvm.PointerType.get(),  # meminfo
                llvm.PointerType.get(),  # parent
            ]
        )
        super().__init__(dmm, fe_type, be_type)

    def has_nrt_meminfo(self):
        return True

    def get_nrt_meminfo(self, value):
        from numba_cuda_mlir.mlir.dialect_exts import llvm as llvm_ext

        return llvm_ext.extractvalue(llvm.PointerType.get(), value, [self._fields.index("meminfo")])

    def get_field_position(self, name):
        return self._fields.index(name)


@register_model(types.CharSeq)
class CharSeqModel(PrimitiveModel):
    """Fixed-length 8-bit character sequence (numpy dtype 'S').

    Represented as !llvm.array<count x i8>.
    """

    def __init__(self, dmm, fe_type):
        be_type = ir.Type.parse(f"!llvm.array<{fe_type.count} x i8>")
        super().__init__(dmm, fe_type, be_type)


@register_model(types.UnicodeCharSeq)
class UnicodeCharSeqModel(PrimitiveModel):
    """Fixed-length unicode character sequence (numpy dtype 'U').

    Represented as !llvm.array<count x i32> (sizeof_unicode_char == 4).
    """

    def __init__(self, dmm, fe_type):
        import numpy as np

        unicode_byte_width = np.dtype("U1").itemsize
        char_bits = unicode_byte_width * 8
        be_type = ir.Type.parse(f"!llvm.array<{fe_type.count} x i{char_bits}>")
        super().__init__(dmm, fe_type, be_type)


@register_model(types.UniTuple)
class UniTupleModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        ele_ty = dmm.lookup(fe_type.dtype).get_value_type()
        be_type = (ele_ty,) * fe_type.count
        super().__init__(dmm, fe_type, be_type)


@register_model(types.Optional)
class OptionalModel(StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(
            dmm,
            fe_type,
            [
                ("data", fe_type.type),
                ("valid", types.boolean),
            ],
        )


@register_model(types.UnionType)
class UnionType(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        """
        All types in a union should result in the same MLIR type.
        Otherwise, raise an error.
        """
        ele_ty = dmm.lookup(fe_type.types[0]).get_value_type()
        for type in fe_type.types[1:]:
            ty = dmm.lookup(type).get_value_type()
            assert ele_ty == ty, (
                f"UnionType results in different MLIR types between {ele_ty} and {ty}."
            )
        super().__init__(dmm, fe_type, ele_ty)


@register_model(types.Array)
class ArrayModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        from numba_cuda_mlir.types import Record

        # For Record, CharSeq, and UnicodeCharSeq arrays, use byte-based
        # memref (memref<?xi8>).  Elements are accessed via byte offset
        # pointer arithmetic.
        if isinstance(fe_type.dtype, (Record, types.CharSeq, types.UnicodeCharSeq)):
            shape = [ShapedType.get_dynamic_size() for _ in range(fe_type.ndim)]

            dyn_stride = MemRefType.get_dynamic_stride_or_offset()
            layout = ir.StridedLayoutAttr.get(
                offset=dyn_stride,
                strides=[dyn_stride] * fe_type.ndim,
            )
            be_type = MemRefType.get(shape, IntegerType.get_signless(8), layout=layout)
            super().__init__(dmm, fe_type, be_type)
            return

        ele_ty = dmm.lookup(fe_type.dtype).get_value_type()
        shape = [ShapedType.get_dynamic_size() for _ in range(fe_type.ndim)]

        # Create strided layout with all dynamic strides
        # This allows the memref to work with any stride pattern (row-major, column-major, etc.)

        dyn_stride = MemRefType.get_dynamic_stride_or_offset()
        layout = ir.StridedLayoutAttr.get(
            offset=dyn_stride,  # dynamic offset
            strides=[dyn_stride] * fe_type.ndim,  # all strides are dynamic
        )

        # This will create memref<?x?xf32> with strided layout for 2D arrays
        if isinstance(ele_ty, IntegerType):
            be_type = MemRefType.get(shape, IntegerType.get_signless(ele_ty.width), layout=layout)
        else:
            be_type = MemRefType.get(shape, ele_ty, layout=layout)
        super().__init__(dmm, fe_type, be_type)


@register_model(types.Boolean)
@register_model(types.BooleanLiteral)
class BooleanModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = IntegerType.get_signless(1)
        super().__init__(dmm, fe_type, be_type)


from numba_cuda_mlir.type_defs.vector_types import VectorType


@register_model(VectorType)
class VectorTypeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        ele_ty = dmm.lookup(fe_type.dtype).get_value_type()
        be_type = ir.VectorType.get(list(fe_type.shape), ele_ty)
        super().__init__(dmm, fe_type, be_type)


@register_model(types.NPDatetime)
@register_model(types.NPTimedelta)
class DatetimeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, T.i64())


@register_model(types.Integer)
@register_model(types.IntegerLiteral)
class IntegerModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = IntegerType.get_signless(fe_type.bitwidth)
        super().__init__(dmm, fe_type, be_type)


@register_model(types.Float)
class FloatModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        match fe_type.bitwidth:
            case 16:
                be_type = F16Type.get()
            case 32:
                be_type = F32Type.get()
            case 64:
                be_type = F64Type.get()
            case _:
                raise ValueError(f"Cannot convert type {str(fe_type)} to MLIR type.")
        super().__init__(dmm, fe_type, be_type)


@register_model(types.Complex)
class ComplexModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        match fe_type.bitwidth:
            case 32:
                be_type = ComplexType.get(F16Type.get())
            case 64:
                be_type = ComplexType.get(F32Type.get())
            case 128:
                be_type = ComplexType.get(F64Type.get())
            case _:
                raise ValueError(f"Cannot convert type {str(fe_type)} to complex MLIR type.")
        super().__init__(dmm, fe_type, be_type)


@register_model(types.RangeType)
class RangeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        element_type = dmm.lookup(fe_type.dtype).get_value_type()
        be_type = T.memref(5, element_type)
        super().__init__(dmm, fe_type, be_type)


@register_model(types.RangeIteratorType)
class RangeIteratorModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        element_type = dmm.lookup(fe_type.yield_type).get_value_type()
        be_type = T.memref(1, element_type)
        super().__init__(dmm, fe_type, be_type)


@register_model(float_types.BFloat16Type)
class BFloat16TypeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = T.bf16()
        super().__init__(dmm, fe_type, be_type)


# Register model for numba-cuda's Bfloat16 type
from numba_cuda_mlir.numba_cuda.types.ext_types import Bfloat16


@register_model(Bfloat16)
class NumbaCudaBfloat16Model(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = T.bf16()
        super().__init__(dmm, fe_type, be_type)


@register_model(float_types.NVFP4Type)
class NVFP4TypeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        raise NotImplementedError("NYI")
        be_type = T.nvfp4()
        super().__init__(dmm, fe_type, be_type)


@register_model(float_types.Float4E2M1FNType)
class Float4E2M1FNTypeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = T.f4E2M1FN()
        super().__init__(dmm, fe_type, be_type)


@register_model(float_types.Float6E2M3FNType)
class Float6E2M3FNTypeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = T.f6E2M3FN()
        super().__init__(dmm, fe_type, be_type)


@register_model(float_types.Float6E3M2FNType)
class Float6E3M2FNTypeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = T.f6E3M2FN()
        super().__init__(dmm, fe_type, be_type)


@register_model(float_types.Float8E3M4Type)
class Float8E3M4TypeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = T.f8E3M4()
        super().__init__(dmm, fe_type, be_type)


@register_model(float_types.Float8E4M3B11FNUZType)
class Float8E4M3B11FNUZTypeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = T.f8E4M3B11FNUZ()
        super().__init__(dmm, fe_type, be_type)


@register_model(float_types.Float8E4M3FNType)
class Float8E4M3FNTypeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = T.f8E4M3FN()
        super().__init__(dmm, fe_type, be_type)


@register_model(float_types.Float8E4M3FNUZType)
class Float8E4M3FNUZTypeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = ir.Float8E4M3FNUZType.get()
        super().__init__(dmm, fe_type, be_type)


@register_model(float_types.Float8E4M3Type)
class Float8E4M3TypeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = T.f8E4M3()
        super().__init__(dmm, fe_type, be_type)


@register_model(float_types.Float8E5M2FNUZType)
class Float8E5M2FNUZTypeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = ir.Float8E5M2FNUZType.get()
        super().__init__(dmm, fe_type, be_type)


@register_model(float_types.Float8E5M2Type)
class Float8E5M2TypeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = T.f8E5M2()
        super().__init__(dmm, fe_type, be_type)


@register_model(float_types.Float8E8M0FNUType)
class Float8E8M0FNUTypeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = T.f8E8M0FNU()
        super().__init__(dmm, fe_type, be_type)


@register_model(float_types.FloatTF32Type)
class FloatTF32TypeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = T.tf32()
        super().__init__(dmm, fe_type, be_type)


@register_model(types.AggregateType)
class AggregateTypeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        # Check if this is a bitfield struct
        if fe_type.is_bitfield_struct:
            # Bitfield struct: all bitfields are packed into a single storage field
            storage_type = fe_type.get_bitfield_storage_type()
            field_types = [dmm.lookup(storage_type).get_value_type()]
        else:
            # Regular struct: create a field for each struct field
            field_types = [
                dmm.lookup(field_type).get_value_type() for _, field_type, *_ in fe_type.fields
            ]
        be_type = llvm.StructType.new_identified(fe_type.name, field_types)
        types.AggregateType.record_named_type(be_type.name, fe_type)
        self._fields = tuple(name for name, *_ in fe_type.fields)
        super().__init__(dmm, fe_type, be_type)

    def get_field_position(self, name):
        return self._fields.index(name)


@register_model(types.UnionType)
class UnionTypeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        # Union is represented as an integer of the size of the largest variant
        # Use the size_bits property which handles all variant types
        max_bits = fe_type.size_bits

        # Round up to nearest power of 2 (8, 16, 32, 64, etc.)
        if max_bits <= 8:
            be_type = T.i8()
        elif max_bits <= 16:
            be_type = T.i16()
        elif max_bits <= 32:
            be_type = T.i32()
        elif max_bits <= 64:
            be_type = T.i64()
        else:
            raise NotImplementedError(
                f"Union '{fe_type.name}' requires {max_bits} bits but only up to 64-bit unions are currently supported"
            )

        types.AggregateType.record_named_type(fe_type.name, fe_type)
        super().__init__(dmm, fe_type, be_type)


@register_model(types.CUTensorMapStorageType)
class CUTensorMapStorageTypeModel(AggregateTypeModel): ...


@register_model(types.ByValPointerType)
class ByValPointerTypeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = llvm.PointerType.get()
        super().__init__(dmm, fe_type, be_type)


@register_model(types.EnumMember)
class EnumMemberModel(PrimitiveModel):
    """EnumMember is represented as its underlying dtype (e.g., int64)."""

    def __init__(self, dmm, fe_type):
        be_type = dmm.lookup(fe_type.dtype).get_value_type()
        super().__init__(dmm, fe_type, be_type)


@register_model(types.IntEnumMember)
class IntEnumMemberModel(PrimitiveModel):
    """IntEnumMember is represented as its underlying integer dtype."""

    def __init__(self, dmm, fe_type):
        be_type = dmm.lookup(fe_type.dtype).get_value_type()
        super().__init__(dmm, fe_type, be_type)


# NumPy Record (structured dtype) support
from numba_cuda_mlir.types import Record, NestedArray


@register_model(NestedArray)
class NestedArrayModel(PrimitiveModel):
    """
    Model for arrays nested within Record types.

    NestedArray is used for array fields in structured dtypes, e.g.:
        np.dtype([('h', np.float32, 2)])  # 'h' is a NestedArray of shape (2,)

    Unlike regular Array, NestedArray has a fixed shape known at compile time.
    We represent it as an LLVM pointer to the data, since NestedArray access
    is lowered directly to LLVM pointer arithmetic (avoiding memref dialect
    issues when we're already in LLVM dialect context).
    """

    def __init__(self, dmm, fe_type):
        from numba_cuda_mlir._mlir.dialects import llvm

        # NestedArrays are represented as raw pointers to their data
        # The shape/stride info is known at compile time from fe_type
        be_type = llvm.PointerType.get()
        super().__init__(dmm, fe_type, be_type)


@register_model(Record)
class RecordModel(PrimitiveModel):
    """
    Model for NumPy structured dtype (Record).

    Records are represented as pointers to byte arrays, matching numba-cuda's
    implementation. Field access is done via byte offset + bitcast.

    The underlying storage is [N x i8] where N is the record's byte size,
    but we pass around an opaque pointer to this storage.
    """

    def __init__(self, dmm, fe_type):
        # Records are passed as opaque pointers to their byte storage
        be_type = llvm.PointerType.get()
        super().__init__(dmm, fe_type, be_type)


_fp8_models_registered = False


def register_fp8_models():
    """
    Register models for FP8 type classes and bfloat16_raw.
    """
    global _fp8_models_registered
    if _fp8_models_registered:
        return
    _fp8_models_registered = True

    from numba_cuda_mlir.types import (
        _type_fp8_e5m2,
        _type_fp8_e4m3,
        _type_fp8_e8m0,
        bfloat16_raw_class,
    )

    @register_model(type(_type_fp8_e5m2))
    class FP8E5M2Model(PrimitiveModel):
        def __init__(self, dmm, fe_type):
            be_type = T.f8E5M2()
            super().__init__(dmm, fe_type, be_type)

    @register_model(type(_type_fp8_e4m3))
    class FP8E4M3Model(PrimitiveModel):
        def __init__(self, dmm, fe_type):
            be_type = T.f8E4M3FN()
            super().__init__(dmm, fe_type, be_type)

    @register_model(type(_type_fp8_e8m0))
    class FP8E8M0Model(PrimitiveModel):
        def __init__(self, dmm, fe_type):
            be_type = T.f8E8M0FNU()
            super().__init__(dmm, fe_type, be_type)

    @register_model(bfloat16_raw_class)
    class Bfloat16RawModel(PrimitiveModel):
        def __init__(self, dmm, fe_type):
            be_type = llvm.StructType.new_identified("bfloat16_raw", [T.i16()])
            super().__init__(dmm, fe_type, be_type)
