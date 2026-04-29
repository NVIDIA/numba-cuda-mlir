# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cusimt.numba_cuda.types import (
    BoundFunction,
    Poison,
    UnicodeType,
    UnionType,
    Boolean,
    StringLiteral,
    BooleanLiteral,
    IntegerLiteral,
    Complex,
    NumberClass,
    NPDatetime,
    NPTimedelta,
    RangeType,
    RangeIteratorType,
    DType,
    DTypeSpec,
    VarArg,
    Any,
    Literal,
    string,
    Function,
    Integer,
    Number,
    Float,
    Module,
    Type,
    NoneType,
    float16,
    float32,
    float64,
    EnumMember,
    IntEnumMember,
    EnumClass,
    # String/char sequence types
    CharSeq,
    UnicodeCharSeq,
    # Numpy aliases
    b1,
    i8,
    u1,
    u2,
    u4,
    u8,
    f2,
    f4,
    f8,
    c8,
    c16,
    i1,
    i2,
    i4,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    complex64,
    complex128,
    bool,
    boolean,
    void,
    none,
    uintp,
    intp,
    voidptr,
    CPointer,
    RawPointer,
    Array,
    ArrayCTypes,
    TypeRef,
    Tuple,
    UniTuple,
    # Record/struct types
    Record,
    NestedArray,
)
from cusimt.numba_cuda.typing import Signature, signature
from cusimt.type_defs.aggregate_types import AggregateType, UnionType
from cusimt.type_defs.cuda_types import CUTensorMapType, CUTensorMapStorageType
from cusimt.type_defs.pointer_types import ByValPointerType
from cusimt.type_defs.float_types import (
    BFloat16Type,
    NVFP4Type,
    Float4E2M1FNType,
    Float6E2M3FNType,
    Float6E3M2FNType,
    Float8E3M4Type,
    Float8E4M3B11FNUZType,
    Float8E4M3FNType,
    Float8E4M3FNUZType,
    Float8E4M3Type,
    Float8E5M2FNUZType,
    Float8E5M2Type,
    Float8E8M0FNUType,
    FloatTF32Type,
)


def get_cusimt_attributes(obj) -> dict:
    return getattr(obj, "__cusimt_attributes__", dict())


def set_cusimt_attributes(obj, attributes: dict):
    setattr(obj, "__cusimt_attributes__", attributes)
    return obj


def grid_constant(type: Type) -> Type:
    attributes = get_cusimt_attributes(type)
    attributes["nvvm.grid_constant"] = True
    set_cusimt_attributes(type, attributes)
    return type


def alignas(type: Type, align: int) -> Type:
    attributes = get_cusimt_attributes(type)
    attributes["llvm.align"] = align
    set_cusimt_attributes(type, attributes)
    return type


i8 = int8
i16 = int16
i32 = int32
i64 = int64
u8 = uint8
u16 = uint16
u32 = uint32
u64 = uint64
f16 = float16
f32 = float32
f64 = float64

# TODO(ajm): alias the real types
bf16 = BFloat16Type()
nvfp4 = NVFP4Type("nvfp4", 4)

f4E2M1FN = Float4E2M1FNType("f4E2M1FN", 4)
f6E2M3FN = Float6E2M3FNType("f6E2M3FN", 6)
f6E3M2FN = Float6E3M2FNType("f6E3M2FN", 6)
f8E3M4 = Float8E3M4Type("f8E3M4", 8)
f8E4M3B11FNUZ = Float8E4M3B11FNUZType("f8E4M3B11FNUZ", 8)
f8E4M3FN = Float8E4M3FNType("f8E4M3FN", 8)
f8E4M3FNUZ = Float8E4M3FNUZType("f8E4M3FNUZ", 8)
f8E4M3 = Float8E4M3Type("f8E4M3", 8)
f8E5M2FNUZ = Float8E5M2FNUZType("f8E5M2FNUZ", 8)
f8E5M2 = Float8E5M2Type("f8E5M2", 8)
f8E8M0FNU = Float8E8M0FNUType("f8E8M0FNU", 8)
tf32 = FloatTF32Type("tf32", 18)
ptr = CPointer(void)
ptr_to = CPointer
byval_ptr = ByValPointerType

# CUDA-specific types
CUTensorMapStorage = CUTensorMapStorageType()
# we need a better representation of byval pointers in the type system.
# Probably we don't expose to users directly but we decay struct parameters
# to byval(struct)
CUTensorMap = alignas(byval_ptr(CUTensorMapStorage), 128)

_FP8_ATTRS = (
    "fp8_e5m2",
    "fp8_e4m3",
    "fp8_e8m0",
    "_type_fp8_e5m2",
    "_type_fp8_e4m3",
    "_type_fp8_e8m0",
    "cvt_e8m0_to_bf16raw",
    "bfloat16_raw_type",
    "bfloat16_raw_class",
)


def __getattr__(name):
    if name not in _FP8_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import sys
    from cusimt.numba_cuda._internal import cuda_fp8

    val = (
        type(cuda_fp8.bfloat16_raw_type)
        if name == "bfloat16_raw_class"
        else getattr(cuda_fp8, name)
    )
    setattr(sys.modules[__name__], name, val)
    return val
