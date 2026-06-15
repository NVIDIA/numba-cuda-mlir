# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
NRTManagedTestType: a minimal NRT-managed type for testing refcounting.

The value is a single NRT_MemInfo* that owns an 8-byte device-heap
allocation.  No C++ shims are needed -- construction calls the
already-linked NRT_MemInfo_alloc_aligned and destruction goes through
the standard NRT_decref → NRT_MemInfo_call_dtor → NRT_Free path.

Import this module before compiling any kernel that uses the type;
importing triggers registration of the data model, typing template,
and lowering implementation.
"""

from numba_cuda_mlir import types as nb_types
from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir._mlir.dialects import llvm
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.extending import lowering_registry, refresh_registries, typing_registry
from numba_cuda_mlir.lowering.nrt import nrt_meminfo_alloc
from numba_cuda_mlir.lowering_utilities import int_of
from numba_cuda_mlir.models import register_model
from numba_cuda_mlir.numba_cuda.datamodel import PrimitiveModel
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AttributeTemplate,
    ConcreteTemplate,
    signature,
)


# ---------------------------------------------------------------------------
# Numba type
# ---------------------------------------------------------------------------


class NRTManagedTestType(nb_types.Type):
    def __init__(self):
        super().__init__(name="NRTManagedTestType")


nrt_managed_test_type = NRTManagedTestType()


# ---------------------------------------------------------------------------
# Data model -- single !llvm.ptr field (the NRT_MemInfo*)
# ---------------------------------------------------------------------------


@register_model(NRTManagedTestType)
class NRTManagedTestTypeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, llvm.PointerType.get())

    def has_nrt_meminfo(self):
        return True

    def get_nrt_meminfo(self, value):
        # just return itself (mi)
        return value


# ---------------------------------------------------------------------------
# Python stub -- called in kernel code to create an instance
# ---------------------------------------------------------------------------


def make_nrt_managed_test_obj_inst():
    raise RuntimeError("Device-only function")


# ---------------------------------------------------------------------------
# Typing
# ---------------------------------------------------------------------------


class MakeNRTManagedTestTemplate(ConcreteTemplate):
    key = make_nrt_managed_test_obj_inst
    cases = [signature(nrt_managed_test_type)]


typing_registry.register_global(
    make_nrt_managed_test_obj_inst,
    nb_types.Function(MakeNRTManagedTestTemplate),
)


# ---------------------------------------------------------------------------
# Lowering -- emits func.call @NRT_MemInfo_alloc_aligned(i64 8, i32 8)
# ---------------------------------------------------------------------------


def _lower_make_nrt_managed_test(builder, target, args, kwargs):
    size = int_of(8, T.i64())
    mi = nrt_meminfo_alloc(builder.mlir_gpu_module, size)
    builder.store_var(target, mi)


lowering_registry.lower(make_nrt_managed_test_obj_inst)(_lower_make_nrt_managed_test)


# ---------------------------------------------------------------------------
# nrt_get_refcount(obj) -> int64
#
# Reads the refcount (first field of NRT_MemInfo, atomic<size_t>)
# directly from the meminfo pointer.  Device-only; used in tests to
# assert exact refcount values at specific points inside a kernel.
# ---------------------------------------------------------------------------


def nrt_get_refcount(obj):
    raise RuntimeError("Device-only function")


class GetRefcountTemplate(ConcreteTemplate):
    key = nrt_get_refcount
    cases = [signature(nb_types.int64, nrt_managed_test_type)]


typing_registry.register_global(
    nrt_get_refcount,
    nb_types.Function(GetRefcountTemplate),
)


def _lower_nrt_get_refcount(builder, target, args, kwargs):
    (obj_var,) = args
    mi = builder.load_var(obj_var)
    refct = llvm.load(T.i64(), mi)
    builder.store_var(target, refct)


lowering_registry.lower(nrt_get_refcount, nrt_managed_test_type)(_lower_nrt_get_refcount)


# ---------------------------------------------------------------------------
# NRTWrapperType: a struct { inner: NRTManagedTestType } for testing getattr
#
# The struct itself does NOT have a meminfo, but its single field is
# NRT-managed.  This exercises the traverse_mlir path: del of the wrapper
# must walk members and decref each NRT-managed one, and getattr extracting
# the inner field must incref it.
# ---------------------------------------------------------------------------


class NRTWrapperType(nb_types.Type):
    def __init__(self):
        super().__init__(name="NRTWrapperType")

    def __iter__(self):
        return iter((nrt_managed_test_type,))


nrt_wrapper_type = NRTWrapperType()


@register_model(NRTWrapperType)
class NRTWrapperModel(PrimitiveModel):
    # A struct containing an NRT-managed field as a member,
    # but not itself directly NRT-managed.
    _fields = ("inner",)

    def __init__(self, dmm, fe_type):
        be_type = llvm.StructType.get_literal([llvm.PointerType.get()])
        super().__init__(dmm, fe_type, be_type)

    def get_field_position(self, name):
        return self._fields.index(name)

    def traverse_mlir(self):
        def getter(value):
            return llvm.extractvalue(
                res=llvm.PointerType.get(),
                container=value,
                position=mlir_ir.DenseI64ArrayAttr.get([0]),
            )

        return [(nrt_managed_test_type, getter)]


class NRTWrapperAttributeTemplate(AttributeTemplate):
    key = NRTWrapperType

    def generic_resolve(self, typ, attr):
        if attr == "inner":
            return nrt_managed_test_type
        return None


typing_registry.register_attr(NRTWrapperAttributeTemplate)


def make_nrt_wrapper():
    raise RuntimeError("Device-only function")


class MakeNRTWrapperTemplate(ConcreteTemplate):
    key = make_nrt_wrapper
    cases = [signature(nrt_wrapper_type)]


typing_registry.register_global(
    make_nrt_wrapper,
    nb_types.Function(MakeNRTWrapperTemplate),
)


def _lower_make_nrt_wrapper(builder, target, args, kwargs):
    size = int_of(8, T.i64())
    mi = nrt_meminfo_alloc(builder.mlir_gpu_module, size)

    struct_type = llvm.StructType.get_literal([llvm.PointerType.get()])
    undef = llvm.UndefOp(struct_type).result
    result = llvm.insertvalue(
        container=undef,
        value=mi,
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )
    builder.store_var(target, result)


lowering_registry.lower(make_nrt_wrapper)(_lower_make_nrt_wrapper)

refresh_registries()
