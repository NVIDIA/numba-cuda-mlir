# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack model data representations into LLVM aggregates at storage boundaries."""

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import builtin, llvm


class _DataLayout:
    """Materialize tuple, complex, and memref data without changing model policy."""

    def __init__(self, storage_type, value_type=None):
        self.storage_type = storage_type
        self.children = None
        if isinstance(value_type, ir.NoneType):
            # A model can have runtime storage despite having no runtime value.
            self.storage_type = value_type
            self.data_type = None
        elif storage_type is None or isinstance(storage_type, ir.NoneType):
            self.data_type = None
        elif isinstance(storage_type, tuple):
            value_types = (
                value_type
                if isinstance(value_type, tuple) and len(value_type) == len(storage_type)
                else (None,) * len(storage_type)
            )
            children = []
            data_types = []
            for typ, value_typ in zip(storage_type, value_types):
                child = _DataLayout(typ, value_typ)
                field = None
                if child.data_type is not None:
                    field = len(data_types)
                    data_types.append(child.data_type)
                children.append((child, field))
            self.children = tuple(children)
            self.data_type = llvm.StructType.get_literal(data_types) if data_types else None
        elif isinstance(storage_type, ir.ComplexType):
            from numba_cuda_mlir.lowering_utilities import get_llvm_struct_for_complex

            self.data_type = get_llvm_struct_for_complex(storage_type)
        elif isinstance(storage_type, ir.MemRefType):
            from numba_cuda_mlir.lowering_utilities import memref_descriptor_type

            self.data_type = memref_descriptor_type(storage_type.rank)
        else:
            # Already packed model data, including nested aggregates, is opaque.
            self.data_type = storage_type

    def pack(self, value):
        if self.children is not None:
            if len(value) != len(self.children):
                raise ValueError("Data does not match its storage layout")
            if self.data_type is None:
                return None
            result = llvm.UndefOp(self.data_type).result
            for (child, field), member in zip(self.children, value):
                if field is not None:
                    result = llvm.insertvalue(result, child.pack(member), [field])
            return result
        if self.data_type is None:
            return None
        if isinstance(self.storage_type, ir.ComplexType):
            from numba_cuda_mlir.lowering_utilities import complex_to_llvm_struct

            return complex_to_llvm_struct(value)
        if isinstance(self.storage_type, ir.MemRefType):
            from numba_cuda_mlir.lowering_utilities import convert

            # Normalize shape, strides, and memory space before descriptor
            # lowering. In particular, shared memrefs need a memory-space cast
            # rather than an unrealized cast directly to a generic descriptor.
            value = convert(value, self.storage_type)
            return builtin.unrealized_conversion_cast([self.data_type], [value])
        return value

    def unpack(self, value):
        if self.children is not None:
            return tuple(
                child.unpack(
                    llvm.extractvalue(child.data_type, value, [field])
                    if field is not None
                    else None
                )
                for child, field in self.children
            )
        if self.data_type is None:
            return self.storage_type
        if isinstance(self.storage_type, ir.ComplexType):
            from numba_cuda_mlir.lowering_utilities import llvm_struct_to_complex

            return llvm_struct_to_complex(value, self.storage_type)
        if isinstance(self.storage_type, ir.MemRefType):
            return builtin.unrealized_conversion_cast([self.storage_type], [value])
        return value


class LLVMStorageAdapter:
    """Adapt one model's data representation to LLVM-storable types.

    The underlying model still owns value/data conversion. Only its resulting
    tuple, complex, or memref representation is encoded for LLVM storage.
    """

    def __init__(self, dmm, fe_type):
        self._model = dmm.lookup(fe_type)
        self._value_type = self._model.get_value_type()
        self._is_static = isinstance(self._value_type, ir.NoneType)
        storage_type = self._value_type if self._is_static else self._model.get_data_type()
        self._layout = _DataLayout(storage_type, self._value_type)
        self._data_type = self._layout.data_type
        if self._data_type is None:
            self._data_type = llvm.StructType.get_literal([])

    def get_data_type(self):
        return self._data_type

    def as_data(self, builder, value):
        if self._is_static:
            return llvm.UndefOp(self._data_type).result
        data = self._layout.pack(self._model.as_data(builder, value))
        return llvm.UndefOp(self._data_type).result if data is None else data

    def from_data(self, builder, value):
        if self._is_static:
            return self._value_type
        return self._model.from_data(builder, self._layout.unpack(value))


class DataPacker:
    """Store front-end values in an LLVM struct using their model conversions.

    Models retain ownership of ``as_data`` / ``from_data`` and their declared
    data types. This packer only materializes data representations that cannot
    be inserted directly into an LLVM aggregate. Static value types occupy no
    fields, including inside tuple-shaped data; an empty pack is an empty struct.
    """

    def __init__(self, dmm, fe_types):
        self._models = tuple(LLVMStorageAdapter(dmm, typ) for typ in fe_types)
        fields = []
        data_types = []
        for model in self._models:
            field = None
            if model._layout.data_type is not None:
                field = len(data_types)
                data_types.append(model.get_data_type())
            fields.append(field)
        self._fields = tuple(fields)
        self.data_type = llvm.StructType.get_literal(data_types)

    def as_data(self, builder, values):
        """Convert value representations and assemble their aggregate storage."""
        if len(values) != len(self._models):
            raise ValueError("Values do not match their data models")
        aggregate = llvm.UndefOp(self.data_type).result
        for model, field, value in zip(self._models, self._fields, values):
            if field is not None:
                aggregate = llvm.insertvalue(aggregate, model.as_data(builder, value), [field])
        return aggregate

    def from_data(self, builder, aggregate):
        """Recover the original tuple of model value representations."""
        return tuple(
            model.from_data(
                builder,
                llvm.extractvalue(model.get_data_type(), aggregate, [field])
                if field is not None
                else None,
            )
            for model, field in zip(self._models, self._fields)
        )
