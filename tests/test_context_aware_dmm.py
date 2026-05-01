# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ContextAwareDataModelManager — verifies that cached MLIR types
are invalidated when the active ir.Context changes."""

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir.numba_cuda import types

from numba_cuda_mlir.models import (
    mlir_data_manager,
    ContextAwareDataModelManager,
    StructModel,
    register_model,
)


def test_cache_invalidated_on_context_switch():
    """Model rebuilt when looked up under a different MLIR context."""
    c1 = ir.Context()
    with c1:
        m1 = mlir_data_manager.lookup(types.float32)
        assert m1.be_type.context is c1

    c2 = ir.Context()
    with c2:
        m2 = mlir_data_manager.lookup(types.float32)
        assert m2.be_type.context is c2
        assert m2.be_type.context is not c1


def test_cache_hit_same_context():
    """Model reused when context hasn't changed."""
    c = ir.Context()
    with c:
        m1 = mlir_data_manager.lookup(types.float64)
        m2 = mlir_data_manager.lookup(types.float64)
        assert m1 is m2


def test_recursive_lookup_rebuilds_nested_types():
    """AggregateType lookup rebuilds its field type models under the new context."""
    from numba_cuda_mlir.type_defs.aggregate_types import AggregateType

    c1 = ir.Context()
    with c1:
        agg = AggregateType("Pt", [("x", types.float32)])
        mlir_data_manager.lookup(agg)
        assert mlir_data_manager.lookup(types.float32).be_type.context is c1

    c2 = ir.Context()
    with c2:
        f32 = mlir_data_manager.lookup(types.float32)
        assert f32.be_type.context is c2


def test_tuple_be_type_invalidated():
    """UniTupleModel stores be_type as a tuple; context check still works."""
    c1 = ir.Context()
    with c1:
        ut = types.UniTuple(types.int32, 3)
        m1 = mlir_data_manager.lookup(ut)
        sample = m1.be_type[0]
        assert sample.context is c1

    c2 = ir.Context()
    with c2:
        m2 = mlir_data_manager.lookup(ut)
        sample = m2.be_type[0]
        assert sample.context is c2


def test_is_subclass():
    """mlir_data_manager is a ContextAwareDataModelManager."""
    assert isinstance(mlir_data_manager, ContextAwareDataModelManager)


def test_struct_model_cache_invalidation_uses_be_type():
    """Vendored StructModel defines be_type for context-aware cache checks."""

    class DummyStructType(types.Type):
        def __init__(self):
            super().__init__(name="DummyStructType")

    @register_model(DummyStructType)
    class DummyStructModel(StructModel):
        def __init__(self, dmm, fe_type):
            super().__init__(dmm, fe_type, [("x", types.int32), ("y", types.float32)])

    dummy = DummyStructType()

    c1 = ir.Context()
    with c1, ir.Location.unknown():
        m1 = mlir_data_manager.lookup(dummy)
        assert m1.be_type.context is c1
        assert mlir_data_manager.lookup(dummy) is m1

    c2 = ir.Context()
    with c2, ir.Location.unknown():
        m2 = mlir_data_manager.lookup(dummy)
        assert m2.be_type.context is c2
        assert m2.be_type.context is not c1
