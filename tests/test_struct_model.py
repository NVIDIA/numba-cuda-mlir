# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vendored StructModel tests adapted from Numba expectations."""

import inspect

from cusimt._mlir import ir
from numba.core import types
import pytest

from cusimt.models import StructModel, mlir_data_manager, register_model


def _make_dummy_model():
    class DummyStructType(types.Type):
        def __init__(self):
            super().__init__(name="DummyStructType")

    @register_model(DummyStructType)
    class DummyStructModel(StructModel):
        def __init__(self, dmm, fe_type):
            super().__init__(dmm, fe_type, [("x", types.int32), ("y", types.float32)])

    frame = inspect.currentframe()
    with ir.Context(), ir.Location.file(frame.f_code.co_filename, frame.f_lineno, 0):
        return mlir_data_manager.lookup(DummyStructType())


def test_struct_model_member_introspection():
    """StructModel exposes member metadata APIs expected by Numba."""
    model = _make_dummy_model()

    assert model.field_count == 2
    assert model.get_field_position("x") == 0
    assert model.get_field_position("y") == 1
    assert model.get_member_fe_type("x") == types.int32
    assert model.get_member_fe_type("y") == types.float32
    assert model.get_type(0) == types.int32
    assert model.get_type(1) == types.float32
    assert model.get_type("x") == types.int32
    assert model.get_type("y") == types.float32

    with pytest.raises(KeyError, match="field named"):
        model.get_field_position("z")
