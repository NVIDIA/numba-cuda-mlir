# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir.mlir.dialect_exts import llvm
from numba_cuda_mlir.models import StructModel, mlir_data_manager
from numba_cuda_mlir.numba_cuda import types

_INIT_OPS = {"zero", "undef"}


class StructProxy:
    """
    Immutable view over a struct SSA value of Numba type fe_type.
    """

    def __init__(
        self,
        fe_type: types.Type,
        value: ir.Value | None = None,
        *,
        init: Literal["zero", "undef"] | None = None,
    ):
        model = mlir_data_manager.lookup(fe_type)

        if not isinstance(model, StructModel):
            raise TypeError(
                f"StructProxy requires a type with a StructModel, got {fe_type} "
                f"with {type(model).__name__}"
            )

        struct_type = model.get_value_type()
        if value is None:
            init = init or "zero"

            if init not in _INIT_OPS:
                raise ValueError(f"init must be one of {sorted(_INIT_OPS)}, got {init!r}")

            if init == "zero":
                value = llvm.mlir_zero(res=struct_type)
            else:
                value = llvm.mlir_undef(res=struct_type)

        elif init is not None:
            raise ValueError("init cannot be combined with an existing value")

        elif value.type != struct_type:
            raise TypeError(f"value for {fe_type} must have type {struct_type}, got {value.type}")

        super().__setattr__("_fe_type", fe_type)
        super().__setattr__("_model", model)
        super().__setattr__("_value", value)

    @property
    def value(self) -> ir.Value:
        return self._value

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(
            f"StructProxy is immutable; use p = p.replace({name}=...) to set a field"
        )

    def _index(self, name: str) -> int:
        try:
            return self._model.get_field_position(name)
        except KeyError:
            raise AttributeError(
                f"{self._fe_type} has no field {name!r}: fields are {self._model._fields}"
            ) from None

    def _field_type(self, index: int) -> ir.Type:
        return self._model.get_model(index).get_value_type()

    def __getattr__(self, name: str) -> ir.Value:
        if name.startswith("_"):
            raise AttributeError(name)

        index = self._index(name)
        return llvm.extractvalue(self._field_type(index), self._value, [index])

    def replace(self, **fields: ir.Value) -> "StructProxy":
        result = self._value

        for name, field_value in fields.items():
            index = self._index(name)
            expected = self._field_type(index)

            if not isinstance(field_value, ir.Value):
                raise TypeError(
                    f"field {name!r} of {self._fe_type} expects an MLIR value, "
                    f"got {type(field_value).__name__}"
                )

            if field_value.type != expected:
                raise TypeError(
                    f"field {name!r} of {self._fe_type} expects {expected}, got {field_value.type}"
                )

            result = llvm.insertvalue(result, field_value, index)
        return StructProxy(self._fe_type, result)
