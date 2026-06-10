# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import functools


class _LLVMLiteRemoved:
    """Tombstone standing in for the removed ``llvmlite.ir`` module.

    These CUDA data models are instantiated during MLIR type inference, but the
    llvmlite ``be_type`` they would build is discarded (the MLIR-typed models
    live in ``numba_cuda_mlir.models``). The eager type construction has been
    replaced by ``None``; any remaining use raises, since llvmlite is no longer
    a dependency.
    """

    def __getattr__(self, name):
        raise NotImplementedError(
            "llvmlite has been removed; llvmlite IR is not available on the "
            f"MLIR path (attempted to use llvmlite.ir.{name})"
        )


ir = _LLVMLiteRemoved()

from numba_cuda_mlir.numba_cuda.datamodel.registry import DataModelManager, register
from numba_cuda_mlir.numba_cuda.datamodel import PrimitiveModel
from numba_cuda_mlir.numba_cuda.datamodel.models import StructModel
from numba_cuda_mlir.numba_cuda.extending import core_models as models
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.types.ext_types import (
    Dim3,
    GridGroup,
    CUDADispatcher,
    Bfloat16,
)


cuda_data_manager = DataModelManager()

register_model = functools.partial(register, cuda_data_manager)


@register_model(Dim3)
class Dim3Model(StructModel):
    def __init__(self, dmm, fe_type):
        members = [("x", types.int32), ("y", types.int32), ("z", types.int32)]
        super().__init__(dmm, fe_type, members)


@register_model(GridGroup)
class GridGroupModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        # llvmlite be_type discarded on the MLIR path
        super().__init__(dmm, fe_type, None)


@register_model(types.Float)
class FloatModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        if fe_type not in (types.float16, types.float32, types.float64):
            raise NotImplementedError(fe_type)
        # llvmlite be_type discarded on the MLIR path
        super().__init__(dmm, fe_type, None)


register_model(CUDADispatcher)(models.OpaqueModel)


@register_model(Bfloat16)
class _model___nv_bfloat16(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        # llvmlite be_type discarded on the MLIR path
        super().__init__(dmm, fe_type, None)
