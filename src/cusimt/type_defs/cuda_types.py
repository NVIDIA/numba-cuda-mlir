# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cusimt.numba_cuda.types import int64, Type
from .aggregate_types import AggregateType


class CUTensorMapStorageType(AggregateType):
    """
    Tensor map descriptor. See the cuda declaration:

    /**
    * Tensor map descriptor. Requires compiler support for aligning to 64 bytes.
    */
    typedef struct CUtensorMap_st {
        alignas(64)
        cuuint64_t opaque[CU_TENSOR_MAP_NUM_QWORDS];
    } CUtensorMap;
    """

    CU_TENSOR_MAP_NUM_QWORDS = 16

    def __init__(self):
        super().__init__(
            "CUTensorMapStorage",
            [(f"f{i}", int64) for i in range(self.CU_TENSOR_MAP_NUM_QWORDS)],
        )


class CUTensorMapType(Type):
    """
    A byval(CUTensorMapStorage) type.
    """

    def __init__(self):
        super().__init__("CUTensorMap")
        self.__cusimt_attributes__ = {"llvm.byval": CUTensorMapStorageType()}
