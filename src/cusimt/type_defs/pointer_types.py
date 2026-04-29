# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cusimt.numba_cuda.types import Type, CPointer


class ByValPointerType(CPointer):
    def __init__(self, dtype: Type):
        super().__init__(dtype)
        self.__cusimt_attributes__ = {"llvm.byval": dtype}
