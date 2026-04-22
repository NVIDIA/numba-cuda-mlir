# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba.core.types import Type, CPointer, void


class ByValPointerType(CPointer):
    def __init__(self, dtype: Type):
        super().__init__(dtype)
        self.__cusimt_attributes__ = {"llvm.byval": dtype}
