# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cusimt.numba_cuda import types
from functools import total_ordering


@total_ordering
class SpecialFloatType(types.Number):
    def __init__(self, name: str, bitwidth: int):
        super().__init__(name)
        self.bitwidth = bitwidth

    def __lt__(self, other):
        if other.__class__ is not self.__class__ and not isinstance(other, types.Float):
            return TypeError(f"Cannot compare {self} and {other}")
        return self.bitwidth < other.bitwidth


class BFloat16Type(SpecialFloatType):
    def __init__(self):
        super().__init__("bf16", 16)


class NVFP4Type(SpecialFloatType): ...


class Float4E2M1FNType(SpecialFloatType): ...


class Float6E2M3FNType(SpecialFloatType): ...


class Float6E3M2FNType(SpecialFloatType): ...


class Float8E3M4Type(SpecialFloatType): ...


class Float8E4M3B11FNUZType(SpecialFloatType): ...


class Float8E4M3FNType(SpecialFloatType): ...


class Float8E4M3FNUZType(SpecialFloatType): ...


class Float8E4M3Type(SpecialFloatType): ...


class Float8E5M2FNUZType(SpecialFloatType): ...


class Float8E5M2Type(SpecialFloatType): ...


class Float8E8M0FNUType(SpecialFloatType): ...


class FloatTF32Type(SpecialFloatType): ...
