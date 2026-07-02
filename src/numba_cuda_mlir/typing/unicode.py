# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import operator
from numba_cuda_mlir import types
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AbstractTemplate,
    Registry,
)

registry = Registry()

_COMPARISON_OPS = [
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
]

for _op in _COMPARISON_OPS:

    class UnicodeComparisonTemplate(AbstractTemplate):
        key = _op
        metadata = {"target": "cuda"}

        def generic(self, args, kws):
            a, b = args
            accepted = (types.UnicodeType, types.StringLiteral)
            if isinstance(a, accepted) and isinstance(b, accepted):
                return types.boolean(a, b)

    registry.register(UnicodeComparisonTemplate)
    registry.register_global(_op, types.Function(UnicodeComparisonTemplate))
