# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Protocol, Any, Tuple, runtime_checkable
from cusimt.numba_cuda.core import types


@runtime_checkable
class ArgumentHandler(Protocol):
    def prepare_args(
        self, ty: types.Type, val: Any, stream: Any = None, retr: list | None = None
    ) -> Tuple[types.Type, Any]: ...
