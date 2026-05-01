# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from .intrin import *
from .libdevice import *
from .jit import *

def syncthreads() -> None: ...
def syncwarp() -> None: ...
def consteval(value: Any) -> Any:
    """
    Force evaluation of an expression at compile time.
    """

def literal_unroll(value: Any) -> Any:
    """
    Unroll a loop at compile time.

    Wrap your range in a loop:

        >>> for i in cuda.literal_unroll(range(10)):
        ...     array[i] = i
    """
