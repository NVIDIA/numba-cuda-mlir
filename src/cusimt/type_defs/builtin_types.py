# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any
from cusimt.numba_cuda.types.abstract import Type


class Namespace(Type):
    """
    A Numba type for something that has a attributes and nothing else.
    """

    def __init__(self, object: Any):
        self.object = object
        super().__init__(f"Namespace({object})")
