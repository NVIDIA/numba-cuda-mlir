# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum


class ErrorCode(IntEnum):
    SUCCESS = 0
    # Codes 1-5 match raise_kernel_error() in cext/launcher/kernel.cpp.
    ASSERTION = 1
    INDEX = 2
    VALUE = 3
    RUNTIME = 4
    ZERO_DIVISION = 5
    RANGE_STEP_ZERO = 6
    SIGNED_DIVISION_OVERFLOW = 7
    FLOAT_TO_INT_NAN = 8
    FLOAT_TO_INT_OVERFLOW = 9
