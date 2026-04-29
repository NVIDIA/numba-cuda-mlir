# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import os
from pathlib import Path


TEST_BIN_DIR = os.getenv(
    "CL_NUMBA_COMPAT_TEST_BIN_DIR",
    str(Path(__file__).resolve().parent / "testing"),
)
