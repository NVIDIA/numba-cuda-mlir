# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause


def initialize_all():
    # Import models to register them with the data model manager
    import cusimt.numba_cuda.models  # noqa: F401
