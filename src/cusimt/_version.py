# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.resources

__version__ = (
    importlib.resources.files("cusimt").joinpath("VERSION").read_text().strip()
)
