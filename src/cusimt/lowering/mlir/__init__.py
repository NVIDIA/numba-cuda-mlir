# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Lower-level interface for interacting with MLIR.
Users should typically not reach into this module, but we can expose
mlir concepts for ninja users.
"""

from . import types
