# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cuda.simt as cuda
from cuda.simt import types
from cusimt._mlir.dialects import nvvm
from cusimt._mlir.extras import types as T


@cuda.intrin.define
def elect_sync() -> types.boolean:
    return nvvm.elect_sync(T.bool())
