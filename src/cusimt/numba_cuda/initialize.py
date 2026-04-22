# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause


def initialize_all():
    # Import models to register them with the data model manager
    import cusimt.numba_cuda.models  # noqa: F401

    from cusimt.numba_cuda import HAS_NUMBA

    # XXX: Return anyway because we don't want to double-register now we have
    # no redirector. Needs fix / tidy-up at some point
    return

    if not HAS_NUMBA:
        return

    from cusimt.numba_cuda.decorators import jit
    from cusimt.numba_cuda.dispatcher import CUDADispatcher
    from numba.core.target_extension import (
        target_registry,
        dispatcher_registry,
        jit_registry,
    )

    cuda_target = target_registry["cuda"]
    jit_registry[cuda_target] = jit
    dispatcher_registry[cuda_target] = CUDADispatcher
