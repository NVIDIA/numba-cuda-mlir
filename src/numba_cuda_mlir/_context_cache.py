# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity for caches whose values belong to one CUDA context lifetime."""


class _ContextToken:
    pass


def current_context_token():
    from numba_cuda_mlir.numba_cuda.cudadrv.devices import get_context

    context = get_context()
    # Context.__eq__ compares raw handles, which CUDA can reuse after reset.
    # extras owns the token; reset removes it, allowing weak caches to expire.
    key = "numba_cuda_mlir.context_token"
    try:
        return context.extras[key]
    except KeyError:
        return context.extras.setdefault(key, _ContextToken())
