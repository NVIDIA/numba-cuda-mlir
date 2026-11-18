# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity for caches whose values belong to one CUDA context lifetime."""

import threading
import weakref

from numba_cuda_mlir._cext import _get_current_context


_current_context = threading.local()
_EXTRAS_KEY = "numba_cuda_mlir.context_token"


class _ContextToken:
    pass


def current_context_token():
    # The handle guard also observes context changes made by other libraries.
    # Only a change or reset needs the full Python context-resolution path.
    handle = _get_current_context()
    cached = getattr(_current_context, "cached", None)
    if cached is not None and cached[0] == handle:
        context = cached[1]()
        if context is not None:
            token = context.extras.get(_EXTRAS_KEY)
            if token is not None:
                return token

    from numba_cuda_mlir.numba_cuda.cudadrv.devices import get_context

    context = get_context()
    # Context.__eq__ compares raw handles, which CUDA can reuse after reset.
    # extras owns the token; reset removes it, allowing weak caches to expire.
    try:
        token = context.extras[_EXTRAS_KEY]
    except KeyError:
        token = context.extras.setdefault(_EXTRAS_KEY, _ContextToken())
    _current_context.cached = (int(context.handle), weakref.ref(context))
    return token
