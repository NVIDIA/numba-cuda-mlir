# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import wraps
from inspect import Parameter, Signature, signature

from numba_cuda_mlir.numba_cuda.core.imputils import Registry


def is_signature_lowering(callback):
    """Identify ``(lowerer, signature, args)`` callbacks by their signature."""
    classified = getattr(callback, "_numba_cuda_mlir_signature_api", None)
    if classified is not None:
        return classified
    try:
        callback_signature = signature(callback)
    except (TypeError, ValueError):
        return False
    return _is_signature_lowering(callback_signature)


def _is_signature_lowering(callback_signature):
    positional_count = 0
    for parameter in callback_signature.parameters.values():
        if parameter.kind is Parameter.VAR_POSITIONAL:
            return False
        if parameter.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
            positional_count += 1
    return positional_count == 3


def register_lowering(callback):
    """Wrap a callback with its once-computed lowering API classification."""
    try:
        callback_signature = signature(callback)
    except (TypeError, ValueError):
        callback_signature = Signature(
            (
                Parameter("args", Parameter.VAR_POSITIONAL),
                Parameter("kwargs", Parameter.VAR_KEYWORD),
            )
        )
    signature_api = getattr(callback, "_numba_cuda_mlir_signature_api", None)
    if signature_api is None:
        signature_api = _is_signature_lowering(callback_signature)

    @wraps(callback)
    def registered(*args, **kwargs):
        return callback(*args, **kwargs)

    registered.__signature__ = callback_signature
    registered._numba_cuda_mlir_signature_api = signature_api
    return registered


class LoweringRegistry(Registry):
    """
    Registry for MLIR-based lowering implementations.

    Each lowering module should create its own instance of this registry:
        registry = LoweringRegistry()
        lower = registry.lower
    """

    def __init__(self):
        super().__init__(name="numba_cuda_mlir")

    def lower(self, func, *argtys):
        """Register a signature or legacy callback without replacing it."""

        def decorate(callback):
            registered = register_lowering(callback)
            super(LoweringRegistry, self).lower(func, *argtys)(registered)
            return callback

        return decorate
