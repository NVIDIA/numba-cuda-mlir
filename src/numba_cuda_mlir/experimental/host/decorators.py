# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

from numba_cuda_mlir.experimental.host.descriptor import Dispatcher
from numba_cuda_mlir.numba_cuda import types, typing
from numba_cuda_mlir.numba_cuda.core import sigutils


def jit(func_or_sig=None, **options):
    targetoptions = {
        "opt_level": 2,
        "dump": False,
        "dump_numba_ir": False,
        "dump_mlir": False,
        "dump_llvm_dialect": False,
    }
    unknown = set(options).difference(targetoptions)
    if unknown:
        raise ValueError(f"Unsupported host.jit option(s): {', '.join(sorted(unknown))}")
    targetoptions.update(options)
    if type(targetoptions["opt_level"]) is not int or not 0 <= targetoptions["opt_level"] <= 3:
        raise ValueError("opt_level must be an integer between 0 and 3")
    for key in targetoptions.keys() - {"opt_level"}:
        if not isinstance(targetoptions[key], bool):
            raise TypeError(f"{key} must be a boolean")

    if sigutils.is_signature(func_or_sig):
        signatures = [func_or_sig]
    elif isinstance(func_or_sig, list):
        signatures = func_or_sig
    elif func_or_sig is None or callable(func_or_sig):
        signatures = None
    else:
        raise TypeError("host.jit expects a function or a Numba signature")

    def decorate(function):
        dispatcher = Dispatcher(function, dict(targetoptions))
        annotation = dispatcher._signature.return_annotation
        if signatures is not None:
            if annotation is not inspect.Signature.empty:
                raise TypeError("Cannot combine an explicit signature with a return annotation")
            for signature in signatures:
                dispatcher.compile(signature)
            dispatcher.disable_compile()
        elif annotation is not inspect.Signature.empty:
            return_type = types.none if annotation is None else annotation
            if not isinstance(return_type, types.Type):
                raise TypeError("host.jit annotations must be Numba scalar types or None")
            arguments = dispatcher._arg_annotations
            if any(argument is inspect.Signature.empty for argument in arguments):
                raise TypeError("A return annotation requires Numba scalar parameter annotations")
            dispatcher.compile(typing.signature(return_type, *arguments))
        return dispatcher

    if callable(func_or_sig) and signatures is None:
        return decorate(func_or_sig)
    return decorate
