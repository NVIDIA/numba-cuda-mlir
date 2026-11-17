# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

from numba_cuda_mlir.experimental.host.descriptor import Dispatcher
from numba_cuda_mlir.numba_cuda import types, typing
from numba_cuda_mlir.numba_cuda.core import sigutils


def host_jit(func_or_sig=None, **options):
    """Compile a scalar Python function for native host execution.

    Uses the same Numba frontend and shared MLIR lowering infrastructure as the
    CUDA path, with three substitutions: a host target context, a host lowering
    subclass, and an in-process native JIT instead of the NVVM/cubin/launcher
    pipeline.

    Parameters
    ----------
    func_or_sig : callable, Numba signature, list of Numba signatures, or None
        Function to compile, or explicit signature(s) to compile when the
        decorated function arrives. With no explicit signature, argument types
        are inferred on first use unless selected by Numba type annotations.
    opt_level : int, optional
        Native optimization level from 0 through 3. Defaults to 2.
    dump : bool, optional
        Print the host MLIR and LLVM-dialect MLIR. Defaults to False.
    dump_numba_ir : bool, optional
        Print the Numba IR. Defaults to False.
    dump_mlir : bool, optional
        Print the host MLIR before conversion to LLVM dialect. Defaults to False.
    dump_llvm_dialect : bool, optional
        Print the LLVM-dialect MLIR. Defaults to False.

    Returns
    -------
    Dispatcher or callable
        Dispatcher when given a function directly (``@host_jit``); otherwise,
        a decorator that builds the dispatcher (``@host_jit()`` or
        ``@host_jit("f8(f8)")``).

    Notes
    -----
    Experimental scalar-only API. Functions accept fixed positional parameters;
    keyword arguments and non-scalar arguments or return values are unsupported.
    A function may return None.

    Explicit signatures fix the available overloads. Parameter annotations must
    agree with explicit signatures, and return annotations cannot be combined
    with them. A return annotation requires all parameters to have Numba scalar
    type annotations; None is also accepted as a return annotation.

    Each argument-type tuple has one compiled return type. Explicit signatures
    requesting a different return type for an existing overload raise TypeError.
    """
    targetoptions = _validated_options(options)
    signatures = _requested_signatures(func_or_sig)

    def decorate(function):
        dispatcher = Dispatcher(function, dict(targetoptions))
        if signatures is not None:
            _compile_requested_signatures(dispatcher, signatures)
        else:
            _compile_annotated_signature(dispatcher)
        return dispatcher

    # Signature interpretation takes precedence over callable interpretation.
    received_the_function = callable(func_or_sig) and signatures is None
    if received_the_function:
        return decorate(func_or_sig)
    return decorate


def _validated_options(options):
    targetoptions = {
        "opt_level": 2,
        "dump": False,
        "dump_numba_ir": False,
        "dump_mlir": False,
        "dump_llvm_dialect": False,
    }
    unknown = set(options).difference(targetoptions)
    if unknown:
        raise ValueError(f"Unsupported host_jit option(s): {', '.join(sorted(unknown))}")
    targetoptions.update(options)
    if type(targetoptions["opt_level"]) is not int or not 0 <= targetoptions["opt_level"] <= 3:
        raise ValueError("opt_level must be an integer between 0 and 3")
    for key in targetoptions.keys() - {"opt_level"}:
        if not isinstance(targetoptions[key], bool):
            raise TypeError(f"{key} must be a boolean")
    return targetoptions


def _requested_signatures(func_or_sig):
    if sigutils.is_signature(func_or_sig):
        return [func_or_sig]
    if isinstance(func_or_sig, list):
        if not func_or_sig:
            raise ValueError("host_jit requires at least one explicit signature")
        return func_or_sig
    if func_or_sig is None or callable(func_or_sig):
        return None
    raise TypeError("host_jit expects a function or a Numba signature")


def _compile_requested_signatures(dispatcher, signatures):
    if dispatcher._signature.return_annotation is not inspect.Signature.empty:
        raise TypeError("Cannot combine an explicit signature with a return annotation")
    for signature in signatures:
        argtypes, _ = sigutils.normalize_signature(signature)
        if any(
            parameter is not inspect.Signature.empty and parameter != argtype
            for parameter, argtype in zip(dispatcher._arg_annotations, argtypes, strict=True)
        ):
            raise TypeError("Explicit signature conflicts with parameter annotations")
        dispatcher.compile(signature)
    dispatcher.disable_compile()


def _compile_annotated_signature(dispatcher):
    annotation = dispatcher._signature.return_annotation
    if annotation is inspect.Signature.empty:
        return
    return_type = types.none if annotation is None else annotation
    if not isinstance(return_type, types.Type):
        raise TypeError("host_jit annotations must be Numba scalar types or None")
    arguments = dispatcher._arg_annotations
    if any(argument is inspect.Signature.empty for argument in arguments):
        raise TypeError("A return annotation requires Numba scalar parameter annotations")
    dispatcher.compile(typing.signature(return_type, *arguments))
