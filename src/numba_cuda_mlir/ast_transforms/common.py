# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Common utilities for AST transformations
import ast
import inspect
import textwrap
import types
from typing import Callable


def get_function_ast(func: Callable) -> ast.Module | None:
    """Parse and return the AST for a function, or None if source unavailable."""
    try:
        source = inspect.getsource(func)
        source = textwrap.dedent(source)
        return ast.parse(source)
    except (OSError, TypeError):
        return None


def get_function_context(func: Callable) -> dict:
    """Build a context dict from a function's globals and closure variables."""
    context = func.__globals__.copy()

    # Add closure variables
    if func.__closure__ is not None:
        for name, cell in zip(func.__code__.co_freevars, func.__closure__):
            context[name] = cell.cell_contents

    return context


def recompile_function(
    func: Callable, tree: ast.Module, stored_values: dict = None
) -> Callable:
    """Compile a modified AST back into a function object.

    Args:
        func: The original function
        tree: The modified AST
        stored_values: Dict of name -> value for complex objects that need to be
            injected into the function's globals
    """
    code = compile(tree, inspect.getfile(func), "exec")

    # Find the function's code object in the compiled module
    func_code = None
    for const in code.co_consts:
        if isinstance(const, types.CodeType) and const.co_name == func.__name__:
            func_code = const
            break

    if func_code is None:
        raise ValueError(f"Could not find compiled code for {func.__name__}")

    # Build closure for the new function - only include cells that are still needed
    new_closure = None
    if func_code.co_freevars and func.__closure__:
        old_freevars = func.__code__.co_freevars
        old_closure = dict(zip(old_freevars, func.__closure__))
        new_closure = tuple(old_closure[name] for name in func_code.co_freevars)

    # Create a new globals dict with closure variables and stored values injected
    new_globals = func.__globals__.copy()

    # Add closure variables to globals - the recompiled code may treat them as globals
    # since it was compiled from source without the original closure context
    if func.__closure__ is not None:
        for name, cell in zip(func.__code__.co_freevars, func.__closure__):
            new_globals[name] = cell.cell_contents

    if stored_values:
        new_globals.update(stored_values)

    # Create new function with the modified code
    new_func = types.FunctionType(
        func_code,
        new_globals,
        func.__name__,
        func.__defaults__,
        new_closure,
    )
    new_func.__annotations__ = func.__annotations__
    new_func.__doc__ = func.__doc__
    new_func.__module__ = func.__module__
    new_func.__qualname__ = func.__qualname__

    return new_func
