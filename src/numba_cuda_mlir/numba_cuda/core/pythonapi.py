# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""CPython C-API helpers.

The bulk of this module - the ``PythonAPI`` class (PyObject C-API codegen),
``ObjModeUtils``, the box/unbox/reflect contexts and ``EnvironmentManager`` -
built llvmlite IR for object-mode boxing/unboxing and ran only on the vendored
``Lower`` path, which is dead on the MLIR path (device code cannot call the
CPython C-API). It has been removed. What remains is the live extension surface:
the ``PY_UNICODE_*_KIND`` constants (used by the MLIR unicode lowering), the
boxing/unboxing/reflection registries and their ``box``/``unbox``/``reflect``
decorators (the public extension API), and ``NativeValue`` (returned by
extension unbox implementations).
"""

from numba_cuda_mlir import _helperlib
from numba_cuda_mlir.numba_cuda.utils import PYVERSION
from numba_cuda_mlir.numba_cuda import types, cgutils

PY_UNICODE_1BYTE_KIND = _helperlib.py_unicode_1byte_kind
PY_UNICODE_2BYTE_KIND = _helperlib.py_unicode_2byte_kind
PY_UNICODE_4BYTE_KIND = _helperlib.py_unicode_4byte_kind
if PYVERSION in ((3, 10), (3, 11)):
    PY_UNICODE_WCHAR_KIND = _helperlib.py_unicode_wchar_kind


class _Registry:
    def __init__(self):
        self.functions = {}

    def register(self, typeclass):
        assert issubclass(typeclass, types.Type)

        def decorator(func):
            if typeclass in self.functions:
                raise KeyError("duplicate registration for %s" % (typeclass,))
            self.functions[typeclass] = func
            return func

        return decorator

    def lookup(self, typeclass, default=None):
        assert issubclass(typeclass, types.Type)
        for cls in typeclass.__mro__:
            func = self.functions.get(cls)
            if func is not None:
                return func
        return default


# Registries of boxing / unboxing implementations
_boxers = _Registry()
_unboxers = _Registry()
_reflectors = _Registry()

box = _boxers.register
unbox = _unboxers.register
reflect = _reflectors.register


class NativeValue:
    """
    Encapsulate the result of converting a Python object to a native value,
    recording whether the conversion was successful and how to cleanup.
    """

    def __init__(self, value, is_error=None, cleanup=None):
        self.value = value
        self.is_error = is_error if is_error is not None else cgutils.false_bit
        self.cleanup = cleanup
