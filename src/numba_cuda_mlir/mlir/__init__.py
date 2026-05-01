# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .ast.py_type import PyTypeObject


def make_nanobind_metaclass_inheritable():
    from numba_cuda_mlir._mlir import ir

    # Hack to allow us to inherit from nanobind's metaclass type
    # https://github.com/wjakob/nanobind/pull/836
    nb_meta_cls = type(ir.Value)
    _Py_TPFLAGS_BASETYPE = 1 << 10
    PyTypeObject.from_object(nb_meta_cls).tp_flags |= _Py_TPFLAGS_BASETYPE


make_nanobind_metaclass_inheritable()
