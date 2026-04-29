# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cusimt.cuda.libdevice as libdevice
import cusimt.cuda.libdevicefuncs as libdevicefuncs
from cusimt.numba_cuda.typing.templates import (
    AttributeTemplate,
    ConcreteTemplate,
    Registry,
)
from cusimt.numba_cuda import types

registry = Registry()
_mapping = dict[str, ConcreteTemplate]()


def libdevice_declare(pyfunc, py_sig):
    class Libdevice_function(ConcreteTemplate):
        key = pyfunc
        cases = [py_sig]

    registry.register_global(pyfunc)(Libdevice_function)
    return Libdevice_function


def _libdevice_register():
    """
    Register type declarations and lowerings for all libdevice functions.
    """
    for descriptor in libdevicefuncs.libdevice_descriptors():
        py_sig = descriptor.py_sig
        name = descriptor.py_name
        pyfunc = getattr(libdevice, descriptor.py_name)
        Libdevice_function = libdevice_declare(pyfunc, py_sig)
        _mapping[name] = Libdevice_function


@registry.register_attr
class LibdeviceModuleTemplate(AttributeTemplate):
    import cusimt.cuda.libdevice as libdevice

    key = types.Module(libdevice)

    def resolve(self, mod, attr):
        if template := _mapping.get(attr):
            return types.Function(template)
        raise AttributeError(f"Libdevice module has no attribute {attr!r}")


_libdevice_register()
