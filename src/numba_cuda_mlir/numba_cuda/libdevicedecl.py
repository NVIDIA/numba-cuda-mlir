# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba_cuda_mlir.numba_cuda import libdevice, libdevicefuncs
from numba_cuda_mlir.numba_cuda.typing.templates import ConcreteTemplate, Registry

registry = Registry()
register_global = registry.register_global


def libdevice_declare(func, retty, args):
    class Libdevice_function(ConcreteTemplate):
        cases = [libdevicefuncs.create_signature(retty, args)]

    pyfunc = getattr(libdevice, func[5:])
    register_global(pyfunc)(Libdevice_function)


for func, (retty, args) in libdevicefuncs.functions.items():
    libdevice_declare(func, retty, args)
