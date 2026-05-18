# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from numba_cuda_mlir.numba_cuda import typing
from numba_cuda_mlir.numba_cuda.core import funcdesc, sigutils
from numba_cuda_mlir.numba_cuda.core.callconv import CUDACABICallConv, CUDACallConv
from numba_cuda_mlir.numba_cuda.codegen import ExternalCodeLibrary


class ExternFunction:
    """A descriptor that can be used to call an external device function."""

    def __init__(self, name, sig):
        self.name = name
        self.sig = sig


@dataclass
class DeviceDeclaration:
    name: str
    restype: object
    argtypes: tuple
    link: tuple
    use_cooperative: bool
    abi: str
    extfn: ExternFunction = field(init=False)
    _typing_context_ids: set[int] = field(default_factory=set, init=False)
    _target_context_ids: set[int] = field(default_factory=set, init=False)

    def __post_init__(self):
        self.sig = typing.signature(self.restype, *self.argtypes)
        self.extfn = ExternFunction(self.name, self.sig)

    def apply(self, typingctx, targetctx):
        self.apply_typing(typingctx)
        self.apply_target(targetctx)

    def apply_typing(self, typingctx):
        context_id = id(typingctx)
        if context_id in self._typing_context_ids:
            return
        device_function_template = typing.make_concrete_template(self.name, self.extfn, [self.sig])
        typingctx.insert_user_function(self.extfn, device_function_template)
        self._typing_context_ids.add(context_id)

    def apply_target(self, targetctx):
        context_id = id(targetctx)
        if context_id in self._target_context_ids:
            return

        lib = ExternalCodeLibrary(f"{self.name}_externals", targetctx.codegen())
        for file in self.link:
            lib.add_linking_file(file)
        lib.use_cooperative = self.use_cooperative

        if self.abi == "numba":
            call_conv = CUDACallConv(targetctx)
        elif self.abi == "c":
            call_conv = CUDACABICallConv(targetctx)
        else:
            raise NotImplementedError(f"Unsupported ABI: {self.abi}")

        fndesc = funcdesc.ExternalFunctionDescriptor(
            self.name, self.restype, self.argtypes, call_conv
        )
        targetctx.insert_user_function(self.extfn, fndesc, libs=(lib,))
        self._target_context_ids.add(context_id)


_device_declarations = []


def normalize_device_declaration(name, sig, link=None, use_cooperative=False, abi="numba"):
    if abi not in ("numba", "c"):
        raise NotImplementedError(f"Unsupported ABI: {abi}")

    if link is None:
        link = tuple()
    elif not isinstance(link, (list, tuple, set)):
        link = (link,)
    else:
        link = tuple(link)

    argtypes, restype = sigutils.normalize_signature(sig)
    if restype is None:
        msg = "Return type must be provided for device declarations"
        raise TypeError(msg)

    return name, restype, tuple(argtypes), link, use_cooperative, abi


def register_device_declaration(name, sig, link=None, use_cooperative=False, abi="numba"):
    declaration = DeviceDeclaration(
        *normalize_device_declaration(name, sig, link, use_cooperative, abi)
    )
    _device_declarations.append(declaration)
    return declaration.extfn


def register_device_declaration_from_parts(name, restype, argtypes, link, use_cooperative, abi):
    declaration = DeviceDeclaration(
        name, restype, tuple(argtypes), tuple(link), use_cooperative, abi
    )
    _device_declarations.append(declaration)
    return declaration


_last_applied_index = {}


def apply_device_declarations(typingctx, targetctx):
    key = (id(typingctx), id(targetctx))
    start = _last_applied_index.get(key, 0)
    end = len(_device_declarations)
    if start >= end:
        return
    for declaration in _device_declarations[start:end]:
        declaration.apply(typingctx, targetctx)
    _last_applied_index[key] = end
