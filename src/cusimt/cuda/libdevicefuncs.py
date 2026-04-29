# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cusimt.numba_cuda import typing, types
from dataclasses import dataclass
from cusimt.numba_cuda.libdevicefuncs import functions, create_signature


@dataclass
class Arg:
    name: str
    ty: types.Type


@dataclass
class Sig:
    name: str
    return_ty: types.Type
    args: tuple[Arg]


@dataclass
class Descriptor:
    py_name: str
    arg_names: tuple[str]
    py_sig: typing.Signature
    c_sig: typing.Signature

    def __str__(self) -> str:
        py_args = ", ".join(
            f"{name}: {ty}" for ty, name in zip(self.py_sig.args, self.arg_names)
        )
        c_args = ", ".join(
            f"{ty} {name}" for ty, name in zip(self.c_sig.args, self.arg_names)
        )
        c_name = f"__nv_{self.py_name}"
        c_ret = (
            "void"
            if self.c_sig.return_type is types.none
            else str(self.c_sig.return_type)
        )
        c_api = f"{c_ret} {c_name}({c_args});"

        match self.py_sig.return_type:
            case types.UniTuple() as ut:
                py_ret = f"UniTuple({ut.dtype}, {ut.count})"
            case types.Tuple() as t:
                ty_str = ", ".join(str(ty) for ty in t.types)
                py_ret = f"Tuple([{ty_str}])"
            case _:
                py_ret = str(self.py_sig.return_type)

        return f'''
def {self.py_name}({py_args}) -> {py_ret}:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/{c_name}.html

    CAPI: {c_api}
    """
'''


def libdevice_descriptors():
    for c_name, (retty, args) in functions.items():
        py_api = create_signature(retty, args)
        c_api_args = tuple(
            types.CPointer(arg.ty) if arg.is_ptr else arg.ty for arg in args
        )
        py_name = c_name[5:]
        arg_names = tuple(arg.name.replace("in", "in_") for arg in args)

        yield Descriptor(
            py_name=py_name,
            arg_names=arg_names,
            py_sig=py_api,
            c_sig=retty(*c_api_args),
        )
