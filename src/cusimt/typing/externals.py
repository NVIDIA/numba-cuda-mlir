# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from cusimt.numba_cuda.extending import typeof_impl
from cusimt.numba_cuda.typing.templates import (
    ConcreteTemplate,
    AttributeTemplate,
    Registry,
)
from cusimt.numba_cuda.typing import Signature
from cusimt.numba_cuda import types
from cusimt.type_defs.builtin_types import Namespace
from cusimt.logging import trace
from functools import lru_cache

registry = Registry()


def _type_to_pyi(ty) -> str:
    """Convert a Numba type to a valid Python type hint string."""
    if ty is None:
        return "None"
    if isinstance(ty, types.UniTuple):
        elem = _type_to_pyi(ty.dtype)
        return f"tuple[{', '.join([elem] * ty.count)}]"
    if isinstance(ty, types.Tuple):
        elems = [_type_to_pyi(t) for t in ty.types]
        return f"tuple[{', '.join(elems)}]"
    s = str(ty)
    s = s.replace("none*", "ptr")
    s = s.replace("none", "None")
    return s


@dataclass
class ExternMLIRLibraryFunction:
    name: str
    sig: Signature
    library: "ExternMLIRLibrary"

    def __hash__(self):
        return hash((self.name, self.sig, self.library))

    def __str__(self):
        tys = [_type_to_pyi(ty) for ty in self.sig.args]
        args = [f"arg{i}: {ty}" for i, ty in enumerate(tys)]
        ret = _type_to_pyi(self.sig.return_type)
        return f"def {self.name}({', '.join(args)}) -> {ret}: ..."


class ExternMLIRLibrary:
    """
    A Python object representing a set of external MLIR functions.

    At typing time we treat an instance of this class as a Numba
    ``types.Module`` whose attributes are callable extern functions
    discovered from the MLIR source.
    """

    def __init__(self, functions: dict[str, Signature], source: str):
        # Map function name -> Numba typing.Signature
        self.functions: dict[str, Signature] = functions
        self.source = source
        from cusimt.descriptor import mlir_target

        typingctx = mlir_target.typing_context

        @registry.register_attr
        class ThisExternMLIRLibraryAttrs(ExternMLIRLibraryAttrs):
            key = Namespace(self)

        # Tell the typing context how to type globals equal to this instance.
        # typingctx.insert_global(self, Namespace(self))
        # And how to resolve attributes on that module type.
        typingctx.insert_attributes(ThisExternMLIRLibraryAttrs(typingctx))
        typingctx.refresh()

    @lru_cache(maxsize=None)
    def _get_or_create_template(self, name: str) -> type[ConcreteTemplate]:
        """
        Create (once) a ConcreteTemplate for the given function name.
        """
        if name not in self.functions:
            raise AttributeError(f"ExternMLIRLibrary has no attribute {name!r}")

        sig = self.functions[name]
        lib = self

        class extern_template(ConcreteTemplate):
            # A unique typing key for this extern function within this library.
            key = (lib, name)
            cases = [sig]

        extern_template.__name__ = f"extern_mlir_{name}"
        return extern_template

    @lru_cache(maxsize=None)
    def __getattr__(self, name: str):
        template = self._get_or_create_template(name)
        libfunc = ExternMLIRLibraryFunction(name, template.cases[0], self)

        from cusimt.descriptor import mlir_target

        typingctx = mlir_target.typing_context
        typingctx.insert_user_function(libfunc, template)
        return libfunc


@registry.register_attr
class ExternMLIRLibraryAttrs(AttributeTemplate):
    key = Namespace

    def generic_resolve(self, ns: Namespace, attr: str):
        lib = ns.object

        if attr not in lib.functions:
            trace("ExternMLIRLibrary %s has no attribute %r, skipping", lib, attr)
            return None

        # Build (or retrieve) a typing template for this function name.
        tmpl = lib._get_or_create_template(attr)
        return types.Function(tmpl)


@typeof_impl.register(ExternMLIRLibrary)
def typeof_extern_mlir_library(val, c):
    return Namespace(val)


@typeof_impl.register(ExternMLIRLibraryFunction)
def typeof_extern_mlir_library_function(val, c):
    template = val.library._get_or_create_template(val.name)
    return types.Function(template)
