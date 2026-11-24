# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import func, gpu


_FUNCTIONS = {"func.func", "gpu.func", "llvm.func"}
_LINKABLE = _FUNCTIONS | {"llvm.mlir.global"}
_FUNCTION_ABI_ATTRIBUTES = ("arg_attrs", "res_attrs", "CConv", "comdat")
_COALESCING_LINKAGES = {
    "#llvm.linkage<common>",
    "#llvm.linkage<linkonce>",
    "#llvm.linkage<linkonce_odr>",
}


def _symbols(scope: ir.Block):
    for view in scope:
        op = view.operation
        if op.name in _LINKABLE:
            yield ir.StringAttr(op.attributes["sym_name"]).value, op


def _symbol_table(scope: ir.Block):
    symbols = {}
    for name, op in _symbols(scope):
        if name in symbols:
            raise ValueError(f"Duplicate symbol {name!r} in source scope")
        symbols[name] = op
    return symbols


def _is_definition(op: ir.Operation) -> bool:
    if op.name == "llvm.mlir.global" and "value" in op.attributes:
        return True
    return any(len(region.blocks) for region in op.regions)


def _attributes(op: ir.Operation, excluded=()):
    return {name: op.attributes[name] for name in op.attributes.keys() if name not in excluded}


def _check_compatible(name: str, left: ir.Operation, right: ir.Operation):
    if left.name != right.name:
        raise ValueError(f"Incompatible symbol kinds for {name!r}: {left.name} and {right.name}")
    if left.name in _FUNCTIONS:
        if left.attributes["function_type"] != right.attributes["function_type"]:
            raise ValueError(f"Incompatible function types for {name!r}")
        # Every explicitly supplied ABI constraint must survive reconciliation,
        # even when this operation is only a declaration.
        for attr in _FUNCTION_ABI_ATTRIBUTES:
            if attr in left.attributes and attr in right.attributes:
                if left.attributes[attr] != right.attributes[attr]:
                    raise ValueError(f"Incompatible function attribute {attr!r} for {name!r}")
    else:
        # Initializers and linkage distinguish definitions, not the storage ABI.
        excluded = {"sym_visibility", "value", "linkage"}
        if _attributes(left, excluded) != _attributes(right, excluded):
            raise ValueError(f"Incompatible global types or attributes for {name!r}")


def _coalescible_globals(left: ir.Operation, right: ir.Operation) -> bool:
    if left.name != "llvm.mlir.global":
        return False
    if str(left.attributes["linkage"]) not in _COALESCING_LINKAGES:
        return False
    if _attributes(left, {"sym_visibility"}) != _attributes(right, {"sym_visibility"}):
        return False
    if not any(len(region.blocks) for op in (left, right) for region in op.regions):
        return True
    # Compare complete initializer bodies, normalizing only the outer attribute
    # already excluded above. Detached clones never mutate retained sources.
    if ("sym_visibility" in left.attributes) != ("sym_visibility" in right.attributes) or (
        "sym_visibility" in left.attributes
        and left.attributes["sym_visibility"] != right.attributes["sym_visibility"]
    ):
        left = left.clone(ip=False).operation
        if "sym_visibility" in right.attributes:
            left.attributes["sym_visibility"] = right.attributes["sym_visibility"]
        elif "sym_visibility" in left.attributes:
            del left.attributes["sym_visibility"]
    return left.get_asm(use_local_scope=True) == right.get_asm(use_local_scope=True)


def _merge_symbol(name, existing, incoming, *, same_origin=False):
    _check_compatible(name, existing, incoming)
    if not _is_definition(incoming):
        selected = existing
    elif not _is_definition(existing):
        selected = incoming
    elif same_origin or existing == incoming or _coalescible_globals(existing, incoming):
        selected = existing
    else:
        raise ValueError(f"Conflicting definitions for symbol {name!r}")
    if selected.name in _FUNCTIONS:
        missing = {
            attr: op.attributes[attr]
            for op in (existing, incoming)
            for attr in _FUNCTION_ABI_ATTRIBUTES
            if attr in op.attributes and attr not in selected.attributes
        }
        if missing:
            selected = selected.clone(ip=False).operation
            for attr, value in missing.items():
                selected.attributes[attr] = value
    return selected


@dataclass(frozen=True)
class FunctionRef:
    """A typed symbol reference retaining the unit that owns its MLIR type."""

    symbol: str
    function_type: ir.FunctionType
    _unit: "CompiledUnit" = field(repr=False, compare=False)


class CompiledUnit:
    """An assembled module and its explicit source symbol scope or scopes.

    The retained scope remains live: runtime emission may add symbols after
    assembly and before the compilation result is published to callers.
    Imported definitions keep their original provenance across further imports.
    """

    def __init__(self, module: ir.Module, scope: ir.Block | tuple[ir.Block, ...]):
        self.module = module
        self.scope = scope
        self._scopes = scope if isinstance(scope, tuple) else (scope,)
        self._origin = object()
        self._origins = {}
        self._dependencies = ()
        self._exports = None

    def _export_symbols(self):
        if self._exports is None:
            exports = {}
            for scope in self._scopes:
                for name, op in _symbol_table(scope).items():
                    exports[name] = (
                        _merge_symbol(name, exports[name], op) if name in exports else op
                    )
            self._exports = exports
        return self._exports

    def _symbol_origin(self, symbol: str):
        return self._origins.get(symbol, self._origin)

    def record_generated_symbols(self, symbols, *, origin):
        """Identify deliberately shared runtime definitions emitted after finish.

        The emitter supplies a stable identity shared by its compilations; normal
        user definitions always retain their distinct compilation-unit identity.
        """
        self._exports = None
        available = self._export_symbols()
        for symbol in symbols:
            if symbol not in available:
                raise ValueError(f"Generated symbol {symbol!r} is absent from its source scope")
            self._origins[symbol] = origin


class ModuleAssembly:
    """Declare dependencies during lowering and import bodies exactly once."""

    def __init__(self, module: ir.Module, target: ir.Block, *, private: bool = False):
        self.module = module
        self.target = target
        self.private = private
        self._unit = CompiledUnit(module, target)
        self._dependencies = {}
        self._declarations = _symbol_table(target)
        self._libraries = {}
        self._finished = False

    def _require_open(self):
        if self._finished:
            raise RuntimeError("Module assembly is already finished")

    def _add_dependency(self, unit: CompiledUnit):
        if unit is self._unit:
            raise ValueError("A module assembly cannot import itself")
        self._dependencies[id(unit)] = unit

    def declare(self, unit: CompiledUnit, symbol: str) -> FunctionRef:
        """Insert a typed declaration without importing any dependency bodies."""
        self._require_open()
        source = unit._export_symbols().get(symbol)
        if source is None:
            raise ValueError(f"Function {symbol!r} is absent from its source scope")
        if source.name != "func.func":
            raise ValueError(f"Symbol {symbol!r} is not a func.func function")
        function_type = ir.FunctionType(ir.TypeAttr(source.attributes["function_type"]).value)
        existing = self._declarations.get(symbol)
        if existing is None:
            # Runtime lowering can insert declarations outside this assembler.
            existing = next((op for name, op in _symbols(self.target) if name == symbol), None)
            if existing is not None:
                self._declarations[symbol] = existing
        if existing is not None:
            _check_compatible(symbol, existing, source)
            for attr in _FUNCTION_ABI_ATTRIBUTES:
                if attr in source.attributes and attr not in existing.attributes:
                    existing.attributes[attr] = source.attributes[attr]
        else:
            with (
                ir.InsertionPoint.at_block_begin(self.target),
                ir.Location.unknown(self.module.context),
            ):
                declaration = func.FuncOp(symbol, function_type, visibility="private")
            for attr in _FUNCTION_ABI_ATTRIBUTES:
                if attr in source.attributes:
                    declaration.attributes[attr] = source.attributes[attr]
            self._declarations[symbol] = declaration.operation
        self._add_dependency(unit)
        return FunctionRef(symbol, function_type, unit)

    def add_library(self, library) -> CompiledUnit:
        """Parse external source once, including its top-level GPU symbol scopes."""
        self._require_open()
        source = library.source
        unit = self._libraries.get(source)
        if unit is None:
            module = ir.Module.parse(source, context=self.module.context)
            scopes = (
                module.body,
                *(op.bodyRegion.blocks[0] for op in module.body if isinstance(op, gpu.GPUModuleOp)),
            )
            unit = CompiledUnit(module, scopes)
            # Source identity, unlike the parsed operation identity, survives
            # independent imports of this library through a dependency diamond.
            unit._origin = ("external-mlir-source", source)
            self._libraries[source] = unit
        self._add_dependency(unit)
        return unit

    def finish(self) -> CompiledUnit:
        """Reconcile declarations and import definitions without editing sources."""
        if self._finished:
            return self._unit
        local = _symbol_table(self.target)
        resolved = {name: (op, self._unit._symbol_origin(name)) for name, op in local.items()}
        # Resolve before cloning so a conflict cannot leave half-imported bodies.
        for unit in self._dependencies.values():
            for name, incoming in unit._export_symbols().items():
                origin = unit._symbol_origin(name)
                if name not in resolved:
                    resolved[name] = incoming, origin
                    continue
                existing, existing_origin = resolved[name]
                selected = _merge_symbol(
                    name, existing, incoming, same_origin=existing_origin == origin
                )
                if not _is_definition(existing) and _is_definition(incoming):
                    existing_origin = origin
                resolved[name] = selected, existing_origin
        for name, (op, origin) in resolved.items():
            if name in local and op == local[name]:
                continue
            clone = op.clone(ip=ir.InsertionPoint.at_block_begin(self.target))
            if self.private and op.name in _FUNCTIONS:
                # Mutate only the clone, never a retained source definition.
                clone.attributes["sym_visibility"] = ir.StringAttr.get(
                    "private", context=self.module.context
                )
            if name in local:
                local[name].erase()
            self._unit._origins[name] = origin
        self._unit._dependencies = tuple(self._dependencies.values())
        self._finished = True
        return self._unit
