# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import func, gpu, llvm
from numba_cuda_mlir.lowering_utilities.link import CompiledUnit, ModuleAssembly


@pytest.fixture(autouse=True)
def mlir_context():
    with ir.Context(), ir.Location.unknown():
        yield


def _definition(name, value=1):
    return (
        f"func.func @{name}() -> i32 {{\n  %c = arith.constant {value} : i32\n  return %c : i32\n}}"
    )


def _unit(*operations):
    module = ir.Module.parse("module {\n" + "\n".join(operations) + "\n}")
    return CompiledUnit(module, module.body)


def _assembly(*operations, private=False):
    unit = _unit(*operations)
    return ModuleAssembly(unit.module, unit.scope, private=private)


def _symbols(scope):
    return {
        ir.StringAttr(op.attributes["sym_name"]).value: op
        for op in scope
        if "sym_name" in op.attributes
    }


def _definitions(scope):
    return {
        name
        for name, op in _symbols(scope).items()
        if any(len(region.blocks) for region in op.regions)
    }


def test_declaration_exposes_exact_layout_without_importing_body():
    source = ir.Module.create()
    packed = llvm.StructType.get_literal(
        [ir.IntegerType.get_signless(8), llvm.PointerType.get(address_space=3)], packed=True
    )
    function_type = ir.FunctionType.get([packed], [packed])
    with ir.InsertionPoint(source.body):
        callee = func.FuncOp("identity", function_type)
        entry = callee.add_entry_block()
        with ir.InsertionPoint(entry):
            func.ReturnOp([entry.arguments[0]])
    unit = CompiledUnit(source, source.body)
    assembly = _assembly()

    ref = assembly.declare(unit, "identity")
    declaration = _symbols(assembly.target)[ref.symbol]
    assert ref.function_type == function_type
    assert ref.function_type.inputs[0] == packed
    assert declaration.is_external
    assert _definitions(assembly.target) == set()

    with ir.InsertionPoint(assembly.target):
        caller = func.FuncOp("caller", ref.function_type)
        entry = caller.add_entry_block()
        with ir.InsertionPoint(entry):
            call = func.CallOp(list(ref.function_type.results), ref.symbol, list(entry.arguments))
            func.ReturnOp(list(call.results))
    result = assembly.finish()
    assert result.module is assembly.module
    assert _definitions(result.scope) == {"identity", "caller"}
    assert result.module.operation.verify()


def test_repeated_declaration_imports_one_body_and_finish_is_idempotent():
    source = _unit(_definition("callee"))
    assembly = _assembly()
    first = assembly.declare(source, "callee")
    second = assembly.declare(source, "callee")
    assert first.symbol == second.symbol
    assert first.function_type == second.function_type
    result = assembly.finish()
    assert len(list(result.scope)) == 1
    assert not _symbols(result.scope)["callee"].is_external
    assert assembly.finish() is result
    with pytest.raises(RuntimeError, match="already finished"):
        assembly.declare(source, "callee")


def test_definition_replaces_existing_compatible_declaration():
    source = _unit(_definition("callee"))
    assembly = _assembly("func.func private @callee() -> i32")
    assembly.declare(source, "callee")
    result = assembly.finish()
    assert len(list(result.scope)) == 1
    assert not _symbols(result.scope)["callee"].is_external


def test_declaration_does_not_replace_existing_definition():
    source = _unit("func.func private @callee() -> i32")
    assembly = _assembly(_definition("callee"))
    original = _symbols(assembly.target)["callee"]
    assembly.declare(source, "callee")
    assert _symbols(assembly.finish().scope)["callee"] == original


def test_distinct_equal_strong_definitions_conflict():
    left = _unit(_definition("shared"))
    right = _unit(_definition("shared"))
    assembly = _assembly()
    assembly.declare(left, "shared")
    assembly.declare(right, "shared")
    with pytest.raises(ValueError, match="Conflicting definitions.*shared"):
        assembly.finish()
    assert _definitions(assembly.target) == set()


def test_local_strong_definition_conflicts_with_import():
    source = _unit(_definition("shared"))
    assembly = _assembly(_definition("shared"))
    assembly.declare(source, "shared")
    with pytest.raises(ValueError, match="Conflicting definitions.*shared"):
        assembly.finish()


def test_incompatible_signature_rejected_during_declaration():
    source = _unit(_definition("callee"))
    assembly = _assembly("func.func private @callee() -> i64")
    with pytest.raises(ValueError, match="Incompatible function types.*callee"):
        assembly.declare(source, "callee")


def test_dependency_signature_conflict_detected_before_import():
    left = _unit(_definition("left"), "func.func private @external(i32)")
    right = _unit(_definition("right"), "func.func private @external(i64)")
    assembly = _assembly()
    assembly.declare(left, "left")
    assembly.declare(right, "right")
    with pytest.raises(ValueError, match="Incompatible function types.*external"):
        assembly.finish()
    assert _definitions(assembly.target) == set()


def test_diamond_import_keeps_original_definition_provenance():
    leaf = _unit(_definition("leaf"))
    left = _assembly(_definition("left"))
    right = _assembly(_definition("right"))
    left.declare(leaf, "leaf")
    right.declare(leaf, "leaf")
    left_unit, right_unit = left.finish(), right.finish()
    source_text = [str(unit.module) for unit in (leaf, left_unit, right_unit)]
    assembly = _assembly()
    assembly.declare(left_unit, "left")
    assembly.declare(right_unit, "right")
    assembly.declare(leaf, "leaf")
    result = assembly.finish()
    assert _definitions(result.scope) == {"leaf", "left", "right"}
    assert len(list(result.scope)) == 3
    assert [str(unit.module) for unit in (leaf, left_unit, right_unit)] == source_text


def test_private_import_does_not_change_source_visibility():
    source = _unit(_definition("callee"))
    original = str(source.module)
    assembly = _assembly(private=True)
    assembly.declare(source, "callee")
    result = assembly.finish()
    assert _symbols(result.scope)["callee"].attributes["sym_visibility"].value == "private"
    assert str(source.module) == original


@pytest.mark.parametrize("linkage", ["common", "linkonce", "linkonce_odr"])
def test_compatible_coalescing_globals_are_preserved(linkage):
    global_op = f"llvm.mlir.global {linkage} @shared(0 : i32) : i32"
    left = _unit(_definition("left"), global_op)
    right = _unit(_definition("right"), global_op)
    original = [str(left.module), str(right.module)]
    assembly = _assembly()
    assembly.declare(left, "left")
    assembly.declare(right, "right")
    symbols = _symbols(assembly.finish().scope)
    assert set(symbols) == {"left", "right", "shared"}
    assert str(symbols["shared"].attributes["linkage"]) == f"#llvm.linkage<{linkage}>"
    assert [str(left.module), str(right.module)] == original


@pytest.mark.parametrize(
    "other, error",
    [
        ("llvm.mlir.global common @shared(0 : i64) : i64", "Incompatible global"),
        ("llvm.mlir.global linkonce @shared(0 : i32) : i32", "Conflicting definitions"),
    ],
)
def test_incompatible_common_globals_are_rejected(other, error):
    left = _unit(_definition("left"), "llvm.mlir.global common @shared(0 : i32) : i32")
    right = _unit(_definition("right"), other)
    assembly = _assembly()
    assembly.declare(left, "left")
    assembly.declare(right, "right")
    with pytest.raises(ValueError, match=error):
        assembly.finish()


def test_linkonce_globals_with_different_initializers_conflict():
    left = _unit(_definition("left"), "llvm.mlir.global linkonce @shared(0 : i32) : i32")
    right = _unit(_definition("right"), "llvm.mlir.global linkonce @shared(1 : i32) : i32")
    assembly = _assembly()
    assembly.declare(left, "left")
    assembly.declare(right, "right")
    with pytest.raises(ValueError, match="Conflicting definitions.*shared"):
        assembly.finish()


def test_distinct_equal_strong_globals_conflict():
    global_op = "llvm.mlir.global @shared(0 : i32) : i32"
    left = _unit(_definition("left"), global_op)
    right = _unit(_definition("right"), global_op)
    assembly = _assembly()
    assembly.declare(left, "left")
    assembly.declare(right, "right")
    with pytest.raises(ValueError, match="Conflicting definitions.*shared"):
        assembly.finish()


def test_global_declaration_is_replaced_by_definition():
    source = _unit(_definition("callee"), "llvm.mlir.global @shared(0 : i32) : i32")
    assembly = _assembly("llvm.mlir.global external @shared() : i32")
    assembly.declare(source, "callee")
    global_op = _symbols(assembly.finish().scope)["shared"]
    assert ir.IntegerAttr(global_op.attributes["value"]).value == 0


def test_library_parsed_once_per_assembly_and_imported_only_at_finish():
    library = SimpleNamespace(source="module {" + _definition("external") + "}")
    assembly = _assembly()
    unit = assembly.add_library(library)
    assert assembly.add_library(library) is unit
    assert list(assembly.target) == []
    ref = assembly.declare(unit, "external")
    assert ref.symbol == "external"
    assert _definitions(assembly.target) == set()
    assert _definitions(assembly.finish().scope) == {"external"}


def test_independently_parsed_library_diamond_has_stable_origin():
    library = SimpleNamespace(source="module {" + _definition("external") + "}")
    left = _assembly(_definition("left"))
    right = _assembly(_definition("right"))
    left.add_library(library)
    right.add_library(library)
    assembly = _assembly()
    assembly.declare(left.finish(), "left")
    assembly.declare(right.finish(), "right")
    assert _definitions(assembly.finish().scope) == {"left", "right", "external"}


def test_import_uses_only_explicit_source_and_target_scopes():
    source = ir.Module.parse(
        "module {"
        + _definition("host_only")
        + "gpu.module @device {"
        + _definition("callee")
        + "}}"
    )
    source_device = next(op for op in source.body if isinstance(op, gpu.GPUModuleOp))
    unit = CompiledUnit(source, source_device.bodyRegion.blocks[0])
    destination = ir.Module.parse("module { gpu.module @first {} gpu.module @second {} }")
    first, second = list(destination.body)
    target = second.bodyRegion.blocks[0]
    source_text = str(source)
    assembly = ModuleAssembly(destination, target)
    assembly.declare(unit, "callee")
    result = assembly.finish()
    assert _definitions(result.scope) == {"callee"}
    assert list(first.bodyRegion.blocks[0]) == []
    assert set(_symbols(destination.body)) == {"first", "second"}
    assert str(source) == source_text


def test_missing_symbol_does_not_search_nested_or_sibling_scopes():
    source = ir.Module.parse("module { gpu.module @device {" + _definition("callee") + "}}")
    assembly = _assembly()
    with pytest.raises(ValueError, match="absent from its source scope"):
        assembly.declare(CompiledUnit(source, source.body), "callee")


def test_unit_includes_later_runtime_emission_with_explicit_provenance():
    origin = object()
    units = []
    for name in ("left", "right"):
        assembly = _assembly(_definition(name))
        unit = assembly.finish()
        with ir.InsertionPoint(unit.scope):
            runtime = func.FuncOp("runtime", ir.FunctionType.get([], []))
            with ir.InsertionPoint(runtime.add_entry_block()):
                func.ReturnOp([])
        unit.record_generated_symbols(["runtime"], origin=origin)
        units.append(unit)
    assembly = _assembly()
    assembly.declare(units[0], "left")
    assembly.declare(units[1], "right")
    assert _definitions(assembly.finish().scope) == {"left", "right", "runtime"}


def test_generated_symbol_registration_requires_actual_symbol():
    unit = _unit(_definition("callee"))
    with pytest.raises(ValueError, match="Generated symbol.*absent"):
        unit.record_generated_symbols(["missing"], origin=object())


def test_generated_symbols_refresh_previously_declared_source_exports():
    source = _unit(_definition("callee"))
    assembly = _assembly()
    assembly.declare(source, "callee")
    with ir.InsertionPoint(source.scope):
        runtime = func.FuncOp("runtime", ir.FunctionType.get([], []))
        with ir.InsertionPoint(runtime.add_entry_block()):
            func.ReturnOp([])
    source.record_generated_symbols(["runtime"], origin=object())
    assembly.declare(source, "runtime")
    assert _definitions(assembly.finish().scope) == {"callee", "runtime"}


def test_new_source_signature_is_checked_against_retained_declaration():
    first = _unit("func.func private @callee(i32)")
    second = _unit("func.func private @callee(i64)")
    assembly = _assembly()
    assembly.declare(first, "callee")
    with pytest.raises(ValueError, match="Incompatible function types.*callee"):
        assembly.declare(second, "callee")


def test_declaration_inserted_after_assembly_creation_is_reused():
    source = _unit(_definition("callee"))
    assembly = _assembly()
    function_type = ir.FunctionType.get([], [ir.IntegerType.get_signless(32)])
    with ir.InsertionPoint(assembly.target):
        declaration = func.FuncOp("callee", function_type, visibility="private")
    assembly.declare(source, "callee")
    assert list(assembly.target) == [declaration]
    result = assembly.finish()
    assert len(list(result.scope)) == 1
    assert _definitions(result.scope) == {"callee"}


def test_mixed_library_scopes_import_callable_functions_and_globals():
    shared = "llvm.mlir.global common @shared(0 : i32) : i32"
    library = SimpleNamespace(
        source="\n".join(
            [
                "module {",
                _definition("top"),
                shared,
                "gpu.module @first {",
                _definition("device"),
                shared,
                "llvm.mlir.global @gpu_global(7 : i32) : i32",
                "} gpu.module @second {",
                _definition("other_device"),
                "}}",
            ]
        )
    )
    assembly = _assembly()
    unit = assembly.add_library(library)
    original = str(unit.module)
    refs = [assembly.declare(unit, name) for name in ("top", "device", "other_device")]
    assert _definitions(assembly.target) == set()
    assert set(_symbols(assembly.target)) == {"top", "device", "other_device"}
    with ir.InsertionPoint(assembly.target):
        caller = func.FuncOp("caller", ir.FunctionType.get([], []))
        with ir.InsertionPoint(caller.add_entry_block()):
            for ref in refs:
                func.CallOp(list(ref.function_type.results), ref.symbol, [])
            func.ReturnOp([])
    result = assembly.finish()
    symbols = _symbols(result.scope)
    assert set(symbols) == {"top", "device", "other_device", "caller", "shared", "gpu_global"}
    assert _definitions(result.scope) == {"top", "device", "other_device", "caller"}
    assert ir.IntegerAttr(symbols["gpu_global"].attributes["value"]).value == 7
    assert result.module.operation.verify()
    assert str(unit.module) == original


@pytest.mark.parametrize("definition_first", [False, True])
def test_library_declaration_and_definition_reconcile_across_scopes(definition_first):
    declaration = "func.func private @callee() -> i32"
    definition = _definition("callee")
    top, nested = (definition, declaration) if definition_first else (declaration, definition)
    library = SimpleNamespace(
        source="module {\n" + top + "\ngpu.module @device {\n" + nested + "\n}}"
    )
    assembly = _assembly()
    unit = assembly.add_library(library)
    assembly.declare(unit, "callee")
    result = assembly.finish()
    assert len(list(result.scope)) == 1
    assert _definitions(result.scope) == {"callee"}


@pytest.mark.parametrize(
    "top, nested, error",
    [
        (_definition("callee"), _definition("callee"), "Conflicting definitions"),
        (
            "func.func private @callee() -> i32",
            "func.func private @callee() -> i64",
            "Incompatible function types",
        ),
    ],
)
def test_conflicting_symbols_in_distinct_library_scopes_are_rejected(top, nested, error):
    library = SimpleNamespace(
        source="module {\n" + top + "\ngpu.module @device {\n" + nested + "\n}}"
    )
    assembly = _assembly()
    unit = assembly.add_library(library)
    with pytest.raises(ValueError, match=error):
        assembly.declare(unit, "callee")
