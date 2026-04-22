# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cusimt._mlir import ir
from cusimt._mlir.dialects import gpu, llvm
from cusimt.logging import trace


def _is_link_target(op: ir.Operation) -> bool:
    return "cusimt.link_target" in op.attributes.keys()


def _ensure_has_regions_and_blocks(op: ir.Operation):
    if len(op.regions) == 0:
        op.regions.append()
    if len(op.regions[0].blocks) == 0:
        op.regions[0].blocks.append()
    return op.regions[-1].blocks[-1]


def _find_link_target(op: ir.Operation) -> ir.Block:
    trace()
    if _is_link_target(op):
        return _ensure_has_regions_and_blocks(op)
    for region in op.regions:
        for block in region.blocks:
            for op in block:
                if target := _find_link_target(op):
                    return target
    return None


def _is_linkable_op(op):
    """Check if an op should be carried across during linking."""
    from cusimt._mlir.dialects import func

    return isinstance(op, (func.FuncOp, gpu.GPUFuncOp, llvm.GlobalOp))


def _extract_linkable_ops(module: ir.Module) -> list[ir.Operation]:
    """Extract all linkable ops (functions, globals) from a module."""
    ops = []
    for op in module.body:
        if _is_linkable_op(op):
            ops.append(op)
        elif isinstance(op, gpu.GPUModuleOp):
            for nested_op in op.regions[0].blocks[0]:
                if _is_linkable_op(nested_op):
                    ops.append(nested_op)
    return ops


def _get_op_sym_name(op) -> str | None:
    """Get the symbol name from an op, if it has one."""
    if "sym_name" in op.attributes.keys():
        return op.attributes["sym_name"].value
    return None


def _get_existing_names(block: ir.Block) -> set[str]:
    """Get symbol names of all linkable ops in a block."""
    names = set()
    for op in block:
        if _is_linkable_op(op):
            if name := _get_op_sym_name(op):
                names.add(name)
    return names


def link_inplace(module: ir.Module | str, other: ir.Module | str):
    """
    Pulls all linkable ops (functions and globals) from `other` and inserts
    them into the body of `module`'s gpu.module so they are available.

    TODO: Use junfeng's MLIR-level lto when available.
    """
    trace("linking %s", other)
    if isinstance(module, str):
        module = ir.Module.parse(module)
    if isinstance(other, str):
        other = ir.Module.parse(other)
    link_target = _find_link_target(module.operation)

    existing_names = _get_existing_names(link_target)
    ops = _extract_linkable_ops(other)
    with ir.InsertionPoint(link_target) as ip:
        for op in ops:
            sym_name = _get_op_sym_name(op)
            if sym_name and sym_name in existing_names:
                trace("skipping duplicate symbol %s", sym_name)
                continue
            if sym_name:
                existing_names.add(sym_name)
            cloned = op.clone()
            ip.insert(cloned.detach_from_parent())

    return module
