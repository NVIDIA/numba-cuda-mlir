# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Empty statement body repair pass
import ast

from numba_cuda_mlir.ast_transforms.pipeline import ASTTransformPass, TransformContext


class EmptyBodyRepairer(ast.NodeTransformer):
    """Fill empty statement bodies with ``pass``."""

    def __init__(self):
        self.modified = False

    def generic_visit(self, node: ast.AST) -> ast.AST:
        node = super().generic_visit(node)
        body = getattr(node, "body", None)
        if isinstance(body, list) and not body:
            node.body = [ast.copy_location(ast.Pass(), node)]
            self.modified = True
        return node


def repair_empty_bodies(tree: ast.Module) -> tuple[ast.Module, bool]:
    """Insert ``pass`` into every empty statement body; returns (tree, was_modified)."""
    repairer = EmptyBodyRepairer()
    new_tree = repairer.visit(tree)
    ast.fix_missing_locations(new_tree)
    return new_tree, repairer.modified


class EmptyBodyRepairPass(ASTTransformPass):
    """Pipeline pass that fills statement bodies emptied by earlier passes with ``pass``."""

    @property
    def name(self) -> str:
        return "EmptyBodyRepair"

    def transform(self, tree: ast.Module, context: TransformContext) -> tuple[ast.Module, bool]:
        return repair_empty_bodies(tree)
