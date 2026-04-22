# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Constant if-statement folding pass
import ast

from cusimt.ast_transforms.pipeline import ASTTransformPass, TransformContext


class ConstantIfTransformer(ast.NodeTransformer):
    """AST transformer that folds if statements with constant conditions."""

    def __init__(self):
        self.modified = False

    def visit_If(self, node: ast.If) -> ast.AST:
        # First visit children (including nested ifs in body/orelse)
        node = self.generic_visit(node)

        # Check if the condition is a constant
        if isinstance(node.test, ast.Constant):
            self.modified = True
            if node.test.value:
                # Condition is truthy - keep only the if body
                return node.body
            else:
                # Condition is falsy - keep only the else body
                return node.orelse if node.orelse else []

        return node


def transform_constant_if(tree: ast.Module) -> tuple[ast.Module, bool]:
    """Fold if statements with constant conditions.

    Returns (transformed_tree, was_modified).
    """
    transformer = ConstantIfTransformer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    return new_tree, transformer.modified


class ConstantIfPass(ASTTransformPass):
    """Pipeline pass that folds if statements with constant conditions."""

    @property
    def name(self) -> str:
        return "ConstantIfFolding"

    def transform(
        self, tree: ast.Module, context: TransformContext
    ) -> tuple[ast.Module, bool]:
        return transform_constant_if(tree)
