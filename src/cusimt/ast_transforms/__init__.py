# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# AST transformation passes for cusimt
# These run before Numba's IR conversion
import ast
from typing import Callable

from cusimt.ast_transforms.common import get_function_ast, recompile_function
from cusimt.ast_transforms.comprehension import ComprehensionPass
from cusimt.ast_transforms.consteval import ConstevalError, ConstevalPass
from cusimt.ast_transforms.constant_if import ConstantIfPass
from cusimt.ast_transforms.pipeline import (
    ASTTransformPass,
    ASTTransformPipeline,
    TransformContext,
)

__all__ = [
    "apply_ast_transforms",
    "ConstevalError",
    "ASTTransformPass",
    "ASTTransformPipeline",
    "TransformContext",
]


class NoneStatementRemover(ast.NodeTransformer):
    """Remove bare None expression statements (e.g., from consteval(print(...)))."""

    def __init__(self):
        self.modified = False

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and node.value.value is None:
            self.modified = True
            return None  # Remove the node
        return node


def remove_none_statements(tree: ast.AST) -> tuple[ast.AST, bool]:
    """Remove bare None statements from the AST."""
    remover = NoneStatementRemover()
    tree = remover.visit(tree)
    ast.fix_missing_locations(tree)
    return tree, remover.modified


class NoneStatementRemovalPass(ASTTransformPass):
    """Pipeline pass that removes bare None expression statements."""

    @property
    def name(self) -> str:
        return "NoneStatementRemoval"

    def transform(
        self, tree: ast.Module, context: TransformContext
    ) -> tuple[ast.Module, bool]:
        return remove_none_statements(tree)


def create_default_pipeline() -> ASTTransformPipeline:
    """Create the default AST transformation pipeline."""
    pipeline = ASTTransformPipeline()
    pipeline.add_pass(ConstevalPass())
    pipeline.add_pass(ConstantIfPass())
    pipeline.add_pass(ComprehensionPass())
    pipeline.add_pass(NoneStatementRemovalPass())
    return pipeline


def apply_ast_transforms(
    func: Callable,
    targetoptions: dict = None,
    argtypes: tuple = None,
) -> tuple[Callable, str | None]:
    """Apply AST transformations to a function at compile time.

    Args:
        func: The function to transform
        targetoptions: Compilation options dict. Supports:
            - experimental_ast_transforms: Enable AST transforms (required)
            - dump_ast: Print AST before/after all transformations
            - dump_ast_after_all: Print AST after each transformation pass
        argtypes: Tuple of Numba types for the function arguments. When provided,
            parameter names in consteval expressions will resolve to their types.

    Returns tuple of (transformed_function, transformed_source).
    transformed_source is None if no transformations were applied.
    """
    targetoptions = targetoptions or {}

    # AST transforms are gated behind experimental flag
    if not targetoptions.get("experimental_ast_transforms", False):
        return func, None

    tree = get_function_ast(func)
    if tree is None:
        return func, None

    dump_ast = targetoptions.get("dump_ast", False)
    dump_ast_after_all = targetoptions.get("dump_ast_after_all", False)

    # Print original source if dump_ast is enabled (but not dump_ast_after_all,
    # which prints its own "before" output)
    if dump_ast and not dump_ast_after_all:
        print(f"=== AST for {func.__name__} (before transforms) ===")
        print(ast.unparse(tree))

    # Create context and pipeline
    context = TransformContext(
        func=func,
        targetoptions=targetoptions,
        argtypes=argtypes or (),
    )
    pipeline = create_default_pipeline()

    # Run the pipeline
    tree, modified = pipeline.run(tree, context, dump_after_all=dump_ast_after_all)

    transformed_source = None
    if modified:
        transformed_source = ast.unparse(tree)
        if dump_ast and not dump_ast_after_all:
            print(f"=== AST for {func.__name__} (after transforms) ===")
            if context.stored_values:
                for name, value in context.stored_values.items():
                    print(f"{name} = {value!r}")
            print(transformed_source)
        func = recompile_function(func, tree, context.stored_values)

    return func, transformed_source
