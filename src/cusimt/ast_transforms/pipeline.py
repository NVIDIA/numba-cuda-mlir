# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# AST transformation pipeline infrastructure
import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class TransformContext:
    """Shared context passed through the AST transform pipeline."""

    func: Callable
    targetoptions: dict = field(default_factory=dict)
    argtypes: tuple = ()
    stored_values: dict = field(default_factory=dict)


class ASTTransformPass(ABC):
    """Base class for AST transformation passes.

    Subclasses must implement:
    - name: Human-readable name of the pass
    - transform(tree, context): Apply the transformation
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this transformation pass."""
        pass

    @abstractmethod
    def transform(
        self, tree: ast.Module, context: TransformContext
    ) -> tuple[ast.Module, bool]:
        """Apply this transformation to the AST.

        Args:
            tree: The AST to transform
            context: Shared context with function info and state

        Returns:
            (transformed_tree, was_modified)
        """
        pass


class ASTTransformPipeline:
    """Pipeline that runs AST transformation passes in sequence."""

    def __init__(self, passes: list[ASTTransformPass] = None):
        self._passes = passes or []

    def add_pass(self, transform_pass: ASTTransformPass) -> "ASTTransformPipeline":
        """Add a pass to the pipeline."""
        self._passes.append(transform_pass)
        return self

    def run(
        self,
        tree: ast.Module,
        context: TransformContext,
        dump_after_all: bool = False,
    ) -> tuple[ast.Module, bool]:
        """Run all passes on the AST.

        Args:
            tree: The AST to transform
            context: Shared context with function info and state
            dump_after_all: If True, print source before and after each pass

        Returns:
            (transformed_tree, any_modified)
        """
        any_modified = False
        func_name = context.func.__name__

        if dump_after_all:
            print(f"# Unparsed source before transforms")
            print(ast.unparse(tree))

        for transform_pass in self._passes:
            tree, modified = transform_pass.transform(tree, context)
            any_modified |= modified

            if dump_after_all:
                print(f"# Unparsed source after {transform_pass.name}")
                print(ast.unparse(tree))

        return tree, any_modified
