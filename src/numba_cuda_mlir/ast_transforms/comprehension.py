# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# local_array_from transformation pass
# Converts local_array_from(genexp, dtype) to local_array + loop
import ast

from numba_cuda_mlir.ast_transforms.pipeline import ASTTransformPass, TransformContext


class LocalArrayFromTransformer(ast.NodeTransformer):
    """AST transformer that converts local_array_from calls to local_array + loop.

    Transforms:
        arr = cuda.local_array_from((expr for var in iterable), dtype=np.float32)

    Into:
        arr = cuda.local_array(len(iterable), dtype=np.float32)
        for __i, var in enumerate(iterable):
            arr[__i] = expr
    """

    def __init__(self):
        self.modified = False
        self._counter = 0

    def _make_temp_name(self) -> str:
        name = f"__laf_i_{self._counter}"
        self._counter += 1
        return name

    def _is_local_array_from(self, node: ast.expr) -> bool:
        """Check if node is a call to local_array_from."""
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        # Match: local_array_from(...) or X.local_array_from(...)
        if isinstance(func, ast.Name) and func.id == "local_array_from":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "local_array_from":
            return True
        return False

    def _get_local_array_func(self, original_func: ast.expr) -> ast.expr:
        """Convert local_array_from func reference to local_array."""
        if isinstance(original_func, ast.Name):
            return ast.Name(id="local_array", ctx=ast.Load())
        elif isinstance(original_func, ast.Attribute):
            # cuda.local_array_from -> cuda.local_array
            return ast.Attribute(
                value=original_func.value,
                attr="local_array",
                ctx=ast.Load(),
            )
        return original_func

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
        node = self.generic_visit(node)

        if len(node.targets) != 1 or not self._is_local_array_from(node.value):
            return node

        call = node.value
        if not call.args or not isinstance(call.args[0], ast.GeneratorExp):
            return node

        genexp = call.args[0]
        if len(genexp.generators) != 1:
            # Only handle single-loop generators for now
            return node

        gen = genexp.generators[0]
        if gen.ifs:
            # Don't handle conditional generators for now
            return node

        self.modified = True
        arr_name = node.targets[0]
        if not isinstance(arr_name, ast.Name):
            return node
        arr_name_str = arr_name.id
        idx_name = self._make_temp_name()

        # Build: arr = cuda.local_array(len(iterable), dtype=dtype)
        len_call = ast.Call(
            func=ast.Name(id="len", ctx=ast.Load()),
            args=[gen.iter],
            keywords=[],
        )
        local_array_call = ast.Call(
            func=self._get_local_array_func(call.func),
            args=[len_call],
            keywords=call.keywords,  # Pass through dtype and other kwargs
        )
        arr_assign = ast.Assign(
            targets=[ast.Name(id=arr_name_str, ctx=ast.Store())],
            value=local_array_call,
        )

        # Build: for idx in range(len(iterable)):
        #            var = iterable[idx]
        #            arr[idx] = expr
        var_assign = ast.Assign(
            targets=[gen.target],
            value=ast.Subscript(
                value=gen.iter,
                slice=ast.Name(id=idx_name, ctx=ast.Load()),
                ctx=ast.Load(),
            ),
        )
        arr_store = ast.Assign(
            targets=[
                ast.Subscript(
                    value=ast.Name(id=arr_name_str, ctx=ast.Load()),
                    slice=ast.Name(id=idx_name, ctx=ast.Load()),
                    ctx=ast.Store(),
                )
            ],
            value=genexp.elt,
        )
        for_loop = ast.For(
            target=ast.Name(id=idx_name, ctx=ast.Store()),
            iter=ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[len_call],
                keywords=[],
            ),
            body=[var_assign, arr_store],
            orelse=[],
        )

        return [arr_assign, for_loop]


class StatementExpander(ast.NodeTransformer):
    """Expand statement lists that contain lists from transforms."""

    def _expand_body(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        result = []
        for stmt in stmts:
            visited = self.visit(stmt)
            if isinstance(visited, list):
                result.extend(visited)
            else:
                result.append(visited)
        return result

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.body = self._expand_body(node.body)
        return node

    def visit_If(self, node: ast.If) -> ast.If:
        node.body = self._expand_body(node.body)
        node.orelse = self._expand_body(node.orelse)
        return node

    def visit_For(self, node: ast.For) -> ast.For:
        node.body = self._expand_body(node.body)
        node.orelse = self._expand_body(node.orelse)
        return node

    def visit_While(self, node: ast.While) -> ast.While:
        node.body = self._expand_body(node.body)
        node.orelse = self._expand_body(node.orelse)
        return node


def transform_comprehensions(tree: ast.Module) -> tuple[ast.Module, bool]:
    """Transform local_array_from calls to local_array + loop.

    Returns (transformed_tree, was_modified).
    """
    transformer = LocalArrayFromTransformer()
    new_tree = transformer.visit(tree)
    if transformer.modified:
        expander = StatementExpander()
        new_tree = expander.visit(new_tree)
    ast.fix_missing_locations(new_tree)
    return new_tree, transformer.modified


class ComprehensionPass(ASTTransformPass):
    """Pipeline pass that transforms local_array_from calls to local_array + loop."""

    @property
    def name(self) -> str:
        return "LocalArrayFrom"

    def transform(self, tree: ast.Module, context: TransformContext) -> tuple[ast.Module, bool]:
        return transform_comprehensions(tree)
