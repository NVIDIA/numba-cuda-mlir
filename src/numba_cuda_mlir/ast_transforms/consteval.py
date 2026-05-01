# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Consteval AST transformation pass
import ast
import copy
import inspect
from typing import Callable

from numba_cuda_mlir.ast_transforms.common import get_function_context
from numba_cuda_mlir.ast_transforms.pipeline import ASTTransformPass, TransformContext


class ConstevalError(Exception):
    """Raised when a consteval expression cannot be evaluated at compile time."""

    pass


class TargetOptionsReplacer(ast.NodeTransformer):
    """Replace numba_cuda_mlir.current_target_options() calls with a context reference."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        # Match: numba_cuda_mlir.current_target_options() or current_target_options()
        if self._is_current_target_options_call(node):
            return ast.copy_location(
                ast.Name(id="__numba_cuda_mlir_target_options__", ctx=ast.Load()), node
            )
        return node

    def _is_current_target_options_call(self, node: ast.Call) -> bool:
        """Check if this is a call to current_target_options()."""
        if node.args or node.keywords:
            return False
        func = node.func
        # current_target_options()
        if isinstance(func, ast.Name) and func.id == "current_target_options":
            return True
        # numba_cuda_mlir.current_target_options() or cuda.current_target_options()
        if isinstance(func, ast.Attribute) and func.attr == "current_target_options":
            return True
        return False


class VariableReplacer(ast.NodeTransformer):
    """Replace a variable name with a constant value throughout an AST."""

    def __init__(self, var_name: str, value):
        self.var_name = var_name
        self.value = value

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.var_name:
            return ast.copy_location(ast.Constant(value=self.value), node)
        return node


class ConstevalTransformer(ast.NodeTransformer):
    """AST transformer that evaluates consteval/literally calls at compile time.

    This transformer:
    1. Tracks local constants from `name = consteval(...)` assignments
    2. Uses those constants when evaluating subsequent consteval expressions
    3. Supports chained constevals like: `a = consteval(X); if consteval(a + 5): ...`
    4. Handles complex objects by storing them in globals and referencing by name
    5. Provides access to argument types (Numba types) via parameter names
    6. Provides access to target options via numba_cuda_mlir.current_target_options()
    7. Supports `with consteval():` blocks for multi-statement compile-time execution
    """

    CONSTEVAL_NAMES = {"consteval", "literally"}
    # Types that can be safely stored in ast.Constant
    CONSTANT_TYPES = (int, float, str, bool, type(None), bytes, tuple)

    def __init__(
        self,
        func: Callable,
        targetoptions: dict = None,
        argtypes: tuple = None,
    ):
        self.func = func
        self.base_context = get_function_context(func)
        self.local_consts = {}  # Track values from consteval assignments
        self.stored_values = {}  # Complex values stored in globals
        self.modified = False
        self._value_counter = 0
        self.targetoptions = targetoptions or {}
        self.argtypes = argtypes or ()
        self.param_type_map = self._build_param_type_map(func, self.argtypes)

    def _build_param_type_map(self, func: Callable, argtypes: tuple) -> dict:
        """Build a mapping from parameter names to their Numba types."""
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        result = {}
        for i, name in enumerate(params):
            if i < len(argtypes):
                result[name] = argtypes[i]
        return result

    @property
    def context(self):
        """Combined context: base (globals + closure) plus local constants and param types."""
        ctx = self.base_context.copy()
        ctx.update(self.local_consts)
        # Add parameter types - these shadow any globals with the same name
        ctx.update(self.param_type_map)
        # Add a resolver for current_target_options()
        ctx["__numba_cuda_mlir_target_options__"] = self.targetoptions
        return ctx

    def _is_consteval_call(self, node: ast.expr) -> bool:
        """Check if a node is a call to consteval or literally."""
        if not isinstance(node, ast.Call):
            return False
        if isinstance(node.func, ast.Name):
            return node.func.id in self.CONSTEVAL_NAMES
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr in self.CONSTEVAL_NAMES
        return False

    def _eval_expr(self, node: ast.expr) -> any:
        """Evaluate an AST expression node using the current context."""
        # Pre-process: replace current_target_options() calls with context reference
        node = TargetOptionsReplacer().visit(copy.deepcopy(node))
        ast.fix_missing_locations(node)
        expr_source = ast.unparse(node)
        try:
            return eval(expr_source, self.context)
        except Exception as e:
            raise ConstevalError(
                f"Cannot evaluate consteval argument '{expr_source}' at compile time: {e}"
            ) from e

    def _can_be_constant(self, value) -> bool:
        """Check if a value can be stored in an ast.Constant node."""
        if isinstance(value, self.CONSTANT_TYPES):
            # For tuples, recursively check all elements
            if isinstance(value, tuple):
                return all(self._can_be_constant(v) for v in value)
            return True
        return False

    def _store_value(self, value) -> str:
        """Store a complex value and return its reference name."""
        name = f"__consteval_{self._value_counter}__"
        self._value_counter += 1
        self.stored_values[name] = value
        return name

    def _transform_consteval(self, node: ast.Call) -> ast.AST:
        """Transform a consteval call to a Constant or Name node."""
        if len(node.args) != 1 or node.keywords:
            raise ConstevalError("consteval expects exactly one positional argument")

        arg = node.args[0]
        value = self._eval_expr(arg)
        self.modified = True

        if self._can_be_constant(value):
            return ast.copy_location(ast.Constant(value=value), node)
        else:
            # Store complex value and reference it by name
            name = self._store_value(value)
            return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)

    def _process_statement_list(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        """Process a list of statements in order, tracking consteval assignments."""
        result = []
        for stmt in stmts:
            transformed = self._transform_statement(stmt)
            if isinstance(transformed, list):
                result.extend(transformed)
            else:
                result.append(transformed)
        return result

    def _get_consteval_value(self, node: ast.AST):
        """Get the actual value from a transformed consteval node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name) and node.id in self.stored_values:
            return self.stored_values[node.id]
        return None

    def _transform_statement(self, stmt: ast.stmt) -> ast.stmt | list[ast.stmt]:
        """Transform a single statement, handling consteval assignments specially."""
        # Check for simple assignment: name = consteval(...)
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name) and self._is_consteval_call(stmt.value):
                # Evaluate and track this consteval assignment
                transformed = self._transform_consteval(stmt.value)
                value = self._get_consteval_value(transformed)
                if value is not None:
                    self.local_consts[target.id] = value
                stmt.value = transformed
                return stmt

        # Check for annotated assignment: name: type = consteval(...)
        if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            if isinstance(stmt.target, ast.Name) and self._is_consteval_call(stmt.value):
                transformed = self._transform_consteval(stmt.value)
                value = self._get_consteval_value(transformed)
                if value is not None:
                    self.local_consts[stmt.target.id] = value
                stmt.value = transformed
                return stmt

        # Check for loop unrolling: for <var> in consteval(...):
        if isinstance(stmt, ast.For) and self._is_consteval_call(stmt.iter):
            return self._unroll_for_loop(stmt)

        # Check for consteval block: with consteval():
        if isinstance(stmt, ast.With) and self._is_consteval_with(stmt):
            return self._transform_consteval_block(stmt)

        # For other statements, use generic visiting
        return self.visit(stmt)

    def _unroll_for_loop(self, node: ast.For) -> list[ast.stmt]:
        """Unroll a for loop with consteval iterator."""
        # Get the loop variable name
        if not isinstance(node.target, ast.Name):
            raise ConstevalError(
                "Loop unrolling only supports simple variable targets, "
                f"got {type(node.target).__name__}"
            )
        var_name = node.target.id

        # Evaluate the iterator
        iter_value = self._eval_expr(node.iter.args[0])
        try:
            items = list(iter_value)
        except TypeError as e:
            raise ConstevalError(
                f"consteval iterator must be iterable, got {type(iter_value).__name__}"
            ) from e

        self.modified = True

        # Unroll: for each value, copy the body and replace the variable
        unrolled = []
        for value in items:
            # Save current local_consts state
            saved_consts = self.local_consts.copy()
            # Add loop variable to local_consts for this iteration
            self.local_consts[var_name] = value

            for body_stmt in node.body:
                # Deep copy the statement
                stmt_copy = copy.deepcopy(body_stmt)
                # Replace the loop variable with the constant value
                replacer = VariableReplacer(var_name, value)
                stmt_copy = replacer.visit(stmt_copy)
                ast.fix_missing_locations(stmt_copy)
                # Process the statement (handles nested constevals)
                transformed = self._transform_statement(stmt_copy)
                if isinstance(transformed, list):
                    unrolled.extend(transformed)
                else:
                    unrolled.append(transformed)

            # Restore local_consts state
            self.local_consts = saved_consts

        # Note: we ignore the else clause (orelse) since unrolled loops
        # don't have a natural "else" semantic
        return unrolled

    def _is_consteval_with(self, node: ast.With) -> bool:
        """Check if this is a 'with consteval():' block."""
        if len(node.items) != 1:
            return False
        item = node.items[0]
        return self._is_consteval_call(item.context_expr)

    def _transform_consteval_block(self, node: ast.With) -> list[ast.stmt]:
        """Transform a 'with consteval():' block by executing it at compile time."""
        item = node.items[0]

        # Error if 'as var:' syntax is used
        if item.optional_vars is not None:
            var_name = ast.unparse(item.optional_vars)
            raise ConstevalError(
                f"'with consteval() as {var_name}:' is not supported.\n"
                "Use 'with consteval():' and extract values with consteval(var) after the block."
            )

        self.modified = True

        # Build execution context
        exec_globals = self.context.copy()
        exec_locals = {}

        # Execute each statement in the block
        for stmt in node.body:
            # Handle nested consteval blocks recursively
            if isinstance(stmt, ast.With) and self._is_consteval_with(stmt):
                # Recursively process nested block - this updates local_consts
                saved_consts = self.local_consts.copy()
                self.local_consts = dict(exec_locals)  # Use current exec state
                self.local_consts.update(exec_globals)
                self._transform_consteval_block(stmt)
                # Capture any new variables from the nested block
                for k, v in self.local_consts.items():
                    if k not in exec_globals:
                        exec_locals[k] = v
                        exec_globals[k] = v
                self.local_consts = saved_consts
                continue

            # Pre-process: replace current_target_options() calls
            stmt = TargetOptionsReplacer().visit(copy.deepcopy(stmt))
            ast.fix_missing_locations(stmt)
            stmt_source = ast.unparse(stmt)
            try:
                exec(stmt_source, exec_globals, exec_locals)
                # Update exec_globals with any new locals so subsequent statements can use them
                exec_globals.update(exec_locals)
            except Exception as e:
                raise ConstevalError(
                    f"Error executing consteval block statement:\n  {stmt_source}\nError: {e}"
                ) from e

        # Track all variables defined in the block for use in subsequent consteval() calls
        self.local_consts.update(exec_locals)

        # Return empty list - the block is removed from output
        return []

    def visit_With(self, node: ast.With) -> ast.AST | list[ast.stmt]:
        """Handle with statements, transforming consteval blocks."""
        if self._is_consteval_with(node):
            return self._transform_consteval_block(node)
        # Normal with statement - process body
        node.body = self._process_statement_list(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Process function body statements in order."""
        node.body = self._process_statement_list(node.body)
        # Don't process decorators - they're not part of the function body
        return node

    def visit_If(self, node: ast.If) -> ast.If:
        """Process if statement bodies in order."""
        node.test = self.visit(node.test)
        node.body = self._process_statement_list(node.body)
        node.orelse = self._process_statement_list(node.orelse)
        return node

    def visit_For(self, node: ast.For) -> ast.For:
        """Process for loop body in order."""
        node.iter = self.visit(node.iter)
        node.body = self._process_statement_list(node.body)
        node.orelse = self._process_statement_list(node.orelse)
        return node

    def visit_While(self, node: ast.While) -> ast.While:
        """Process while loop body in order."""
        node.test = self.visit(node.test)
        node.body = self._process_statement_list(node.body)
        node.orelse = self._process_statement_list(node.orelse)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Transform consteval calls to constants."""
        # First visit children (for nested calls)
        node = self.generic_visit(node)

        if not self._is_consteval_call(node):
            return node

        return self._transform_consteval(node)


def transform_consteval(
    func: Callable,
    tree: ast.Module,
    targetoptions: dict = None,
    argtypes: tuple = None,
) -> tuple[ast.Module, bool, dict]:
    """Transform consteval calls in the AST to literal constants.

    Args:
        func: The function being transformed
        tree: The AST to transform
        targetoptions: Target options dict from @jit decorator
        argtypes: Tuple of Numba types for the function arguments

    Returns (transformed_tree, was_modified, stored_values).
    stored_values is a dict of name -> value for complex objects that couldn't
    be stored as AST constants. These should be injected into the function's globals.
    """
    transformer = ConstevalTransformer(func, targetoptions, argtypes)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    return new_tree, transformer.modified, transformer.stored_values


class ConstevalPass(ASTTransformPass):
    """Pipeline pass that evaluates consteval/literally calls at compile time."""

    @property
    def name(self) -> str:
        return "Consteval"

    def transform(self, tree: ast.Module, context: TransformContext) -> tuple[ast.Module, bool]:
        tree, modified, stored_values = transform_consteval(
            context.func, tree, context.targetoptions, context.argtypes
        )
        context.stored_values.update(stored_values)
        return tree, modified
