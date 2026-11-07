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


_UNSET = object()

# Nodes that evaluate code with side effects or new scopes; an operand containing one
# is not evaluated a second time just to check it.
_NOT_REEVALUATED = (
    ast.Call,
    ast.Lambda,
    ast.NamedExpr,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
)


def _is_numba_type(value) -> bool:
    from numba_cuda_mlir.numba_cuda import types

    return isinstance(value, types.Type)


def _is_type_like(value) -> bool:
    """A Numba type, or a container whose members (dict keys) are all Numba types."""
    if _is_numba_type(value):
        return True
    if isinstance(value, (tuple, list, set, frozenset)):
        return all(_is_type_like(v) for v in value)
    if isinstance(value, dict):
        return all(_is_type_like(k) for k in value)
    return False


class _TypeAsValueChecker(ast.NodeVisitor):
    """Reject a consteval expression that uses a Numba type as if it were a runtime value.

    Inside ``consteval``, a parameter name evaluates to the parameter's Numba type. A
    comparison such as ``x == 0`` then compares a type with an int: Python returns
    ``False`` instead of raising, and the fold silently picks the wrong branch. The same
    holds for a type's truth value. A comparison whose other operand is a type, such as
    ``x == types.int32`` or ``x in (types.int32, types.int64)``, is a type check and is
    left alone.
    """

    def __init__(self, context: dict, source: str):
        self.context = context
        self.source = source
        self.shadowed: list[set[str]] = []

    def check(self, node: ast.expr, truth_test: bool) -> None:
        if truth_test:
            self._check_truth(node)
        self.visit(node)

    def _is_shadowed(self, node: ast.expr) -> bool:
        return any(
            isinstance(n, ast.Name) and any(n.id in names for names in self.shadowed)
            for n in ast.walk(node)
        )

    def _value(self, node: ast.expr):
        """The compile-time value of ``node``, or ``_UNSET`` if it is not safe to evaluate."""
        if self._is_shadowed(node) or any(isinstance(n, _NOT_REEVALUATED) for n in ast.walk(node)):
            return _UNSET
        try:
            return eval(ast.unparse(node), self.context)
        except Exception:
            return _UNSET  # the evaluation proper reports the error

    def _type_of(self, node: ast.expr):
        """The Numba type ``node`` names, for a name or attribute chain such as ``arr.dtype``."""
        root = node
        while isinstance(root, ast.Attribute):
            root = root.value
        if not isinstance(root, ast.Name):
            return None
        value = self._value(node)
        return value if _is_numba_type(value) else None

    def _error(self, node: ast.expr, numba_type, problem: str) -> ConstevalError:
        name = ast.unparse(node)
        return ConstevalError(
            f"Cannot evaluate consteval argument '{self.source}' at compile time: inside "
            f"consteval, '{name}' is the Numba type {numba_type}, not a runtime value, so "
            f"{problem}. Compare it with a Numba type (for example '{name} == types.int32'), "
            "or use a regular 'if' to test the runtime value."
        )

    def _check_truth(self, node: ast.expr) -> None:
        numba_type = self._type_of(node)
        if numba_type is not None:
            raise self._error(
                node, numba_type, "its truth value says nothing about the runtime value"
            )

    def visit_Compare(self, node: ast.Compare) -> None:
        operands = [node.left, *node.comparators]
        for op, left, right in zip(node.ops, operands, operands[1:]):
            for typed, other in ((left, right), (right, left)):
                numba_type = self._type_of(typed)
                if numba_type is None:
                    continue
                other_value = self._value(other)
                if other_value is _UNSET or _is_type_like(other_value):
                    continue
                pair = ast.Compare(left=left, ops=[op], comparators=[right])
                outcome = self._value(pair)
                problem = f"'{ast.unparse(pair)}' compares a type with {other_value!r}"
                if isinstance(outcome, bool):
                    problem += f" and is always {outcome}"
                raise self._error(typed, numba_type, problem)
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if isinstance(node.op, ast.Not):
            self._check_truth(node.operand)
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        for value in node.values:
            self._check_truth(value)
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._check_truth(node.test)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if (
            isinstance(func, ast.Name)
            and func.id == "bool"
            and self.context.get("bool", bool) is bool
            and len(node.args) == 1
            and not node.keywords
        ):
            self._check_truth(node.args[0])
        self.generic_visit(node)

    def _visit_scope(self, node: ast.AST, names: set[str]) -> None:
        self.shadowed.append(names)
        self.generic_visit(node)
        self.shadowed.pop()

    def _visit_comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> None:
        names = {
            n.id
            for generator in node.generators
            for n in ast.walk(generator.target)
            if isinstance(n, ast.Name)
        }
        self._visit_scope(node, names)

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _visit_comprehension

    def visit_Lambda(self, node: ast.Lambda) -> None:
        args = node.args
        params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        params += [a for a in (args.vararg, args.kwarg) if a is not None]
        self._visit_scope(node, {a.arg for a in params})


class VariableReplacer(ast.NodeTransformer):
    """Replace variable names with expressions throughout an AST."""

    def __init__(self, replacements: dict[str, ast.expr]):
        self.replacements = replacements

    def visit_Name(self, node: ast.Name) -> ast.AST:
        replacement = self.replacements.get(node.id)
        if replacement is None:
            return node
        return ast.copy_location(copy.deepcopy(replacement), node)


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
        # consteval arguments in an ``if``/``while`` test, whose truth value picks the branch
        self._truth_tests: list[ast.expr] = []

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
        ctx.update(self.stored_values)
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
        truth_test = any(node is test for test in self._truth_tests)
        # Pre-process: replace current_target_options() calls with context reference
        node = TargetOptionsReplacer().visit(copy.deepcopy(node))
        ast.fix_missing_locations(node)
        expr_source = ast.unparse(node)
        context = self.context
        _TypeAsValueChecker(context, expr_source).check(node, truth_test)
        try:
            return eval(expr_source, context)
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

        value = self._eval_expr(node.args[0])
        self.modified = True
        return ast.copy_location(self._value_expr(value), node)

    def _value_expr(self, value) -> ast.expr:
        """An expression yielding ``value``: a constant, or a reference to a stored value."""
        if isinstance(value, ast.expr):
            return value
        if self._can_be_constant(value):
            return ast.Constant(value=value)
        return ast.Name(id=self._store_value(value), ctx=ast.Load())

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
        self._check_unroll_loop_control(node)
        self._check_unroll_loop_variables(node)

        # Evaluate the iterator. A parameter name evaluates to its Numba type,
        # so it yields element accesses rather than compile-time values.
        items = self._parameter_element_exprs(node.iter.args[0])
        if items is None:
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
        bindings = {}
        for value in items:
            bindings = self._bind_loop_target(node.target, value)
            unrolled.extend(self._unroll_statements(node.body, bindings))

        # The loop variables keep their final values after the loop, and
        # ``break`` is rejected above, so the ``else`` clause always runs.
        self.local_consts.update(bindings)
        for name, value in bindings.items():
            target = ast.Name(id=name, ctx=ast.Store())
            assign = ast.Assign(targets=[target], value=self._value_expr(value))
            unrolled.append(ast.fix_missing_locations(ast.copy_location(assign, node)))
        unrolled.extend(self._process_statement_list(node.orelse))
        return unrolled

    def _unroll_statements(self, stmts: list[ast.stmt], bindings: dict) -> list[ast.stmt]:
        """Copy ``stmts`` with the loop variables in ``bindings`` substituted."""
        replacer = VariableReplacer({name: self._value_expr(v) for name, v in bindings.items()})
        copies = [ast.fix_missing_locations(replacer.visit(copy.deepcopy(s))) for s in stmts]

        # Make the loop variables visible to nested constevals for these statements only
        saved_consts = self.local_consts.copy()
        self.local_consts.update(bindings)
        result = self._process_statement_list(copies)
        self.local_consts = saved_consts
        return result

    def _parameter_element_exprs(self, iter_arg: ast.expr) -> list[ast.expr] | None:
        """Element accesses for ``consteval(<tuple parameter>)``, else ``None``.

        Evaluating a parameter name yields the parameter's Numba type, whose
        members are *types*. Substituting those into the body would silently
        replace each element with its type, so a tuple parameter is unrolled
        into ``p[0]``, ``p[1]``, ... instead, leaving the values at runtime.
        """
        if not isinstance(iter_arg, ast.Name):
            return None
        # ``context`` lets param types shadow other bindings, so match that here.
        param_type = self.param_type_map.get(iter_arg.id)
        if param_type is None:
            return None

        from numba_cuda_mlir.numba_cuda import types

        if not isinstance(param_type, types.BaseTuple):
            raise ConstevalError(
                f"Cannot unroll over parameter '{iter_arg.id}' of type {param_type}: "
                "only a tuple parameter has a compile-time length"
            )
        return [
            ast.Subscript(
                value=ast.Name(id=iter_arg.id, ctx=ast.Load()),
                slice=ast.Constant(value=index),
                ctx=ast.Load(),
            )
            for index in range(len(param_type))
        ]

    def _check_unroll_loop_control(self, node: ast.For) -> None:
        """Reject loop control statements that would escape an unrolled loop."""

        class LoopControlFinder(ast.NodeVisitor):
            control = None

            def visit_Break(self, node: ast.Break) -> None:
                self.control = "break"

            def visit_Continue(self, node: ast.Continue) -> None:
                self.control = "continue"

            def visit_For(self, node: ast.For) -> None:
                for stmt in node.orelse:
                    self.visit(stmt)

            def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
                for stmt in node.orelse:
                    self.visit(stmt)

            def visit_While(self, node: ast.While) -> None:
                for stmt in node.orelse:
                    self.visit(stmt)

        finder = LoopControlFinder()
        for stmt in node.body:
            finder.visit(stmt)
        if finder.control:
            raise ConstevalError(f"Loop unrolling does not support {finder.control} statements")

    def _check_unroll_loop_variables(self, node: ast.For) -> None:
        """Reject bodies that rebind a loop variable of an unrolled loop.

        Unrolling substitutes the bound value for every use of the variable,
        which has no meaning for an assignment target: ``v = 0`` would become
        ``3 = 0`` or ``t[0] = 0``.
        """
        loop_vars = {name.id for name in ast.walk(node.target) if isinstance(name, ast.Name)}
        for stmt in node.body:
            for name in ast.walk(stmt):
                if (
                    isinstance(name, ast.Name)
                    and name.id in loop_vars
                    and isinstance(name.ctx, (ast.Store, ast.Del))
                ):
                    raise ConstevalError(
                        f"Loop unrolling does not support rebinding loop variable '{name.id}'"
                    )

    def _bind_loop_target(self, target: ast.expr, value) -> dict[str, object]:
        """Bind a consteval loop target to a compile-time value."""
        if isinstance(target, ast.Name):
            return {target.id: value}
        if isinstance(target, ast.Starred):
            raise ConstevalError("Loop unrolling does not support starred targets")
        if not isinstance(target, (ast.Tuple, ast.List)):
            raise ConstevalError(
                "Loop unrolling only supports name, tuple, or list targets, "
                f"got {type(target).__name__}"
            )

        try:
            values = list(value)
        except TypeError as e:
            raise ConstevalError(
                f"Cannot unpack {type(value).__name__} into {ast.unparse(target)}"
            ) from e

        if len(values) != len(target.elts):
            direction = "not enough" if len(values) < len(target.elts) else "too many"
            raise ConstevalError(
                f"{direction} values to unpack (expected {len(target.elts)}, got {len(values)})"
            )

        bindings = {}
        for element, element_value in zip(target.elts, values):
            bindings.update(self._bind_loop_target(element, element_value))
        return bindings

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

    def _visit_test(self, test: ast.expr) -> ast.expr:
        """Visit an ``if``/``while`` test, marking a ``consteval(...)`` test as a truth test."""
        if self._is_consteval_call(test) and len(test.args) == 1:
            self._truth_tests.append(test.args[0])
            try:
                return self.visit(test)
            finally:
                self._truth_tests.pop()
        return self.visit(test)

    def visit_If(self, node: ast.If) -> ast.If:
        """Process if statement bodies in order."""
        node.test = self._visit_test(node.test)
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
        node.test = self._visit_test(node.test)
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
