# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Numba IR transformation passes for cusimt
from cusimt.numba_cuda.core.compiler_lock import global_compiler_lock
from cusimt.numba_cuda.core.untyped_passes import (
    FunctionPass,
    register_pass,
    TransformLiteralUnrollConstListToTuple,
    IterLoopCanonicalization,
    RewriteSemanticConstants,
    MixedContainerUnroller,
    GenericRewrites,
    InlineInlinables,
)
from cusimt.numba_cuda.core import untyped_passes as untyped_passes_module
from cusimt.numba_cuda.core.typed_passes import PartialTypeInference
from cusimt.numba_cuda.core import ir
from cusimt.numba_cuda.misc.special import literal_unroll


@register_pass(mutates_CFG=True, analysis_only=False)
class CusimtLiteralUnroll(FunctionPass):
    """
    Implement the literal_unroll semantics.
    This is a cusimt-specific version that accepts both numba.misc.special.literal_unroll
    and numba.cuda.misc.special.literal_unroll.
    """

    _name = "cusimt_literal_unroll"

    def __init__(self):
        FunctionPass.__init__(self)

    def _is_literal_unroll(self, value):
        """Check if value is either version of literal_unroll."""
        if value is literal_unroll:
            return True
        return False

    def run_pass(self, state):
        # Determine whether to even attempt this pass... if there's no
        # `literal_unroll` as a global or as a freevar then just skip.
        found = False
        func_ir = state.func_ir
        for blk in func_ir.blocks.values():
            for asgn in blk.find_insts(ir.Assign):
                if isinstance(asgn.value, (ir.Global, ir.FreeVar)):
                    if self._is_literal_unroll(asgn.value.value):
                        found = True
                        break
            if found:
                break
        if not found:
            return False

        # run as subpipeline
        from cusimt.numba_cuda.core.compiler_machinery import PassManager

        pm = PassManager("literal_unroll_subpipeline")
        # get types where possible to help with list->tuple change
        pm.add_pass(PartialTypeInference, "performs partial type inference")
        # make const lists tuples
        pm.add_pass(
            TransformLiteralUnrollConstListToTuple, "switch const list for tuples"
        )
        # recompute partial typemap following IR change
        pm.add_pass(PartialTypeInference, "performs partial type inference")
        # canonicalise loops - use our patched version
        pm.add_pass(
            CusimtIterLoopCanonicalization, "switch iter loops for range driven loops"
        )
        # rewrite consts
        pm.add_pass(RewriteSemanticConstants, "rewrite semantic constants")
        # do the unroll - we patched the module-level literal_unroll above,
        # so the standard MixedContainerUnroller will work
        pm.add_pass(MixedContainerUnroller, "performs mixed container unroll")
        # rewrite dynamic getitem to static getitem as it's possible some more
        # getitems will now be statically resolvable
        pm.add_pass(GenericRewrites, "Generic Rewrites")
        pm.add_pass(RewriteSemanticConstants, "rewrite semantic constants")
        pm.finalize()
        pm.run(state)
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class CusimtIterLoopCanonicalization(IterLoopCanonicalization):
    """
    cusimt-specific version of IterLoopCanonicalization that accepts both
    numba.misc.special.literal_unroll and numba.cuda.misc.special.literal_unroll.
    """

    _name = "cusimt_iter_loop_canonicalisation"

    _accepted_calls = (literal_unroll,)


@register_pass(mutates_CFG=True, analysis_only=False)
class CusimtInlineInlinables(InlineInlinables):
    """InlineInlinables that skips self-recursive functions to avoid infinite inlining."""

    _name = "cusimt_inline_inlinables"

    def _do_work(self, state, work_list, block, i, expr, inline_worker):
        try:
            to_inline = state.func_ir.get_definition(expr.func)
            val = getattr(to_inline, "value", None)
            if val and hasattr(val, "py_func") and self._is_self_recursive(val.py_func):
                return False
        except Exception:
            pass
        return super()._do_work(state, work_list, block, i, expr, inline_worker)

    @staticmethod
    def _is_self_recursive(pyfunc):
        """Check if a function's bytecode references its own name."""
        import dis

        for instr in dis.get_instructions(pyfunc):
            if (
                instr.opname in ("LOAD_GLOBAL", "LOAD_DEREF")
                and instr.argval == pyfunc.__name__
            ):
                return True
        return False
