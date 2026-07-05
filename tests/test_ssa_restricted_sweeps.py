# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression pins for SSA reconstruction with restricted sweeps.

The SSA rewrite passes visit only blocks that define or use the
variable being processed; the test compares them against unrestricted
sweeps (``relevant_labels=None``), which rewrite every block exactly
as the pre-restriction implementation did.

The hand-built IR includes the shape that makes the restriction
subtle: an assignment with the same variable on both sides
(``x = x + x``). ``_GatherDefsHandler`` drops every mention of a
variable in a statement that assigns it, so such a block appears in
the def map but not the use map — the use-fixup pass must visit the
union of use and def blocks or the self-referential operand silently
keeps the unversioned name.
"""

import operator

import numpy as np
from numba_cuda_mlir import cuda
from numba_cuda_mlir.numba_cuda.core import ir
from numba_cuda_mlir.numba_cuda.core import ssa as ssa_mod


def _build_self_use_blocks():
    """CFG with an SSA violator whose only non-terminal mention in
    the loop block is ``x = x + x``:

        0:  x = 0.0; cond = True; jump 8
        8:  x = x + x; branch cond ? 8 : 16
        16: return x
    """
    loc = ir.Loc("test_ssa", 1)
    scope = ir.Scope(parent=None, loc=loc)
    x = scope.define("x", loc)
    cond = scope.define("cond", loc)

    b0 = ir.Block(scope, loc)
    b0.body = [
        ir.Assign(ir.Const(0.0, loc), x, loc),
        ir.Assign(ir.Const(True, loc), cond, loc),
        ir.Jump(8, loc),
    ]
    b8 = ir.Block(scope, loc)
    b8.body = [
        ir.Assign(
            ir.Expr.binop(operator.add, x, x, loc),
            x,
            loc,
        ),
        ir.Branch(cond, 8, 16, loc),
    ]
    b16 = ir.Block(scope, loc)
    b16.body = [ir.Return(x, loc)]
    return {0: b0, 8: b8, 16: b16}


def _build_diamond_blocks():
    """Diamond with a pass-through join: x is assigned in both arms
    and used two blocks later, so the phi for x lands in a block that
    neither defines nor uses it.

        0:  cond = True; branch cond ? 8 : 16
        8:  x = 1.0; jump 24
        16: x = 2.0; jump 24
        24: jump 32                 (pass-through phi location)
        32: return x
    """
    loc = ir.Loc("test_ssa", 2)
    scope = ir.Scope(parent=None, loc=loc)
    x = scope.define("x", loc)
    cond = scope.define("cond", loc)

    b0 = ir.Block(scope, loc)
    b0.body = [
        ir.Assign(ir.Const(True, loc), cond, loc),
        ir.Branch(cond, 8, 16, loc),
    ]
    b8 = ir.Block(scope, loc)
    b8.body = [
        ir.Assign(ir.Const(1.0, loc), x, loc),
        ir.Jump(24, loc),
    ]
    b16 = ir.Block(scope, loc)
    b16.body = [
        ir.Assign(ir.Const(2.0, loc), x, loc),
        ir.Jump(24, loc),
    ]
    b24 = ir.Block(scope, loc)
    b24.body = [ir.Jump(32, loc)]
    b32 = ir.Block(scope, loc)
    b32.body = [ir.Return(x, loc)]
    return {0: b0, 8: b8, 16: b16, 24: b24, 32: b32}


def _reference_run_ssa(blocks):
    """The production driver with the block restriction disabled:
    every rewrite pass visits every block, as the pre-restriction
    implementation did."""
    if not blocks:
        return {}
    cfg = ssa_mod.compute_cfg_from_blocks(blocks)
    df_plus = ssa_mod._iterated_domfronts(cfg)
    violators, defs, uses = ssa_mod._find_defs_violators(blocks, cfg)
    cache_list_vars = ssa_mod._CacheListVars()
    for varname in violators:
        blocks, defmap = ssa_mod._fresh_vars(blocks, varname, None)
        blocks = ssa_mod._fix_ssa_vars(blocks, varname, defmap, cfg, df_plus, cache_list_vars, None)
    return blocks


def _dump(blocks):
    return [(label, [str(stmt) for stmt in blk.body]) for label, blk in sorted(blocks.items())]


def _assert_restricted_matches_unrestricted(builder):
    # Build twice so each run gets its own scope and the SSA version
    # counters start identically.
    got = _dump(ssa_mod._run_ssa(builder()))
    expected = _dump(_reference_run_ssa(builder()))
    assert got == expected


def test_self_use_assignment():
    _assert_restricted_matches_unrestricted(_build_self_use_blocks)


def test_phi_in_passthrough_block():
    _assert_restricted_matches_unrestricted(_build_diamond_blocks)


def test_phi_insertion_does_not_mutate_input_blocks():
    # Restricted sweeps pass untouched blocks through by reference,
    # so the phi-insertion step must build a fresh block: prepending
    # in place would push the phi into the caller's IR as well.
    blocks = _build_diamond_blocks()
    input_bodies = {label: list(blk.body) for label, blk in blocks.items()}
    ssa_mod._run_ssa(blocks)
    for label, blk in blocks.items():
        assert blk.body == input_bodies[label], f"input block {label} was mutated in place"


def test_end_to_end_accumulators():
    # End-to-end sanity: two SSA violators rewritten one after the
    # other in a compiled kernel produce correct values.
    @cuda.jit
    def k(inp, out):
        s = 0.0
        p = 1.0
        for i in range(inp.shape[0]):
            s = s + inp[i]
            p = p * inp[i]
        out[0] = s
        out[1] = p

    inp = np.array([1.0, 2.0, 3.0, 4.0])
    out = cuda.to_device(np.zeros(2))
    k[1, 1](cuda.to_device(inp), out)
    got = out.copy_to_host()
    assert got[0] == 10.0
    assert got[1] == 24.0
