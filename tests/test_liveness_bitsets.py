# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Equivalence tests for the bitset ``compute_live_map``.

The production implementation runs both dataflow fix points on
bitsets in reverse-topological sweep order. Every case is checked
against a set-based fix-point implementation kept in this file as
the reference.

Coverage is deterministic and enumerable rather than sampled:

* every 2-block universe over 2 variables - all 16 edge subsets
  crossed with all 256 per-block use/def assignments - which covers
  empty sets, self-loops, mutual edges, unreachable blocks,
  def-without-use and use-without-def exhaustively;
* every 3-block digraph (all 512 edge subsets) with a fixed
  demonstrative use/def pattern, covering joins, diamonds, cycles
  and disconnected components;
* named large shapes that stress the implementation specifically:
  variable counts straddling the bitset byte boundaries, and a deep
  chain with a backedge (the worst case for sweep ordering and the
  classic shape of flattened inline='always' functions).
"""

import itertools
from collections import defaultdict

from numba_cuda_mlir.numba_cuda.core.analysis import compute_live_map


class _StubCFG:
    """Adjacency-only stand-in: compute_live_map consumes just
    ``successors``/``predecessors`` iterables of (label, data)."""

    def __init__(self, edges, labels):
        self._succ = {label: [] for label in labels}
        self._pred = {label: [] for label in labels}
        for src, dst in edges:
            self._succ[src].append((dst, None))
            self._pred[dst].append((src, None))

    def successors(self, label):
        return iter(self._succ[label])

    def predecessors(self, label):
        return iter(self._pred[label])


def _reference_live_map(cfg, blocks, var_use_map, var_def_map):
    """Set-based fix point used as the reference implementation."""

    def fix_point_progress(dct):
        return tuple(len(v) for v in dct.values())

    def fix_point(fn, dct):
        old_point = None
        new_point = fix_point_progress(dct)
        while old_point != new_point:
            fn(dct)
            old_point = new_point
            new_point = fix_point_progress(dct)

    def def_reach(dct):
        for offset in var_def_map:
            used_or_defined = var_def_map[offset] | var_use_map[offset]
            dct[offset] |= used_or_defined
            for out_blk, _ in cfg.successors(offset):
                dct[out_blk] |= dct[offset]

    def liveness(dct):
        for offset in dct:
            live_vars = dct[offset]
            for inc_blk, _data in cfg.predecessors(offset):
                reachable = live_vars & def_reach_map[inc_blk]
                dct[inc_blk] |= reachable - var_def_map[inc_blk]

    live_map = {}
    for offset in blocks.keys():
        live_map[offset] = set(var_use_map[offset])

    def_reach_map = defaultdict(set)
    fix_point(def_reach, def_reach_map)
    fix_point(liveness, live_map)
    return live_map


def _assert_matches_reference(cfg, blocks, use_map, def_map):
    got = compute_live_map(cfg, blocks, use_map, def_map)
    expected = _reference_live_map(cfg, blocks, use_map, def_map)
    assert got == expected, (
        f"edges={[(s, list(d)) for s, d in cfg._succ.items()]} "
        f"use={use_map} def={def_map}: {got} != {expected}"
    )


def test_exhaustive_two_block_worlds():
    # All 16 directed-edge subsets over labels {0, 8} (including
    # self-loops) crossed with all 256 use/def assignments of the
    # variables {a, b} to the two blocks.
    labels = (0, 8)
    possible_edges = [(s, d) for s in labels for d in labels]
    subsets = [
        frozenset(c)
        for n in range(len(possible_edges) + 1)
        for c in itertools.combinations(possible_edges, n)
    ]
    var_subsets = [set(), {"a"}, {"b"}, {"a", "b"}]
    blocks = {label: None for label in labels}
    for edges in subsets:
        cfg = _StubCFG(edges, labels)
        for u0, d0, u1, d1 in itertools.product(var_subsets, repeat=4):
            use_map = {0: u0, 8: u1}
            def_map = {0: d0, 8: d1}
            _assert_matches_reference(cfg, blocks, use_map, def_map)


def test_all_three_block_graphs():
    # Every 3-block digraph (512 edge subsets), with a fixed
    # demonstrative use/def pattern: block 0 defines a and uses b,
    # block 8 defines b and uses a, block 16 only uses both. Under
    # some orientations this makes 16 a join, a loop member, or
    # unreachable.
    labels = (0, 8, 16)
    possible_edges = [(s, d) for s in labels for d in labels]
    use_map = {0: {"b"}, 8: {"a"}, 16: {"a", "b"}}
    def_map = {0: {"a"}, 8: {"b"}, 16: set()}
    blocks = {label: None for label in labels}
    for n in range(len(possible_edges) + 1):
        for edges in itertools.combinations(possible_edges, n):
            cfg = _StubCFG(edges, labels)
            _assert_matches_reference(cfg, blocks, use_map, def_map)


def test_byte_boundary_widths():
    # Variable counts straddling the byte boundaries of the bitset
    # encoding, on a diamond with a backedge. Use/def membership is
    # a fixed arithmetic pattern so every width places variables in
    # the first, middle and last bytes of the encoding.
    labels = [0, 8, 16, 24]
    edges = [(0, 8), (0, 16), (8, 24), (16, 24), (24, 0)]
    blocks = {label: None for label in labels}
    cfg = _StubCFG(edges, labels)
    for n_vars in (1, 7, 8, 9, 63, 64, 65, 200):
        variables = [f"v{i}" for i in range(n_vars)]
        use_map = {}
        def_map = {}
        for pos, label in enumerate(labels):
            use_map[label] = {v for i, v in enumerate(variables) if i % 4 == pos}
            def_map[label] = {v for i, v in enumerate(variables) if (i * 7) % 4 == pos}
        _assert_matches_reference(cfg, blocks, use_map, def_map)


def test_deep_chain_with_backedge():
    # One long chain plus a loop backedge: the worst case for the
    # backward sweep order and the classic shape of flattened
    # inline='always' functions. Membership follows a fixed stride
    # pattern so variables span short and whole-chain lifetimes.
    labels = list(range(0, 400, 4))
    edges = [(a, b) for a, b in zip(labels, labels[1:])]
    edges.append((labels[-1], labels[1]))
    variables = [f"v{i}" for i in range(50)]
    use_map = {}
    def_map = {}
    for pos, label in enumerate(labels):
        use_map[label] = {v for i, v in enumerate(variables) if (pos + i) % 5 == 0}
        def_map[label] = {v for i, v in enumerate(variables) if (pos * 3 + i) % 11 == 0}
    blocks = {label: None for label in labels}
    cfg = _StubCFG(edges, labels)
    _assert_matches_reference(cfg, blocks, use_map, def_map)


def test_empty_and_single_block():
    cfg = _StubCFG([], [0])
    _assert_matches_reference(cfg, {0: None}, {0: set()}, {0: set()})
    _assert_matches_reference(cfg, {0: None}, {0: {"x", "y"}}, {0: {"y"}})
