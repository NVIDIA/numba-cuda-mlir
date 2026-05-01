# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
NRT stats tests using NRTManagedTestType.

Each test launches a single-thread kernel that exercises a specific
incref/decref code path and then asserts that:

1. NRT allocation stats are balanced (alloc == free, mi_alloc == mi_free).
2. The refcount at key points inside the kernel matches expectations
   (via ``nrt_get_refcount``).

Refcount assertions catch missing-incref bugs (premature frees) that
stats alone cannot detect -- see ``test_device_func_arg_noinline`` for
a detailed explanation.
"""

import numpy as np
import pytest
from numba_cuda_mlir.cuda import jit
from numba_cuda_mlir.memory_management.rtsys import rtsys

# Importing registers the type, model, typing, and lowering.
from .nrt_test_type import (  # noqa: F401
    make_nrt_managed_test_obj_inst,
    make_nrt_wrapper,
    nrt_get_refcount,
)


@pytest.fixture(autouse=True)
def nrt_stats():
    """Enable NRT stats and yield a helper for reading deltas."""
    rtsys.ensure_initialized()
    rtsys.memsys_enable_stats()
    assert rtsys.memsys_stats_enabled()
    yield


class TestNRTCreateDestroy:
    """Basic create + destroy: tests the Del decref site."""

    def test_single_create_destroy(self, nrt_stats):
        """One object created, falls out of scope -> decref frees it."""

        @jit
        def kernel(rc):
            # born: refcount 1
            # should be 1 in the rc output
            x = make_nrt_managed_test_obj_inst()
            rc[0] = nrt_get_refcount(x)

        before = rtsys.get_allocation_stats()
        rc = np.zeros(1, dtype=np.int64)
        kernel[1, 1](rc)
        after = rtsys.get_allocation_stats()

        assert rc[0] == 1, f"expected refcount 1 after alloc, got {rc[0]}"

        d_alloc = after.alloc - before.alloc
        d_free = after.free - before.free
        d_mi_alloc = after.mi_alloc - before.mi_alloc
        d_mi_free = after.mi_free - before.mi_free

        assert d_alloc == 1, f"expected 1 alloc, got {d_alloc}"
        assert d_free == 1, f"expected 1 free, got {d_free}"
        assert d_mi_alloc == 1, f"expected 1 mi_alloc, got {d_mi_alloc}"
        assert d_mi_free == 1, f"expected 1 mi_free, got {d_mi_free}"

    def test_multiple_create_destroy(self, nrt_stats):
        """Three independent objects, each freed at end of scope."""

        @jit
        def kernel(rc):
            a = make_nrt_managed_test_obj_inst()
            rc[0] = nrt_get_refcount(a)
            b = make_nrt_managed_test_obj_inst()
            rc[1] = nrt_get_refcount(b)
            c = make_nrt_managed_test_obj_inst()
            rc[2] = nrt_get_refcount(c)

        before = rtsys.get_allocation_stats()
        rc = np.zeros(3, dtype=np.int64)
        kernel[1, 1](rc)
        after = rtsys.get_allocation_stats()

        for i in range(3):
            assert rc[i] == 1, f"object {i}: expected refcount 1, got {rc[i]}"

        assert after.alloc - before.alloc == 3
        assert after.free - before.free == 3
        assert after.mi_alloc - before.mi_alloc == 3
        assert after.mi_free - before.mi_free == 3


class TestNRTVarCopy:
    """Variable copy: tests the incref-on-assignment site.

    ``y = x`` must emit an incref so the object survives both
    ``del x`` and ``del y``.  Without the incref the second decref
    would hit refcount 0 twice -> double-free (free > alloc).
    """

    def test_copy_alias(self, nrt_stats):
        @jit
        def kernel(rc):
            x = make_nrt_managed_test_obj_inst()
            rc[0] = nrt_get_refcount(x)  # 1: just allocated
            y = x  # incref → 2
            rc[1] = nrt_get_refcount(x)  # 2: x and y both alive
            # x's last use was rc[1], so del x fires here → 1
            # numba frontend detects the last use of a var and
            # inserts a del at the earliest possible point
            # see _patch_var_dels in postproc.py for details
            rc[2] = nrt_get_refcount(y)  # 1: only y remains

        before = rtsys.get_allocation_stats()
        rc = np.zeros(3, dtype=np.int64)
        kernel[1, 1](rc)
        after = rtsys.get_allocation_stats()

        assert rc[0] == 1, f"after alloc: expected 1, got {rc[0]}"
        assert rc[1] == 2, f"after y=x (via x): expected 2, got {rc[1]}"
        assert (
            rc[2] == 1
        ), f"after del x (via y): expected 1 (x dead, only y alive), got {rc[2]}"

        d_alloc = after.alloc - before.alloc
        d_free = after.free - before.free

        assert d_alloc == 1, f"expected 1 alloc, got {d_alloc}"
        assert d_free == d_alloc, (
            f"free ({d_free}) != alloc ({d_alloc}): "
            "missing incref on variable copy causes double-free"
        )


class TestNRTFunctionArg:
    """Function argument: tests incref when binding a parameter.

    When a device function receives an NRT-managed argument, it must
    incref on entry so its scope-exit decref doesn't steal the
    caller's reference.  Without the incref the callee's ``del``
    drops refcount to 0 and frees the object while the caller still
    holds a pointer to it -> use-after-free / double-free.
    """

    def test_device_func_arg_inlined(self, nrt_stats):
        """Var-copy incref after inlining (NOT arg-binding incref).

        With default ``inline="always"``, the child's body is spliced
        into the kernel and the arg-binding IR disappears entirely.
        What remains is a variable alias (``obj = x``) which must be
        incref'd so both ``del x`` and ``del obj`` can decref safely.

        This does NOT test the function-argument-binding incref path;
        see ``test_device_func_arg_noinline`` for that.

        Refcount trace after inlining:

            alloc              → rc 1
            obj = x  (incref)  → rc 2   (two live aliases)
            rc[1] read         → rc 2
            del obj  (decref)  → rc 1
            rc[2] read         → rc 1
            del x    (decref)  → rc 0   (freed)
        """

        @jit(device=True)
        def child(obj, out, rc):
            rc[1] = nrt_get_refcount(obj)
            out[0] = 1

        @jit
        def kernel(out, rc):
            x = make_nrt_managed_test_obj_inst()
            rc[0] = nrt_get_refcount(x)
            child(x, out, rc)
            rc[2] = nrt_get_refcount(x)

        before = rtsys.get_allocation_stats()
        out = np.zeros(1, dtype=np.int32)
        rc = np.zeros(3, dtype=np.int64)
        kernel[1, 1](out, rc)
        after = rtsys.get_allocation_stats()

        assert out[0] == 1, "child function did not execute"
        assert rc[0] == 1, f"before call: expected refcount 1, got {rc[0]}"
        assert rc[1] == 2, (
            f"inside child: expected refcount 2 "
            f"(caller + inlined var copy), got {rc[1]}"
        )
        assert rc[2] == 1, f"after call: expected refcount 1, got {rc[2]}"

        d_alloc = after.alloc - before.alloc
        d_free = after.free - before.free

        assert d_alloc == 1, f"expected 1 alloc, got {d_alloc}"
        assert d_free == d_alloc, (
            f"free ({d_free}) != alloc ({d_alloc}): "
            "missing incref on function argument causes double-free"
        )

    def test_device_func_arg_noinline(self, nrt_stats):
        """Non-inlined child -- the callee's ``obj = arg.obj`` must
        incref so its scope-exit decref doesn't steal the caller's
        reference.  Without the arg-binding incref, the child's
        scope-exit decref frees the object while the kernel still
        holds a live reference -> the kernel's final decref is a
        use-after-free.

        The child deliberately aliases its argument (``y = obj``) so
        that its scope exit has two decrefs.  Combined with the
        missing arg-binding incref the refcount goes:

            alloc -> 1   (make obj)
            +1    -> 2   (y = obj incref)
            -1    -> 1   (del y)
            -1    -> 0   (del obj) <- freed here
            -1    -> -1  (del x in kernel) <- use-after-free

        Important caveat: the stats-based assertion (free == alloc)
        alone does NOT catch this bug.  NRT_decref on already-freed
        memory reads a stale refcount and doesn't bump the free
        counter a second time, so stats still show alloc=1, free=1
        even when the bug is present.

        To actually observe the failure, run under compute-sanitizer:

            compute-sanitizer --tool memcheck pytest <this file> \\
                -k test_device_func_arg_noinline

        Without the incref in ``lower_arg_assign`` this produces:

            Invalid __global__ atomic of size 8 bytes
                at NRT_decref+0x1b0
                Address ... is out of bounds
                and is located in the device heap

        The ``inline="never"`` is critical: with the default
        ``inline="always"`` the child is inlined into the kernel,
        the arg-binding IR disappears, and the var-copy incref on
        ``y = obj`` is sufficient.
        """

        @jit(device=True, inline="never")
        def child(obj, out, rc):
            y = obj  # incref → 3 (caller + arg + y)
            rc[1] = nrt_get_refcount(obj)  # 3: obj, y both alive
            out[0] = 1
            # obj's last use was rc[1], so del obj fires → 2
            rc[2] = nrt_get_refcount(y)  # 2: only caller x + y

        @jit
        def kernel(out, rc):
            x = make_nrt_managed_test_obj_inst()
            rc[0] = nrt_get_refcount(x)  # 1: just allocated
            child(x, out, rc)
            rc[3] = nrt_get_refcount(x)  # 1: child's refs released

        before = rtsys.get_allocation_stats()
        out = np.zeros(1, dtype=np.int32)
        rc = np.zeros(4, dtype=np.int64)
        kernel[1, 1](out, rc)
        after = rtsys.get_allocation_stats()

        d_alloc = after.alloc - before.alloc
        d_free = after.free - before.free
        d_mi_alloc = after.mi_alloc - before.mi_alloc
        d_mi_free = after.mi_free - before.mi_free

        assert out[0] == 1, "child function did not execute"

        assert rc[0] == 1, f"before call: expected refcount 1, got {rc[0]}"
        assert rc[1] == 3, (
            f"inside child (obj): expected refcount 3 "
            f"(caller + arg + alias), got {rc[1]}"
        )
        assert rc[2] == 2, (
            f"inside child (y): expected refcount 2 "
            f"(caller + alias, obj dead), got {rc[2]}"
        )
        assert rc[3] == 1, f"after call: expected refcount 1, got {rc[3]}"

        assert d_alloc == 1, f"expected 1 alloc, got {d_alloc}"
        assert d_free == d_alloc, (
            f"free ({d_free}) != alloc ({d_alloc}): "
            "missing incref on function argument causes double-free"
        )
        assert d_mi_alloc == 1, f"expected 1 mi_alloc, got {d_mi_alloc}"
        assert d_mi_free == d_mi_alloc, f"expected mi_free == mi_alloc, got {d_mi_free}"


class TestNRTCallResult:
    """Call result: tests that a non-inlined device function returning an
    NRT-managed type transfers an *owned* reference to the caller.

    The Numba IR for a return statement includes a ``cast`` of the local
    variable to the return type.  Numba's PostProcessor inserts a
    ``del`` for the original variable after the cast but before the
    ``return``.  Without an incref on the cast, the del drops the
    refcount to 0 (freeing the object), this gives you a double free.

    This is fixed by the cast-incref in ``lower_cast_expr_assign``.
    The callee IR now looks like::

        %0 = call @NRT_MemInfo_alloc_aligned(...)  ; refcount 1
        call @NRT_incref(%0)                        ; cast incref → 2
        call @NRT_decref(%0)                        ; del original → 1
        return %0                                   ; owned return
    """

    def test_call_result_noinline(self, nrt_stats):
        """Non-inlined device function returns an NRT-managed object.

        Refcount trace (with cast-incref fix):

            callee: alloc          → 1
            callee: cast incref    → 2
            callee: del original   → 1   (del before return)
            callee: return         → 1   (owned reference)
            caller: x = result     → ...
            caller: rc[0] read     → 1
            caller: del x          → 0   (freed)

        Without the fix, the callee's del drops the refcount to 0
        before the return, and the caller receives a dangling pointer.
        """

        @jit(device=True, inline="never")
        def make_obj():
            return make_nrt_managed_test_obj_inst()

        @jit
        def kernel(rc):
            x = make_obj()
            rc[0] = nrt_get_refcount(x)

        before = rtsys.get_allocation_stats()
        rc = np.zeros(1, dtype=np.int64)
        kernel[1, 1](rc)
        after = rtsys.get_allocation_stats()

        d_alloc = after.alloc - before.alloc
        d_free = after.free - before.free
        d_mi_alloc = after.mi_alloc - before.mi_alloc
        d_mi_free = after.mi_free - before.mi_free

        assert (
            rc[0] == 1
        ), f"expected refcount 1 in caller after non-inlined call, got {rc[0]}; "
        assert d_alloc == 1, f"expected 1 alloc, got {d_alloc}"
        assert d_free == d_alloc, (
            f"free ({d_free}) != alloc ({d_alloc}): "
            "call result refcounting is broken"
        )
        assert d_mi_alloc == 1, f"expected 1 mi_alloc, got {d_mi_alloc}"
        assert d_mi_free == d_mi_alloc, f"expected mi_free == mi_alloc, got {d_mi_free}"

    def test_call_result_chained_noinline(self, nrt_stats):
        """Non-inlined device function calls another non-inlined device
        function and returns the result -- two levels of cast-incref.

        Verifies that the owned-return convention composes across
        multiple non-inlined call boundaries.
        """

        @jit(device=True, inline="never")
        def inner():
            return make_nrt_managed_test_obj_inst()

        @jit(device=True, inline="never")
        def outer():
            return inner()

        @jit
        def kernel(rc):
            x = outer()
            rc[0] = nrt_get_refcount(x)

        before = rtsys.get_allocation_stats()
        rc = np.zeros(1, dtype=np.int64)
        kernel[1, 1](rc)
        after = rtsys.get_allocation_stats()

        d_alloc = after.alloc - before.alloc
        d_free = after.free - before.free

        assert rc[0] == 1, (
            f"expected refcount 1 after chained call, got {rc[0]}; "
            "cast-incref must compose across call boundaries"
        )
        assert d_alloc == 1, f"expected 1 alloc, got {d_alloc}"
        assert d_free == d_alloc, (
            f"free ({d_free}) != alloc ({d_alloc}): "
            "chained call result refcounting is broken"
        )


class TestNRTBuildTuple:
    """Build-tuple: tests incref when packing NRT-managed values into a tuple.

    When ``t = (x,)`` is lowered, each NRT-managed element must be
    incref'd so the tuple holds its own reference.  The corresponding
    ``del t`` must decref each element.  Without the incref, the tuple
    element shares the single reference owned by ``x``; if ``x`` is
    deleted first the tuple holds a dangling pointer.
    """

    def test_build_tuple_incref(self, nrt_stats):
        """Pack an NRT object into a 1-tuple and verify the refcount
        reflects the extra reference while both x and t are alive.

        Refcount trace (with fix):

            x = alloc           → 1
            rc[0] = refcount(x) → 1
            t = (x,)            → incref → 2
            rc[1] = refcount(x) → 2  (x alive, t alive because used later)
            _ = t               → keeps t alive past rc[1]
            del x               → decref → 1
            del t               → decref element → 0 (freed)
            del _               → no-op (alias of dead tuple)

        Without build-tuple incref: rc[1] == 1 (missing reference).
        """

        @jit
        def kernel(rc):
            x = make_nrt_managed_test_obj_inst()
            rc[0] = nrt_get_refcount(x)  # 1
            t = (x,)  # build_tuple
            rc[1] = nrt_get_refcount(x)  # 2 with fix, 1 without
            _ = t  # keep t alive past rc[1]

        before = rtsys.get_allocation_stats()
        rc = np.zeros(2, dtype=np.int64)
        kernel[1, 1](rc)
        after = rtsys.get_allocation_stats()

        d_alloc = after.alloc - before.alloc
        d_free = after.free - before.free

        assert rc[0] == 1, f"after alloc: expected 1, got {rc[0]}"
        assert rc[1] == 2, (
            f"after build_tuple: expected 2 (x + tuple element), got {rc[1]}; "
            "build_tuple is not incref'ing NRT-managed elements"
        )
        assert d_alloc == 1, f"expected 1 alloc, got {d_alloc}"
        assert d_free == d_alloc, (
            f"free ({d_free}) != alloc ({d_alloc}): "
            "build_tuple/del refcounting is unbalanced"
        )


class TestNRTGetattr:
    """Getattr: tests incref when extracting an NRT-managed field from a struct.

    ``NRTWrapperType`` is a struct ``{ inner: NRTManagedTestType }`` whose
    data model implements ``traverse_mlir`` so that ``del`` of the wrapper
    decrefs the inner member, and ``y = wrapper`` increfs it.

    When ``inner = wrapper.inner`` extracts the NRT-managed field via
    ``llvm.extractvalue``, the result must be incref'd.  Without the
    incref, ``del inner`` drops the refcount to 0 while the wrapper
    still holds a pointer to the same object -- a use-after-free that
    causes a double-free when the wrapper is later deleted.
    """

    def test_getattr_nrt_field(self, nrt_stats):
        """Extract an NRT-managed field from a struct wrapper.

        Refcount trace (with getattr incref):

            wrapper = make_nrt_wrapper()  → alloc, refcount 1
            inner = wrapper.inner         → getattr incref → 2
            rc[0] = refcount(inner)       → 2
            del inner                     → decref → 1
            _ = wrapper                   → var-assign incref (traverse) → 2
            del wrapper                   → traverse decref → 1
            del _                         → traverse decref → 0 (freed)

        Without the incref: rc[0] == 1 and del inner frees the object
        while the wrapper still holds a pointer to it.
        """

        @jit
        def kernel(rc):
            wrapper = make_nrt_wrapper()
            inner = wrapper.inner
            rc[0] = nrt_get_refcount(inner)
            _ = wrapper

        before = rtsys.get_allocation_stats()
        rc = np.zeros(1, dtype=np.int64)
        kernel[1, 1](rc)
        after = rtsys.get_allocation_stats()

        d_alloc = after.alloc - before.alloc
        d_free = after.free - before.free

        assert rc[0] == 2, (
            f"after getattr: expected 2 (wrapper + extracted ref), got {rc[0]}; "
            "getattr is not incref'ing the extracted NRT-managed field"
        )
        assert d_alloc == 1, f"expected 1 alloc, got {d_alloc}"
        assert d_free == d_alloc, (
            f"free ({d_free}) != alloc ({d_alloc}): "
            "getattr/del refcounting is unbalanced"
        )


class TestNRTTupleUnpack:
    """Tuple unpacking: tests the full incref chain for tuple pack/unpack.

    Three sites must incref for this to work:

    1. ``build_tuple`` increfs each element (existing).
    2. ``exhaust_iter`` increfs each element when forwarding the tuple,
       because both the original and the forwarded copy will be del'd
       (each del decrefs elements via tuple-traverse).
    3. ``static_getitem`` increfs the extracted element, because del of
       the tuple(s) will decref it but the extracted variable still
       needs its own reference.

    Numba IR for ``a, = t`` is::

        $exhaust = exhaust_iter(t, count=1)
        del t                               # last use was exhaust_iter
        a = static_getitem($exhaust, 0)
        del $exhaust                        # last use was static_getitem

    Without the exhaust_iter incref, del t frees the element.
    Without the getitem incref, del $exhaust frees the element.
    Either way, ``a`` becomes a dangling pointer.
    """

    def test_tuple_unpack_single(self, nrt_stats):
        """Pack an NRT object into a 1-tuple, then unpack it.

        Refcount trace (with all increfs correct):

            x = alloc                → 1
            t = (x,)                 → build_tuple incref → 2
            $exhaust = exhaust_iter  → exhaust incref → 3
            del t                    → tuple-traverse decref → 2
            a = static_getitem       → getitem incref → 3
            del $exhaust             → tuple-traverse decref → 2
            rc[0] = refcount(a)      → 2  (x and a alive)
            _ = x                    → var-copy incref → 3
            del a                    → decref → 2
            del x                    → decref → 1
            del _                    → decref → 0 (freed)

        Without the fixes: object freed before rc[0] read.
        """

        @jit
        def kernel(rc):
            x = make_nrt_managed_test_obj_inst()
            t = (x,)
            (a,) = t
            rc[0] = nrt_get_refcount(a)
            _ = x

        before = rtsys.get_allocation_stats()
        rc = np.zeros(1, dtype=np.int64)
        kernel[1, 1](rc)
        after = rtsys.get_allocation_stats()

        d_alloc = after.alloc - before.alloc
        d_free = after.free - before.free

        assert rc[0] == 2, (
            f"after unpack: expected 2 (x + a alive, t and $exhaust dead), "
            f"got {rc[0]}; tuple unpack incref chain is broken"
        )
        assert d_alloc == 1, f"expected 1 alloc, got {d_alloc}"
        assert d_free == d_alloc, (
            f"free ({d_free}) != alloc ({d_alloc}): "
            "tuple unpack refcounting is unbalanced"
        )

    def test_tuple_unpack_two_elements(self, nrt_stats):
        """Unpack a 2-tuple of NRT objects.

        Two independent allocations, packed into a tuple, unpacked.
        Each must survive independently after the tuple is deleted.
        """

        @jit
        def kernel(rc):
            x = make_nrt_managed_test_obj_inst()
            y = make_nrt_managed_test_obj_inst()
            t = (x, y)
            (a, b) = t
            rc[0] = nrt_get_refcount(a)
            rc[1] = nrt_get_refcount(b)
            _ = x
            __ = y

        before = rtsys.get_allocation_stats()
        rc = np.zeros(2, dtype=np.int64)
        kernel[1, 1](rc)
        after = rtsys.get_allocation_stats()

        d_alloc = after.alloc - before.alloc
        d_free = after.free - before.free

        assert rc[0] == 2, f"element 0 after unpack: expected 2, got {rc[0]}"
        assert rc[1] == 2, f"element 1 after unpack: expected 2, got {rc[1]}"
        assert d_alloc == 2, f"expected 2 allocs, got {d_alloc}"
        assert d_free == d_alloc, (
            f"free ({d_free}) != alloc ({d_alloc}): "
            "two-element tuple unpack refcounting is unbalanced"
        )

    def test_var_copy_tuple(self, nrt_stats):
        """``y = t`` where t is a tuple of NRT objects must incref elements.

        Without the var-copy incref on tuple elements, del t would free
        the elements while y still holds references to them.
        """

        @jit
        def kernel(rc):
            x = make_nrt_managed_test_obj_inst()
            t = (x,)
            y = t
            rc[0] = nrt_get_refcount(x)
            _ = y

        before = rtsys.get_allocation_stats()
        rc = np.zeros(1, dtype=np.int64)
        kernel[1, 1](rc)
        after = rtsys.get_allocation_stats()

        d_alloc = after.alloc - before.alloc
        d_free = after.free - before.free

        assert rc[0] >= 2, (
            f"after tuple var-copy: expected >= 2, got {rc[0]}; "
            "var-copy of tuple is not incref'ing NRT elements"
        )
        assert d_alloc == 1, f"expected 1 alloc, got {d_alloc}"
        assert d_free == d_alloc, (
            f"free ({d_free}) != alloc ({d_alloc}): "
            "tuple var-copy refcounting is unbalanced"
        )


class TestNRTForLoop:
    """For-loop over a tuple of NRT objects.

    ``for x in (obj1, obj2):`` lowers to getiter + iternext + pair_first.
    The ``pair_first`` extracts the loop variable from the iterator result.
    If it doesn't incref, the loop body operates on a borrowed reference
    that can become dangling if the tuple is deleted mid-iteration.
    """

    def test_for_loop_tuple_nrt(self, nrt_stats):
        """Iterate over a 1-tuple of NRT objects.

        Refcount trace (1-element tuple, single iteration):
            x = make_nrt_managed_test_obj_inst()       → alloc, rc=1
            t = (x,)                                   → incref(x), rc=2
            getiter(t) + iternext + pair_first(elem)   → incref(elem), rc=3
            nrt_get_refcount(elem)                     → reads rc
            del elem (end of iteration)                → decref, rc decreases
            del t (dead after loop)                    → decref_tuple_elements
            del x / del _                              → final decref → rc=0
        """

        @jit
        def kernel(rc):
            x = make_nrt_managed_test_obj_inst()
            t = (x,)
            for elem in t:
                rc[0] = nrt_get_refcount(elem)
            _ = x

        before = rtsys.get_allocation_stats()
        rc = np.zeros(1, dtype=np.int64)
        kernel[1, 1](rc)
        after = rtsys.get_allocation_stats()

        d_alloc = after.alloc - before.alloc
        d_free = after.free - before.free

        assert rc[0] >= 2, (
            f"inside for-loop: expected refcount >= 2 "
            f"(original + tuple + pair_first incref, minus any early dels), "
            f"got {rc[0]}"
        )
        assert d_alloc == 1, f"expected 1 alloc, got {d_alloc}"
        assert d_free == d_alloc, (
            f"free ({d_free}) != alloc ({d_alloc}): "
            "for-loop over NRT tuple refcounting is unbalanced"
        )


class TestNRTTraverseMlir:
    """Tests that del and reassignment of a struct with ``traverse_mlir``
    correctly walks into inner NRT-managed fields.

    ``NRTWrapperType`` is a struct ``{ inner: NRTManagedTestType }`` that
    implements ``traverse_mlir`` but does NOT have its own meminfo.  This
    the outer struct has no meminfo, but its inner value does.

    The NRT context discovers the inner meminfo via ``traverse_mlir``
    and emits incref/decref through the traversal path.
    """

    def test_del_wrapper_frees_inner(self, nrt_stats):
        """Deleting a wrapper must decref its inner NRT field.

        Without traverse_mlir support in lower_del, the inner object
        would never be freed — a memory leak.

        Refcount trace:
            make_nrt_wrapper           → inner rc=1
            $tmp = getattr(w, "inner") → incref → rc=2
            del w                      → traverse decref → rc=1
            nrt_get_refcount($tmp)     → reads 1
            del $tmp                   → decref → rc=0 (freed)
        """

        @jit
        def kernel(rc):
            w = make_nrt_wrapper()
            rc[0] = nrt_get_refcount(w.inner)

        before = rtsys.get_allocation_stats()
        rc = np.zeros(1, dtype=np.int64)
        kernel[1, 1](rc)
        after = rtsys.get_allocation_stats()

        d_alloc = after.alloc - before.alloc
        d_free = after.free - before.free

        assert rc[0] == 1, (
            f"expected 1 (wrapper already del'd via traverse, only getattr ref remains), "
            f"got {rc[0]}"
        )
        assert d_alloc == 1, f"expected 1 alloc, got {d_alloc}"
        assert d_free == d_alloc, (
            f"free ({d_free}) != alloc ({d_alloc}): "
            "del of traverse_mlir wrapper is not freeing inner NRT field"
        )

    def test_reassign_wrapper_frees_old_inner(self, nrt_stats):
        """Reassigning a wrapper variable must decref the OLD inner field.

        When a wrapper variable is reassigned, store_var must decref
        the old wrapper's inner NRT field before overwriting.  Without
        traverse_mlir in store_var's decref-old path, the first inner
        object leaks.

        The key assertion is alloc == free == 2: both wrappers' inner
        NRT objects are freed.
        """

        @jit
        def kernel(rc):
            w1 = make_nrt_wrapper()
            w1 = make_nrt_wrapper()
            rc[0] = nrt_get_refcount(w1.inner)

        before = rtsys.get_allocation_stats()
        rc = np.zeros(1, dtype=np.int64)
        kernel[1, 1](rc)
        after = rtsys.get_allocation_stats()

        d_alloc = after.alloc - before.alloc
        d_free = after.free - before.free

        assert rc[0] >= 1, f"expected >= 1, got {rc[0]}"
        assert d_alloc == 2, f"expected 2 allocs (two wrappers), got {d_alloc}"
        assert d_free == d_alloc, (
            f"free ({d_free}) != alloc ({d_alloc}): "
            "reassigning a traverse_mlir wrapper leaks the old inner NRT field"
        )

    def test_dead_intermediate_wrapper(self, nrt_stats):
        """An intermediate wrapper that goes out of scope must free its inner.

        Two wrappers are created but only one survives to the end;
        the other goes out of scope and its inner NRT field must be
        freed via traverse_mlir.  The key assertion is alloc == free == 2.
        """

        @jit
        def kernel(rc):
            w1 = make_nrt_wrapper()
            w2 = make_nrt_wrapper()
            rc[0] = nrt_get_refcount(w1.inner)
            _ = w2

        before = rtsys.get_allocation_stats()
        rc = np.zeros(1, dtype=np.int64)
        kernel[1, 1](rc)
        after = rtsys.get_allocation_stats()

        d_alloc = after.alloc - before.alloc
        d_free = after.free - before.free

        assert rc[0] >= 1, f"expected >= 1, got {rc[0]}"
        assert d_alloc == 2, f"expected 2 allocs, got {d_alloc}"
        assert d_free == d_alloc, (
            f"free ({d_free}) != alloc ({d_alloc}): "
            "dead intermediate wrapper is not freeing its inner NRT field"
        )
