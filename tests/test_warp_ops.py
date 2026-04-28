# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import cuda.simt as cuda
from cusimt.numba_cuda.testing import cc_X_or_above


def _safe_cc_check(cc):
    return cc_X_or_above(*cc)


# vote_sync kernels
def use_any_sync(ary_in, ary_out):
    i = cuda.grid(1)
    pred = cuda.any_sync(0xFFFFFFFF, ary_in[i])
    ary_out[i] = pred


def use_all_sync(ary_in, ary_out):
    i = cuda.grid(1)
    pred = cuda.all_sync(0xFFFFFFFF, ary_in[i])
    ary_out[i] = pred


def use_eq_sync(ary_in, ary_out):
    i = cuda.grid(1)
    pred = cuda.eq_sync(0xFFFFFFFF, ary_in[i])
    ary_out[i] = pred


def use_ballot_sync(ary_in, ary_out):
    i = cuda.grid(1)
    ballot = cuda.ballot_sync(0xFFFFFFFF, ary_in[i])
    ary_out[i] = ballot


# match_sync kernels
def use_match_any_sync(ary_in, ary_out):
    i = cuda.grid(1)
    mask = cuda.match_any_sync(0xFFFFFFFF, ary_in[i])
    ary_out[i] = mask


def use_match_all_sync(ary_in, ary_mask_out, ary_pred_out):
    i = cuda.grid(1)
    mask, pred = cuda.match_all_sync(0xFFFFFFFF, ary_in[i])
    ary_mask_out[i] = mask
    ary_pred_out[i] = pred


# activemask kernel
def use_activemask(ary_out):
    i = cuda.grid(1)
    mask = cuda.activemask()
    ary_out[i] = mask


@pytest.mark.skipif(not _safe_cc_check((7, 0)), reason="Vote sync requires CC 7.0 or greater")
class TestVoteSyncOperations:
    def test_any_sync_one_true(self):
        compiled = cuda.jit(use_any_sync)
        ary_in = np.zeros(32, dtype=np.int32)
        ary_in[5] = 1  # One thread has true predicate
        ary_out = np.zeros(32, dtype=np.int32)
        ary_in_d = cuda.to_device(ary_in)
        ary_out_d = cuda.to_device(ary_out)
        compiled[1, 32](ary_in_d, ary_out_d)
        ary_out = ary_out_d.copy_to_host()
        assert np.all(ary_out == 1)

    def test_any_sync_all_false(self):
        compiled = cuda.jit(use_any_sync)
        ary_in = np.zeros(32, dtype=np.int32)
        ary_out = np.zeros(32, dtype=np.int32)
        ary_in_d = cuda.to_device(ary_in)
        ary_out_d = cuda.to_device(ary_out)
        compiled[1, 32](ary_in_d, ary_out_d)
        ary_out = ary_out_d.copy_to_host()
        assert np.all(ary_out == 0)

    def test_all_sync_all_true(self):
        compiled = cuda.jit(use_all_sync)
        ary_in = np.ones(32, dtype=np.int32)
        ary_out = np.zeros(32, dtype=np.int32)
        ary_in_d = cuda.to_device(ary_in)
        ary_out_d = cuda.to_device(ary_out)
        compiled[1, 32](ary_in_d, ary_out_d)
        ary_out = ary_out_d.copy_to_host()
        assert np.all(ary_out == 1)

    def test_all_sync_one_false(self):
        compiled = cuda.jit(use_all_sync)
        ary_in = np.ones(32, dtype=np.int32)
        ary_in[15] = 0  # One thread has false predicate
        ary_out = np.zeros(32, dtype=np.int32)
        ary_in_d = cuda.to_device(ary_in)
        ary_out_d = cuda.to_device(ary_out)
        compiled[1, 32](ary_in_d, ary_out_d)
        ary_out = ary_out_d.copy_to_host()
        assert np.all(ary_out == 0)

    def test_eq_sync_all_same(self):
        compiled = cuda.jit(use_eq_sync)
        ary_in = np.ones(32, dtype=np.int32)
        ary_out = np.zeros(32, dtype=np.int32)
        ary_in_d = cuda.to_device(ary_in)
        ary_out_d = cuda.to_device(ary_out)
        compiled[1, 32](ary_in_d, ary_out_d)
        ary_out = ary_out_d.copy_to_host()
        assert np.all(ary_out == 1)

    def test_eq_sync_all_same_zeros(self):
        compiled = cuda.jit(use_eq_sync)
        ary_in = np.zeros(32, dtype=np.int32)
        ary_out = np.zeros(32, dtype=np.int32)
        ary_in_d = cuda.to_device(ary_in)
        ary_out_d = cuda.to_device(ary_out)
        compiled[1, 32](ary_in_d, ary_out_d)
        ary_out = ary_out_d.copy_to_host()
        assert np.all(ary_out == 1)

    def test_eq_sync_different(self):
        compiled = cuda.jit(use_eq_sync)
        ary_in = np.zeros(32, dtype=np.int32)
        ary_in[0] = 1  # First thread different
        ary_out = np.zeros(32, dtype=np.int32)
        ary_in_d = cuda.to_device(ary_in)
        ary_out_d = cuda.to_device(ary_out)
        compiled[1, 32](ary_in_d, ary_out_d)
        ary_out = ary_out_d.copy_to_host()
        assert np.all(ary_out == 0)

    def test_ballot_sync(self):
        compiled = cuda.jit(use_ballot_sync)
        ary_in = np.zeros(32, dtype=np.int32)
        ary_in[0] = 1
        ary_in[5] = 1
        ary_in[31] = 1
        expected = (1 << 0) | (1 << 5) | (1 << 31)
        ary_out = np.zeros(32, dtype=np.uint32)
        ary_in_d = cuda.to_device(ary_in)
        ary_out_d = cuda.to_device(ary_out)
        compiled[1, 32](ary_in_d, ary_out_d)
        ary_out = ary_out_d.copy_to_host()
        assert np.all(ary_out == expected)

    def test_ballot_sync_all_true(self):
        compiled = cuda.jit(use_ballot_sync)
        ary_in = np.ones(32, dtype=np.int32)
        ary_out = np.zeros(32, dtype=np.uint32)
        ary_in_d = cuda.to_device(ary_in)
        ary_out_d = cuda.to_device(ary_out)
        compiled[1, 32](ary_in_d, ary_out_d)
        ary_out = ary_out_d.copy_to_host()
        assert np.all(ary_out == 0xFFFFFFFF)

    def test_ballot_sync_none_true(self):
        compiled = cuda.jit(use_ballot_sync)
        ary_in = np.zeros(32, dtype=np.int32)
        ary_out = np.zeros(32, dtype=np.uint32)
        ary_in_d = cuda.to_device(ary_in)
        ary_out_d = cuda.to_device(ary_out)
        compiled[1, 32](ary_in_d, ary_out_d)
        ary_out = ary_out_d.copy_to_host()
        assert np.all(ary_out == 0)


@pytest.mark.skipif(not _safe_cc_check((7, 0)), reason="Match sync requires CC 7.0 or greater")
class TestMatchSyncOperations:
    def test_match_any_sync_all_same(self):
        compiled = cuda.jit(use_match_any_sync)
        ary_in = np.full(32, 42, dtype=np.int32)
        ary_out = np.zeros(32, dtype=np.uint32)
        ary_in_d = cuda.to_device(ary_in)
        ary_out_d = cuda.to_device(ary_out)
        compiled[1, 32](ary_in_d, ary_out_d)
        ary_out = ary_out_d.copy_to_host()
        assert np.all(ary_out == 0xFFFFFFFF)

    def test_match_any_sync_two_groups(self):
        compiled = cuda.jit(use_match_any_sync)
        ary_in = np.array([1 if i < 16 else 2 for i in range(32)], dtype=np.int32)
        ary_out = np.zeros(32, dtype=np.uint32)
        ary_in_d = cuda.to_device(ary_in)
        ary_out_d = cuda.to_device(ary_out)
        compiled[1, 32](ary_in_d, ary_out_d)
        ary_out = ary_out_d.copy_to_host()
        # First 16 threads should match with mask 0x0000FFFF
        assert np.all(ary_out[:16] == 0x0000FFFF)
        # Last 16 threads should match with mask 0xFFFF0000
        assert np.all(ary_out[16:] == 0xFFFF0000)

    def test_match_all_sync_all_same(self):
        compiled = cuda.jit(use_match_all_sync)
        ary_in = np.full(32, 42, dtype=np.int32)
        ary_mask_out = np.zeros(32, dtype=np.uint32)
        ary_pred_out = np.zeros(32, dtype=np.int32)
        ary_in_d = cuda.to_device(ary_in)
        ary_mask_out_d = cuda.to_device(ary_mask_out)
        ary_pred_out_d = cuda.to_device(ary_pred_out)
        compiled[1, 32](ary_in_d, ary_mask_out_d, ary_pred_out_d)
        ary_mask_out = ary_mask_out_d.copy_to_host()
        ary_pred_out = ary_pred_out_d.copy_to_host()
        assert np.all(ary_mask_out == 0xFFFFFFFF)
        assert np.all(ary_pred_out == 1)

    def test_match_all_sync_not_all_same(self):
        compiled = cuda.jit(use_match_all_sync)
        ary_in = np.array([1 if i < 16 else 2 for i in range(32)], dtype=np.int32)
        ary_mask_out = np.zeros(32, dtype=np.uint32)
        ary_pred_out = np.zeros(32, dtype=np.int32)
        ary_in_d = cuda.to_device(ary_in)
        ary_mask_out_d = cuda.to_device(ary_mask_out)
        ary_pred_out_d = cuda.to_device(ary_pred_out)
        compiled[1, 32](ary_in_d, ary_mask_out_d, ary_pred_out_d)
        ary_pred_out = ary_pred_out_d.copy_to_host()
        # Predicate should be false since not all threads have same value
        assert np.all(ary_pred_out == 0)


@pytest.mark.skipif(not _safe_cc_check((7, 0)), reason="Activemask requires CC 7.0 or greater")
class TestActivemask:
    def test_activemask_full_warp(self):
        compiled = cuda.jit(use_activemask)
        ary_out = np.zeros(32, dtype=np.uint32)
        ary_out_d = cuda.to_device(ary_out)
        compiled[1, 32](ary_out_d)
        ary_out = ary_out_d.copy_to_host()
        assert np.all(ary_out == 0xFFFFFFFF)


@pytest.mark.skipif(
    not _safe_cc_check((7, 0)), reason="Vote sync with partial mask requires CC 7.0+"
)
class TestVoteSyncPartialMask:
    def test_any_sync_partial_mask(self):
        @cuda.jit
        def use_any_sync_partial(ary_in, ary_out):
            i = cuda.grid(1)
            # Only first 16 threads participate
            if i < 16:
                pred = cuda.any_sync(0x0000FFFF, ary_in[i])
                ary_out[i] = pred

        ary_in = np.zeros(32, dtype=np.int32)
        ary_in[5] = 1  # Thread 5 has true
        ary_out = np.zeros(32, dtype=np.int32)
        ary_in_d = cuda.to_device(ary_in)
        ary_out_d = cuda.to_device(ary_out)
        use_any_sync_partial[1, 32](ary_in_d, ary_out_d)
        ary_out = ary_out_d.copy_to_host()
        # First 16 threads should all see true
        assert np.all(ary_out[:16] == 1)

    def test_ballot_sync_partial_mask(self):
        @cuda.jit
        def use_ballot_sync_partial(ary_in, ary_out):
            i = cuda.grid(1)
            # Only first 16 threads participate
            if i < 16:
                ballot = cuda.ballot_sync(0x0000FFFF, ary_in[i])
                ary_out[i] = ballot

        ary_in = np.zeros(32, dtype=np.int32)
        ary_in[0] = 1
        ary_in[5] = 1
        ary_in[15] = 1
        expected = (1 << 0) | (1 << 5) | (1 << 15)
        ary_out = np.zeros(32, dtype=np.uint32)
        ary_in_d = cuda.to_device(ary_in)
        ary_out_d = cuda.to_device(ary_out)
        use_ballot_sync_partial[1, 32](ary_in_d, ary_out_d)
        ary_out = ary_out_d.copy_to_host()
        assert np.all(ary_out[:16] == expected)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    pytest.main([__file__, "-v"])
