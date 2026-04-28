# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
from gpu_utils import check_cc_min, check_cc_exact
from cusimt.numba_cuda.core import config as cuda_config

TEST_BIN_DIR = os.getenv(
    "CL_NUMBA_COMPAT_TEST_BIN_DIR",
    str(Path(__file__).resolve().parent / "numba_cuda_tests" / "testing"),
)


@pytest.fixture(scope="session", autouse=True)
def disable_low_occupancy_warnings():
    """Disable low occupancy warnings during tests (similar to numba-cuda CUDATestCase)."""
    original_value = cuda_config.CUDA_LOW_OCCUPANCY_WARNINGS
    cuda_config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
    yield
    cuda_config.CUDA_LOW_OCCUPANCY_WARNINGS = original_value


@pytest.fixture(scope="session", autouse=True)
def show_full_ice_tracebacks():
    """Show full tracebacks for internal compiler errors during tests."""
    import os

    original = os.environ.get("CUSIMT_ICE_FULL_TB")
    os.environ["CUSIMT_ICE_FULL_TB"] = "1"
    yield
    if original is None:
        os.environ.pop("CUSIMT_ICE_FULL_TB", None)
    else:
        os.environ["CUSIMT_ICE_FULL_TB"] = original


def pytest_addoption(parser):
    parser.addoption(
        "--dump-failed-filechecks",
        action="store_true",
        help="Dump reproducers for FileCheck tests that fail.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_cc_min(cc, feature): skip test if GPU compute capability < cc",
    )
    config.addinivalue_line(
        "markers",
        "requires_cc_exact(cc, feature): skip test if GPU compute capability != cc",
    )
    if config.getoption("--pdb"):
        import logging

        logging.basicConfig(level=logging.DEBUG)
        config.option.reruns = 0
        config.option.capture = "yes"
        config.option.maxfail = 1
        config.option.verbose = 1
        config.option.showcapture = 1
        config.option.numprocesses = 0


def pytest_runtest_setup(item):
    for marker in item.iter_markers("requires_cc_min"):
        min_cc = marker.args[0]
        feature = marker.args[1] if len(marker.args) > 1 else "This feature"
        should_skip, msg = check_cc_min(min_cc, feature)
        if should_skip:
            pytest.skip(msg)

    for marker in item.iter_markers("requires_cc_exact"):
        exact_cc = marker.args[0]
        feature = marker.args[1] if len(marker.args) > 1 else "This feature"
        should_skip, msg = check_cc_exact(exact_cc, feature)
        if should_skip:
            pytest.skip(msg)


@pytest.fixture(scope="class")
def initialize_from_pytest_config(request):
    """
    Fixture to initialize the test case with pytest configuration options.
    """
    request.cls._dump_failed_filechecks = request.config.getoption(
        "dump_failed_filechecks"
    )
