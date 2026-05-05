# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda


def test_help_message():
    HELP_PREFIX = "Printing the options because: help was specified"
    try:

        @cuda.jit(help=True)
        def test_kernel():
            pass

    except SystemExit as e:
        assert str(e).strip().startswith(HELP_PREFIX), (
            f"Expected SystemExit message to start with '{HELP_PREFIX}', got {e}"
        )


def test_help_message_because_bad_argument():
    HELP_PREFIX = "Printing the options because: Got invalid options: foo."
    try:

        @cuda.jit(foo="bar")
        def test_kernel():
            pass

    except ValueError as e:
        assert str(e).strip().startswith(HELP_PREFIX), (
            f"Expected ValueError message to start with '{HELP_PREFIX}', got {e}"
        )


def test_help_message_because_bad_type():
    HELP_PREFIX = (
        "Printing the options because: Expected opt_level to be of type <class 'int'>, got foo"
    )
    try:

        @cuda.jit(opt_level="foo")
        def test_kernel():
            pass

    except TypeError as e:
        assert str(e).strip().startswith(HELP_PREFIX), (
            f"Expected TypeError message to start with '{HELP_PREFIX}', got {e}"
        )


if __name__ == "__main__":
    test_help_message()
    test_help_message_because_bad_argument()
    test_help_message_because_bad_type()
