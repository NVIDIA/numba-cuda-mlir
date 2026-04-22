# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from importlib.util import find_spec
from cusimt._mlir import ir
from textwrap import indent
from pathlib import Path
from io import StringIO
import re
from textwrap import dedent
import inspect
from cusimt._mlir.ir import Module, Operation
from subprocess import PIPE, Popen, run
from typing import Iterable, Union
import tempfile
import sys
import shutil

# For legacy Numba-CUDA filecheck test cases
from filecheck.matcher import Matcher
from filecheck.options import Options
from filecheck.parser import Parser, pattern_for_opts
from filecheck.finput import FInput
from cusimt.descriptor import MLIRDispatcher
import cmath
import datetime
import enum
import math
import numpy as np
import pytest
import unittest

np_version = tuple(map(int, np.__version__.split(".")[:2]))
IS_NUMPY_2 = np_version >= (2, 0)


def get_filecheck_path():
    for filecheck_name in ("FileCheck", "filecheck"):
        path = Path(sys.prefix) / "bin" / filecheck_name
        if path.exists():
            return path
        found = shutil.which(filecheck_name)
        if found and Path(found).exists():
            return found
    return None


def filecheck_with_comments(module):
    if isinstance(module, Module):
        module = module.operation
    if isinstance(module, Operation):
        assert module.verify()

    op = str(module).strip()
    op = "\n".join(filter(None, op.splitlines()))
    op = dedent(op)

    fun = inspect.currentframe().f_back.f_code
    _, lnum = inspect.findsource(fun)
    fun_with_checks = inspect.getsource(fun)

    filecheck_path = get_filecheck_path()
    assert filecheck_path is not None, "couldn't find FileCheck"
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(("\n" * lnum + fun_with_checks).encode())
        tmp.flush()
        p = Popen(
            [filecheck_path, tmp.name],
            stdout=PIPE,
            stdin=PIPE,
            stderr=PIPE,
        )
        out, err = map(lambda o: o.decode(), p.communicate(input=op.encode()))
        if p.returncode:
            err = err.replace(tmp.name, inspect.getfile(fun))
            updated_checks = generate_test_checks(op)
            raise ValueError(f"\n{err}\n\nUpdated checks:\n{updated_checks}")


def filecheck(checks: str, actual: str | bytes | ir.Module):
    if isinstance(actual, ir.Module):
        actual = str(actual)
    if isinstance(actual, bytes):
        actual = actual.decode()
    checks = dedent(checks)

    filecheck_path = get_filecheck_path()
    assert filecheck_path is not None, "couldn't find FileCheck"

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
        tmp.write(checks)
        tmp.flush()
        tmp_name = tmp.name

    try:
        p = Popen(
            [
                filecheck_path,
                tmp_name,
            ],
            stdout=PIPE,
            stdin=PIPE,
            stderr=PIPE,
        )
        out, err = p.communicate(input=actual.encode())
        out = out.decode()
        err = err.decode()

        if p.returncode != 0:
            raise ValueError(f"\n{out}\n{err}")
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def _strip_gpu_module(mlir_module_str: str):
    """
    gpu.module confuses the test generator, so we strip it out before generating checks.
    """
    lines = mlir_module_str.strip().splitlines()
    lines = [line for line in lines if "@cusimt_gpu_module" not in line]
    lines = list(reversed(lines))
    for i, line in enumerate(lines):
        if line.strip() == "}":
            lines = lines[i + 1 :]
            break
    lines = reversed(lines)
    return "\n".join(lines)


def generate_test_checks(mlir_module_str: str | ir.Module, generic=True):
    mlir_module_str = str(mlir_module_str)
    stripped_input = _strip_gpu_module(mlir_module_str)

    import cusimt

    root = Path(cusimt.__path__[0]).parent.parent
    script = root / "scripts" / "generate-test-checks.py"

    result = run(
        [sys.executable, str(script)],
        input=stripped_input.encode(),
        stdout=PIPE,
        stderr=PIPE,
    )

    if result.returncode != 0:
        raise ValueError(
            result.stderr.decode()
            + "\n"
            + stripped_input
            + "\n"
            + result.stdout.decode()
        )

    out = indent(result.stdout.decode().replace("// ", "# "), " " * 4)
    out = re.sub("# CHECK:", "# CHECK-NEXT:", out)
    if generic:
        out = re.sub(r"\[\[\S+:\.\*\]\]", "{{.*}}", out)
        out = re.sub(r"\[\[[A-Z0-9_]+\]\]", "{{.*}}", out)
    return out


# From legacy Numba-CUDA testsuite


@pytest.mark.usefixtures("initialize_from_pytest_config")
class NumbaCUDATestCase(unittest.TestCase):
    """
    For tests copied from the legacy Numba-CUDA testsuite that use FileCheck.

    Method assertFileCheckAsm will inspect an MLIRDispatcher and assert that
    the compilation artifacts match the FileCheck checks given in the kernel's
    docstring.

    Method assertFileCheckMatches can be used to assert that a given string
    matches FileCheck checks, and is not specific to MLIRDispatcher.
    """

    Signature = Union[tuple[type, ...], None]

    _bool_types = (bool, np.bool_)
    _exact_typesets = [_bool_types, (int,), (str,), (np.integer,), (bytes, np.bytes_)]
    _approx_typesets = [(float,), (complex,), (np.inexact)]
    _sequence_typesets = [(tuple, list)]
    _float_types = (float, np.floating)
    _complex_types = (complex, np.complexfloating)

    FLOAT16_RTOL = np.finfo(np.float16).eps

    def _getIRContents(
        self,
        ir_result: Union[dict[Signature, str], str],
        signature: Union[Signature, None] = None,
    ) -> Iterable[str]:
        if isinstance(ir_result, str):
            assert (
                signature is None
            ), "Cannot use signature because the kernel was only compiled for one signature"
            return [ir_result]

        if signature is None:
            return list(ir_result.values())

        return [ir_result[signature]]

    def assertFileCheckAsm(
        self,
        ir_producer: MLIRDispatcher,
        signature: Union[tuple[type, ...], None] = None,
        check_prefixes: tuple[str] = ("ASM",),
        **extra_filecheck_options,
    ) -> None:
        """
        Assert that the assembly output of the given MLIRDispatcher matches
        the FileCheck checks given in the kernel's docstring.
        """
        ir_contents = self._getIRContents(ir_producer.inspect_asm(), signature)
        assert ir_contents, "No assembly output found for the given signature."
        assert (
            ir_producer.__doc__ is not None
        ), "Kernel docstring is required. To pass checks explicitly, use assertFileCheckMatches."
        check_patterns = ir_producer.__doc__
        for ir_content in ir_contents:
            self.assertFileCheckMatches(
                ir_content,
                check_patterns=check_patterns,
                check_prefixes=check_prefixes,
                **extra_filecheck_options,
            )

    def assertFileCheckMatches(
        self,
        ir_content: str,
        check_patterns: str,
        check_prefixes: tuple[str] = ("CHECK",),
        **extra_filecheck_options,
    ) -> None:
        """
        Assert that the given string matches the passed FileCheck checks.

        Args:
            ir_content: The string to check against.
            check_patterns: The FileCheck checks to use.
            check_prefixes: The prefixes to use for the FileCheck checks.
            extra_filecheck_options: Extra options to pass to FileCheck.
        """
        opts = Options(
            match_filename="-",
            check_prefixes=list(check_prefixes),
            **extra_filecheck_options,
        )
        input_file = FInput(fname="-", content=ir_content)
        parser = Parser(opts, StringIO(check_patterns), *pattern_for_opts(opts))
        matcher = Matcher(opts, input_file, parser)
        matcher.stderr = StringIO()
        result = matcher.run()
        if result != 0:
            if self._dump_failed_filechecks:
                dump_directory = Path(
                    datetime.now().strftime("numba-ir-%Y_%m_%d_%H_%M_%S")
                )
                if not dump_directory.exists():
                    dump_directory.mkdir(parents=True, exist_ok=True)
                base_path = self.id().replace(".", "_")
                ir_dump = dump_directory / Path(base_path).with_suffix(".ll")
                checks_dump = dump_directory / Path(base_path).with_suffix(".checks")
                with (
                    open(ir_dump, "w") as ir_file,
                    open(checks_dump, "w") as checks_file,
                ):
                    _ = ir_file.write(ir_content + "\n")
                    _ = checks_file.write(check_patterns)
                    dump_instructions = f"Reproduce with:\n\nfilecheck --check-prefixes={','.join(check_prefixes)} {checks_dump} --input-file {ir_dump}"
            else:
                dump_instructions = (
                    "Rerun with --dump-failed-filechecks to generate a reproducer."
                )

            self.fail(
                f"FileCheck failed:\n{matcher.stderr.getvalue()}\n\n"
                + dump_instructions
            )

    def assertPreciseEqual(
        self,
        first,
        second,
        prec="exact",
        ulps=1,
        msg=None,
        ignore_sign_on_zero=False,
        abs_tol=None,
    ):
        """
        Versatile equality testing function with more built-in checks than
        standard assertEqual().

        For arrays, test that layout, dtype, shape are identical, and
        recursively call assertPreciseEqual() on the contents.

        For other sequences, recursively call assertPreciseEqual() on
        the contents.

        For scalars, test that two scalars or have similar types and are
        equal up to a computed precision.
        If the scalars are instances of exact types or if *prec* is
        'exact', they are compared exactly.
        If the scalars are instances of inexact types (float, complex)
        and *prec* is not 'exact', then the number of significant bits
        is computed according to the value of *prec*: 53 bits if *prec*
        is 'double', 24 bits if *prec* is single.  This number of bits
        can be lowered by raising the *ulps* value.
        ignore_sign_on_zero can be set to True if zeros are to be considered
        equal regardless of their sign bit.
        abs_tol if this is set to a float value its value is used in the
        following. If, however, this is set to the string "eps" then machine
        precision of the type(first) is used in the following instead. This
        kwarg is used to check if the absolute difference in value between first
        and second is less than the value set, if so the numbers being compared
        are considered equal. (This is to handle small numbers typically of
        magnitude less than machine precision).

        Any value of *prec* other than 'exact', 'single' or 'double'
        will raise an error.
        """
        try:
            self._assertPreciseEqual(
                first, second, prec, ulps, msg, ignore_sign_on_zero, abs_tol
            )
        except AssertionError as exc:
            failure_msg = str(exc)
            # Fall off of the 'except' scope to avoid Python 3 exception
            # chaining.
        else:
            return
        # Decorate the failure message with more information
        self.fail("when comparing %s and %s: %s" % (first, second, failure_msg))

    def _assertPreciseEqual(
        self,
        first,
        second,
        prec="exact",
        ulps=1,
        msg=None,
        ignore_sign_on_zero=False,
        abs_tol=None,
    ):
        """Recursive workhorse for assertPreciseEqual()."""

        def _assertNumberEqual(first, second, delta=None):
            if (
                delta is None
                or first == second == 0.0
                or math.isinf(first)
                or math.isinf(second)
            ):
                self.assertEqual(first, second, msg=msg)
                # For signed zeros
                if not ignore_sign_on_zero:
                    try:
                        if math.copysign(1, first) != math.copysign(1, second):
                            self.fail(
                                self._formatMessage(msg, "%s != %s" % (first, second))
                            )
                    except TypeError:
                        pass
            else:
                self.assertAlmostEqual(first, second, delta=delta, msg=msg)

        first_family = self._detect_family(first)
        second_family = self._detect_family(second)

        assertion_message = "Type Family mismatch. (%s != %s)" % (
            first_family,
            second_family,
        )
        if msg:
            assertion_message += ": %s" % (msg,)
        self.assertEqual(first_family, second_family, msg=assertion_message)

        # We now know they are in the same comparison family
        compare_family = first_family

        # For recognized sequences, recurse
        if compare_family == "ndarray":
            dtype = self._fix_dtype(first.dtype)
            self.assertEqual(dtype, self._fix_dtype(second.dtype))
            self.assertEqual(first.ndim, second.ndim, "different number of dimensions")
            self.assertEqual(first.shape, second.shape, "different shapes")
            self.assertEqual(
                first.flags.writeable, second.flags.writeable, "different mutability"
            )
            # itemsize is already checked by the dtype test above
            self.assertEqual(
                self._fix_strides(first), self._fix_strides(second), "different strides"
            )
            if first.dtype != dtype:
                first = first.astype(dtype)
            if second.dtype != dtype:
                second = second.astype(dtype)
            for a, b in zip(first.flat, second.flat):
                self._assertPreciseEqual(
                    a, b, prec, ulps, msg, ignore_sign_on_zero, abs_tol
                )
            return

        elif compare_family == "sequence":
            self.assertEqual(len(first), len(second), msg=msg)
            for a, b in zip(first, second):
                self._assertPreciseEqual(
                    a, b, prec, ulps, msg, ignore_sign_on_zero, abs_tol
                )
            return

        elif compare_family == "exact":
            exact_comparison = True

        elif compare_family in ["complex", "approximate"]:
            exact_comparison = False

        elif compare_family == "enum":
            self.assertIs(first.__class__, second.__class__)
            self._assertPreciseEqual(
                first.value, second.value, prec, ulps, msg, ignore_sign_on_zero, abs_tol
            )
            return

        elif compare_family == "unknown":
            # Assume these are non-numeric types: we will fall back
            # on regular unittest comparison.
            self.assertIs(first.__class__, second.__class__)
            exact_comparison = True

        else:
            assert 0, "unexpected family"

        # If a Numpy scalar, check the dtype is exactly the same too
        # (required for datetime64 and timedelta64).
        if hasattr(first, "dtype") and hasattr(second, "dtype"):
            self.assertEqual(first.dtype, second.dtype)

        # Mixing bools and non-bools should always fail
        if isinstance(first, self._bool_types) != isinstance(second, self._bool_types):
            assertion_message = "Mismatching return types (%s vs. %s)" % (
                first.__class__,
                second.__class__,
            )
            if msg:
                assertion_message += ": %s" % (msg,)
            self.fail(assertion_message)

        try:
            if cmath.isnan(first) and cmath.isnan(second):
                # The NaNs will compare unequal, skip regular comparison
                return
        except TypeError:
            # Not floats.
            pass

        # if absolute comparison is set, use it
        if abs_tol is not None:
            if abs_tol == "eps":
                rtol = np.finfo(type(first)).eps
            elif isinstance(abs_tol, float):
                rtol = abs_tol
            else:
                raise ValueError('abs_tol is not "eps" or a float, found %s' % abs_tol)
            if abs(first - second) < rtol:
                return

        exact_comparison = exact_comparison or prec == "exact"

        if not exact_comparison and prec != "exact":
            if prec == "single":
                bits = 24
            elif prec == "double":
                bits = 53
            else:
                raise ValueError("unsupported precision %r" % (prec,))
            k = 2 ** (ulps - bits - 1)
            delta = k * (abs(first) + abs(second))
        else:
            delta = None
        if isinstance(first, self._complex_types):
            _assertNumberEqual(first.real, second.real, delta)
            _assertNumberEqual(first.imag, second.imag, delta)
        elif isinstance(first, (np.timedelta64, np.datetime64)):
            # Since Np 1.16 NaT == NaT is False, so special comparison needed
            if np.isnat(first):
                self.assertEqual(np.isnat(first), np.isnat(second))
            else:
                _assertNumberEqual(first, second, delta)
        else:
            _assertNumberEqual(first, second, delta)

    def _detect_family(self, numeric_object):
        """
        This function returns a string description of the type family
        that the object in question belongs to.  Possible return values
        are: "exact", "complex", "approximate", "sequence", and "unknown"
        """
        if isinstance(numeric_object, np.ndarray):
            return "ndarray"

        if isinstance(numeric_object, enum.Enum):
            return "enum"

        for tp in self._sequence_typesets:
            if isinstance(numeric_object, tp):
                return "sequence"

        for tp in self._exact_typesets:
            if isinstance(numeric_object, tp):
                return "exact"

        for tp in self._complex_types:
            if isinstance(numeric_object, tp):
                return "complex"

        for tp in self._approx_typesets:
            if isinstance(numeric_object, tp):
                return "approximate"

        return "unknown"

    def _fix_dtype(self, dtype):
        """
        Fix the given *dtype* for comparison.
        """
        # Under 64-bit Windows, Numpy may return either int32 or int64
        # arrays depending on the function.
        if (
            sys.platform == "win32"
            and sys.maxsize > 2**32
            and dtype == np.dtype("int32")
        ):
            return np.dtype("int64")
        else:
            return dtype

    def _fix_strides(self, arr):
        """
        Return the strides of the given array, fixed for comparison.
        Strides for 0- or 1-sized dimensions are ignored.
        """
        if arr.size == 0:
            return [0] * arr.ndim
        else:
            return [
                stride / arr.itemsize
                for (stride, shape) in zip(arr.strides, arr.shape)
                if shape > 1
            ]
