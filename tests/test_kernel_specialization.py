# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from numba_cuda_mlir import cuda, models, types
from numba_cuda_mlir.extending import overload, refresh_registries, typing_registry
from numba_cuda_mlir.models import register_model
from numba_cuda_mlir.numba_cuda.cudadrv.devicearray import DeviceRecord
from numba_cuda_mlir.numba_cuda.extending import typeof_impl

pytestmark = pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")


@pytest.mark.parametrize("scalar_only", [False, True], ids=["array-output", "native-fast"])
@pytest.mark.parametrize("reverse", [False, True])
def test_float_specializations_repeated_both_orders(scalar_only, reverse):
    output = cuda.device_array(1, dtype=np.float64)
    if scalar_only:
        # Capture the output so the launch has only scalar arguments.
        @cuda.jit
        def kernel(x, y):
            output[0] = x + y
    else:

        @cuda.jit
        def kernel(output, x, y):
            output[0] = x + y

    cases = [(np.float32(1), np.float32(2**-25)), (np.float64(1), np.float64(2**-25))]
    if reverse:
        cases.reverse()
    launch = kernel[1, 1]
    for x, y in cases + cases[::-1]:
        args = (x, y) if scalar_only else (output, x, y)
        launch(*args)
        assert output.copy_to_host()[0] == float(x + y)


@pytest.mark.parametrize("scalar_only", [False, True], ids=["array-output", "native-fast"])
def test_python_integer_signed_unsigned_boundaries(scalar_only):
    output = cuda.device_array(2, dtype=np.int64)
    if scalar_only:

        @cuda.jit
        def kernel(value):
            output[0] = value > 0
            output[1] = value == 0

    else:

        @cuda.jit
        def kernel(output, value):
            output[0] = value > 0
            output[1] = value == 0

    launch = kernel[1, 1]
    values = [-(2**63), -1, 0, 2**63 - 1, 2**63, 2**64 - 1]
    for value in values + values[::-1]:
        args = (value,) if scalar_only else (output, value)
        launch(*args)
        np.testing.assert_array_equal(output.copy_to_host(), [int(value > 0), int(value == 0)])


@pytest.mark.parametrize("launch_sensitive", [False, True], ids=["generic", "launch-config"])
@pytest.mark.parametrize("kind", ["signedness", "width"])
def test_numpy_integer_specializations(kind, launch_sensitive):
    class LaunchConfigHandler:
        uses_launch_config = True

        def prepare_args(self, ty, val, stream=None, retr=None):
            return ty, val

    extensions = [LaunchConfigHandler()] if launch_sensitive else []

    @cuda.jit(extensions=extensions)
    def kernel(output, value):
        output[0] = value > 0
        output[1] = value >> 16

    if kind == "signedness":
        values = [np.int64(-1), np.uint64(2**64 - 1)]
    else:
        values = [np.int32(-7), np.int64(2**40 + 3)]
    output = cuda.device_array(2, dtype=np.int64)
    launch = kernel[1, 1]
    for value in values + values[::-1]:
        launch(output, value)
        expected = [int(value) > 0, int(value) >> 16]
        np.testing.assert_array_equal(output.copy_to_host(), expected)


@pytest.mark.parametrize("device_record", [False, True], ids=["numpy-void", "device-record"])
def test_record_layout_specializations(device_record):
    @cuda.jit
    def kernel(output, value):
        output[0] = value.x
        output[1] = value.y

    records = []
    for offsets, x, y in (([0, 8], 11, 23), ([8, 16], 37, 41)):
        dtype = np.dtype(
            {
                "names": ["x", "y"],
                "formats": [np.int64, np.int64],
                "offsets": offsets,
                "itemsize": 24,
            }
        )
        host = np.zeros((), dtype=dtype)[()]
        host["x"], host["y"] = x, y
        if device_record:
            value = DeviceRecord(dtype)
            value.copy_to_device(host)
        else:
            value = host
        records.append((value, [x, y]))

    output = cuda.device_array(2, dtype=np.int64)
    launch = kernel[1, 1]
    for value, expected in records + records[::-1]:
        launch(output, value)
        np.testing.assert_array_equal(output.copy_to_host(), expected)


def test_numpy_record_alignment_specializations():
    def record_is_aligned(value):
        raise NotImplementedError("device-only typing helper")

    @overload(record_is_aligned, inline="always", typing_registry=typing_registry)
    def overload_record_is_aligned(value):
        if isinstance(value, types.Record):
            aligned = int(value.aligned)
            return lambda value: aligned

    refresh_registries(include_uninitialized_cuda=False)

    @cuda.jit
    def kernel(output, value):
        output[0] = record_is_aligned(value)
        output[1] = value.x

    # Equal NumPy dtypes can still carry distinct Numba record alignment.
    records = []
    for aligned in (False, True):
        record = np.zeros((), dtype=np.dtype([("x", np.int64)], align=aligned))[()]
        record["x"] = 19 if aligned else 7
        records.append(record)
    output = cuda.device_array(2, dtype=np.int64)
    launch = kernel[1, 1]
    for record in records + records[::-1]:
        launch(output, record)
        np.testing.assert_array_equal(
            output.copy_to_host(), [int(record.dtype.isalignedstruct), record["x"]]
        )


@pytest.mark.parametrize("scalar_type", [np.datetime64, np.timedelta64])
def test_temporal_unit_specializations(scalar_type):
    @cuda.jit
    def kernel(output, value, step):
        output[0] = value + step

    values = [scalar_type(7, "s"), scalar_type(11, "ms")]
    step = np.timedelta64(3, "ms")
    output = cuda.device_array(1, dtype=(values[0] + step).dtype)
    launch = kernel[1, 1]
    for value in values + values[::-1]:
        launch(output, value, step)
        assert output.copy_to_host()[0] == value + step


@pytest.mark.parametrize("launch_sensitive", [False, True], ids=["generic", "launch-config"])
def test_value_dependent_type_survives_extension_erasure_and_recompile(launch_sensitive):
    class TaggedType(types.Type):
        def __init__(self, tag):
            self.tag = tag
            super().__init__(name=f"KernelSpecializationTag({tag})")

    class TaggedValue:
        def __init__(self, tag):
            self.tag = tag

    @typeof_impl.register(TaggedValue)
    def typeof_tagged_value(value, context):
        return TaggedType(value.tag)

    register_model(TaggedType)(models.CPointerModel)

    def tag_of(value):
        raise NotImplementedError("device-only typing helper")

    @overload(tag_of, inline="always", typing_registry=typing_registry)
    def overload_tag_of(value):
        if isinstance(value, TaggedType):
            tag = value.tag
            return lambda value: tag

    refresh_registries(include_uninitialized_cuda=False)
    events = []

    class ErasingHandler:
        uses_launch_config = launch_sensitive

        def prepare_args(self, ty, val, stream=None, retr=None):
            if isinstance(val, TaggedValue):
                assert isinstance(ty, TaggedType)
                assert ty.tag == val.tag
                tag = val.tag
                events.append(("prepare", tag))
                retr.append(lambda: events.append(("callback", tag)))
                # Both semantic types become the exact same runtime type/value.
                return ty, np.uint64(0)
            return ty, val

    @cuda.jit(extensions=[ErasingHandler()])
    def kernel(output, value):
        output[0] = tag_of(value)

    output = cuda.device_array(1, dtype=np.int64)
    retained = kernel[1, 1]
    expected_events = []

    def check(launcher, tag):
        launcher(output, TaggedValue(tag))
        assert output.copy_to_host()[0] == tag
        expected_events.extend([("prepare", tag), ("callback", tag)])
        assert events == expected_events

    for tag in (3, 9, 3, 9):
        check(retained, tag)
    kernel.recompile()
    # Reverse first use after recompile, then mix retained and fresh launchers.
    for tag in (9, 3, 9, 3):
        check(retained, tag)
        check(kernel[1, 1], tag)


def test_fixed_integer_signature_converts_before_packing():
    @cuda.jit("void(int64[::1], int32)")
    def kernel(output, value):
        output[0] = value

    output = cuda.device_array(1, dtype=np.int64)
    launch = kernel[1, 1]
    for value, expected in ((2**32 - 1, -1), (1, 1), (-(2**31), -(2**31)), (2**32 + 7, 7)):
        launch(output, value)
        assert output.copy_to_host()[0] == expected


def test_fixed_float_signature_rounds_before_device_arithmetic():
    @cuda.jit("void(float64[::1], float32, float32)")
    def kernel(output, x, y):
        output[0] = x + y

    output = cuda.device_array(1, dtype=np.float64)
    launch = kernel[1, 1]
    for x, y in ((1.0, 2**-25), (2.0, 2**-24), (1.0, 2**-23)):
        launch(output, x, y)
        assert output.copy_to_host()[0] == float(np.float32(x) + np.float32(y))


def test_vararg_specializations_preserve_selected_scalar_widths():
    @cuda.jit
    def kernel(output, *values):
        output[0] = values[0] + values[1]

    output = cuda.device_array(1, dtype=np.float64)
    launch = kernel[1, 1]
    cases = [
        (np.float32(1), np.float32(2**-25)),
        (np.float64(1), np.float64(2**-25)),
        (np.float32(1), np.float64(2**-25)),
    ]
    for values in cases + cases[::-1]:
        launch(output, *values)
        assert output.copy_to_host()[0] == float(values[0] + values[1])


@pytest.mark.parametrize("scalar_only", [False, True], ids=["array-output", "native-fast"])
def test_frozen_varargs_convert_selected_scalar_widths(scalar_only):
    output = cuda.device_array(1, dtype=np.float64)
    if scalar_only:

        @cuda.jit
        def kernel(*values):
            output[0] = values[0] + values[1]

    else:

        @cuda.jit
        def kernel(output, *values):
            output[0] = values[0] + values[1]

    launch = kernel[1, 1]
    prefix = () if scalar_only else (output,)
    launch(*prefix, np.float32(1), np.float32(2**-25))
    assert output.copy_to_host()[0] == 1.0
    kernel.disable_compile()
    for scalar_type in (np.float32, np.float64, np.float32):
        launch(*prefix, scalar_type(1), scalar_type(2**-25))
        assert output.copy_to_host()[0] == 1.0
