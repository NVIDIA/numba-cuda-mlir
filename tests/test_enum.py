# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Python enum support in CUDA kernels."""

from enum import Enum, IntEnum

import numpy as np
import pytest
import cuda.simt as cs


class Color(Enum):
    red = 1
    green = 2
    blue = 3


class Shape(IntEnum):
    circle = 2
    square = 500


class RequestError(IntEnum):
    not_found = 404
    internal_error = 500


class Temperature(Enum):
    freezing = 0.0
    boiling = 100.0


class Planet(Enum):
    mercury = (3.303e23, 2.4397e6)


class TestEnum:
    def test_attribute_access(self):
        @cs.jit
        def kernel(out):
            out[0] = Color.red.value
            out[1] = Color.green.value

        out = cs.device_array(2, dtype=np.int64)
        kernel[1, 1](out)
        np.testing.assert_array_equal(out.copy_to_host(), [1, 2])

    def test_as_argument(self):
        @cs.jit
        def kernel(color, out):
            out[0] = color.value

        out = cs.device_array(1, dtype=np.int64)
        kernel[1, 1](Color.red, out)
        assert out.copy_to_host()[0] == 1

    def test_equality(self):
        @cs.jit
        def kernel(out):
            out[0] = Color.red == Color.red
            out[1] = Color.red == Color.green
            out[2] = Color.red != Color.green

        out = cs.device_array(3, dtype=np.bool_)
        kernel[1, 1](out)
        np.testing.assert_array_equal(out.copy_to_host(), [True, False, True])

    @pytest.mark.parametrize(
        "color,idx", [(Color.red, 0), (Color.green, 1), (Color.blue, 2)]
    )
    def test_argument_comparison(self, color, idx):
        @cs.jit
        def kernel(color, out):
            out[0] = color == Color.red
            out[1] = color == Color.green
            out[2] = color == Color.blue

        out = cs.device_array(3, dtype=np.bool_)
        kernel[1, 1](color, out)
        expected = [i == idx for i in range(3)]
        np.testing.assert_array_equal(out.copy_to_host(), expected)

    def test_conditional(self):
        @cs.jit
        def kernel(color, out):
            if color == Color.red:
                out[0] = 10
            elif color == Color.green:
                out[0] = 20
            else:
                out[0] = 30

        out = cs.device_array(1, dtype=np.int64)
        for color, expected in [(Color.red, 10), (Color.green, 20), (Color.blue, 30)]:
            kernel[1, 1](color, out)
            assert out.copy_to_host()[0] == expected

    def test_ternary(self):
        @cs.jit
        def kernel(pred, out):
            out[0] = (Color.red if pred else Color.green).value

        out = cs.device_array(1, dtype=np.int64)
        kernel[1, 1](True, out)
        assert out.copy_to_host()[0] == 1
        kernel[1, 1](False, out)
        assert out.copy_to_host()[0] == 2


class TestIntEnum:
    def test_value_and_argument(self):
        @cs.jit
        def kernel(shape, out):
            out[0] = Shape.circle.value
            out[1] = shape.value

        out = cs.device_array(2, dtype=np.int64)
        kernel[1, 1](Shape.square, out)
        np.testing.assert_array_equal(out.copy_to_host(), [2, 500])

    def test_arithmetic(self):
        @cs.jit
        def kernel(x, out):
            out[0] = (
                x + Shape.circle if x <= RequestError.not_found else x - Shape.circle
            )

        out = cs.device_array(1, dtype=np.int64)
        kernel[1, 1](300, out)
        assert out.copy_to_host()[0] == 302
        kernel[1, 1](500, out)
        assert out.copy_to_host()[0] == 498

    def test_equality_cross_type(self):
        @cs.jit
        def kernel(out):
            out[0] = Shape.square == Shape.square
            out[1] = Shape.square == RequestError.internal_error

        out = cs.device_array(2, dtype=np.bool_)
        kernel[1, 1](out)
        np.testing.assert_array_equal(out.copy_to_host(), [True, True])


class TestFloatEnum:
    def test_value_and_argument(self):
        @cs.jit
        def kernel(temp, out):
            out[0] = Temperature.freezing.value
            out[1] = temp.value

        out = cs.device_array(2, dtype=np.float64)
        kernel[1, 1](Temperature.boiling, out)
        np.testing.assert_array_equal(out.copy_to_host(), [0.0, 100.0])

    def test_comparison(self):
        @cs.jit
        def kernel(out):
            out[0] = Temperature.freezing == Temperature.freezing
            out[1] = Temperature.freezing == Temperature.boiling

        out = cs.device_array(2, dtype=np.bool_)
        kernel[1, 1](out)
        np.testing.assert_array_equal(out.copy_to_host(), [True, False])


class TestDeviceFunction:
    def test_enum_return(self):
        @cs.jit(device=True)
        def choose(pred):
            return Color.red if pred else Color.green

        @cs.jit
        def kernel(pred, out):
            out[0] = choose(pred) == Color.red
            out[1] = choose(not pred) == Color.green

        out = cs.device_array(2, dtype=np.bool_)
        kernel[1, 1](True, out)
        np.testing.assert_array_equal(out.copy_to_host(), [True, True])


class TestUnsupported:
    def test_tuple_enum(self):
        @cs.jit
        def kernel(p, out):
            out[0] = 1

        out = cs.device_array(1, dtype=np.int64)
        with pytest.raises(TypeError, match="tuple values are not supported"):
            kernel[1, 1](Planet.mercury, out)

    def test_string_enum(self):
        class StringEnum(Enum):
            hello = "world"

        @cs.jit
        def kernel(s, out):
            out[0] = 1

        out = cs.device_array(1, dtype=np.int64)
        with pytest.raises(TypeError, match="str values are not supported"):
            kernel[1, 1](StringEnum.hello, out)
