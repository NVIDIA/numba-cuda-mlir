# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import sys

import pytest

from numba_cuda_mlir import _cext


class _Resource:
    def __init__(self, events):
        self.events = events
        self.loaded_modules = []

    def __del__(self):
        self.events.append("resource")


class _CompiledImage(bytes):
    pass


class _LoadCallback:
    def __init__(self, resource):
        self.resource = resource
        self.specialization = None

    def __call__(self, module):
        self.resource.loaded_modules.append(module)

    def __del__(self):
        self.resource.events.append("callback")


def _compile_result(resource, tuple_size):
    # No load is requested: bytes need not contain a real CUDA image. Retain an
    # artifact owner even in the three-tuple form, which has no load callback.
    image = _CompiledImage(b"unlaunched image")
    image.resource = resource
    result = (image, "unlaunched_kernel", False)
    if tuple_size >= 4:
        result += (_LoadCallback(resource),)
    if tuple_size == 5:
        result += (((4, 4),),)
    return result


@pytest.mark.parametrize("tuple_size", [3, 4, 5])
def test_unlaunched_specialization_releases_compile_result(tuple_size):
    events = []
    resource = _Resource(events)
    resource_refs = sys.getrefcount(resource)
    result = _compile_result(resource, tuple_size)
    result_refs = sys.getrefcount(result)

    specialization = _cext.KernelSpecialization(result)
    assert sys.getrefcount(result) == result_refs + 1
    assert resource.loaded_modules == []
    assert events == []

    del specialization
    gc.collect()
    # The tuple itself must be released, not only its specialization wrapper.
    assert sys.getrefcount(result) == result_refs
    assert resource.loaded_modules == []

    del result
    gc.collect()
    assert sys.getrefcount(resource) == resource_refs
    expected_events = [] if tuple_size == 3 else ["callback"]
    assert events == expected_events

    del resource
    gc.collect()
    assert events == expected_events + ["resource"]


@pytest.mark.parametrize("tuple_size", [4, 5])
def test_unlaunched_specialization_cycle_releases_payload(tuple_size):
    events = []
    resource = _Resource(events)
    resource_refs = sys.getrefcount(resource)
    result = _compile_result(resource, tuple_size)
    specialization = _cext.KernelSpecialization(result)
    # Model a callback closure that retains its owning compiled specialization.
    result[3].specialization = specialization
    assert sys.getrefcount(resource) > resource_refs
    assert resource.loaded_modules == []
    assert events == []

    del result, specialization
    gc.collect()
    # GC may finalize the callback even if a native tuple reference leaks. Both
    # actual resource references and destruction must therefore be checked.
    assert sys.getrefcount(resource) == resource_refs
    assert resource.loaded_modules == []
    assert events == ["callback"]

    del resource
    gc.collect()
    assert events == ["callback", "resource"]
