# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import threading


if hasattr(threading, "serialize_iterator"):

    def _LockedCounter(start=0):
        return threading.serialize_iterator(itertools.count(start))

else:

    class _LockedCounter:
        def __init__(self, start=0):
            self._value = start
            self._lock = threading.Lock()

        def __iter__(self):
            return self

        def __next__(self):
            with self._lock:
                value = self._value
                self._value += 1
                return value
