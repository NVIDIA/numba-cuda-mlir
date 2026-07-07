# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import nanobind


version_parts = nanobind.__version__.split(".")
if version_parts[:2] != ["2", "13"]:
    print(f"Skipping nanobind patch for version {nanobind.__version__}")
    raise SystemExit(0)

path = Path(nanobind.source_dir()) / "nb_type.cpp"
text = path.read_text()

# nanobind 2.13's exact check rejects types using a derived metaclass, such as
# ArithValueMeta, even though their instances retain the nanobind layout.
old = """bool nb_type_check(PyObject *t) noexcept {
    return internals->nb_type == Py_TYPE(t);
}
"""
new = """bool nb_type_check(PyObject *t) noexcept {
    PyTypeObject *metaclass = Py_TYPE(t);
    return metaclass == internals->nb_type ||
           PyType_IsSubtype(metaclass, internals->nb_type);
}
"""

if new not in text:
    if old not in text:
        raise RuntimeError(f"Could not find nanobind 2.13 nb_type_check in {path}")
    path.write_text(text.replace(old, new, 1))
