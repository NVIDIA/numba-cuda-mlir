#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write a .def file exporting the LLVM C API defined in static .lib archives.

Reads each archive's symbol index (the first linker member, "/") and keeps the
C API names (`LLVM*`; C++ symbols are mangled and start with "?").
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import sys


def _archive_symbols(path: pathlib.Path) -> list[str]:
    data = path.read_bytes()
    if not data.startswith(b"!<arch>\n"):
        raise ValueError(f"{path} is not an archive")
    # Member header: name[16] date[12] uid[6] gid[6] mode[8] size[10] fmag[2].
    header = data[8:68]
    if header[:16].rstrip() != b"/":
        raise ValueError(f"{path} has no symbol index")
    size = int(header[48:58])
    member = data[68 : 68 + size]
    (count,) = struct.unpack(">I", member[:4])
    names = member[4 + 4 * count :].split(b"\0")[:count]
    return [name.decode("ascii") for name in names]


def main() -> int:
    parser = argparse.ArgumentParser("gen-llvm-c-exports")
    parser.add_argument("--def", dest="def_file", type=pathlib.Path, required=True)
    parser.add_argument("libs", nargs="+", type=pathlib.Path)
    ns = parser.parse_args()

    exports = sorted(
        {name for lib in ns.libs for name in _archive_symbols(lib) if name.startswith("LLVM")}
    )
    if "LLVMContextCreate" not in exports:
        raise RuntimeError("LLVMContextCreate not found in the given libraries")

    ns.def_file.write_text(
        "LIBRARY LLVM-C\nEXPORTS\n" + "".join(f"{name}\n" for name in exports),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
