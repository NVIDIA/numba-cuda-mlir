#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate LLVM C API export list/DEF file from Windows LLVM static libs.

This script extracts defined external symbols from `.lib` archives via
`dumpbin /symbols`, keeps only LLVM C API names (`LLVM*`), and writes:
  1) a newline-separated exports file (response-file friendly), and optionally
  2) a `.def` file suitable for `link.exe /DEF:...`.

Use `--libsfile` to pass a response-style file containing one library path per
line (quotes around paths are allowed).
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from typing import Iterable
import re


_SYMBOL_RE = re.compile(r"\|\s+((?:__imp_)?_?LLVM[A-Za-z0-9_@]+)\s*$")


def _normalize_symbol(symbol: str, strip_leading_underscore: bool) -> str:
    # Import-libraries often expose __imp_ thunks alongside function symbols.
    if symbol.startswith("__imp_"):
        symbol = symbol[len("__imp_") :]

    # x86 stdcall decoration: LLVMFoo@8 -> LLVMFoo
    if "@" in symbol:
        base, sep, suffix = symbol.rpartition("@")
        if sep and suffix.isdigit():
            symbol = base

    if strip_leading_underscore and symbol.startswith("_"):
        symbol = symbol[1:]

    # For imported x86 cdecl symbols this can still be present after __imp_.
    if symbol.startswith("_LLVM"):
        symbol = symbol[1:]

    return symbol


def _read_libsfile(path: pathlib.Path) -> list[pathlib.Path]:
    libs: list[pathlib.Path] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line[0] == line[-1] and line[0] in {"'", '"'}:
            line = line[1:-1]
        libs.append(pathlib.Path(line))
    return libs


def _extract_lib_exports(
    lib_path: pathlib.Path, dumpbin: str, strip_leading_underscore: bool
) -> set[str]:
    cmd = [dumpbin, "/nologo", "/symbols", str(lib_path)]
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"dumpbin failed for {lib_path} (exit {proc.returncode})\n{proc.stdout}\n{proc.stderr}"
        )

    exports: set[str] = set()
    for line in proc.stdout.splitlines():
        if "External" not in line:
            continue
        if "UNDEF" in line:
            continue
        m = _SYMBOL_RE.search(line)
        if not m:
            continue
        symbol = _normalize_symbol(m.group(1), strip_leading_underscore)
        if not symbol.startswith("LLVM"):
            continue
        exports.add(symbol)

    return exports


def _collect_exports(
    libs: Iterable[pathlib.Path], dumpbin: str, strip_leading_underscore: bool
) -> set[str]:
    exports: set[str] = set()
    for lib in libs:
        exports |= _extract_lib_exports(lib, dumpbin, strip_leading_underscore)
    return exports


def _write_exports(path: pathlib.Path, exports: Iterable[str]) -> None:
    path.write_text("".join(f"{name}\n" for name in sorted(exports)), encoding="utf-8")


def _write_def(path: pathlib.Path, dll_name: str, exports: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"LIBRARY {dll_name}\n")
        f.write("EXPORTS\n")
        for name in sorted(exports):
            f.write(f"{name}\n")


def main() -> int:
    parser = argparse.ArgumentParser("gen-llvm-c-exports")
    parser.add_argument(
        "-i",
        "--libsfile",
        type=pathlib.Path,
        default=None,
        help="response-style file with one library path per line",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("LLVM-C.exports"),
        help="output file with newline-separated export names",
    )
    parser.add_argument(
        "--deffile",
        type=pathlib.Path,
        default=None,
        help="optional module-definition file to emit",
    )
    parser.add_argument(
        "--dll-name",
        default="LLVM-C",
        help="DLL name used in the generated .def LIBRARY directive",
    )
    parser.add_argument(
        "-u",
        "--underscore",
        action="store_true",
        help="strip one leading underscore (for 32-bit symbol decoration)",
    )
    parser.add_argument(
        "--dumpbin",
        default="dumpbin.exe",
        help="path to dumpbin executable",
    )
    parser.add_argument(
        "libs",
        nargs="*",
        help="library paths to scan",
    )
    ns = parser.parse_args()

    libs = [pathlib.Path(p) for p in ns.libs]
    if ns.libsfile is not None:
        libs.extend(_read_libsfile(ns.libsfile))
    if not libs:
        parser.error("no libraries provided; pass libs and/or --libsfile")

    missing = [str(p) for p in libs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing library files: {', '.join(missing)}")

    exports = _collect_exports(
        libs=libs,
        dumpbin=ns.dumpbin,
        strip_leading_underscore=ns.underscore,
    )
    if not exports:
        raise RuntimeError("no LLVM C API symbols found")

    ns.output.parent.mkdir(parents=True, exist_ok=True)
    _write_exports(ns.output, exports)
    if ns.deffile is not None:
        ns.deffile.parent.mkdir(parents=True, exist_ok=True)
        _write_def(ns.deffile, ns.dll_name, exports)

    return 0


if __name__ == "__main__":
    sys.exit(main())
