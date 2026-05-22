#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print LLVM C API symbols from the LLVM_CAPI_REQUIRED macro list."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


SYMBOL_RE = re.compile(r"^\s*X\([^,]+,\s*(LLVM[A-Za-z0-9_]+)\s*,")
DEFAULT_EXTRA_SYMBOLS = ("LLVMContextCreate",)


def extract_symbols(path: pathlib.Path) -> list[str]:
    symbols: list[str] = []
    in_required_list = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#define LLVM_CAPI_REQUIRED"):
            in_required_list = True
            continue
        if in_required_list and line.startswith("#define LLVM_CAPI_OPTIONAL"):
            break
        if not in_required_list:
            continue
        match = SYMBOL_RE.match(line)
        if match:
            symbols.append(match.group(1))
    return symbols


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=pathlib.Path)
    ns = parser.parse_args()

    symbols = [*DEFAULT_EXTRA_SYMBOLS, *extract_symbols(ns.source)]
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        raise RuntimeError(f"no LLVM C API symbols found in {ns.source}")
    for symbol in symbols:
        print(symbol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
