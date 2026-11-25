# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generate the ELF export map from the bridge's native C++ dependencies."""

import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("archives", nargs="+")
    args = parser.parse_args()

    symbols = set()
    for archive in args.archives:
        output = subprocess.check_output(
            [args.nm, "--defined-only", "--extern-only", "--format=posix", archive],
            text=True,
        )
        for line in output.splitlines():
            fields = line.split()
            # Archive member headers have no symbol type or value.
            if len(fields) >= 3:
                symbols.add(fields[0])

    args.output.write_text(
        "{\n  global:\n"
        "    mlir*;\n"
        "    LLVM*;\n"
        "    numbaCudaMlir*;\n"
        # Operation/type casts must use the same IDs in the bridge and Python.
        "    _ZN4mlir6detail14TypeIDResolver*;\n"
        + "".join(f"    {symbol};\n" for symbol in sorted(symbols))
        + "  local: *;\n};\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
