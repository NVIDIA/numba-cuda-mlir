# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check security release floors; also verify the venv when PYTHON is supplied."""

import os
import sys


FIXED_PATCHES = {(3, 11): 18, (3, 12): 16, (3, 13): 18, (3, 14): 10, (3, 15): 2}


def main():
    expected = os.environ.get("PYTHON")
    if expected:
        if os.path.normcase(sys.executable) != os.path.normcase(expected):
            sys.exit(f"PATH resolves Python to {sys.executable}, not {expected}")
        if sys.prefix == sys.base_prefix:
            sys.exit("Hardened Python must run inside the selected virtual environment")
    version = sys.version_info
    series = (version.major, version.minor)
    minimum = FIXED_PATCHES.get(series)
    if minimum is None:
        sys.exit(f"No patched release is known for Python {version.major}.{version.minor}")
    if version.micro < minimum or version.releaselevel != "final":
        sys.exit(
            f"Python {sys.version} predates the required stable {version.major}."
            f"{version.minor}.{minimum} eMudhra/emSign security release"
        )
    print(f"Python {version.major}.{version.minor}.{version.micro}: patched interpreter verified")


if __name__ == "__main__":
    main()
