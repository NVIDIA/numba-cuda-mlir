#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _get_date_str():
    return datetime.now().strftime("%Y-%m-%d")


def _get_time_str():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def _get_version_base():
    with open(ROOT / "numba_cuda_mlir" / "VERSION") as f:
        return f.read().strip()


def get_tag():
    return f"v{_get_version_base()}-{_get_date_str()}"


def get_version():
    return f"{_get_version_base()}.{_get_time_str()}"


if __name__ == "__main__":
    print(get_tag())
