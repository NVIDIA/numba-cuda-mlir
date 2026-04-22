# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from cuda.simt import tools

_GPU_CC = None


def get_gpu_cc():
    """Get the GPU compute capability as a tuple, cached after first call."""
    global _GPU_CC
    if _GPU_CC is None:
        _GPU_CC = tools.get_gpu_compute_capability(tuple)
    return _GPU_CC


def check_cc_min(min_cc, feature="This feature"):
    """Check if GPU meets minimum compute capability requirement.

    Returns:
        (should_skip, skip_message) tuple
    """
    cc = get_gpu_cc()
    if cc < min_cc:
        return True, f"{feature} requires CC {min_cc[0]}.{min_cc[1]}+, got {cc}"
    return False, None


def check_cc_exact(exact_cc, feature="This feature"):
    """Check if GPU matches exact compute capability requirement.

    Returns:
        (should_skip, skip_message) tuple
    """
    cc = get_gpu_cc()
    if cc != exact_cc:
        return True, f"{feature} requires CC {exact_cc[0]}.{exact_cc[1]}, got {cc}"
    return False, None


def require_cc_min(min_cc, feature="This feature"):
    """Exit with message if GPU doesn't meet minimum CC requirement."""
    should_skip, msg = check_cc_min(min_cc, feature)
    if should_skip:
        print(f"SKIP: {msg}")
        sys.exit(0)


def require_cc_exact(exact_cc, feature="This feature"):
    """Exit with message if GPU doesn't match exact CC requirement."""
    should_skip, msg = check_cc_exact(exact_cc, feature)
    if should_skip:
        print(f"SKIP: {msg}")
        sys.exit(0)
