# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal launch-configuration normalization shared by dispatch and planning."""

_LAUNCH_CONFIG_TRACKER_METADATA_KEY = "launch_config_tracker"
_LAUNCH_CONFIG_TRACKER_OPTION = "__launch_config_tracker__"


class _LaunchConfigTracker:
    """Lazily normalize and record launch metadata used by one compiler attempt."""

    __slots__ = ("_available_launch_config", "_launch_config", "required")

    def __init__(self, available_launch_config):
        self._available_launch_config = available_launch_config
        self._launch_config = None
        self.required = False

    def require(self):
        if self._launch_config is None:
            self._launch_config = _normalize_available_launch_config(self._available_launch_config)
        self.required = True
        return self._launch_config

    @property
    def launch_config(self):
        return self._launch_config


def _is_launch_config_key_tuple(value):
    return (
        isinstance(value, tuple)
        and len(value) == 4
        and all(isinstance(item, tuple) and len(item) == 2 for item in value)
        and value[0][0] == "grid"
        and value[1][0] == "block"
        and value[2][0] == "sharedmem"
        and value[3][0] == "cluster"
    )


def _launch_config_key(launch_config):
    """Return the specialization key for a configure-produced launch config."""

    if launch_config is None:
        return None
    block = launch_config.get("block")
    if block is None:
        raise ValueError("launch_config must contain a 'block' entry")
    if not isinstance(block, tuple):
        raise TypeError("launch_config 'block' must be a normalized tuple")
    grid = launch_config.get("grid")
    if grid is None:
        raise ValueError("launch_config must contain a 'grid' entry")
    if not isinstance(grid, tuple):
        raise TypeError("launch_config 'grid' must be a normalized tuple")
    cluster = launch_config.get("cluster")
    if cluster is not None and not isinstance(cluster, tuple):
        raise TypeError("launch_config 'cluster' must be a normalized tuple or None")
    sharedmem = launch_config.get("sharedmem", 0)
    if sharedmem is None:
        sharedmem = 0
    try:
        sharedmem = int(sharedmem)
    except (TypeError, ValueError):
        raise TypeError("launch_config 'sharedmem' must be integer-convertible") from None
    return (
        ("grid", grid),
        ("block", block),
        ("sharedmem", sharedmem),
        ("cluster", cluster),
    )


def _launch_config_dict_from_key(launch_config_key):
    if not _is_launch_config_key_tuple(launch_config_key):
        raise TypeError("launch_config_key must be a normalized launch-config key")
    return {name: value for name, value in launch_config_key}


def _normalize_available_launch_config(launch_config):
    launch_config_key = _launch_config_key(launch_config)
    if launch_config_key is None:
        return None
    return _launch_config_dict_from_key(launch_config_key)
