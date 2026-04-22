# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import inspect
from pathlib import Path

_logger = logging.getLogger(__name__)


def trace(fmt=None, *args):
    if not _logger.isEnabledFor(logging.DEBUG):
        return
    frame = inspect.stack()[1]
    file = Path(frame.filename).name
    msg = fmt % args if args else (fmt or "")
    _logger.debug(f"TRACE: {file}:{frame.lineno} {frame.function}(): {msg}")
