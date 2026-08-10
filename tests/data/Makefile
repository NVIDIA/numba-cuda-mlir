# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds the isolation-test poisoning shim. Consumed by
# tests/test_symbol_isolation.py via NUMBA_CUDA_MLIR_TEST_BIN_DIR.

CC ?= cc
CFLAGS ?= -O0 -fPIC
OUT ?= fake_mlir_shim.so

$(OUT): fake_mlir_shim.c
	$(CC) $(CFLAGS) -shared -o $@ $<

clean:
	rm -f $(OUT)

.PHONY: clean
