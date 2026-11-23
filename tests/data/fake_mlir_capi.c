// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Decoy for tests/test_symbol_isolation.py: an unrelated library that carries
// upstream MLIR's default CAPI soname (libMLIRPythonCAPI.so), as another
// MLIR-based package would. It defines none of the MLIR C API, so if our
// bindings resolve their CAPI dependency to it, they fail to load.
int fake_mlir_capi_marker = 0;
