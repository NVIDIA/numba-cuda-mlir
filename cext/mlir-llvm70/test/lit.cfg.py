# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import os
import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

config.name = "llvm70"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.llvm70_tools_dir

llvm_config.use_default_substitutions()

# Make llvm70-translate findable.
llvm_config.with_environment("PATH", config.llvm70_tools_dir, append_path=True)
# Also add llvm tools dir for FileCheck.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

# Tool substitutions.
tools = [ToolSubst("llvm70-translate", unresolved="fatal")]
llvm_config.add_tool_substitutions(tools, [config.llvm70_tools_dir])

# Resolve library paths from environment variables so that the
# multi-cuda test script can swap libnvvm at runtime.
llvm70_libllvm = os.environ.get("LIBLLVM7", "")
llvm70_libnvvm = os.environ.get("LLVM70_LIBNVVM", "")

if llvm70_libllvm:
    config.environment["LIBLLVM7"] = llvm70_libllvm
if llvm70_libnvvm:
    config.environment["LLVM70_LIBNVVM"] = llvm70_libnvvm

# SM override for multi-SM testing (set by scripts/run-test-matrix.sh).
llvm70_chip = os.environ.get("LLVM70_CHIP", "")
if llvm70_chip:
    config.environment["LLVM70_CHIP"] = llvm70_chip
    config.available_features.add("chip-override")

# CUDA toolkit for e2e tests.
cuda_home = os.environ.get("CUDA_HOME", "")
if cuda_home:
    config.available_features.add("cuda-exec")
    config.substitutions.append(("%cuda_home", cuda_home))
    config.substitutions.append(("%cc", "cc"))
