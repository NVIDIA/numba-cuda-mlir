# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import os
import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

config.name = "nvvm70"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.nvvm70_tools_dir

llvm_config.use_default_substitutions()

# Make nvvm70-translate findable.
llvm_config.with_environment("PATH", config.nvvm70_tools_dir, append_path=True)
# Also add llvm tools dir for FileCheck.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

# Tool substitutions.
tools = [ToolSubst("nvvm70-translate", unresolved="fatal")]
llvm_config.add_tool_substitutions(tools, [config.nvvm70_tools_dir])

# Resolve library paths from environment variables so that the
# multi-cuda test script can swap libnvvm at runtime.
nvvm70_libllvm = os.environ.get("LIBLLVM7", "")
nvvm70_libnvvm = os.environ.get("NVVM70_LIBNVVM", "")

if nvvm70_libllvm:
    config.environment["LIBLLVM7"] = nvvm70_libllvm
if nvvm70_libnvvm:
    config.environment["NVVM70_LIBNVVM"] = nvvm70_libnvvm

# SM override for multi-SM testing (set by scripts/run-test-matrix.sh).
nvvm70_chip = os.environ.get("NVVM70_CHIP", "")
if nvvm70_chip:
    config.environment["NVVM70_CHIP"] = nvvm70_chip
    config.available_features.add("chip-override")

# CUDA toolkit for e2e tests.
cuda_home = os.environ.get("CUDA_HOME", "")
if cuda_home:
    config.available_features.add("cuda-exec")
    config.substitutions.append(("%cuda_home", cuda_home))
    config.substitutions.append(("%cc", "cc"))
