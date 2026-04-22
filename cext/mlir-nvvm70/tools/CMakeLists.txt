# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_executable(nvvm70-translate
  nvvm70-translate.cpp
)

target_link_libraries(nvvm70-translate
  PRIVATE
  NVVM70Translate
  MLIRGPUTransforms
  MLIRParser
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
  LLVMSupport
)

install(TARGETS nvvm70-translate DESTINATION bin)
