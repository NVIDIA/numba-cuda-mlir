# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_executable(llvm70-translate
  llvm70-translate.cpp
)
if(MSVC)
  target_compile_options(llvm70-translate PRIVATE /wd4706)
endif()

target_link_libraries(llvm70-translate
  PRIVATE
  LLVM70Translate
  MLIRGPUTransforms
  MLIRParser
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
  LLVMSupport
)

install(TARGETS llvm70-translate DESTINATION bin)
