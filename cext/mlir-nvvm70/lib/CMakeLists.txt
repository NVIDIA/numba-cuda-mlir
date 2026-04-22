# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_library(NVVM70Translate
  NVVM70IRBuilder.cpp
  LibNVVMCompiler.cpp
  NVVM70Target.cpp
  Dialect/NVVM70Dialect.cpp
  Target/Target.cpp
)

add_dependencies(NVVM70Translate NVVM70OpsIncGen)

target_include_directories(NVVM70Translate PUBLIC
  ${NVVM70_SOURCE_DIR}/include
  ${NVVM70_BINARY_DIR}/include
)

set_target_properties(NVVM70Translate PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(NVVM70Translate
  PUBLIC
  MLIRGPUDialect
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIROpenMPDialect
  MLIRIR
  MLIRSupport
  LLVMSupport
  ${CMAKE_DL_LIBS}
)

install(TARGETS NVVM70Translate DESTINATION lib)

# Shared library with C API for in-process use from Python.
# Set NVVM70_MLIR_PYTHON_CAPI to the path of libMLIRPythonCAPI.so to link
# against it and share TypeIDs with the Python MLIR bindings, enabling
# raw Operation* passing without text roundtrip.
add_library(MLIRToNVVM70 SHARED CAPI.cpp)
add_dependencies(MLIRToNVVM70 NVVM70OpsIncGen)
target_include_directories(MLIRToNVVM70 PRIVATE
  ${NVVM70_SOURCE_DIR}/include
  ${NVVM70_BINARY_DIR}/include
)
target_link_libraries(MLIRToNVVM70 PRIVATE NVVM70Translate ${CMAKE_DL_LIBS})

if(NVVM70_MLIR_PYTHON_CAPI)
  message(STATUS "MLIRToNVVM70: linking against ${NVVM70_MLIR_PYTHON_CAPI}")
  target_link_libraries(MLIRToNVVM70 PRIVATE ${NVVM70_MLIR_PYTHON_CAPI})
endif()

install(TARGETS MLIRToNVVM70 DESTINATION lib)
