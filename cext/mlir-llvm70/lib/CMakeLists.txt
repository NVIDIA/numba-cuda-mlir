# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_library(LLVM70Translate
  LLVM70IRBuilder.cpp
  LibNVVMCompiler.cpp
  LLVM70Target.cpp
  Dialect/LLVM70Dialect.cpp
  Target/Target.cpp
)

add_dependencies(LLVM70Translate LLVM70OpsIncGen)

target_include_directories(LLVM70Translate PUBLIC
  ${LLVM70_SOURCE_DIR}/include
  ${LLVM70_BINARY_DIR}/include
)

set_target_properties(LLVM70Translate PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(LLVM70Translate
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

install(TARGETS LLVM70Translate DESTINATION lib)

# Shared library with C API for in-process use from Python.
# Set LLVM70_MLIR_PYTHON_CAPI to the path of libMLIRPythonCAPI.so to link
# against it and share TypeIDs with the Python MLIR bindings, enabling
# raw Operation* passing without text roundtrip.
add_library(MLIRToLLVM70 SHARED CAPI.cpp)
add_dependencies(MLIRToLLVM70 LLVM70OpsIncGen)
target_include_directories(MLIRToLLVM70 PRIVATE
  ${LLVM70_SOURCE_DIR}/include
  ${LLVM70_BINARY_DIR}/include
)
if(WIN32)
  set_target_properties(MLIRToLLVM70 PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
target_link_libraries(MLIRToLLVM70 PRIVATE LLVM70Translate ${CMAKE_DL_LIBS})

if(WIN32 AND LLVM70_ROOT)
  file(GLOB LLVM70_STATIC_LIBS "${LLVM70_ROOT}/lib/*.lib")
  if(NOT LLVM70_STATIC_LIBS)
    message(FATAL_ERROR "LLVM70_ROOT was set but no static libraries were found under ${LLVM70_ROOT}/lib")
  endif()
  target_link_libraries(MLIRToLLVM70 PRIVATE ${LLVM70_STATIC_LIBS})
  foreach(LLVM70_STATIC_LIB IN LISTS LLVM70_STATIC_LIBS)
    target_link_options(MLIRToLLVM70 PRIVATE "/WHOLEARCHIVE:${LLVM70_STATIC_LIB}")
  endforeach()
endif()

if(LLVM70_MLIR_PYTHON_CAPI)
  message(STATUS "MLIRToLLVM70: linking against ${LLVM70_MLIR_PYTHON_CAPI}")
  target_link_libraries(MLIRToLLVM70 PRIVATE ${LLVM70_MLIR_PYTHON_CAPI})
endif()

install(TARGETS MLIRToLLVM70 DESTINATION lib)
