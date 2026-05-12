# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(MSVC)
    set(CEXT_OPTIONS /EHs-c- /GR-)
else()
    set(CEXT_OPTIONS -fno-exceptions -fno-rtti)
endif()

add_library(_typeconv MODULE
    typeconv.cpp
    _typeconv.cpp
)

target_compile_options(_typeconv PUBLIC ${CEXT_OPTIONS})

target_include_directories(_typeconv PRIVATE ${Python_INCLUDE_DIRS})

set_target_properties(_typeconv PROPERTIES SKIP_BUILD_RPATH TRUE)

if (APPLE)
    target_link_options(_typeconv PRIVATE -undefined dynamic_lookup)
endif()
