# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(CEXT_OPTIONS -fno-exceptions -fno-rtti)

add_library(_helperlib MODULE
    _helpermod.cpp
)

target_compile_options(_helperlib PUBLIC ${CEXT_OPTIONS})

target_include_directories(_helperlib PRIVATE ${Python_INCLUDE_DIRS})

set_target_properties(_helperlib PROPERTIES SKIP_BUILD_RPATH TRUE)

if (APPLE)
    target_link_options(_helperlib PRIVATE -undefined dynamic_lookup)
endif()

