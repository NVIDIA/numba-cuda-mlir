# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(CEXT_OPTIONS -fno-exceptions -fno-rtti)

add_library(_mviewbuf MODULE
    mviewbuf.cpp
)

target_compile_options(_mviewbuf PUBLIC ${CEXT_OPTIONS})

target_include_directories(_mviewbuf PRIVATE ${Python_INCLUDE_DIRS})

set_target_properties(_mviewbuf PROPERTIES SKIP_BUILD_RPATH TRUE)

if (APPLE)
    target_link_options(_mviewbuf PRIVATE -undefined dynamic_lookup)
endif()
