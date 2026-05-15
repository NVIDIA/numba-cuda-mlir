# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(MSVC)
    set(CEXT_OPTIONS /EHs-c- /GR- /wd4100 /wd4530)
else()
    set(CEXT_OPTIONS -fno-exceptions -fno-rtti)
endif()

add_library(_mviewbuf MODULE
    mviewbuf.cpp
)

target_compile_options(_mviewbuf PUBLIC ${CEXT_OPTIONS})

target_include_directories(_mviewbuf PRIVATE ${Python_INCLUDE_DIRS})
if(WIN32)
    target_link_libraries(_mviewbuf PRIVATE Python::Module)
endif()

set_target_properties(_mviewbuf PROPERTIES SKIP_BUILD_RPATH TRUE)

if (APPLE)
    target_link_options(_mviewbuf PRIVATE -undefined dynamic_lookup)
endif()
