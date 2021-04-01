# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "hello_batched_blob")

if (WIN32)
ie_add_sample(NAME ${TARGET_NAME}
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

set(OCL_SEARCH_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/clDNN/common/khronos_ocl_icd/windows/Release/lib/x64")
set(OCL_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/clDNN/common/khronos_ocl_icd/windows/include")
else()
ie_add_sample(NAME ${TARGET_NAME}
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
set(OCL_SEARCH_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/clDNN/common/khronos_ocl_icd/linux/Release/bin/x64")
set(OCL_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/clDNN/common/include")
endif ()

find_library(OPENCL_LIB OpenCL PATHS ${OCL_SEARCH_PATH} NO_DEFAULT_PATH)

if (NOT OPENCL_LIB)
	message(FATAL_ERROR "OpenCL library not found")
endif ()

message("-----------------------------------------------------")
message("-----------------------------------------------------")
message("-----------------------------------------------------")
message("-----------------------------------------------------")
message(${OCL_INCLUDE_PATH})
target_include_directories(${TARGET_NAME} PRIVATE ${OCL_INCLUDE_PATH})
include_directories(${TARGET_NAME} PRIVATE ${OCL_INCLUDE_PATH})
target_link_libraries(${TARGET_NAME} PRIVATE ${OPENCL_LIB})

add_dependencies(${TARGET_NAME} clDNNPlugin)
