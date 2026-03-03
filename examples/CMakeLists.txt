# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_directories(
    ${PROJECT_SOURCE_DIR}/src/device/
)

if(DEFINED ENABLE_ASCENDC_DUMP AND ENABLE_ASCENDC_DUMP)
    add_compile_definitions(ENABLE_ASCENDC_DUMP)
    add_compile_definitions(ASCENDC_DUMP=1)
    add_compile_definitions(ASCENDC_DEBUG)
    link_libraries(ascend_dump)
endif()

function(aclshmem_add_fusion_example NAME)
    add_executable(${NAME} ${ARGN})
    
    target_compile_options(${NAME} PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} ${CCE_AICORE_ARCH} ${SANITIZER_FLAGS})
    target_include_directories(${NAME} PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/3rdparty/catlass/include
        ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
        ${PROJECT_SOURCE_DIR}/examples/${NAME}
        ${PROJECT_SOURCE_DIR}/examples/templates/include
        ${PROJECT_SOURCE_DIR}/examples/utils
        ${PROJECT_SOURCE_DIR}/src/device/
        ${PROJECT_SOURCE_DIR}/src/
        ${MPI_INCLUDE_PATH}
    )
    target_link_options(${NAME} PRIVATE --cce-fatobj-link ${CCE_AICORE_ARCH} ${SANITIZER_FLAGS})

    target_link_libraries(${NAME} PRIVATE shmem)
    target_compile_options(${NAME} PRIVATE ${MPI_CXX_COMPILE_FLAGS})

endfunction()

function(aclshmem_add_collective_example NAME)
    add_library(${NAME}_kernel SHARED ${NAME}_kernel.cpp)
    target_compile_options(${NAME}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} ${CCE_AICORE_ARCH_VEC} ${SANITIZER_FLAGS})
    target_include_directories(${NAME}_kernel PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/3rdparty/catlass/include
        ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
        ${PROJECT_SOURCE_DIR}/examples/${NAME}
        ${PROJECT_SOURCE_DIR}/examples/utils
        ${PROJECT_SOURCE_DIR}/src/device/
        ${PROJECT_SOURCE_DIR}/src/
    )
    target_link_options(${NAME}_kernel PRIVATE --cce-fatobj-link ${CCE_AICORE_ARCH_VEC} ${SANITIZER_FLAGS})

    add_executable(${NAME} main.cpp)
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    target_include_directories(${NAME} PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
        ${PROJECT_SOURCE_DIR}/examples/${NAME}
        ${PROJECT_SOURCE_DIR}/examples/templates/include
        ${PROJECT_SOURCE_DIR}/examples/utils
        ${PROJECT_SOURCE_DIR}/src/device/
        ${PROJECT_SOURCE_DIR}/src/host
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/include
        ${PROJECT_SOURCE_DIR}/src/host/mem/heap
        ${MPI_INCLUDE_PATH}
    )
    target_link_libraries(${NAME} PRIVATE shmem ${NAME}_kernel)
    target_compile_options(${NAME} PRIVATE ${MPI_CXX_COMPILE_FLAGS})

endfunction()

foreach(EXAMPLE
    kv_shuffle
    allgather
    allgather_matmul
    allgather_matmul_with_gather_result
    allgather_matmul_padding
    dispatch_gmm_combine
    matmul_allreduce
    matmul_reduce_scatter
    matmul_reduce_scatter_padding
    dynamic_tiling
    rma_d2h_demo
    sdma
)
    add_subdirectory(${EXAMPLE})
endforeach()

if(ACLSHMEM_RDMA_SUPPORT)
    foreach(EXAMPLE
        rdma_perftest
        rdma_demo
        rdma_handlewait_test/unuse_handlewait
        rdma_handlewait_test/use_handlewait
    )
        add_subdirectory(${EXAMPLE})
endforeach()
endif()