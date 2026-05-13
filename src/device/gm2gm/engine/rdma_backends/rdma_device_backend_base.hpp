/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEM_RDMA_DEVICE_BACKEND_BASE_HPP
#define ACLSHMEM_RDMA_DEVICE_BACKEND_BASE_HPP

#include "kernel_operator.h"
#include "rdma_device_backend_in_die.hpp"
#include "rdma_device_backend_xscale.hpp"
#include "rdma_device_backend_base.h"

template <typename T, bool IS_MASKED, aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE T aclshmemi_roce_atomic_fetch_and_add(
    __gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qp_idx, uint64_t add_val, uint64_t boundary,
    AscendC::LocalTensor<uint64_t>& ub_local64, AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    static_assert(
        aclshmemi_rdma_backend_dependent_false<B>::value, "aclshmemi_roce_atomic_fetch_and_add not implemented\n");
    return T(1);
}

template <typename T, bool IS_MASKED, aclshmemi_rdma_backend_t B>
ACLSHMEM_DEVICE T aclshmemi_roce_atomic_compare_and_swap(
    __gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qp_idx, uint64_t swap_val, uint64_t comp_val,
    uint64_t swap_mask, uint64_t comp_mask, AscendC::LocalTensor<uint64_t>& ub_local64,
    AscendC::LocalTensor<uint32_t>& ub_local32, uint32_t sync_id)
{
    static_assert(
        aclshmemi_rdma_backend_dependent_false<B>::value, "aclshmemi_roce_atomic_compare_and_swap not implemented\n");
    return T(1);
}
#endif // ACLSHMEM_RDMA_DEVICE_BACKEND_BASE_HPP
