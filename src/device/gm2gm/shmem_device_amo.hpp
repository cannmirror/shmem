/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_AMO_HPP
#define SHMEM_DEVICE_AMO_HPP

#include "kernel_operator.h"
#include "host/shmem_host_def.h"
#include "shmemi_device_cc.h"
#include "device/gm2gm/engine/shmem_device_udma.h"

#define ACLSHMEM_ATOMIC_ADD_TYPENAME(NAME, TYPE)                                                                       \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_add(__gm__ TYPE *dst, TYPE value, int32_t pe)                        \
    {                                                                                                                  \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                     \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                                    \
            if constexpr (std::is_same_v<TYPE, uint32_t> || std::is_same_v<TYPE, int64_t> ||                           \
                std::is_same_v<TYPE, uint64_t>) {                                                                      \
                AscendC::printf("MTE atomic add supports: int8, int16, int32, float, half, bfloat16. \n");             \
            } else {                                                                                                   \
                auto ptr = aclshmem_ptr(dst, pe);                                                                      \
                AscendC::TEventID my_sync_id = (AscendC::TEventID)device_state->mte_config.sync_id;                    \
                __gm__ TYPE *remote_ptr = reinterpret_cast<__gm__ TYPE *>(ptr);                                        \
                __ubuf__ TYPE *buf = reinterpret_cast<__ubuf__ TYPE *>(device_state->mte_config.aclshmem_ub);          \
                *buf = value;                                                                                          \
                AscendC::SetAtomicAdd<TYPE>();                                                                         \
                aclshmemi_copy_ub2gm(remote_ptr, buf, sizeof(TYPE));                                                   \
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(my_sync_id);                                           \
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(my_sync_id);                                          \
                AscendC::SetAtomicNone();                                                                              \
            }                                                                                                          \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_UDMA) {                                            \
            aclshmemx_udma_atomic_add(dst, value, pe);                                                                 \
        }                                                                                                              \
    }                                                                                                                  \

ACLSHMEM_TYPE_FUNC_ATOMIC_ADD(ACLSHMEM_ATOMIC_ADD_TYPENAME);


#define ACLSHMEM_ATOMIC_FETCH_ADD_TYPENAME(NAME, TYPE)                                                                 \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_fetch_add(__gm__ TYPE *dst, TYPE value, int32_t pe)                  \
    {                                                                                                                  \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                     \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_UDMA) {                                                   \
            return aclshmemx_udma_atomic_fetch_add(dst, value, pe);                                                    \
        } else {                                                                                                       \
            AscendC::printf("FAA is only supported on UDMA. \n");                                                      \
            return 0;                                                                                                  \
        }                                                                                                              \
    }                                                                                                                  \

ACLSHMEM_TYPE_FUNC_ATOMIC_FETCH(ACLSHMEM_ATOMIC_FETCH_ADD_TYPENAME);


#define ACLSHMEM_ATOMIC_COMPARE_SWAP_TYPENAME(NAME, TYPE)                                                              \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_compare_swap(__gm__ TYPE *dst, TYPE cond, TYPE value, int32_t pe)    \
    {                                                                                                                  \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                     \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_UDMA) {                                                   \
            return aclshmemx_udma_atomic_compare_swap(dst, cond, value, pe);                                           \
        } else {                                                                                                       \
            AscendC::printf("CAS is only supported on UDMA. \n");                                                      \
            return 0;                                                                                                  \
        }                                                                                                              \
    }                                                                                                                  \

ACLSHMEM_TYPE_FUNC_ATOMIC_FETCH(ACLSHMEM_ATOMIC_COMPARE_SWAP_TYPENAME);

#endif // SHMEM_DEVICE_AMO_HPP
