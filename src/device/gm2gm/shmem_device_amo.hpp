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
#include "shmemi_device_cc.h"

#define ACLSHMEM_ATOMIC_ADD_TYPENAME(NAME, TYPE)                                                                         \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_add(__gm__ TYPE *dst, TYPE value, int32_t pe)                          \
    {                                                                                                                    \
        auto ptr = aclshmem_ptr(dst, pe);                                                                                \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                       \
        AscendC::TEventID my_sync_id = (AscendC::TEventID)device_state->mte_config.sync_id;                              \
        __gm__ TYPE *remote_ptr = reinterpret_cast<__gm__ TYPE *>(ptr);                                                  \
        __ubuf__ TYPE *buf = reinterpret_cast<__ubuf__ TYPE *>(device_state->mte_config.aclshmem_ub);                    \
        *buf = value;                                                                                                    \
        AscendC::SetAtomicAdd<TYPE>();                                                                                   \
        aclshmemi_copy_ub2gm(remote_ptr, buf, sizeof(TYPE));                                                             \
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(my_sync_id);                                                     \
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(my_sync_id);                                                    \
        AscendC::SetAtomicNone();                                                                                        \
    }

ACLSHMEM_TYPE_FUNC_ATOMIC_ADD(ACLSHMEM_ATOMIC_ADD_TYPENAME);

#endif