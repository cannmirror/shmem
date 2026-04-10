/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SHMEMI_MSTX_REPORT_H__
#define __SHMEMI_MSTX_REPORT_H__

#include <cstdint>

enum mstx_dfx_event_type {
    MSTX_EVENT_CROSS_CORE_BARRIER = 4,
    MSTX_SCOPE_START = 1000,
    MSTX_SCOPE_END = 1001
};

struct mstx_cross_core_barrier {
    uint32_t used_core_num;
    uint32_t *used_core_id;
    bool is_aiv_only;
    bool pipe_barrier_all;
};

#ifdef __MSTX_DFX_REPORT__

template <typename StructType, typename... Args>
ACLSHMEM_DEVICE void mstx_dfx_report_start(mstx_dfx_event_type event_id, Args... args)
{
    StructType mstx_info{args...};
    __mstx_dfx_report_stub(event_id, sizeof(StructType), &mstx_info);
    __mstx_dfx_report_stub(MSTX_SCOPE_START, 0, nullptr);
}

ACLSHMEM_DEVICE void mstx_dfx_report_end()
{
    __mstx_dfx_report_stub(MSTX_SCOPE_END, 0, nullptr);
}

#define MSTX_DFX_REPORT_START(event_id, struct_type, ...) \
    mstx_dfx_report_start<struct_type>(event_id, ##__VA_ARGS__)

#define MSTX_DFX_REPORT_END() \
    mstx_dfx_report_end()

#else

template <typename StructType, typename... Args>
ACLSHMEM_DEVICE void mstx_dfx_report_start(mstx_dfx_event_type, Args...) {}

ACLSHMEM_DEVICE void mstx_dfx_report_end() {}

#define MSTX_DFX_REPORT_START(event_id, struct_type, ...)
#define MSTX_DFX_REPORT_END()

#endif

#endif
