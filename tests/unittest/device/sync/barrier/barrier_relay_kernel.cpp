/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "shmem.h"
#include "shmemi_device_common.h"

extern "C" ACLSHMEM_GLOBAL void relay_put_barrier_perf(uint64_t config, GM_ADDR slots_addr, int rank_id,
    int rank_size, int iters)
{
    util_set_ffts_config(config);

#ifdef __DAV_C220_VEC__
    __gm__ int32_t *slots = reinterpret_cast<__gm__ int32_t *>(slots_addr);
    int32_t my_pe = aclshmem_my_pe();
    (void)rank_id;

    for (int i = 1; i <= iters; ++i) {
        int32_t tag = (my_pe + 1) * 10000 + i;
        for (int dst = 0; dst < rank_size; ++dst) {
            if (dst == my_pe) {
                continue;
            }
            aclshmemi_signal_set((__gm__ int32_t *)(slots + my_pe), dst, tag);
        }
        aclshmemx_barrier_all_vec_relay();
    }
    aclshmemi_store(slots + rank_size + 1, iters);
#endif
}

extern "C" ACLSHMEM_GLOBAL void barrier_perf_v3(uint64_t config, int iters)
{
    util_set_ffts_config(config);
#ifdef __DAV_C220_VEC__
    for (int i = 0; i < iters; ++i) {
        aclshmemx_barrier_all_vec();
    }
#endif
}

extern "C" ACLSHMEM_GLOBAL void barrier_perf_relay(uint64_t config, int iters)
{
    util_set_ffts_config(config);
#ifdef __DAV_C220_VEC__
    for (int i = 0; i < iters; ++i) {
        aclshmemx_barrier_all_vec_relay();
    }
#endif
}

void relay_put_barrier_perf_do(void *stream, uint64_t config, uint8_t *slots, int rank_id, int rank_size, int iters)
{
    relay_put_barrier_perf<<<16, nullptr, stream>>>(config, slots, rank_id, rank_size, iters);
}

void relay_put_barrier_test_do(void *stream, uint64_t config, uint8_t *slots, int rank_id, int rank_size, int rounds)
{
    relay_put_barrier_perf_do(stream, config, slots, rank_id, rank_size, rounds);
}

void barrier_perf_v3_do(void *stream, uint64_t config, int iters)
{
    barrier_perf_v3<<<16, nullptr, stream>>>(config, iters);
}

void barrier_perf_relay_do(void *stream, uint64_t config, int iters)
{
    barrier_perf_relay<<<16, nullptr, stream>>>(config, iters);
}
