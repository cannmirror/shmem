/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <chrono>
#include <iostream>
#include <string>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmem.h"
#include "shmemi_host_common.h"
#include "unittest_main_test.h"
#include "barrier_relay_kernel.h"

constexpr int32_t kRelayRounds = 3;
constexpr int32_t kPerfIters = 50;

static void test_relay_put_barrier(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    const size_t slot_bytes = static_cast<size_t>(n_ranks + 2) * sizeof(int32_t);
    int32_t *slots_dev = static_cast<int32_t *>(aclshmem_malloc(slot_bytes));
    ASSERT_NE(slots_dev, nullptr);
    ASSERT_EQ(aclrtMemset(slots_dev, slot_bytes, 0, slot_bytes), 0);

    relay_put_barrier_perf_do(stream, util_get_ffts_config(), reinterpret_cast<uint8_t *>(slots_dev), rank_id,
        n_ranks, kRelayRounds);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

    int32_t *slots_host = nullptr;
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&slots_host), slot_bytes), 0);
    ASSERT_EQ(aclrtMemcpy(slots_host, slot_bytes, slots_dev, slot_bytes, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    EXPECT_EQ(slots_host[n_ranks + 1], kRelayRounds) << "rank_id=" << rank_id;

    aclshmemi_control_barrier_all();
    aclrtFreeHost(slots_host);
    aclshmem_free(slots_dev);
    test_finalize(stream, device_id);
}

static void test_barrier_relay_perf(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    std::chrono::steady_clock::time_point t0;
    std::chrono::steady_clock::time_point t1;

    aclshmemi_control_barrier_all();
    if (rank_id == 0) {
        t0 = std::chrono::steady_clock::now();
    }
    barrier_perf_v3_do(stream, util_get_ffts_config(), kPerfIters);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    if (rank_id == 0) {
        t1 = std::chrono::steady_clock::now();
    }
    const double ms_v3 = (rank_id == 0)
        ? std::chrono::duration<double, std::milli>(t1 - t0).count()
        : 0.0;

    aclshmemi_control_barrier_all();
    if (rank_id == 0) {
        t0 = std::chrono::steady_clock::now();
    }
    barrier_perf_relay_do(stream, util_get_ffts_config(), kPerfIters);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    if (rank_id == 0) {
        t1 = std::chrono::steady_clock::now();
    }
    const double ms_relay = (rank_id == 0)
        ? std::chrono::duration<double, std::milli>(t1 - t0).count()
        : 0.0;

    const size_t slot_bytes = static_cast<size_t>(n_ranks + 2) * sizeof(int32_t);
    int32_t *slots_dev = static_cast<int32_t *>(aclshmem_malloc(slot_bytes));
    ASSERT_NE(slots_dev, nullptr);
    ASSERT_EQ(aclrtMemset(slots_dev, slot_bytes, 0, slot_bytes), 0);

    aclshmemi_control_barrier_all();
    if (rank_id == 0) {
        t0 = std::chrono::steady_clock::now();
    }
    relay_put_barrier_perf_do(stream, util_get_ffts_config(), reinterpret_cast<uint8_t *>(slots_dev), rank_id,
        n_ranks, kPerfIters);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    if (rank_id == 0) {
        t1 = std::chrono::steady_clock::now();
    }
    const double ms_put_relay = (rank_id == 0)
        ? std::chrono::duration<double, std::milli>(t1 - t0).count()
        : 0.0;

    if (rank_id == 0) {
        const double per_v3 = ms_v3 / kPerfIters;
        const double per_relay = ms_relay / kPerfIters;
        const double per_put_relay = ms_put_relay / kPerfIters;
        const double ratio = (per_v3 > 0 ? per_relay / per_v3 : 0);
        std::cout << "[PERF] npes=" << n_ranks << " iters=" << kPerfIters << std::endl;
        std::cout << "  barrier_all_vec (v3 LEGACY):       " << per_v3 << " ms/iter (total " << ms_v3 << " ms)"
                  << std::endl;
        std::cout << "  barrier_all_vec_relay:             " << per_relay << " ms/iter (total " << ms_relay
                  << " ms)" << std::endl;
        std::cout << "  (N-1) put + barrier_all_vec_relay: " << per_put_relay << " ms/iter (total "
                  << ms_put_relay << " ms)" << std::endl;
        std::cout << "  relay/v3 barrier ratio:            " << ratio << std::endl;
        std::cout << "[PERF_CSV] npes=" << n_ranks << ",v3_ms=" << per_v3 << ",relay_ms=" << per_relay
                  << ",put_relay_ms=" << per_put_relay << ",ratio=" << ratio << std::endl;
    }

    aclshmem_free(slots_dev);
    test_finalize(stream, device_id);
}

TEST(TEST_SYNC_API, test_relay_put_barrier)
{
    const int32_t process_count = test_global_ranks;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_relay_put_barrier, local_mem_size, process_count);
}

TEST(TEST_SYNC_API, test_barrier_relay_perf)
{
    const int32_t process_count = test_global_ranks;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_barrier_relay_perf, local_mem_size, process_count);
}
