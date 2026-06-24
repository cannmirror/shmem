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
#include <vector>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmem.h"
#include "shmemi_host_common.h"
#include "unittest_main_test.h"
#include "team_allgather_kernel.h"

constexpr int32_t kChunkElems = 16;
constexpr int32_t kPerfIters = 20;

static size_t allgather_buf_bytes(int32_t n_ranks)
{
    return static_cast<size_t>(n_ranks) * kChunkElems * sizeof(int32_t);
}

static double allgather_put_bytes_per_iter(int32_t n_ranks)
{
    return static_cast<double>(n_ranks) * (n_ranks - 1) * kChunkElems * sizeof(int32_t);
}

static void test_allgather_sync_relay_func(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    const size_t buf_bytes = allgather_buf_bytes(n_ranks);
    int32_t *gva_dev = static_cast<int32_t *>(aclshmem_malloc(buf_bytes));
    ASSERT_NE(gva_dev, nullptr);
    ASSERT_EQ(aclrtMemset(gva_dev, buf_bytes, 0, buf_bytes), 0);

    const int32_t rank_tag = 1000 + rank_id;
    std::vector<int32_t> local_chunk(kChunkElems, rank_tag);
    ASSERT_EQ(aclrtMemcpy(gva_dev + kChunkElems * rank_id, kChunkElems * sizeof(int32_t), local_chunk.data(),
                          kChunkElems * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE),
              0);

    allgather_put_sync_do(stream, util_get_ffts_config(), reinterpret_cast<uint8_t *>(gva_dev), 1, 1);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

    std::vector<int32_t> result(n_ranks * kChunkElems, 0);
    ASSERT_EQ(aclrtMemcpy(result.data(), buf_bytes, gva_dev, buf_bytes, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (int32_t src = 0; src < n_ranks; ++src) {
        EXPECT_EQ(result[kChunkElems * src], 1000 + src) << "rank_id=" << rank_id << " src=" << src;
    }

    aclshmem_free(gva_dev);
    test_finalize(stream, device_id);
}

static void test_allgather_sync_relay_perf(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    const size_t buf_bytes = allgather_buf_bytes(n_ranks);
    int32_t *gva_dev = static_cast<int32_t *>(aclshmem_malloc(buf_bytes));
    ASSERT_NE(gva_dev, nullptr);
    ASSERT_EQ(aclrtMemset(gva_dev, buf_bytes, 0, buf_bytes), 0);

    std::chrono::steady_clock::time_point t0;
    std::chrono::steady_clock::time_point t1;

    aclshmemi_control_barrier_all();
    if (rank_id == 0) {
        t0 = std::chrono::steady_clock::now();
    }
    allgather_put_sync_do(stream, util_get_ffts_config(), reinterpret_cast<uint8_t *>(gva_dev), kPerfIters, 0);
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
    allgather_put_sync_do(stream, util_get_ffts_config(), reinterpret_cast<uint8_t *>(gva_dev), kPerfIters, 1);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    if (rank_id == 0) {
        t1 = std::chrono::steady_clock::now();
    }
    const double ms_relay = (rank_id == 0)
        ? std::chrono::duration<double, std::milli>(t1 - t0).count()
        : 0.0;

    if (rank_id == 0) {
        const double per_v3 = ms_v3 / kPerfIters;
        const double per_relay = ms_relay / kPerfIters;
        const double ratio = (per_v3 > 0 ? per_relay / per_v3 : 0);
        const double put_bytes = allgather_put_bytes_per_iter(n_ranks);
        const double bw_v3_gbps = put_bytes / (per_v3 * 1e-3) / 1e9;
        const double bw_relay_gbps = put_bytes / (per_relay * 1e-3) / 1e9;
        std::cout << "[ALLGATHER_PERF] npes=" << n_ranks << " iters=" << kPerfIters
                  << " chunk=" << (kChunkElems * sizeof(int32_t)) << "B" << std::endl;
        std::cout << "  put + barrier_all_vec (v3):        " << per_v3 << " ms/iter"
                  << "  put_bw=" << bw_v3_gbps << " GB/s" << std::endl;
        std::cout << "  put + sync_all_vec_relay:          " << per_relay << " ms/iter"
                  << "  put_bw=" << bw_relay_gbps << " GB/s" << std::endl;
        std::cout << "  relay/v3 ratio:                    " << ratio << std::endl;
        std::cout << "[ALLGATHER_PERF_CSV] npes=" << n_ranks << ",v3_ms=" << per_v3 << ",relay_ms=" << per_relay
                  << ",ratio=" << ratio << ",v3_wall_us=" << (per_v3 * 1000.0)
                  << ",relay_wall_us=" << (per_relay * 1000.0) << ",v3_put_bw_gbps=" << bw_v3_gbps
                  << ",relay_put_bw_gbps=" << bw_relay_gbps << std::endl;
    }

    aclshmem_free(gva_dev);
    test_finalize(stream, device_id);
}

TEST(TEST_TEAM_ALLGATHER, test_allgather_sync_relay_func)
{
    const int32_t process_count = test_global_ranks;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_allgather_sync_relay_func, local_mem_size, process_count);
}

TEST(TEST_TEAM_ALLGATHER, test_allgather_sync_relay_perf)
{
    const int32_t process_count = test_global_ranks;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_allgather_sync_relay_perf, local_mem_size, process_count);
}
