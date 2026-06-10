/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
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
constexpr int32_t kPerfBlockNum = 16;
constexpr int32_t kStripKMax = 8;
constexpr double kFitIntercept = -0.68265914;
constexpr double kFitIntraCoef = 0.267035791;
constexpr double kFitInterCoef = 0.949747778;

struct BarrierVisitModel {
    int32_t remote_rd;
    int32_t remote_wr;
    int32_t local_rd;
    int32_t local_wr;
    int32_t inter_rounds;
    int32_t intra_rounds;
    double fit_us;
};

static int32_t effective_k(int32_t n_ranks, int32_t block_num)
{
    return std::min({kStripKMax, block_num, n_ranks});
}

static int32_t inter_rounds(int32_t n_ranks, int32_t block_num)
{
    const int32_t k = effective_k(n_ranks, block_num);
    return (k <= 1) ? (n_ranks - 1) : ((n_ranks - 1 + k - 1) / k);
}

static int32_t intra_core_rounds(int32_t n_ranks, int32_t block_num)
{
    const int32_t k = effective_k(n_ranks, block_num);
    if (k <= 1) {
        return 0;
    }
    int32_t intra = 4 * static_cast<int32_t>(std::log2(k));
    if (block_num > 8) {
        intra += 4 * ((std::min(block_num, 32) - 8) / 8);
    }
    return intra;
}

static BarrierVisitModel model_v3(int32_t n_ranks, int32_t block_num = kPerfBlockNum)
{
    BarrierVisitModel m{};
    m.remote_rd = n_ranks - 1;
    m.remote_wr = 0;
    m.local_rd = 0;
    m.local_wr = 2;
    m.inter_rounds = inter_rounds(n_ranks, block_num);
    m.intra_rounds = intra_core_rounds(n_ranks, block_num) + 1;
    m.fit_us = kFitIntercept + kFitIntraCoef * m.intra_rounds + kFitInterCoef * m.inter_rounds;
    return m;
}

static BarrierVisitModel model_relay(int32_t n_ranks, int32_t block_num = kPerfBlockNum)
{
    BarrierVisitModel m{};
    m.remote_rd = 0;
    m.remote_wr = n_ranks - 1;
    m.local_rd = n_ranks - 1;
    m.local_wr = 1;
    m.inter_rounds = inter_rounds(n_ranks, block_num);
    m.intra_rounds = intra_core_rounds(n_ranks, block_num) + m.inter_rounds;
    m.fit_us = kFitIntercept + kFitIntraCoef * m.intra_rounds + kFitInterCoef * m.inter_rounds;
    return m;
}

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
        std::cout << "  sync_all_vec_relay:                " << per_relay << " ms/iter (total " << ms_relay
                  << " ms)" << std::endl;
        std::cout << "  (N-1) put + sync_all_vec_relay:    " << per_put_relay << " ms/iter (total "
                  << ms_put_relay << " ms)" << std::endl;
        std::cout << "  relay/v3 sync ratio:               " << ratio << std::endl;
        const BarrierVisitModel v3m = model_v3(n_ranks);
        const BarrierVisitModel rlym = model_relay(n_ranks);
        std::cout << "[PERF_CSV] npes=" << n_ranks << ",v3_ms=" << per_v3 << ",relay_ms=" << per_relay
                  << ",put_relay_ms=" << per_put_relay << ",ratio=" << ratio << std::endl;
        std::cout << "[PERF_OPS] npes=" << n_ranks << " block_num=" << kPerfBlockNum
                  << " K=" << effective_k(n_ranks, kPerfBlockNum) << std::endl;
        std::cout << "  v3    remote_rd=" << v3m.remote_rd << " remote_wr=" << v3m.remote_wr
                  << " local_rd=" << v3m.local_rd << " local_wr=" << v3m.local_wr
                  << " inter_rounds=" << v3m.inter_rounds << " intra_rounds=" << v3m.intra_rounds
                  << " fit_us=" << v3m.fit_us << std::endl;
        std::cout << "  relay remote_rd=" << rlym.remote_rd << " remote_wr=" << rlym.remote_wr
                  << " local_rd=" << rlym.local_rd << " local_wr=" << rlym.local_wr
                  << " inter_rounds=" << rlym.inter_rounds << " intra_rounds=" << rlym.intra_rounds
                  << " fit_us=" << rlym.fit_us << std::endl;
        std::cout << "[PERF_OPS_CSV] npes=" << n_ranks << ",v3_rr=" << v3m.remote_rd << ",v3_rw="
                  << v3m.remote_wr << ",v3_lr=" << v3m.local_rd << ",v3_lw=" << v3m.local_wr
                  << ",relay_rr=" << rlym.remote_rd << ",relay_rw=" << rlym.remote_wr << ",relay_lr="
                  << rlym.local_rd << ",relay_lw=" << rlym.local_wr << ",v3_fit_us=" << v3m.fit_us
                  << ",relay_fit_us=" << rlym.fit_us << ",v3_wall_us=" << (per_v3 * 1000.0)
                  << ",relay_wall_us=" << (per_relay * 1000.0) << std::endl;
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
