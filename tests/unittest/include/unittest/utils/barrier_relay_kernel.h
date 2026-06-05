/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef BARRIER_RELAY_KERNEL_H
#define BARRIER_RELAY_KERNEL_H

void relay_put_barrier_test_do(void *stream, uint64_t config, uint8_t *slots, int rank_id, int rank_size,
    int rounds);
void barrier_perf_v3_do(void *stream, uint64_t config, int iters);
void barrier_perf_relay_do(void *stream, uint64_t config, int iters);
void relay_put_barrier_perf_do(void *stream, uint64_t config, uint8_t *slots, int rank_id, int rank_size, int iters);

#endif // BARRIER_RELAY_KERNEL_H
