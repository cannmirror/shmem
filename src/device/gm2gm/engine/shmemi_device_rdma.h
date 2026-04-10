/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_DEVICE_RDMA_H
#define SHMEMI_DEVICE_RDMA_H

#include "kernel_operator.h"
#include "device/shmem_def.h"

enum class AclShmemRdmaBackend : uint32_t {
    inDie  = 0,
};

enum class AclShmemRdmaOpcode : uint32_t {
    OP_RDMA_READ = 0,
    OP_RDMA_WRITE,
    OP_RDMA_WRITE_WITH_IMM
};

struct ACLSHMEMRDMAInfo {
    uint32_t qpNum; // number of QP per connection
    uint64_t sqPtr; // pointer to send queue address array of size [PE_NUM][qpNum]
    uint64_t rqPtr; // pointer to receive queue address array of size [PE_NUM][qpNum]
    uint64_t scqPtr; // pointer to send completion queue address array of size [PE_NUM][qpNum]
    uint64_t rcqPtr; // pointer to receive completion queue address array of size [PE_NUM][qpNum]
    uint64_t memPtr; // pointer to memory region array of size [MAX_PE_NUM]
};

struct ACLSHMEMmemInfo {
    uint64_t size; // size of the memory region
    uint64_t addr; // start address of the memory region
    uint32_t lkey; // local key of the memory region
    uint32_t rkey; // remote key of the memory region
};

enum class ACLSHMEMDBMode : int32_t { INVALID_DB = -1, HW_DB = 0, SW_DB };

struct ACLSHMEMWQCtx {
    uint32_t wqn; // work queue number
    uint64_t bufAddr; // start address of ring buffer
    uint32_t wqeSize; // size of each WQE
    uint32_t depth; // depth of ring buffer
    uint64_t headAddr; // work queue head (Producer Index) address
    uint64_t tailAddr; // work queue tail (Consumer Index) address
    ACLSHMEMDBMode dbMode;
    uint64_t dbAddr; // doorbell address
    uint32_t sl; // service level
};

struct ACLSHMEMCQCtx {
    uint32_t cqn; // completion queue number
    uint64_t bufAddr; // start address of ring buffer
    uint32_t cqeSize; // size of each CQE
    uint32_t depth; // depth of ring buffer
    uint64_t headAddr; // work queue head (Producer Index) address
    uint64_t tailAddr; // work queue tail (Consumer Index) address
    ACLSHMEMDBMode dbMode;
    uint64_t dbAddr; // doorbell address
};

ACLSHMEM_DEVICE __gm__ ACLSHMEMRDMAInfo* aclshmemi_qp_info_fetch();

/**
 * @brief Asynchronous RDMA Write function.
 *
 * @param dst                    [in] destination address in remote HBM
 * @param src                    [in] source address in local HBM
 * @param pe                     [in] PE number of the remote PE.
 * @param qpIdx                  [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param messageLen             [in] message length in Bytes
 * @param ubLocal64              [in] temporary UB local tensor of uint64_t used as workspace
 * @param ubLocal32              [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\\MTE3 Event.
 */
template<typename T>
ACLSHMEM_DEVICE void aclshmemi_roce_write(__gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qpIdx,
                                          uint64_t messageLen, AscendC::LocalTensor<uint64_t> ubLocal64,
                                          AscendC::LocalTensor<uint32_t> ubLocal32, uint32_t sync_id);

/**
 * @brief Asynchronous RDMA READ function.
 *
 * @param dst                    [in] destination address in local HBM
 * @param src                    [in] source address in remote HBM
 * @param pe                     [in] PE number of the remote PE.
 * @param qpIdx                  [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param messageLen             [in] message length in Bytes
 * @param ubLocal64              [in] temporary UB local tensor of uint64_t used as workspace
 * @param ubLocal32              [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\\MTE3 Event.
 */
template<typename T>
ACLSHMEM_DEVICE void aclshmemi_roce_read(__gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qpIdx,
                                         uint64_t messageLen, AscendC::LocalTensor<uint64_t> ubLocal64,
                                         AscendC::LocalTensor<uint32_t> ubLocal32, uint32_t sync_id);

/**
 * @brief RDMA Quiet function. This synchronous function ensures all previous RDMA WQEs are completed
 * (data has arrived at the destination NIC).
 *
 * @param pe                     [in] PE number of the remote PE.
 * @param qpIdx                  [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param ubLocal64              [in] temporary UB local tensor of uint64_t used as workspace
 * @param ubLocal32              [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\\MTE3 Event.
 */
ACLSHMEM_DEVICE void aclshmemi_roce_quiet(uint32_t pe, uint32_t qpIdx, AscendC::LocalTensor<uint64_t> ubLocal64,
                                          AscendC::LocalTensor<uint32_t> ubLocal32, uint32_t sync_id);

#endif