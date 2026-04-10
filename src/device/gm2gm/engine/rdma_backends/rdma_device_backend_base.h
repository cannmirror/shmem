/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RDMA_DEVICE_BACKEND_BASE_H
#define RDMA_DEVICE_BACKEND_BASE_H

#include "device/shmem_def.h"
#include "gm2gm/engine/shmemi_device_rdma.h"

/*
 *  =====================================================================================================
 *  SHMEM RDMA Architecture Structure —— Base Header
 *  =====================================================================================================
 *  We provide two categories of APIs:
 *  1. Fine-grained primitives (ibverbs-style)
 *      - aclshmemi_roce_fill_wqe
 *      - aclshmemi_roce_ring_sq_doorbell
 *      - aclshmemi_roce_post_send
 *      - aclshmemi_roce_ring_cq_doorbell
 *      - aclshmemi_roce_poll_cq
 * 
 *  2. Coarse-grained operations
 *      - aclshmemi_roce_write
 *      - aclshmemi_roce_read
 * =====================================================================================================
 *  Internal call chain (base -> backend-specialized):
 *      aclshmemi_roce_post_send<B, OP_CODE>()                  // compile-time dispatched entry
 *          -> aclshmemi_roce_post_send_read_write()            // shared post-send helper for READ/WRITE
 *              -> aclshmemi_roce_fill_wqe<B, OP_CODE>()        // compile-time dispatched fill_wqe
 *                  -> aclshmemi_rdma_fill_wqe_write_read()     // shared fill_wqe helper for READ/WRITE
 *          -> aclshmemi_roce_ring_sq_doorbell<B>()             // backend-dispatched doorbell 
 * =====================================================================================================
 */

/**
 * @brief Fill WQE for RDMA operation
 *
 * @tparam B                     RDMA Backend type
 * @tparam OP_CODE               rdma opcode in AclShmemRdmaOpcode enum class
 * @param dst                    [in] Remote GM memory address to access
 * @param src                    [in] Local GM memory address to access
 * @param pe                     [in] PE number of the remote PE.
 * @param qpIdx                  [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param messageLen             [in] Message length in bytes
 * @param ubLocal64              [in] temporary UB local tensor of uint64_t used as workspace
 * @param ubLocal32              [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\MTE3 Event.
 * @param wqeAddr                [in] WQE address to fill
 * @param curHead                [in] Current head of WQE queue
 * @param ACLSHMEMmemInfoTable   [in] Memory info table GM address
 * @return uint32_t              [out] total size of WQE in bytes
 */
template<AclShmemRdmaBackend B, AclShmemRdmaOpcode OP_CODE>
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_fill_wqe(__gm__ uint8_t* dst, __gm__ uint8_t* src, uint32_t pe,
                                                 uint32_t qpIdx, uint64_t messageLen,
                                                 AscendC::LocalTensor<uint64_t>& ubLocal64,
                                                 AscendC::LocalTensor<uint32_t>& ubLocal32, uint32_t sync_id,
                                                 __gm__ uint8_t* wqeAddr, uint32_t curHead,
                                                 uint64_t ACLSHMEMmemInfoTable);

/**
 * @brief Ring SQ DB for RDMA operation
 *
 * @tparam B                     RDMA Backend type
 * @param ubLocal64              [in] temporary UB local tensor of uint64_t used as workspace
 * @param ubLocal32              [in] temporary UB local tensor of uint32_t used as workspace
 * @param curHead                [in] Current head of SQ WQE
 * @param qpCtxEntry             [in] Current QP Context
 * @param sync_id                [in] ID used to Sync S\MTE3 Event.
 */
template<AclShmemRdmaBackend B>
ACLSHMEM_DEVICE void aclshmemi_roce_ring_sq_doorbell(AscendC::LocalTensor<uint64_t>& ubLocal64,
                                                     AscendC::LocalTensor<uint32_t>& ubLocal32, uint32_t curHead,
                                                     __gm__ ACLSHMEMWQCtx*& qpCtxEntry, uint32_t sync_id);

/**
 * @brief AIV direct RDMA helper function for post send, prepare WQE and ring doorbell.
 * Directly calls the underlying implementation through template parameters
 * without using switch, resulting in higher performance.
 *
 * @tparam B                     RDMA Backend type
 * @tparam OP_CODE               rdma opcode in AclShmemRdmaOpcode enum class
 * @param dst                    [in] address in remote HBM
 * @param src                    [in] address in local HBM
 * @param pe                     [in] PE number of the remote PE.
 * @param qpIdx                  [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param messageLen             [in] message length in Bytes
 * @param ubLocal64              [in] temporary UB local tensor of uint64_t used as workspace
 * @param ubLocal32              [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\MTE3 Event.
 */
template<AclShmemRdmaBackend B, AclShmemRdmaOpcode OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_roce_post_send(__gm__ uint8_t* dst, __gm__ uint8_t* src, uint32_t pe,
                                              uint32_t qpIdx, uint64_t messageLen,
                                              AscendC::LocalTensor<uint64_t>& ubLocal64,
                                              AscendC::LocalTensor<uint32_t>& ubLocal32, uint32_t sync_id);

/**
 * @brief Ring CQ DB for RDMA operation
 *
 * @tparam B                     RDMA Backend type
 * @param ubLocal64              [in] temporary UB local tensor of uint64_t used as workspace
 * @param ubLocal32              [in] temporary UB local tensor of uint32_t used as workspace
 * @param curTail                [in] Current tail of CQ WQE
 * @param pe                     [in] PE number of the remote PE.
 * @param qpIdx                  [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param sync_id                [in] ID used to Sync S\MTE3 Event.
 */
template<AclShmemRdmaBackend B>
ACLSHMEM_DEVICE void aclshmemi_roce_ring_cq_doorbell(AscendC::LocalTensor<uint64_t>& ubLocal64,
                                                     AscendC::LocalTensor<uint32_t>& ubLocal32, uint32_t curTail,
                                                     uint32_t pe, uint32_t qpIdx, uint32_t sync_id);

/**
 * @brief RDMA Poll Completion Queue (CQ) function. Return status: 0 means success, non-zero means error.
 *
 * @tparam B                     RDMA Backend type
 * @param pe                     [in] PE number of the remote PE.
 * @param qpIdx                  [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param idx                    [in] expect completion queue consumer index after polling
 * @param ubLocal64              [in] temporary UB local tensor of uint64_t used as workspace
 * @param ubLocal32              [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\MTE3 Event.
 */
template<AclShmemRdmaBackend B>
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_poll_cq(uint32_t pe, uint32_t qpIdx, uint32_t idx,
                                                AscendC::LocalTensor<uint64_t>& ubLocal64,
                                                AscendC::LocalTensor<uint32_t>& ubLocal32, uint32_t sync_id);

/**
 * @brief Asynchronous RDMA Write function.
 *
 * @tparam B                     RDMA Backend type
 * @param dst                    [in] destination address in remote HBM
 * @param src                    [in] source address in local HBM
 * @param pe                     [in] PE number of the remote PE.
 * @param qpIdx                  [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param messageLen             [in] message length in Bytes
 * @param ubLocal64              [in] temporary UB local tensor of uint64_t used as workspace
 * @param ubLocal32              [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\\MTE3 Event.
 */
template<typename T, AclShmemRdmaBackend B>
ACLSHMEM_DEVICE void aclshmemi_roce_write(__gm__ T* dst, __gm__ T* src, 
                                          uint32_t pe, uint32_t qpIdx, uint64_t messageLen,
                                          AscendC::LocalTensor<uint64_t>& ubLocal64,
                                          AscendC::LocalTensor<uint32_t>& ubLocal32, uint32_t sync_id)
{
    aclshmemi_roce_post_send<B, AclShmemRdmaOpcode::OP_RDMA_WRITE>(
        dst, src, pe, qpIdx, messageLen, ubLocal64, ubLocal32, sync_id);
}

/**
 * @brief Asynchronous RDMA READ function.
 *
 * @tparam B                     RDMA Backend type
 * @param dst                    [in] destination address in local HBM
 * @param src                    [in] source address in remote HBM
 * @param pe                     [in] PE number of the remote PE.
 * @param qpIdx                  [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param messageLen             [in] message length in Bytes
 * @param ubLocal64              [in] temporary UB local tensor of uint64_t used as workspace
 * @param ubLocal32              [in] temporary UB local tensor of uint32_t used as workspace
 * @param sync_id                [in] ID used to Sync S\\MTE3 Event.
 */
template<typename T, AclShmemRdmaBackend B>
ACLSHMEM_DEVICE void aclshmemi_roce_read(__gm__ T* dst, __gm__ T* src,
                                         uint32_t pe, uint32_t qpIdx, uint64_t messageLen,
                                         AscendC::LocalTensor<uint64_t>& ubLocal64,
                                         AscendC::LocalTensor<uint32_t>& ubLocal32, uint32_t sync_id)
{
    // READ semantics require swapping the positions of dst and src here
    aclshmemi_roce_post_send<B, AclShmemRdmaOpcode::OP_RDMA_READ>(
        src, dst, pe, qpIdx, messageLen, ubLocal64, ubLocal32, sync_id);
}

#endif  // RDMA_DEVICE_BACKEND_BASE_H