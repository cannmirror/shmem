/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_RDMA_HPP
#define SHMEM_DEVICE_RDMA_HPP

#include "kernel_operator.h"
#include "device/shmem_def.h"
#include "shmemi_device_rdma.h"
#include "rdma_backends/rdma_device_backend_base.h"

// Decide Current RDMA Backend
#include "rdma_backends/rdma_device_backend_inDie.hpp"
#define K_RDMA_BACKEND (AclShmemRdmaBackend::inDie)

ACLSHMEM_DEVICE __gm__ ACLSHMEMRDMAInfo* aclshmemi_qp_info_fetch()
{
    __gm__ ACLSHMEMRDMAInfo* RDMAInfo = (__gm__ ACLSHMEMRDMAInfo*)(aclshmemi_get_qp_info_address(0));
    return RDMAInfo;
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemi_roce_write(__gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qpIdx,
                                          uint64_t messageLen, AscendC::LocalTensor<uint64_t> ubLocal64,
                                          AscendC::LocalTensor<uint32_t> ubLocal32, uint32_t sync_id)
{
    aclshmemi_roce_write<T, K_RDMA_BACKEND>(dst, src, pe, qpIdx, messageLen, ubLocal64, ubLocal32, sync_id);
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemi_roce_read(__gm__ T* dst, __gm__ T* src, uint32_t pe, uint32_t qpIdx,
                                         uint64_t messageLen, AscendC::LocalTensor<uint64_t> ubLocal64,
                                         AscendC::LocalTensor<uint32_t> ubLocal32, uint32_t sync_id)
{
    aclshmemi_roce_read<T, K_RDMA_BACKEND>(dst, src, pe, qpIdx, messageLen, ubLocal64, ubLocal32, sync_id);
}

ACLSHMEM_DEVICE void aclshmemi_roce_quiet(uint32_t pe, uint32_t qpIdx, AscendC::LocalTensor<uint64_t> ubLocal64,
                                          AscendC::LocalTensor<uint32_t> ubLocal32, uint32_t sync_id)
{
    __gm__ ACLSHMEMRDMAInfo* RDMAInfo = aclshmemi_qp_info_fetch();

    uint32_t qpNum = RDMAInfo->qpNum;
    __gm__ ACLSHMEMWQCtx* qpCtxEntry =
        (__gm__ ACLSHMEMWQCtx*)(RDMAInfo->sqPtr + (pe * qpNum + qpIdx) * sizeof(ACLSHMEMWQCtx));
    auto curHardwareHeadAddr = qpCtxEntry->headAddr;
    dcci_cachelines((__gm__ uint8_t*)curHardwareHeadAddr, 8);
    uint32_t curHead = *(__gm__ uint32_t*)(curHardwareHeadAddr);
    aclshmemi_roce_poll_cq<K_RDMA_BACKEND>(pe, qpIdx, curHead, ubLocal64, ubLocal32, sync_id);
}

ACLSHMEM_DEVICE __gm__ void* aclshmem_roce_ptr(__gm__ void* ptr, int pe)
{
    // Get Global State
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();

    // Back to root address
    uint64_t offset = reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(device_state->heap_base);
    uint64_t remote_ptr = reinterpret_cast<uint64_t>(device_state->p2p_device_heap_base[pe]) + offset;

    return reinterpret_cast<__gm__ void*>(remote_ptr);
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe)
{
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    uint32_t sync_id = device_state->rdma_config.sync_id;
    auto ptr = aclshmem_ptr(src, pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_read((__gm__ uint8_t*)dst, (__gm__ uint8_t*)ptr, pe, 0, elem_size * sizeof(T), ub_tensor_64,
                        ub_tensor_32, sync_id);
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe,
                                            uint32_t sync_id)
{
    auto ptr = aclshmem_ptr(src, pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_read((__gm__ uint8_t*)dst, (__gm__ uint8_t*)ptr, pe, 0, elem_size * sizeof(T), ub_tensor_64,
                        ub_tensor_32, sync_id);
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                            AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe)
{
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    uint32_t sync_id = device_state->rdma_config.sync_id;
    auto ptr = aclshmem_ptr((__gm__ void*)src.GetPhyAddr(), pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr()) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_read((__gm__ uint8_t*)dst.GetPhyAddr(), (__gm__ uint8_t*)ptr, pe, 0, elem_size * sizeof(T),
                        ub_tensor_64, ub_tensor_32, sync_id);
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                            AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe, uint32_t sync_id)
{
    auto ptr = aclshmem_ptr((__gm__ void*)src.GetPhyAddr(), pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr()) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_read((__gm__ uint8_t*)dst.GetPhyAddr(), (__gm__ uint8_t*)ptr, pe, 0, elem_size * sizeof(T),
                        ub_tensor_64, ub_tensor_32, sync_id);
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe)
{
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    uint32_t sync_id = device_state->rdma_config.sync_id;
    auto ptr = aclshmem_ptr(dst, pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_write((__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, pe, 0, elem_size * sizeof(T), ub_tensor_64,
                         ub_tensor_32, sync_id);
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe,
                                            uint32_t sync_id)
{
    auto ptr = aclshmem_ptr(dst, pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_write((__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, pe, 0, elem_size * sizeof(T), ub_tensor_64,
                         ub_tensor_32, sync_id);
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                            AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe)
{
    __gm__ aclshmem_device_host_state_t* device_state = aclshmemi_get_state();
    uint32_t sync_id = device_state->rdma_config.sync_id;
    auto ptr = aclshmem_ptr((__gm__ void*)dst.GetPhyAddr(), pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr()) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_write((__gm__ uint8_t*)ptr, (__gm__ uint8_t*)(src.GetPhyAddr()), pe, 0, elem_size * sizeof(T),
                         ub_tensor_64, ub_tensor_32, sync_id);
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                            AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe, uint32_t sync_id)
{
    auto ptr = aclshmem_ptr((__gm__ void*)dst.GetPhyAddr(), pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr()) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_write((__gm__ uint8_t*)ptr, (__gm__ uint8_t*)(src.GetPhyAddr()), pe, 0, elem_size * sizeof(T),
                         ub_tensor_64, ub_tensor_32, sync_id);
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_quiet(uint32_t pe, __ubuf__ T* buf, uint32_t sync_id)
{
    __gm__ ACLSHMEMRDMAInfo* RDMAInfo = aclshmemi_qp_info_fetch();
    uint32_t qpNum = RDMAInfo->qpNum;

    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;

    for (uint32_t qpIdx = 0; qpIdx < qpNum; qpIdx++) {
        __gm__ ACLSHMEMWQCtx* qpCtxEntry =
            (__gm__ ACLSHMEMWQCtx*)(RDMAInfo->sqPtr + (pe * qpNum + qpIdx) * sizeof(ACLSHMEMWQCtx));
        auto curHardwareHeadAddr = qpCtxEntry->headAddr;
        dcci_cachelines((__gm__ uint8_t*)curHardwareHeadAddr, 8);
        uint32_t curHead = *(__gm__ uint32_t*)(curHardwareHeadAddr);
        aclshmemi_roce_poll_cq<K_RDMA_BACKEND>(pe, qpIdx, curHead, ub_tensor_64, ub_tensor_32, sync_id);
    }
}

#endif