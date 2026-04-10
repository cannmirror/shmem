/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEM_RDMA_DEVICE_BACKEND_INDIE_HPP
#define ACLSHMEM_RDMA_DEVICE_BACKEND_INDIE_HPP

#include "rdma_device_backend_base.h"

// vendor-specfic header
struct ACLSHMEMwqeCtx {
    uint32_t byte4;
    uint32_t msgLen;
    uint32_t immtdata;
    uint32_t byte16;
    uint32_t byte20;
    uint32_t rkey;
    uint64_t va;
};

struct ACLSHMEMsegCtx {
    uint32_t len;
    uint32_t lkey;
    uint64_t addr;
};

struct ACLSHMEMcqeCtx {
    uint32_t byte4;
    uint32_t immtdata;
    uint32_t byte12;
    uint32_t byte16;
    uint32_t byteCnt;
    uint32_t smac;
    uint32_t byte28;
    uint32_t byte32;
};

enum class AclShmemRdmaInDieOpcode : uint32_t {
    OP_RDMA_WRITE = 3,
    OP_RDMA_WRITE_WITH_IMM,
    OP_RDMA_READ
};

// Shard Helper: WQE file for RDMA WRITE/READ, Returns the final size of WQE
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_fill_wqe_write_read(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint32_t pe, uint64_t messageLen,
    __gm__ uint8_t* wqeAddr, uint32_t curHead, AclShmemRdmaInDieOpcode opcode, uint64_t ACLSHMEMmemInfoTable)
{
    constexpr uint32_t shift = 13;
    // Write WQE to HBM
    uint64_t ownBit = (curHead >> shift) & 0x1;
    uint32_t byte4 = (uint32_t)opcode & 0x1F; // [0:4] opcode
    byte4 |= ((~ownBit) << 7) & (1 << 7); // [7] owner_bit
    byte4 |= 1 << 8; // [8] IBV_SEND_SINGNALED
    *(__gm__ uint32_t*)(wqeAddr) = byte4; // control set by local parameter, see above lines
    *(__gm__ uint32_t*)(wqeAddr + 4) = messageLen; // message size in bytes
    *(__gm__ uint32_t*)(wqeAddr + 8) = 0; // immtdata is always 0 till we provide poll CQ flow in AIV
    *(__gm__ uint32_t*)(wqeAddr + 12) = 1 << 24; // [120:127] num_sge = 1
    *(__gm__ uint32_t*)(wqeAddr + 16) = 0; // [128:151] start_sge_index = 0
    __gm__ ACLSHMEMmemInfo* remoteMemInfo =
        (__gm__ ACLSHMEMmemInfo*)(ACLSHMEMmemInfoTable + sizeof(ACLSHMEMmemInfo) * pe);
    *(__gm__ uint32_t*)(wqeAddr + 20) = remoteMemInfo->rkey; // rkey
    *(__gm__ uint64_t*)(wqeAddr + 24) = (uint64_t)dst; // remote VA

    // Write SGE to HBM
    __gm__ uint8_t* sgeAddr = wqeAddr + sizeof(ACLSHMEMwqeCtx);
    *(__gm__ uint32_t*)(sgeAddr) = messageLen; // message size in bytes
    __gm__ ACLSHMEMmemInfo* localMemInfo =
        (__gm__ ACLSHMEMmemInfo*)(ACLSHMEMmemInfoTable + sizeof(ACLSHMEMmemInfo) * aclshmemi_get_my_pe());
    *(__gm__ uint32_t*)(sgeAddr + 4) = localMemInfo->lkey; // lkey
    *(__gm__ uint64_t*)(sgeAddr + 8) = (uint64_t)src; // local VA
    return sizeof(ACLSHMEMwqeCtx) + sizeof(ACLSHMEMsegCtx);
}

// Note: Here Map public AclShmemRdmaOpcode to vendor-specific Opcode here
template<>
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_fill_wqe<AclShmemRdmaBackend::inDie, AclShmemRdmaOpcode::OP_RDMA_WRITE>(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint32_t pe, uint32_t qpIdx,
    uint64_t messageLen, AscendC::LocalTensor<uint64_t>& ubLocal64, AscendC::LocalTensor<uint32_t>& ubLocal32,
    uint32_t sync_id, __gm__ uint8_t* wqeAddr, uint32_t curHead, uint64_t ACLSHMEMmemInfoTable)
{
    return aclshmemi_roce_fill_wqe_write_read(
        dst, src, pe, messageLen, wqeAddr, curHead, AclShmemRdmaInDieOpcode::OP_RDMA_WRITE, ACLSHMEMmemInfoTable);
}

// Note: Here Map public AclShmemRdmaOpcode to vendor-specific Opcode here
template<>
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_fill_wqe<AclShmemRdmaBackend::inDie, AclShmemRdmaOpcode::OP_RDMA_READ>(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, uint32_t pe, uint32_t qpIdx,
    uint64_t messageLen, AscendC::LocalTensor<uint64_t>& ubLocal64, AscendC::LocalTensor<uint32_t>& ubLocal32,
    uint32_t sync_id, __gm__ uint8_t* wqeAddr, uint32_t curHead, uint64_t ACLSHMEMmemInfoTable)
{
    return aclshmemi_roce_fill_wqe_write_read(
        dst, src, pe, messageLen, wqeAddr, curHead, AclShmemRdmaInDieOpcode::OP_RDMA_READ, ACLSHMEMmemInfoTable);
}

template<>
ACLSHMEM_DEVICE void aclshmemi_roce_ring_sq_doorbell<AclShmemRdmaBackend::inDie>(
    AscendC::LocalTensor<uint64_t>& ubLocal64, AscendC::LocalTensor<uint32_t>& ubLocal32, uint32_t curHead,
    __gm__ ACLSHMEMWQCtx*& qpCtxEntry, uint32_t sync_id)
{
    uint64_t doorBellInfo = 0;
    doorBellInfo |= qpCtxEntry->wqn; // [0:23] DB_TAG = qp_num
    doorBellInfo |= 0 << 24; // [24:27] DB_CMD = HNS_ROCE_V2_SQ_DB(0)
    doorBellInfo |= ((uint64_t)curHead % 65536) << 32; // [32:47] DB_PI = sq.head
    doorBellInfo |= (uint64_t)(qpCtxEntry->sl) << 48; // [48:50] DB_SL = qp.sl

    auto curHardwareHeadAddr = qpCtxEntry->headAddr;

    __gm__ uint64_t* doorBellAddr = (__gm__ uint64_t*)(qpCtxEntry->dbAddr);

    ubLocal64.SetValue(0, doorBellInfo);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::GlobalTensor<uint64_t> DBGlobalTensor;
    DBGlobalTensor.SetGlobalBuffer(doorBellAddr);
    AscendC::DataCopyExtParams copyParams{1, 1 * sizeof(uint64_t), 0, 0, 0};
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::DataCopyPad(DBGlobalTensor, ubLocal64, copyParams);

    ubLocal32.SetValue(0, (uint32_t)curHead);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::GlobalTensor<uint32_t> HeadGlobalTensor;
    HeadGlobalTensor.SetGlobalBuffer((__gm__ uint32_t*)curHardwareHeadAddr);
    AscendC::DataCopyExtParams copyParamsHead{1, 1 * sizeof(uint32_t), 0, 0, 0};
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::DataCopyPad(HeadGlobalTensor, ubLocal32, copyParamsHead);
    AscendC::PipeBarrier<PIPE_MTE3>();
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(sync_id);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(sync_id);
}

// Shard Helper: Post-Send orchestration for RDMA WRITE/READ
template<AclShmemRdmaOpcode OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_roce_post_send_read_write(
    __gm__ uint8_t* dst, __gm__ uint8_t* src,
    uint32_t pe, uint32_t qpIdx, uint64_t messageLen,
    AscendC::LocalTensor<uint64_t>& ubLocal64, AscendC::LocalTensor<uint32_t>& ubLocal32, uint32_t sync_id)
{
    __gm__ ACLSHMEMRDMAInfo* RDMAInfo = aclshmemi_qp_info_fetch();

    uint32_t qpNum = RDMAInfo->qpNum;
    __gm__ ACLSHMEMWQCtx* qpCtxEntry =
        (__gm__ ACLSHMEMWQCtx*)(RDMAInfo->sqPtr + (pe * qpNum + qpIdx) * sizeof(ACLSHMEMWQCtx));
    auto ACLSHMEMmemInfoTable = RDMAInfo->memPtr;
    auto sqBaseAddr = qpCtxEntry->bufAddr;
    auto wqeSize = qpCtxEntry->wqeSize;
    auto curHardwareHeadAddr = qpCtxEntry->headAddr;
    dcci_cachelines((__gm__ uint8_t*)curHardwareHeadAddr, 8);
    uint32_t curHead = *(__gm__ uint32_t*)(curHardwareHeadAddr);
    auto curHardwareTailAddr = qpCtxEntry->tailAddr;
    auto depth = qpCtxEntry->depth;
    uint32_t wqeTotalSize = 0;

    // Poll CQ if send queue is full
    dcci_cachelines((__gm__ uint8_t*)curHardwareTailAddr, 8);
    if ((curHead + 10) % depth == (*(__gm__ uint32_t*)(curHardwareTailAddr)) % depth) {
        aclshmemi_roce_poll_cq<AclShmemRdmaBackend::inDie>(pe, qpIdx,
            *(__gm__ uint32_t*)(curHardwareTailAddr) + ACLSHMEM_NUM_CQE_PER_POLL_CQ, ubLocal64, ubLocal32, sync_id);
    }

    __gm__ uint8_t* wqeAddr = (__gm__ uint8_t*)(sqBaseAddr + wqeSize * (curHead % depth));

    // Write WQE to HBM
    wqeTotalSize = aclshmemi_roce_fill_wqe<AclShmemRdmaBackend::inDie, OP_CODE>(dst, src, 
        pe, qpIdx, messageLen, ubLocal64, ubLocal32, sync_id, wqeAddr, curHead, ACLSHMEMmemInfoTable);

#ifdef ENABLE_ASCENDC_DUMP
    if (wqeTotalSize == 0) {
        AscendC::PRINTF("Invalid Opcode %u for backend %u\n", (uint32_t)OP_CODE, (uint32_t)AclShmemRdmaBackend::inDie);
    }
#endif // ENABLE_ASCENDC_DUMP

    // WQE & SGE cache flush
    dcci_cachelines(wqeAddr, wqeTotalSize);
    curHead++;

    aclshmemi_roce_ring_sq_doorbell<AclShmemRdmaBackend::inDie>(ubLocal64, ubLocal32, curHead, qpCtxEntry, sync_id);
}

template<>
ACLSHMEM_DEVICE void aclshmemi_roce_post_send<AclShmemRdmaBackend::inDie, AclShmemRdmaOpcode::OP_RDMA_WRITE>(
    __gm__ uint8_t* dst, __gm__ uint8_t* src,
    uint32_t pe, uint32_t qpIdx, uint64_t messageLen,
    AscendC::LocalTensor<uint64_t>& ubLocal64, AscendC::LocalTensor<uint32_t>& ubLocal32, uint32_t sync_id)
{
    aclshmemi_roce_post_send_read_write<AclShmemRdmaOpcode::OP_RDMA_WRITE>(
        dst, src, pe, qpIdx, messageLen, ubLocal64, ubLocal32, sync_id);
}

template<>
ACLSHMEM_DEVICE void aclshmemi_roce_post_send<AclShmemRdmaBackend::inDie, AclShmemRdmaOpcode::OP_RDMA_READ>(
    __gm__ uint8_t* dst, __gm__ uint8_t* src,
    uint32_t pe, uint32_t qpIdx, uint64_t messageLen,
    AscendC::LocalTensor<uint64_t>& ubLocal64, AscendC::LocalTensor<uint32_t>& ubLocal32, uint32_t sync_id)
{
    aclshmemi_roce_post_send_read_write<AclShmemRdmaOpcode::OP_RDMA_READ>(
        dst, src, pe, qpIdx, messageLen, ubLocal64, ubLocal32, sync_id);
}

template<>
ACLSHMEM_DEVICE void aclshmemi_roce_ring_cq_doorbell<AclShmemRdmaBackend::inDie>(
    AscendC::LocalTensor<uint64_t>& ubLocal64, AscendC::LocalTensor<uint32_t>& ubLocal32, uint32_t curTail,
    uint32_t pe, uint32_t qpIdx, uint32_t sync_id)
{
    __gm__ ACLSHMEMRDMAInfo* RDMAInfo = aclshmemi_qp_info_fetch();

    uint32_t qpNum = RDMAInfo->qpNum;
    __gm__ ACLSHMEMCQCtx* cqCtxEntry =
        (__gm__ ACLSHMEMCQCtx*)(RDMAInfo->scqPtr + (pe * qpNum + qpIdx) * sizeof(ACLSHMEMCQCtx));
    AscendC::DataCopyExtParams copyParamsTail{1, 1 * sizeof(uint32_t), 0, 0, 0};

    // Ring CQ Doorbell
    auto cqDBAddr = cqCtxEntry->dbAddr;
    if (cqCtxEntry->dbMode == ACLSHMEMDBMode::SW_DB) {
        ubLocal32.SetValue(0, (uint32_t)(curTail & 0xFFFFFF));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(sync_id);
        AscendC::GlobalTensor<uint32_t> CQDBGlobalTensor;
        CQDBGlobalTensor.SetGlobalBuffer((__gm__ uint32_t*)cqDBAddr);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(sync_id);
        AscendC::DataCopyPad(CQDBGlobalTensor, ubLocal32, copyParamsTail);
    } else if (cqCtxEntry->dbMode == ACLSHMEMDBMode::HW_DB) {
        uint64_t doorBellInfo = 0;
        doorBellInfo |= cqCtxEntry->cqn; // [0:23] DB_TAG = qp_num
        doorBellInfo |= 3 << 24; // [24:27] DB_CMD = HNS_ROCE_V2_CQ_DB_PTR(3)
        doorBellInfo |= (uint64_t)(curTail & 0xFFFFFF) << 32; // [32:55] DB_CQ_CI = cq.tail
        doorBellInfo |= (uint64_t)1 << 56; // [56:56] DB_CQ_CMD_SN = 1
        ubLocal64.SetValue(0, doorBellInfo);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(sync_id);
        AscendC::GlobalTensor<uint64_t> DBGlobalTensor;
        DBGlobalTensor.SetGlobalBuffer((__gm__ uint64_t*)cqDBAddr);
        AscendC::DataCopyExtParams copyParams{1, 1 * sizeof(uint64_t), 0, 0, 0};
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(sync_id);
        AscendC::DataCopyPad(DBGlobalTensor, ubLocal64, copyParams);
    }

    // Update WQ tail
    __gm__ ACLSHMEMWQCtx* wqCtxEntry =
        (__gm__ ACLSHMEMWQCtx*)(RDMAInfo->sqPtr + (pe * qpNum + qpIdx) * sizeof(ACLSHMEMWQCtx));
    auto curWQTailAddr = wqCtxEntry->tailAddr;
    uint32_t curWQTail = *(__gm__ uint32_t*)(curWQTailAddr);
    ubLocal32.SetValue(0, curTail);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::GlobalTensor<uint32_t> WQTailGlobalTensor;
    WQTailGlobalTensor.SetGlobalBuffer((__gm__ uint32_t*)curWQTailAddr);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(sync_id);
    AscendC::DataCopyPad(WQTailGlobalTensor, ubLocal32, copyParamsTail);
    AscendC::PipeBarrier<PIPE_MTE3>();
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(sync_id);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(sync_id);
}

template<>
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_poll_cq<AclShmemRdmaBackend::inDie>(
    uint32_t pe, uint32_t qpIdx, uint32_t idx,
    AscendC::LocalTensor<uint64_t>& ubLocal64,
    AscendC::LocalTensor<uint32_t>& ubLocal32, uint32_t sync_id)
{
    if (idx == 0) {
        return 0;
    }
    __gm__ ACLSHMEMRDMAInfo* RDMAInfo = aclshmemi_qp_info_fetch();

    uint32_t qpNum = RDMAInfo->qpNum;
    __gm__ ACLSHMEMCQCtx* cqCtxEntry =
        (__gm__ ACLSHMEMCQCtx*)(RDMAInfo->scqPtr + (pe * qpNum + qpIdx) * sizeof(ACLSHMEMCQCtx));
    auto cqBaseAddr = cqCtxEntry->bufAddr;
    auto cqeSize = cqCtxEntry->cqeSize;
    auto depth = cqCtxEntry->depth;
    auto curHardwareTailAddr = cqCtxEntry->tailAddr;
    dcci_cachelines((__gm__ uint8_t*)curHardwareTailAddr, 8);
    uint32_t curTail = *(__gm__ uint32_t*)(curHardwareTailAddr);

    const uint32_t shiftWidth = 7;
    AscendC::DataCopyExtParams copyParamsTail{1, 1 * sizeof(uint32_t), 0, 0, 0};
    while (curTail != idx) {
        __gm__ ACLSHMEMcqeCtx* cqeAddr = (__gm__ ACLSHMEMcqeCtx*)(cqBaseAddr + cqeSize * (curTail & (depth - 1)));
        uint32_t cqeByte4 = *(__gm__ uint32_t*)cqeAddr;
        while (((cqeByte4 & (1 << shiftWidth)) != 0) == ((curTail & depth) != 0)) {
            int64_t tmp = AscendC::GetSystemCycle(); // reserved for timeout check
            dcci_cachelines((__gm__ uint8_t*)cqeAddr, 32);
            cqeByte4 = *(__gm__ uint32_t*)cqeAddr;
        }
        curTail++;
        uint32_t wqn = cqeAddr->byte16 & 0xFFFFFF; // reserved for multi WQ share the same CQ

        // Check CQE status
        uint32_t status = (cqeAddr->byte4 >> 8) & 0xFF;
        if (status) {
            return status;
        }
    }

    // Update CQ tail
    ubLocal32.SetValue(0, (uint32_t)curTail);
    AscendC::GlobalTensor<uint32_t> TailGlobalTensor;
    TailGlobalTensor.SetGlobalBuffer((__gm__ uint32_t*)curHardwareTailAddr);
    AscendC::DataCopyPad(TailGlobalTensor, ubLocal32, copyParamsTail);

    aclshmemi_roce_ring_cq_doorbell<AclShmemRdmaBackend::inDie>(ubLocal64, ubLocal32, curTail, pe, qpIdx, sync_id);
    return 0;
}

#endif  // ACLSHMEM_RDMA_DEVICE_BACKEND_INDIE_HPP