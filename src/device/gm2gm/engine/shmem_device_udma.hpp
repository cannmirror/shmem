/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef SHMEM_DEVICE_UDMA_HPP
#define SHMEM_DEVICE_UDMA_HPP

#include "kernel_operator.h"
#include "device/shmem_def.h"
#include "shmemi_device_common.h"
#include "shmemi_device_udma.h"
#include "utils/shmemi_kernel_debug.h"
#include "../host_device/shmemi_host_device_constant.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#define ACLSHMEM_UDMA_SUPPORTED 1
#else
#define ACLSHMEM_UDMA_SUPPORTED 0
#endif

constexpr uint32_t MAX_RETRY_TIMES = 1000000;

ACLSHMEM_DEVICE void aclshmemi_dump_sge(__gm__ uint8_t* wqeAddr, uint32_t sge_num);
ACLSHMEM_DEVICE void aclshmemi_dump_wqe(__gm__ uint8_t* wqeAddr, uint32_t atomicLen);

ACLSHMEM_DEVICE __gm__ ACLSHMEMAIVUDMAInfo* aclshmemi_udma_qp_info_fetch()
{
    __gm__ ACLSHMEMAIVUDMAInfo* udmaInfo = (__gm__ ACLSHMEMAIVUDMAInfo*)(aclshmemi_get_qp_info_address(0));
    return udmaInfo;
}

ACLSHMEM_DEVICE void aclshmemi_dump_cqe(__gm__ ACLSHMEMJfcCqeCtx* cqeAddr)
{
    if (cqeAddr == nullptr) {
        AscendC::printf("CQE: nullptr pointer\n");
        return;
    }
    uint32_t sR = cqeAddr->sR;
    uint32_t isJetty = cqeAddr->isJetty;
    uint32_t owner = cqeAddr->owner;
    uint32_t inlineEn = cqeAddr->inlineEn;
    uint32_t opcode = cqeAddr->opcode;
    uint32_t fd = cqeAddr->fd;
    uint32_t substatus = cqeAddr->substatus;
    uint32_t status = cqeAddr->status;
    uint32_t entryIdx = cqeAddr->entryIdx;
    uint32_t localNumL = cqeAddr->localNumL;
    uint32_t localNumH = cqeAddr->localNumH;
    uint32_t rmtIdx = cqeAddr->rmtIdx;
    uint32_t tpn = cqeAddr->tpn;
    AscendC::printf(
        "CQE: DW0 - sR: %d, isJetty: %d, owner: %d, inlineEn: %d, opcode: %d, fd: %d, substatus: %d, status: %d\n", sR,
        isJetty, owner, inlineEn, opcode, fd, substatus, status);
    AscendC::printf("CQE: DW1 - entryIdx: %d, localNumL: %d\n", entryIdx, localNumL);
    AscendC::printf("CQE: DW2 - localNumH: %d, rmtIdx: %d\n", localNumH, rmtIdx);
    AscendC::printf("CQE: DW3 - tpn: %d\n", tpn);
    AscendC::printf("CQE: DW4 - byteCnt: %d\n", cqeAddr->byteCnt);
    AscendC::printf("CQE: DW5-DW6 - userData: 0x%x%x\n", cqeAddr->userDataH, cqeAddr->userDataL);
    AscendC::printf(
        "CQE: DW7-DW10 - rmtEid: [0x%x, 0x%x, 0x%x, 0x%x]\n", cqeAddr->rmtEid[0], cqeAddr->rmtEid[1],
        cqeAddr->rmtEid[2], cqeAddr->rmtEid[3]);
    AscendC::printf("CQE: DW11-DW12 - data: 0x%x%x\n", cqeAddr->dataH, cqeAddr->dataL);
    AscendC::printf(
        "CQE: DW13-DW15 - inlineData: [0x%x, 0x%x, 0x%x]\n", cqeAddr->inlineData[0], cqeAddr->inlineData[1],
        cqeAddr->inlineData[2]);
}

ACLSHMEM_DEVICE uint32_t aclshmemi_udma_poll_cq(uint32_t pe, uint32_t qpIdx, uint32_t idx)
{
    if (idx == 0) {
        return 0;
    }
    __gm__ ACLSHMEMAIVUDMAInfo* udmaInfo = aclshmemi_udma_qp_info_fetch();
    uint32_t qpNum = udmaInfo->qpNum;
    __gm__ ACLSHMEMUDMACqCtx* cqCtxEntry =
        (__gm__ ACLSHMEMUDMACqCtx*)(udmaInfo->scqPtr + (pe * qpNum + qpIdx) * sizeof(ACLSHMEMUDMACqCtx));
    auto cqBaseAddr = cqCtxEntry->bufAddr;
    auto bbShift = cqCtxEntry->baseBkShift;
    auto cqeSize = 1 << bbShift;
    uint32_t curTail = cqCtxEntry->tail;
    while (curTail != idx) {
        __gm__ ACLSHMEMJfcCqeCtx* cqeAddr =
            (__gm__ ACLSHMEMJfcCqeCtx*)(cqBaseAddr + cqeSize * (curTail & (shm::UDMA_CQ_DEPTH_DEFAULT - 1)));
        bool validOwner = (curTail / shm::UDMA_CQ_DEPTH_DEFAULT) & 1;
        uint32_t times = 0;
        while ((validOwner ^ cqeAddr->owner) == 0 && times < MAX_RETRY_TIMES) { // util cqeAddr->owner changed
            dcci_cachelines((__gm__ uint8_t*)cqeAddr, sizeof(ACLSHMEMJfcCqeCtx));
            times++;
        }
        if (times >= MAX_RETRY_TIMES) {
            ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Poll cq timeout! curTail=%d idx=%d\n", curTail, idx);
            return 0xFF;
        }
        // Check CQE status
        uint8_t status = cqeAddr->status & 0xFF;
        uint8_t subStatus = cqeAddr->substatus & 0xFF;
        constexpr uint8_t statusShift = 8;
        if (status != 0 || subStatus != 0) {
            ACLSHMEM_DEBUG_FUNC(aclshmemi_dump_cqe, cqeAddr);
            return (status << statusShift) | subStatus;
        }
        curTail++;
    }

    // Update CQ tail
    cqCtxEntry->tail = curTail;
    __gm__ ACLSHMEMUDMAWQCtx* wqCtxEntry =
        (__gm__ ACLSHMEMUDMAWQCtx*)(udmaInfo->sqPtr + (pe * qpNum + qpIdx) * sizeof(ACLSHMEMUDMAWQCtx));
    aclshmemi_udma_poll_cq_update_info(curTail, qpIdx, cqCtxEntry, wqCtxEntry);
    return 0;
}

ACLSHMEM_DEVICE void aclshmemi_udma_poll_cq_update_info(
    uint32_t curTail, uint32_t qpIdx, __gm__ ACLSHMEMUDMACqCtx* cqCtxEntry, __gm__ ACLSHMEMUDMAWQCtx* wqCtxEntry)
{
    // Ring CQ Doorbell (reference URMA implementation)
    auto cqDBAddr = cqCtxEntry->dbAddr;
    // For JFC, we write the consumer index (curTail) directly
    __gm__ uint32_t* dbAddr = (__gm__ uint32_t*)cqCtxEntry->dbAddr;
    st_dev((uint32_t)(curTail & 0xFFFFFF), dbAddr, 0);
    // Update WQ tail
    wqCtxEntry->tail = curTail;
}

ACLSHMEM_DEVICE void aclshmemi_dump_wqe(__gm__ uint8_t* wqeAddr, uint32_t atomicLen)
{
    if (wqeAddr == nullptr) {
        AscendC::printf("WQE: nullptr pointer\n");
        return;
    }
    __gm__ ACLSHMEMSqeCtx* sqeCtx = (__gm__ ACLSHMEMSqeCtx*)wqeAddr;
    auto sqeBbIdx = sqeCtx->sqeBbIdx;
    auto flag = sqeCtx->flag;
    auto rsv0 = sqeCtx->rsv0;
    auto nf = sqeCtx->nf;
    auto tokenEn = sqeCtx->tokenEn;
    auto rmtJettyType = sqeCtx->rmtJettyType;
    AscendC::printf(
        "WQE: sqe_bb_idx: %x flag: %x rsv0: %x nf: %x token_en: %x rmt_jetty_type: %x\n", sqeBbIdx, flag, rsv0, nf,
        tokenEn, rmtJettyType);
    auto owner = sqeCtx->owner;
    auto targetHint = sqeCtx->targetHint;
    auto opcode = sqeCtx->opcode;
    auto rsv1 = sqeCtx->rsv1;
    auto inlineMsgLen = sqeCtx->inlineMsgLen;
    auto tpId = sqeCtx->tpId;
    AscendC::printf(
        "WQE: owner: %x target_hint: %x opcode: %x rsv1: %x inline_msg_len: %x tp_id: %x\n", owner, targetHint, opcode,
        rsv1, inlineMsgLen, tpId);
    auto sgeNum = sqeCtx->sgeNum;
    auto rmtJettyOrSegId = sqeCtx->rmtJettyOrSegId;
    auto rsv2 = sqeCtx->rsv2;
    AscendC::printf("WQE: sge_num: %x rmt_jetty_or_seg_id: %x rsv2: %x\n", sgeNum, rmtJettyOrSegId, rsv2);
    auto rmtEidL = sqeCtx->rmtEidL;
    auto rmtEidH = sqeCtx->rmtEidH;
    AscendC::printf("WQE: rmt_eid: %x, %x\n", rmtEidL, rmtEidH);
    auto rmtTokenValue = sqeCtx->rmtTokenValue;
    auto udfType = sqeCtx->udfType;
    auto reduceDataType = sqeCtx->reduceDataType;
    auto reduceOpcode = sqeCtx->reduceOpcode;
    auto rmtAddrLOrTokenId = sqeCtx->rmtAddrLOrTokenId;
    auto rmtAddrHOrTokenValue = sqeCtx->rmtAddrHOrTokenValue;
    AscendC::printf(
        "WQE: rmt_token_value: %x udf_type: %x reduce_data_type: %x reduce_opcode: %x\n", rmtTokenValue, udfType,
        reduceDataType, reduceOpcode);
    AscendC::printf(
        "WQE: rmt_addr_l_or_token_id: %x rmt_addr_h_or_token_value: %x\n", rmtAddrLOrTokenId, rmtAddrHOrTokenValue);
    if (opcode == static_cast<uint32_t>(aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY)) {
        __gm__ ACLSHMEMNotifyCtx* notifyCtx =
            (__gm__ ACLSHMEMNotifyCtx*)((__gm__ uint8_t*)sqeCtx + sizeof(ACLSHMEMSqeCtx));
        auto notifyTokenId = notifyCtx->notifyTokenId;
        auto notifyTokenValue = notifyCtx->notifyTokenValue;
        auto notifyAddrL = notifyCtx->notifyAddrL;
        auto notifyAddrH = notifyCtx->notifyAddrH;
        auto notifyDataL = notifyCtx->notifyDataL;
        auto notifyDataH = notifyCtx->notifyDataH;
        AscendC::printf(
            "WQE: notifyTokenId: %x notifyTokenValue: %x notifyAddrL: %x notifyAddrH: %x notifyDataL: %x "
            "notifyDataH: %x \n",
            notifyTokenId, notifyTokenValue, notifyAddrL, notifyAddrH, notifyDataL, notifyDataH);
    }
    aclshmemi_dump_sge(wqeAddr, sgeNum);
    if (opcode == static_cast<uint32_t>(aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA)) {
        __gm__ uint8_t* amoDataAddr = (__gm__ uint8_t*)sqeCtx + sizeof(ACLSHMEMSqeCtx) + sizeof(ACLSHMEMSgeCtx);
        uint64_t addData = (atomicLen == sizeof(uint32_t)) ? static_cast<uint64_t>(*(__gm__ uint32_t*)amoDataAddr) :
                                                             *(__gm__ uint64_t*)amoDataAddr;
        AscendC::printf("SGE: add_data: 0x%llx \n", (unsigned long long)addData);
    } else if (opcode == static_cast<uint32_t>(aclshmemi_udma_opcode_t::UDMA_OP_CAS)) {
        __gm__ uint8_t* amoDataAddr = (__gm__ uint8_t*)sqeCtx + sizeof(ACLSHMEMSqeCtx) + sizeof(ACLSHMEMSgeCtx);
        uint64_t swapData = (atomicLen == sizeof(uint32_t)) ? static_cast<uint64_t>(*(__gm__ uint32_t*)amoDataAddr) :
                                                              *(__gm__ uint64_t*)amoDataAddr;
        uint64_t condData = (atomicLen == sizeof(uint32_t)) ?
                                static_cast<uint64_t>(*(__gm__ uint32_t*)(amoDataAddr + atomicLen)) :
                                *(__gm__ uint64_t*)(amoDataAddr + atomicLen);
        AscendC::printf(
            "SGE: cond_data: 0x%llx, swap_data: 0x%llx\n", (unsigned long long)condData, (unsigned long long)swapData);
    }
}

ACLSHMEM_DEVICE void aclshmemi_dump_sge(__gm__ uint8_t* wqeAddr, uint32_t sge_num)
{
    if (wqeAddr == nullptr) {
        AscendC::printf("WQE: nullptr pointer\n");
        return;
    }
    __gm__ ACLSHMEMSgeCtx* sgeCtx = (__gm__ ACLSHMEMSgeCtx*)(wqeAddr + sizeof(ACLSHMEMSqeCtx));
    for (size_t i = 0; i < sge_num; i++) {
        auto sgeLen = sgeCtx->len;
        auto sgeRmtAddr = sgeCtx->va;
        AscendC::printf("SGE: sge idx: %d, va: %p sge_len: %d\n", i, sgeRmtAddr, sgeLen);
        sgeCtx++;
    }
}

template <aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE constexpr uint32_t get_wqe_bb_cnt()
{
    // 暂时不考虑inline功能，inline场景需要考虑inline data的大小
    if constexpr (
        OP_CODE == aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA || OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_CAS ||
        OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY) {
        return 2;
    } else {
        return 1;
    }
}

template <typename T, aclshmemi_udma_opcode_t OP_CODE, typename SQE_PTR>
ACLSHMEM_DEVICE void aclshmemi_fill_reduce(SQE_PTR sqeCtx)
{
    if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE) {
        static_assert(
            AscendC::IsSameType<SQE_PTR, __gm__ ACLSHMEMSqeCtx*>::value,
            "WRITE_WITH_REDUCE requires SQE in HBM (PIPE_S path); UB-staged SQE not supported");

        sqeCtx->udfType = 0;        // inline reduce
        sqeCtx->reduceOpcode = 0xa; // reduce add
        if constexpr (AscendC::IsSameType<T, float>::value) {
            sqeCtx->reduceDataType = 0x7; // fp32
        }
    }
}

template <typename T, aclshmemi_udma_opcode_t OP_CODE, typename SQE_PTR>
ACLSHMEM_DEVICE void aclshmemi_fill_notify_data(
    SQE_PTR sqeCtx, uint32_t tid, uint32_t tokenValue, const aclshmemi_udma_params_t<T, OP_CODE>& params)
{
    if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY) {
        // The notify ctx lives immediately after the SQE in the same address space.
        using NotifyPtr = typename AscendC::Std::conditional<
            AscendC::IsSameType<SQE_PTR, __ubuf__ ACLSHMEMSqeCtx*>::value, __ubuf__ ACLSHMEMNotifyCtx*,
            __gm__ ACLSHMEMNotifyCtx*>::type;
        using BytePtr = typename AscendC::Std::conditional<
            AscendC::IsSameType<SQE_PTR, __ubuf__ ACLSHMEMSqeCtx*>::value, __ubuf__ uint8_t*, __gm__ uint8_t*>::type;
        NotifyPtr notifyCtx = (NotifyPtr)((BytePtr)sqeCtx + sizeof(ACLSHMEMSqeCtx));
        notifyCtx->notifyTokenId = tid & 0xFFFFF; // 20 bits
        notifyCtx->notifyTokenValue = tokenValue;
        notifyCtx->notifyAddrL = reinterpret_cast<uint64_t>(params.sig_addr) & 0xFFFFFFFF;
        notifyCtx->notifyAddrH = (reinterpret_cast<uint64_t>(params.sig_addr) >> 32) & 0xFFFFFFFF;
        notifyCtx->notifyDataL = params.signal & 0xFFFFFFFF;
        notifyCtx->notifyDataH = (params.signal >> 32) & 0xFFFFFFFF;
    }
}

template <typename T, aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_fill_sge_ctx(
    __gm__ ACLSHMEMSgeCtx* sgeCtx, uint64_t messageLen, __gm__ uint8_t* localAddr, __gm__ ACLSHMEMUDMAWQCtx* qpCtxEntry,
    const aclshmemi_udma_params_t<T, OP_CODE>& params)
{
    // default
    sgeCtx->len = messageLen;
    if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA) { // fetch and add
        auto amoAddr = qpCtxEntry->amoAddr;
        sgeCtx->va = amoAddr;
        __gm__ T* addDataAddr = (__gm__ T*)((__gm__ uint8_t*)sgeCtx + sizeof(ACLSHMEMSgeCtx));
        *addDataAddr = params.value;                                        // fill in add_data
    } else if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_CAS) { // compare and swap
        auto amoAddr = qpCtxEntry->amoAddr;
        sgeCtx->va = amoAddr;
        __gm__ T* swapDataAddr = (__gm__ T*)((__gm__ uint8_t*)sgeCtx + sizeof(ACLSHMEMSgeCtx));
        *swapDataAddr = params.value; // fill in swap_data
        __gm__ T* cmpDataAddr = (__gm__ T*)((__gm__ uint8_t*)swapDataAddr + sizeof(T));
        *cmpDataAddr = params.cond; // fill in cmp_data
    } else if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE) {
        auto amoAddr = qpCtxEntry->amoAddr;
        *(__gm__ T*)amoAddr = params.value;
        dcci_cachelines((__gm__ uint8_t*)amoAddr, sizeof(T));
        sgeCtx->va = amoAddr;
    } else {
        sgeCtx->va = reinterpret_cast<uint64_t>(localAddr);
    }
}

template <aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE __gm__ uint8_t* aclshmemi_udma_get_sge_ctx(__gm__ uint8_t* wqe_addr)
{
    if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY) {
        constexpr size_t offset = sizeof(ACLSHMEMSqeCtx) + sizeof(ACLSHMEMNotifyCtx);
        return (wqe_addr + offset);
    } else {
        return (wqe_addr + sizeof(ACLSHMEMSqeCtx));
    }
}

ACLSHMEM_DEVICE void poll_cq_when_sq_overflow(
    __gm__ ACLSHMEMUDMAWQCtx* qpCtxEntry, uint32_t wqeCnt, uint32_t pe, uint32_t qpIdx)
{
    // Poll CQ if send queue is full
    constexpr uint32_t POLL_CQ_THRESHOLD = 10;
    uint32_t curTail = qpCtxEntry->tail;
    if ((wqeCnt + POLL_CQ_THRESHOLD) % shm::UDMA_SQ_BASKBLK_CNT == (curTail) % shm::UDMA_SQ_BASKBLK_CNT) {
        uint32_t idx =
            (curTail + ACLSHMEM_NUM_CQE_PER_POLL_CQ) > wqeCnt ? wqeCnt : curTail + ACLSHMEM_NUM_CQE_PER_POLL_CQ;
        (void)aclshmemi_udma_poll_cq(pe, qpIdx, idx);
    }
}

template <typename T, aclshmemi_udma_opcode_t OP_CODE, typename SQE_PTR = __gm__ ACLSHMEMSqeCtx*>
ACLSHMEM_DEVICE void aclshmemi_udma_fill_sqe_ctx(
    SQE_PTR sqeCtx, __gm__ uint8_t* remoteAddr, __gm__ ACLSHMEMUBmemInfo* remoteMemInfo, uint32_t curHead,
    const aclshmemi_udma_params_t<T, OP_CODE>& params)
{
    // Fill SQE control information (reference udma_fill_write_sqe logic).
    // Templated on SQE_PTR so the same fill logic targets either HBM (PIPE_S path)
    // or UB scratch (PIPE_MTE3 path).
    if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE) {
        sqeCtx->opcode = static_cast<uint32_t>(aclshmemi_udma_opcode_t::UDMA_OP_WRITE);
        // udf_flag = 1 for write with reduce; last 2 bits = 00 (NO, no ordering) for higher throughput.
        sqeCtx->flag = 0b10100000;
    } else {
        sqeCtx->opcode = static_cast<uint32_t>(OP_CODE);
        // Last 2 bits = 00 (NO, no ordering) for higher throughput.
        // Note: aclshmemx_udma_quiet only guarantees completion of submitted WQEs,
        // it does NOT restore inter-submission ordering — the framework currently
        // does not provide a strong-ordering primitive for this path.
        sqeCtx->flag = 0b00100000;
    }
    sqeCtx->nf = 0; // Need fence
    sqeCtx->tokenEn = remoteMemInfo->tokenValueValid;
    sqeCtx->rmtJettyType = remoteMemInfo->rmtJettyType;
    sqeCtx->owner = (curHead & shm::UDMA_SQ_BASKBLK_CNT) == 0 ? 1 : 0; // depth: baseblk_cnt
    sqeCtx->targetHint = remoteMemInfo->targetHint;
    sqeCtx->inlineMsgLen = 0; // No inline data
    sqeCtx->tpId = remoteMemInfo->tpn;
    sqeCtx->sgeNum = 1; // Single SGE
    sqeCtx->rmtJettyOrSegId = remoteMemInfo->tid;
    sqeCtx->rmtTokenValue = remoteMemInfo->rmtTokenValue;
    aclshmemi_fill_reduce<T, OP_CODE>(sqeCtx);
    // Set remote address (reference udma_fill_write_sqe logic)
    uint64_t remoteAddrValue = reinterpret_cast<uint64_t>(remoteAddr);
    sqeCtx->rmtAddrLOrTokenId = remoteAddrValue & 0xFFFFFFFF;
    sqeCtx->rmtAddrHOrTokenValue = (remoteAddrValue >> 32) & 0xFFFFFFFF;
    auto rmtEid = (__gm__ uint64_t*)(remoteMemInfo->eidAddr);
    sqeCtx->rmtEidL = rmtEid[0];
    sqeCtx->rmtEidH = rmtEid[1];
    aclshmemi_fill_notify_data<T, OP_CODE>(sqeCtx, remoteMemInfo->tid, remoteMemInfo->rmtTokenValue, params);
}

ACLSHMEM_DEVICE __gm__ ACLSHMEMUDMAWQCtx* aclshmemi_udma_get_qp_ctx(
    __gm__ ACLSHMEMAIVUDMAInfo* udma_info, uint32_t pe, uint32_t qp_idx)
{
    uint32_t qp_num = udma_info->qpNum;
    __gm__ ACLSHMEMUDMAWQCtx* qp_ctx_entry =
        (__gm__ ACLSHMEMUDMAWQCtx*)(udma_info->sqPtr + (pe * qp_num + qp_idx) * sizeof(ACLSHMEMUDMAWQCtx));
    return qp_ctx_entry;
}

ACLSHMEM_DEVICE __gm__ ACLSHMEMSqeCtx* aclshmemi_udma_get_sqe_ctx(
    __gm__ ACLSHMEMUDMAWQCtx* qp_ctx_entry, uint32_t cur_head, uint32_t wqe_size)
{
    auto sq_base_addr = qp_ctx_entry->bufAddr;
    __gm__ uint8_t* wqe_addr = (__gm__ uint8_t*)(sq_base_addr + wqe_size * (cur_head % shm::UDMA_SQ_BASKBLK_CNT));
    __gm__ ACLSHMEMSqeCtx* sqe_ctx = (__gm__ ACLSHMEMSqeCtx*)wqe_addr;
    return sqe_ctx;
}

ACLSHMEM_DEVICE void assert_qp_params_valid(__gm__ ACLSHMEMUDMAWQCtx* qpCtxEntry)
{
    auto depth = qpCtxEntry->depth;
    bool valid = (shm::UDMA_SQ_BASKBLK_CNT == depth);
    if (!valid) {
        AscendC::printf("sq depth [%d] is not equal with baskblkcnt [%d].\n", depth, shm::UDMA_SQ_BASKBLK_CNT);
        trap();
    }
}

ACLSHMEM_DEVICE void assert_not_self_send(uint32_t pe)
{
    if (pe == static_cast<uint32_t>(aclshmem_my_pe())) {
        AscendC::printf("udma_post_send: pe(%d) == my_pe, self-send not allowed\n", pe);
        trap();
    }
}

// Converge the UDMA slot index computation into a single inline helper. The signature is
// identical for both builds -- the (pe, relay_pe) parameters are unconditionally present, no
// #if on the parameter list. Only the function body is compile-time gated:
//   * OFF (direct path, original v1.5.0 behavior): slot == pe. relay_pe is ignored, and neither
//     aclshmemi_get_my_pe nor aclshmemi_get_total_pe is read -- zero extra work on the hot path.
//   * ON (relay path): slot == pe * rankCount + actualRelayPe, where actualRelayPe falls back to
//     myPe when relay_pe == -1.
ACLSHMEM_DEVICE uint32_t aclshmemi_udma_compute_slot(uint32_t pe, uint32_t relay_pe = static_cast<uint32_t>(-1))
{
#if defined(ACLSHMEM_RELAY_SUPPORT)
    uint32_t myPe = static_cast<uint32_t>(aclshmemi_get_my_pe());
    uint32_t rankCount = static_cast<uint32_t>(aclshmemi_get_total_pe());
    uint32_t actualRelayPe = (relay_pe == static_cast<uint32_t>(-1)) ? myPe : relay_pe;
    return pe * rankCount + actualRelayPe;
#else
    (void)relay_pe; // direct path: ignore relay_pe, no extra computation
    return pe;
#endif
}

template <typename T, aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_udma_post_send(
    __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr, uint32_t pe, uint32_t qpIdx, uint64_t messageLen,
    const aclshmemi_udma_params_t<T, OP_CODE>& params, uint32_t relay_pe)
{
    __gm__ ACLSHMEMAIVUDMAInfo* udmaInfo = aclshmemi_udma_qp_info_fetch();
    ACLSHMEM_DEBUG_FUNC(assert_not_self_send, pe);
    // Unified slot computation for both builds (call site is identical, no #if here). OFF returns
    // pe (direct path); ON returns pe*N+actualRelayPe. Direct/OFF callers pass relay_pe == -1,
    // which compute_slot ignores while returning pe.
    uint32_t slot = aclshmemi_udma_compute_slot(pe, relay_pe);
    __gm__ ACLSHMEMUDMAWQCtx* qpCtxEntry = aclshmemi_udma_get_qp_ctx(udmaInfo, slot, qpIdx);
    auto wqeSize = 1 << qpCtxEntry->baseBkShift; // basebk_shift
    uint32_t curHead = qpCtxEntry->head;
    ACLSHMEM_DEBUG_FUNC(assert_qp_params_valid, qpCtxEntry);
    uint32_t wqeCnt = qpCtxEntry->wqeCnt;
    // Poll CQ if send queue is full
    poll_cq_when_sq_overflow(qpCtxEntry, wqeCnt, slot, qpIdx);
    // Get memory info for this (actualPe, relayPe) slot
    __gm__ ACLSHMEMUBmemInfo* remoteMemInfo =
        (__gm__ ACLSHMEMUBmemInfo*)(udmaInfo->memPtr + sizeof(ACLSHMEMUBmemInfo) * slot);
    // Write SQE to HBM
    __gm__ ACLSHMEMSqeCtx* sqeCtx = aclshmemi_udma_get_sqe_ctx(qpCtxEntry, curHead, wqeSize);
    aclshmemi_udma_fill_sqe_ctx<T, OP_CODE>(sqeCtx, remoteAddr, remoteMemInfo, curHead, params);
    // Write SGE to HBM
    __gm__ ACLSHMEMSgeCtx* sgeCtx =
        (__gm__ ACLSHMEMSgeCtx*)(aclshmemi_udma_get_sge_ctx<OP_CODE>((__gm__ uint8_t*)sqeCtx));
    aclshmemi_fill_sge_ctx<T, OP_CODE>(sgeCtx, messageLen, localAddr, qpCtxEntry, params);
    // WQE & SGE cache flush
    constexpr uint32_t wqeBbCnt = get_wqe_bb_cnt<OP_CODE>();
    dcci_cachelines((__gm__ uint8_t*)sqeCtx, wqeSize * wqeBbCnt);
    curHead += wqeBbCnt;
    aclshmemi_udma_post_send_update_info(curHead, qpCtxEntry);
    wqeCnt++;
    qpCtxEntry->wqeCnt = wqeCnt;
#if defined(ACLSHMEM_RELAY_SUPPORT)
    dcci_cachelines((__gm__ uint8_t*)qpCtxEntry, sizeof(ACLSHMEMUDMAWQCtx));
#endif
    ACLSHMEM_DEBUG_FUNC(aclshmemi_dump_wqe, (__gm__ uint8_t*)sqeCtx, (uint32_t)sizeof(T));
}

ACLSHMEM_DEVICE void aclshmemi_udma_post_send_update_info(uint32_t curHead, __gm__ ACLSHMEMUDMAWQCtx*& qpCtxEntry)
{
    // Ring SQ Doorbell (reference udma_update_sq_db in UDMA)
    // Note: db address is 64-bit, but we only update 32-bit value
    __gm__ uint32_t* doorBellAddr = (__gm__ uint32_t*)qpCtxEntry->dbAddr;
    st_dev(curHead, doorBellAddr, 0);
    qpCtxEntry->head = curHead;
    return;
}

ACLSHMEM_DEVICE void aclshmemi_udma_copy_wqe_from_ub(
    __gm__ uint8_t* dstGm, AscendC::LocalTensor<uint8_t>& ubLocal, uint32_t copyLen, uint32_t syncId)
{
    AscendC::GlobalTensor<uint8_t> gmTensor;
    gmTensor.SetGlobalBuffer(dstGm);
    AscendC::DataCopyExtParams copyParams{1, copyLen, 0, 0, 0};

    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(syncId);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(syncId);
    AscendC::DataCopyPad(gmTensor, ubLocal, copyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(syncId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(syncId);
}

// ---- MTE3-staged WQE construction (PIPE_MTE3 path) -------------------------------
// The PIPE_S path above (aclshmemi_udma_post_send) writes the SQE/SGE directly to HBM
// via scalar stores + dcci_cachelines. The MTE3 path stages the full WQE block in
// caller-provided UB scratch and lands it on the SQ ring with a single DataCopyPad.
// Useful for hot loops where the per-iteration scalar->HBM bursts dominate.
//
// MTE3 path only supports UDMA_OP_WRITE / UDMA_OP_WRITE_WITH_NOTIFY / UDMA_OP_READ
// (the data-mover opcodes). FAA / CAS / WRITE_WITH_REDUCE remain on PIPE_S because
// their SGE side fills GM-resident AMO data which is incompatible with UB staging.
template <typename T, aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_udma_post_send_mte3(
    __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr, uint32_t pe, uint32_t qpIdx, uint64_t messageLen,
    __ubuf__ uint8_t* ubScratch, uint32_t syncId, const aclshmemi_udma_params_t<T, OP_CODE>& params = {},
    uint32_t relay_pe = static_cast<uint32_t>(-1))
{
    static_assert(
        OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE ||
            OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY ||
            OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_READ,
        "PIPE_MTE3 WQE path only supports UDMA_OP_WRITE / UDMA_OP_WRITE_WITH_NOTIFY / UDMA_OP_READ");

    __gm__ ACLSHMEMAIVUDMAInfo* udmaInfo = aclshmemi_udma_qp_info_fetch();
    // Unified slot computation for both builds (call site is identical, no #if here). OFF returns
    // pe (direct path); ON returns pe*N+actualRelayPe. Direct/OFF callers pass relay_pe == -1,
    // which compute_slot ignores while returning pe.
    uint32_t slot = aclshmemi_udma_compute_slot(pe, relay_pe);
    __gm__ ACLSHMEMUDMAWQCtx* qpCtxEntry = aclshmemi_udma_get_qp_ctx(udmaInfo, slot, qpIdx);
    auto wqeSize = 1 << qpCtxEntry->baseBkShift;
    uint32_t curHead = qpCtxEntry->head;
    ACLSHMEM_DEBUG_FUNC(assert_qp_params_valid, qpCtxEntry);
    uint32_t wqeCnt = qpCtxEntry->wqeCnt;
    poll_cq_when_sq_overflow(qpCtxEntry, wqeCnt, slot, qpIdx);

    __gm__ ACLSHMEMUBmemInfo* remoteMemInfo =
        (__gm__ ACLSHMEMUBmemInfo*)(udmaInfo->memPtr + sizeof(ACLSHMEMUBmemInfo) * slot);

    // Stage WQE (SQE + optional notify + SGE) in caller's UB scratch. Reuse the
    // address-space-templated fill helper so SQE field assignments are not duplicated.
    __ubuf__ ACLSHMEMSqeCtx* sqeUb = (__ubuf__ ACLSHMEMSqeCtx*)ubScratch;
    aclshmemi_udma_fill_sqe_ctx<T, OP_CODE, __ubuf__ ACLSHMEMSqeCtx*>(
        sqeUb, remoteAddr, remoteMemInfo, curHead, params);

    constexpr size_t sgeOff = (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY) ?
                                  sizeof(ACLSHMEMSqeCtx) + sizeof(ACLSHMEMNotifyCtx) :
                                  sizeof(ACLSHMEMSqeCtx);
    __ubuf__ ACLSHMEMSgeCtx* sgeUb = (__ubuf__ ACLSHMEMSgeCtx*)((__ubuf__ uint8_t*)ubScratch + sgeOff);
    // OP_CODE is restricted to non-AMO/non-REDUCE here, so the SGE only needs
    // len + va. The full FAA/CAS/REDUCE fan-out is intentionally unused in this path.
    sgeUb->len = messageLen;
    sgeUb->va = reinterpret_cast<uint64_t>(localAddr);

    // Single-shot DMA UB -> SQ ring entry. The helper uses the same S->MTE3 and
    // MTE3->S ordering as the RDMA backend before ringing the SQ doorbell.
    __gm__ ACLSHMEMSqeCtx* sqeGm = aclshmemi_udma_get_sqe_ctx(qpCtxEntry, curHead, wqeSize);
    constexpr uint32_t wqeBbCnt = get_wqe_bb_cnt<OP_CODE>();
    AscendC::LocalTensor<uint8_t> ubLocal;
    ubLocal.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ubLocal.address_.bufferAddr = reinterpret_cast<uint64_t>(ubScratch);
    aclshmemi_udma_copy_wqe_from_ub((__gm__ uint8_t*)sqeGm, ubLocal, static_cast<uint32_t>(wqeSize * wqeBbCnt), syncId);

    curHead += wqeBbCnt;
    aclshmemi_udma_post_send_update_info(curHead, qpCtxEntry);
    wqeCnt++;
    qpCtxEntry->wqeCnt = wqeCnt;
#if defined(ACLSHMEM_RELAY_SUPPORT)
    dcci_cachelines((__gm__ uint8_t*)qpCtxEntry, sizeof(ACLSHMEMUDMAWQCtx));
#endif
    ACLSHMEM_DEBUG_FUNC(aclshmemi_dump_wqe, (__gm__ uint8_t*)sqeGm, (uint32_t)sizeof(T));
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_write_mte3(
    __gm__ T* destDmaAddr, __gm__ T* srcDmaAddr, uint32_t pe, uint32_t qpIdx, uint64_t messageLen,
    __ubuf__ uint8_t* ubScratch, uint32_t syncId, uint32_t relay_pe = static_cast<uint32_t>(-1))
{
    aclshmemi_udma_post_send_mte3<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE>(
        reinterpret_cast<__gm__ uint8_t*>(destDmaAddr), reinterpret_cast<__gm__ uint8_t*>(srcDmaAddr), pe, qpIdx,
        messageLen, ubScratch, syncId, {}, relay_pe);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_write_notify_mte3(
    __gm__ T* destDmaAddr, __gm__ T* srcDmaAddr, uint32_t pe, uint32_t qpIdx, uint64_t messageLen,
    const aclshmemi_udma_params_t<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY>& params,
    __ubuf__ uint8_t* ubScratch, uint32_t syncId)
{
    aclshmemi_udma_post_send_mte3<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY>(
        reinterpret_cast<__gm__ uint8_t*>(destDmaAddr), reinterpret_cast<__gm__ uint8_t*>(srcDmaAddr), pe, qpIdx,
        messageLen, ubScratch, syncId, params);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_write(
    __gm__ T* destDmaAddr, __gm__ T* srcDmaAddr, uint32_t pe, uint32_t qpIdx, uint64_t messageLen, uint32_t relay_pe)
{
    aclshmemi_udma_post_send<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE>(
        reinterpret_cast<__gm__ uint8_t*>(destDmaAddr), reinterpret_cast<__gm__ uint8_t*>(srcDmaAddr), pe, qpIdx,
        messageLen, {}, relay_pe);
}

template <typename T, aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_udma_write_notify(
    __gm__ T* destDmaAddr, __gm__ T* srcDmaAddr, uint32_t pe, uint32_t qpIdx, uint64_t messageLen,
    const aclshmemi_udma_params_t<T, OP_CODE>& params)
{
    aclshmemi_udma_post_send<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY>(
        reinterpret_cast<__gm__ uint8_t*>(destDmaAddr), reinterpret_cast<__gm__ uint8_t*>(srcDmaAddr), pe, qpIdx,
        messageLen, params);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_read(
    __gm__ T* destDmaAddr, __gm__ T* srcDmaAddr, uint32_t srcPe, uint32_t qpIdx, uint64_t messageLen, uint32_t relay_pe)
{
    aclshmemi_udma_post_send<T, aclshmemi_udma_opcode_t::UDMA_OP_READ>(
        reinterpret_cast<__gm__ uint8_t*>(srcDmaAddr), reinterpret_cast<__gm__ uint8_t*>(destDmaAddr), srcPe, qpIdx,
        messageLen, {}, relay_pe);
}

ACLSHMEM_DEVICE void aclshmemx_udma_quiet(int pe)
{
    __gm__ ACLSHMEMAIVUDMAInfo* udmaInfo = aclshmemi_udma_qp_info_fetch();
    uint32_t qpNum = udmaInfo->qpNum;
    uint32_t actualPe = static_cast<uint32_t>(pe);
#if defined(ACLSHMEM_RELAY_SUPPORT)
    uint32_t rankCount = static_cast<uint32_t>(aclshmemi_get_total_pe());
    // Drain every (pe, relay_pe) slot. Slots that never received a post_send have wqeCnt==0
    // and poll_cq returns immediately. UDMA does not support self ops (pe == myPe), so the
    // (pe, pe) diagonal is never posted to and host leaves it unfilled -- skip it.
    for (uint32_t relayPe = 0; relayPe < rankCount; ++relayPe) {
        if (relayPe == actualPe) {
            continue;
        }
        uint32_t slot = actualPe * rankCount + relayPe;
        __gm__ ACLSHMEMUDMAWQCtx* qpCtxEntry =
            (__gm__ ACLSHMEMUDMAWQCtx*)(udmaInfo->sqPtr + (slot * qpNum + 0) * sizeof(ACLSHMEMUDMAWQCtx));
        uint32_t wqeCnt = qpCtxEntry->wqeCnt;
        if (wqeCnt == 0) {
            continue;
        }
        aclshmemi_udma_poll_cq(slot, 0, wqeCnt);
    }
#else
    // Original direct path: poll only the single slot == pe. Does not read rankCount.
    __gm__ ACLSHMEMUDMAWQCtx* qpCtxEntry =
        (__gm__ ACLSHMEMUDMAWQCtx*)(udmaInfo->sqPtr + (actualPe * qpNum + 0) * sizeof(ACLSHMEMUDMAWQCtx));
    uint32_t wqeCnt = qpCtxEntry->wqeCnt;
    if (wqeCnt != 0) {
        aclshmemi_udma_poll_cq(actualPe, 0, wqeCnt);
    }
#endif
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_get_nbi(__gm__ T* dst, __gm__ T* src, uint32_t elem_size, int pe)
{
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        auto ptr = aclshmem_ptr(src, pe);
        aclshmemi_udma_read((__gm__ uint8_t*)dst, (__gm__ uint8_t*)ptr, pe, 0, elem_size * sizeof(T));
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE>
ACLSHMEM_DEVICE void aclshmemx_udma_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id)
{
    static_assert(
        WQE_PIPE == PIPE_S || WQE_PIPE == PIPE_MTE3, "Only PIPE_S and PIPE_MTE3 are supported for UDMA WQE_PIPE");
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        if constexpr (WQE_PIPE == PIPE_MTE3) {
            auto ptr = aclshmem_ptr(src, pe);
            // For UDMA_OP_READ, the SQE's "remoteAddr" slot carries the src (remote)
            // and "localAddr" carries dst (local), matching aclshmemi_udma_read().
            aclshmemi_udma_post_send_mte3<T, aclshmemi_udma_opcode_t::UDMA_OP_READ>(
                (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)dst, static_cast<uint32_t>(pe), 0, elem_size * sizeof(T),
                reinterpret_cast<__ubuf__ uint8_t*>(buf), sync_id);
        } else {
            (void)buf;
            (void)sync_id;
            aclshmemi_udma_get_nbi(dst, src, elem_size, pe);
        }
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE>
ACLSHMEM_DEVICE void aclshmemx_udma_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t sync_id)
{
    aclshmemx_udma_get_nbi<T, WQE_PIPE>(
        (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
        elem_size, pe, sync_id);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_put_nbi(__gm__ T* dst, __gm__ T* src, uint32_t elem_size, int pe)
{
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        auto ptr = aclshmem_ptr(dst, pe);
        aclshmemi_udma_write((__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, pe, 0, elem_size * sizeof(T));
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE>
ACLSHMEM_DEVICE void aclshmemx_udma_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id)
{
    static_assert(
        WQE_PIPE == PIPE_S || WQE_PIPE == PIPE_MTE3, "Only PIPE_S and PIPE_MTE3 are supported for UDMA WQE_PIPE");
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        if constexpr (WQE_PIPE == PIPE_MTE3) {
            auto ptr = aclshmem_ptr(dst, pe);
            aclshmemi_udma_write_mte3<T>(
                (__gm__ T*)ptr, src, static_cast<uint32_t>(pe), 0, elem_size * sizeof(T),
                reinterpret_cast<__ubuf__ uint8_t*>(buf), sync_id);
        } else {
            (void)buf;
            (void)sync_id;
            aclshmemi_udma_put_nbi(dst, src, elem_size, pe);
        }
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE>
ACLSHMEM_DEVICE void aclshmemx_udma_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t sync_id)
{
    static_assert(
        WQE_PIPE == PIPE_S || WQE_PIPE == PIPE_MTE3, "Only PIPE_S and PIPE_MTE3 are supported for UDMA WQE_PIPE");
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        auto dstPhyAddr = (__gm__ T*)dst.GetPhyAddr();
        auto srcPhyAddr = (__gm__ T*)src.GetPhyAddr();
        if constexpr (WQE_PIPE == PIPE_MTE3) {
            auto ptr = aclshmem_ptr(dstPhyAddr, pe);
            aclshmemi_udma_write_mte3<T>(
                (__gm__ T*)ptr, srcPhyAddr, static_cast<uint32_t>(pe), 0, elem_size * sizeof(T),
                reinterpret_cast<__ubuf__ uint8_t*>(buf.GetPhyAddr()), sync_id);
        } else {
            (void)buf;
            (void)sync_id;
            aclshmemi_udma_put_nbi(dstPhyAddr, srcPhyAddr, elem_size, pe);
        }
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id)
{
    static_assert(
        WQE_PIPE == PIPE_S || WQE_PIPE == PIPE_MTE3, "Only PIPE_S and PIPE_MTE3 are supported for UDMA WQE_PIPE");
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
#if defined(ACLSHMEM_RELAY_SUPPORT)
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_printf, "udma relay put: pe=%d relay_pe=%d\n", pe, relay_pe);
        int rankCount = aclshmemi_get_total_pe();
        int myPe = aclshmemi_get_my_pe();
        if (pe < 0 || pe >= rankCount || relay_pe < 0 || relay_pe >= rankCount || pe == relay_pe || pe == myPe ||
            relay_pe == myPe) {
            ACLSHMEM_DEBUG_FUNC(
                aclshmemi_kernel_abort,
                "udma relay put: invalid pe=%d relay_pe=%d (myPe=%d rankCount=%d); "
                "require 0<=actual,relay<rankCount, actual!=relay, neither equals myPe\n",
                pe, relay_pe, myPe, rankCount);
            return;
        }
        auto ptr = aclshmem_ptr(dst, pe);
        if constexpr (WQE_PIPE == PIPE_MTE3) {
            aclshmemi_udma_write_mte3<uint8_t>(
                (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, static_cast<uint32_t>(pe), 0u, elem_size * sizeof(T),
                reinterpret_cast<__ubuf__ uint8_t*>(buf), sync_id, static_cast<uint32_t>(relay_pe));
        } else {
            (void)buf;
            (void)sync_id;
            aclshmemi_udma_write<uint8_t>(
                (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, static_cast<uint32_t>(pe), 0u, elem_size * sizeof(T),
                static_cast<uint32_t>(relay_pe));
        }
#else
        // sizeof(T) == 0 is always false but depends on T, so it only fires when this template is
        // actually instantiated (i.e. relay API is called) rather than at parse time. A plain
        // static_assert(false, ...) would break the whole OFF build even when relay is never used.
        static_assert(
            sizeof(T) == 0, "aclshmemx_udma_relay_put_nbi requires ACLSHMEM_RELAY_SUPPORT=ON; rebuild with it enabled");
#endif
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id)
{
#if defined(ACLSHMEM_RELAY_SUPPORT)
    aclshmemx_udma_relay_put_nbi<T, WQE_PIPE>(
        (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
        elem_size, pe, relay_pe, sync_id);
#else
    (void)dst;
    (void)src;
    (void)buf;
    (void)elem_size;
    (void)pe;
    (void)relay_pe;
    (void)sync_id;
    static_assert(
        sizeof(T) == 0, "aclshmemx_udma_relay_put_nbi requires ACLSHMEM_RELAY_SUPPORT=ON; rebuild with it enabled");
#endif
}

template <typename T, pipe_t WQE_PIPE>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id)
{
    static_assert(
        WQE_PIPE == PIPE_S || WQE_PIPE == PIPE_MTE3, "Only PIPE_S and PIPE_MTE3 are supported for UDMA WQE_PIPE");
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
#if defined(ACLSHMEM_RELAY_SUPPORT)
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_printf, "udma relay get: pe=%d relay_pe=%d\n", pe, relay_pe);
        int rankCount = aclshmemi_get_total_pe();
        int myPe = aclshmemi_get_my_pe();
        if (pe < 0 || pe >= rankCount || relay_pe < 0 || relay_pe >= rankCount || pe == relay_pe || pe == myPe ||
            relay_pe == myPe) {
            ACLSHMEM_DEBUG_FUNC(
                aclshmemi_kernel_abort,
                "udma relay get: invalid pe=%d relay_pe=%d (myPe=%d rankCount=%d); "
                "require 0<=actual,relay<rankCount, actual!=relay, neither equals myPe\n",
                pe, relay_pe, myPe, rankCount);
            return;
        }
        auto ptr = aclshmem_ptr(src, pe);
        if constexpr (WQE_PIPE == PIPE_MTE3) {
            // For UDMA_OP_READ the SQE's "remoteAddr" slot carries src (remote) and
            // "localAddr" carries dst (local), matching aclshmemi_udma_read().
            aclshmemi_udma_post_send_mte3<uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_READ>(
                (__gm__ uint8_t*)ptr, (__gm__ uint8_t*)dst, static_cast<uint32_t>(pe), 0u, elem_size * sizeof(T),
                reinterpret_cast<__ubuf__ uint8_t*>(buf), sync_id, {}, static_cast<uint32_t>(relay_pe));
        } else {
            (void)buf;
            (void)sync_id;
            aclshmemi_udma_read<uint8_t>(
                (__gm__ uint8_t*)dst, (__gm__ uint8_t*)ptr, static_cast<uint32_t>(pe), 0u, elem_size * sizeof(T),
                static_cast<uint32_t>(relay_pe));
        }
#else
        // sizeof(T) == 0 is always false but depends on T, so it only fires when this template is
        // actually instantiated (i.e. relay API is called) rather than at parse time. A plain
        // static_assert(false, ...) would break the whole OFF build even when relay is never used.
        static_assert(
            sizeof(T) == 0, "aclshmemx_udma_relay_get_nbi requires ACLSHMEM_RELAY_SUPPORT=ON; rebuild with it enabled");
#endif
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T, pipe_t WQE_PIPE>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id)
{
#if defined(ACLSHMEM_RELAY_SUPPORT)
    aclshmemx_udma_relay_get_nbi<T, WQE_PIPE>(
        (__gm__ T*)dst.GetPhyAddr(), (__gm__ T*)src.GetPhyAddr(), reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr()),
        elem_size, pe, relay_pe, sync_id);
#else
    (void)dst;
    (void)src;
    (void)buf;
    (void)elem_size;
    (void)pe;
    (void)relay_pe;
    (void)sync_id;
    static_assert(
        sizeof(T) == 0, "aclshmemx_udma_relay_get_nbi requires ACLSHMEM_RELAY_SUPPORT=ON; rebuild with it enabled");
#endif
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_put_signal_nbi(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, __gm__ uint64_t* sig_addr, uint64_t signal, int pe)
{
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        auto ptr = aclshmem_ptr(dst, pe);
        auto sig_addr_dst = aclshmem_ptr(sig_addr, pe);
        aclshmemi_udma_params_t<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY> signal_params{
            .sig_addr = (__gm__ uint64_t*)(sig_addr_dst), .signal = signal};
        aclshmemi_udma_write_notify<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY>(
            (__gm__ T*)ptr, src, pe, 0, elem_size * sizeof(T), signal_params);
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

// Buf-taking overload. PIPE_MTE3 (default) stages the WRITE_WITH_NOTIFY WQE in the
// caller-provided UB scratch (size one full WQE block; wqe_size * 2 bytes, 256 B is
// safe for the current SQ basebk_shift). PIPE_S falls through to the no-buf path and
// ignores buf/sync_id, mirroring the put_nbi(buf) overload's S/MTE3 split.
template <typename T, pipe_t WQE_PIPE>
ACLSHMEM_DEVICE void aclshmemx_udma_put_signal_nbi(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, __gm__ uint64_t* sig_addr, uint64_t signal, int pe,
    __ubuf__ uint8_t* buf, uint32_t sync_id)
{
    static_assert(
        WQE_PIPE == PIPE_S || WQE_PIPE == PIPE_MTE3, "Only PIPE_S and PIPE_MTE3 are supported for UDMA WQE_PIPE");
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        if constexpr (WQE_PIPE == PIPE_MTE3) {
            auto ptr = aclshmem_ptr(dst, pe);
            auto sig_addr_dst = aclshmem_ptr(sig_addr, pe);
            aclshmemi_udma_params_t<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY> signal_params{
                .sig_addr = (__gm__ uint64_t*)(sig_addr_dst), .signal = signal};
            aclshmemi_udma_write_notify_mte3<T>(
                (__gm__ T*)ptr, src, static_cast<uint32_t>(pe), 0, elem_size * sizeof(T), signal_params, buf, sync_id);
        } else {
            (void)buf;
            (void)sync_id;
            aclshmemx_udma_put_signal_nbi<T>(dst, src, elem_size, sig_addr, signal, pe);
        }
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}
template <typename T, aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE constexpr bool aclshmemi_udma_check_atomic_len()
{
    size_t atomic_len = sizeof(T);
    if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA) {
        if (atomic_len != 4 && atomic_len != 8) {
            return false;
        }
    } else if constexpr (OP_CODE == aclshmemi_udma_opcode_t::UDMA_OP_CAS) {
        if (atomic_len != 4 && atomic_len != 8 && atomic_len != 16) {
            return false;
        }
    }
    return true;
}

ACLSHMEM_DEVICE uint64_t aclshmemi_udma_get_amo_addr(uint32_t pe, uint32_t qp_idx)
{
    __gm__ ACLSHMEMAIVUDMAInfo* udmaInfo = aclshmemi_udma_qp_info_fetch();
    uint32_t qp_num = udmaInfo->qpNum;
    // Atomic ops only run on the direct path, so reuse the same slot as aclshmemi_udma_post_send's
    // default (relay = self). Gate the layout exactly like compute_slot: OFF is the original
    // v1.5.0 single-dimension table where slot = pe; ON is the N*N table where the direct bucket
    // is (actual=pe, relay=self) = pe*N + myPe. Using the N*N formula on OFF would read amoAddr
    // from the wrong (out-of-bounds) slot and corrupt atomic fetch data.
    uint32_t slot = aclshmemi_udma_compute_slot(pe);
    __gm__ ACLSHMEMUDMAWQCtx* qpCtxEntry =
        (__gm__ ACLSHMEMUDMAWQCtx*)(udmaInfo->sqPtr + (slot * qp_num + qp_idx) * sizeof(ACLSHMEMUDMAWQCtx));
    auto amo_addr = qpCtxEntry->amoAddr;
    return amo_addr;
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemi_udma_get_amo_addr_value(uint64_t amo_addr)
{
    dcci_cachelines((__gm__ uint8_t*)amo_addr, sizeof(T));
    __gm__ T* fetch_addr = reinterpret_cast<__gm__ T*>(amo_addr);
    T fetch_data = *fetch_addr;
    return fetch_data;
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemi_udma_get_atomic_fetch_data(uint32_t pe, uint32_t qp_idx)
{
    auto amo_addr = aclshmemi_udma_get_amo_addr(pe, qp_idx);
    return aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_add(__gm__ T* dst, T value, int32_t pe)
{
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        if constexpr (!aclshmemi_udma_check_atomic_len<T, aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA>()) {
            ACLSHMEM_DEBUG_FUNC(
                aclshmemi_kernel_abort, "Atomic size %u is not supported for UDMA atomic add\n", sizeof(T));
        }
        auto ptr = aclshmem_ptr(dst, pe);
        aclshmemi_udma_params_t<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE> atomic_params{
            .value = value, .cond = 0};
        if constexpr (AscendC::IsSameType<T, float>::value) { // float使用write with reduce逻辑处理
            aclshmemi_udma_post_send<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE>(
                reinterpret_cast<__gm__ uint8_t*>(ptr), nullptr, pe, 0, sizeof(T), atomic_params);
        } else {
            aclshmemi_udma_post_send<T, aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA>(
                reinterpret_cast<__gm__ uint8_t*>(ptr), nullptr, pe, 0, sizeof(T), atomic_params);
        }
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
    }
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_add(__gm__ T* dst, T value, int32_t pe)
{
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        if constexpr (!aclshmemi_udma_check_atomic_len<T, aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA>()) {
            ACLSHMEM_DEBUG_FUNC(
                aclshmemi_kernel_abort, "Atomic size %u is not supported for UDMA atomic fetch add\n", sizeof(T));
        }

        auto ptr = aclshmem_ptr(dst, pe);
        aclshmemi_udma_params_t<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE> atomic_params{
            .value = value, .cond = 0};
        aclshmemi_udma_post_send<T, aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA>(
            reinterpret_cast<__gm__ uint8_t*>(ptr), nullptr, pe, 0, sizeof(T), atomic_params);

        aclshmemx_udma_quiet(pe);
        return aclshmemi_udma_get_atomic_fetch_data<T>(pe, 0);
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
        return 0;
    }
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_compare_swap(__gm__ T* dst, T cond, T value, int32_t pe)
{
    if constexpr (ACLSHMEM_UDMA_SUPPORTED) {
        if constexpr (!aclshmemi_udma_check_atomic_len<T, aclshmemi_udma_opcode_t::UDMA_OP_CAS>()) {
            ACLSHMEM_DEBUG_FUNC(
                aclshmemi_kernel_abort, "Atomic size %u is not supported for UDMA atomic compare swap\n", sizeof(T));
        }
        auto ptr = aclshmem_ptr(dst, pe);
        aclshmemi_udma_params_t<T, aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE> cas_params{
            .value = value, .cond = cond};
        aclshmemi_udma_post_send<T, aclshmemi_udma_opcode_t::UDMA_OP_CAS>(
            reinterpret_cast<__gm__ uint8_t*>(ptr), nullptr, pe, 0, sizeof(T), cas_params);
        aclshmemx_udma_quiet(pe);
        return aclshmemi_udma_get_atomic_fetch_data<T>(pe, 0);
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on Ascend950 or later\n");
        return 0;
    }
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch(__gm__ T* dst, int32_t pe)
{
    return aclshmemx_udma_atomic_fetch_add<T>(dst, 0, pe);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_set(__gm__ T* dst, T value, int32_t pe)
{
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        if (aclshmemx_udma_atomic_compare_swap(dst, old_value, value, pe) == old_value) {
            break;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_set timeout!\n");
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_swap(__gm__ T* dst, T value, int32_t pe)
{
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        if (aclshmemx_udma_atomic_compare_swap(dst, old_value, value, pe) == old_value) {
            return old_value;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_swap timeout!\n");
    return 0;
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_inc(__gm__ T* dst, int32_t pe)
{
    return aclshmemx_udma_atomic_fetch_add<T>(dst, 1, pe);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_inc(__gm__ T* dst, int32_t pe)
{
    aclshmemx_udma_atomic_add<T>(dst, 1, pe);
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_and(__gm__ T* dst, T value, int32_t pe)
{
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        T new_value = old_value & value;
        if (aclshmemx_udma_atomic_compare_swap(dst, old_value, new_value, pe) == old_value) {
            return old_value;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_fetch_and timeout!\n");
    return 0;
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_and(__gm__ T* dst, T value, int32_t pe)
{
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        T new_value = old_value & value;
        if (aclshmemx_udma_atomic_compare_swap(dst, old_value, new_value, pe) == old_value) {
            break;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_and timeout!\n");
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_or(__gm__ T* dst, T value, int32_t pe)
{
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        T new_value = old_value | value;
        if (aclshmemx_udma_atomic_compare_swap(dst, old_value, new_value, pe) == old_value) {
            return old_value;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_fetch_or timeout!\n");
    return 0;
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_or(__gm__ T* dst, T value, int32_t pe)
{
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        T new_value = old_value | value;
        if (aclshmemx_udma_atomic_compare_swap(dst, old_value, new_value, pe) == old_value) {
            break;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_or timeout!\n");
}

template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_xor(__gm__ T* dst, T value, int32_t pe)
{
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        T new_value = old_value ^ value;
        if (aclshmemx_udma_atomic_compare_swap(dst, old_value, new_value, pe) == old_value) {
            return old_value;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_fetch_xor timeout!\n");
    return 0;
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_xor(__gm__ T* dst, T value, int32_t pe)
{
    uint32_t times = 0;
    while (times < MAX_RETRY_TIMES) {
        auto amo_addr = aclshmemi_udma_get_amo_addr(pe, 0);
        aclshmemi_udma_get_nbi((__gm__ T*)amo_addr, dst, 1, pe);
        aclshmemx_udma_quiet(pe);
        T old_value = aclshmemi_udma_get_amo_addr_value<T>(amo_addr);
        T new_value = old_value ^ value;
        if (aclshmemx_udma_atomic_compare_swap(dst, old_value, new_value, pe) == old_value) {
            break;
        }
        times++;
    }
    ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Atomic_xor timeout!\n");
}

#endif
