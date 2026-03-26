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
#include "shmemi_device_udma.h"

#if defined(ASCENDC_DUMP) && (ASCENDC_DUMP == 1) && defined(DEBUG_MODE)
#define ACLSHMEM_DEBUG_FUNC(func, ...) do { func(__VA_ARGS__); } while(0)
#else
#define ACLSHMEM_DEBUG_FUNC(func, ...) ((void)0)
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#define ACLSHMEM_UDMA_SUPPORTED 1
#else
#define ACLSHMEM_UDMA_SUPPORTED 0
#endif

template<class... Args>
ACLSHMEM_DEVICE void aclshmemi_kernel_abort(__gm__ const char* fmt, Args&&... args)
{
    AscendC::printf(fmt, args...);
    trap();
}

ACLSHMEM_DEVICE void aclshmemi_dump_sge(__gm__ uint8_t* wqeAddr, uint32_t sge_num);
ACLSHMEM_DEVICE void aclshmemi_dump_wqe(__gm__ uint8_t* wqeAddr);

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
    AscendC::printf("CQE: DW0 - sR: %d, isJetty: %d, owner: %d, inlineEn: %d, opcode: %d, fd: %d, substatus: %d, status: %d\n",
                   sR, isJetty, owner, inlineEn, opcode, fd, substatus, status);
    AscendC::printf("CQE: DW1 - entryIdx: %d, localNumL: %d\n",
                   entryIdx, localNumL);
    AscendC::printf("CQE: DW2 - localNumH: %d, rmtIdx: %d\n",
                   localNumH, rmtIdx);
    AscendC::printf("CQE: DW3 - tpn: %d\n", tpn);
    AscendC::printf("CQE: DW4 - byteCnt: %d\n", cqeAddr->byteCnt);
    AscendC::printf("CQE: DW5-DW6 - userData: 0x%08x%08x\n",
                   cqeAddr->userDataH, cqeAddr->userDataL);
    AscendC::printf("CQE: DW7-DW10 - rmtEid: [0x%08x, 0x%08x, 0x%08x, 0x%08x]\n",
                   cqeAddr->rmtEid[0], cqeAddr->rmtEid[1], cqeAddr->rmtEid[2], cqeAddr->rmtEid[3]);
    AscendC::printf("CQE: DW11-DW12 - data: 0x%08x%08x\n",
                   cqeAddr->dataH, cqeAddr->dataL);
    AscendC::printf("CQE: DW13-DW15 - inlineData: [0x%08x, 0x%08x, 0x%08x]\n",
                   cqeAddr->inlineData[0], cqeAddr->inlineData[1], cqeAddr->inlineData[2]);
}

ACLSHMEM_DEVICE uint32_t aclshmemi_udma_poll_cq(uint32_t remoteRankId, uint32_t qpIdx, uint32_t idx)
{
    if (idx == 0) {
        return 0;
    }
    __gm__ ACLSHMEMAIVUDMAInfo* udmaInfo = aclshmemi_udma_qp_info_fetch();
    uint32_t qpNum = udmaInfo->qpNum;
    __gm__ ACLSHMEMUDMACqCtx* cqCtxEntry = (__gm__ ACLSHMEMUDMACqCtx*)(udmaInfo->scqPtr
        + (remoteRankId * qpNum + qpIdx) * sizeof(ACLSHMEMUDMACqCtx));
    auto cqBaseAddr = cqCtxEntry->bufAddr;
    auto bbShift = cqCtxEntry->baseBkShift;
    auto cqeSize = 1 << bbShift;
    auto depth = cqCtxEntry->depth;
    auto curHardwareTailAddr = cqCtxEntry->tailAddr;
    uint32_t curTail = ld_dev((__gm__ uint32_t*)(curHardwareTailAddr), 0);
    const uint32_t maxTimes = 1000000;
    while (curTail != idx) {
        __gm__ ACLSHMEMJfcCqeCtx* cqeAddr = (__gm__ ACLSHMEMJfcCqeCtx*)(cqBaseAddr + cqeSize * (curTail & (depth - 1)));
        bool validOwner = (curTail / depth) & 1;
        uint32_t times = 0;
        while ((validOwner ^ cqeAddr->owner) == 0 && times < maxTimes) { // util cqeAddr->owner changed
            dcci_cachelines((__gm__ uint8_t*)cqeAddr, sizeof(ACLSHMEMJfcCqeCtx));
            times++;
        }
        if (times >= maxTimes) {
            ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Poll cq timeout\n");
            return 0xFF;
        }
        curTail++;
        // Check CQE status
        uint8_t status = cqeAddr->status & 0xFF;
        uint8_t subStatus = cqeAddr->substatus & 0xFF;
        constexpr uint8_t statusShift = 8;
        if (status != 0 || subStatus != 0) {
            ACLSHMEM_DEBUG_FUNC(aclshmemi_dump_cqe, cqeAddr);
            return (status << statusShift) | subStatus;
        }
    }

    // Update CQ tail
    st_dev(curTail, (__gm__ uint32_t*)(cqCtxEntry->tailAddr), 0);
    __gm__ ACLSHMEMUDMAWQCtx* wqCtxEntry = (__gm__ ACLSHMEMUDMAWQCtx*)(udmaInfo->sqPtr
        + (remoteRankId * qpNum + qpIdx) * sizeof(ACLSHMEMUDMAWQCtx));
    aclshmemi_udma_poll_cq_update_info(curTail, qpIdx, cqCtxEntry, wqCtxEntry);
    return 0;
}

ACLSHMEM_DEVICE void aclshmemi_udma_poll_cq_update_info(uint32_t curTail, uint32_t qpIdx,
    __gm__ ACLSHMEMUDMACqCtx* cqCtxEntry, __gm__ ACLSHMEMUDMAWQCtx* wqCtxEntry)
{
    // Ring CQ Doorbell (reference URMA implementation)
    auto cqDBAddr = cqCtxEntry->dbAddr;
    // For JFC, we write the consumer index (curTail) directly
    __gm__ uint32_t* dbAddr = (__gm__ uint32_t*)cqCtxEntry->dbAddr;
    st_dev((uint32_t)(curTail & 0xFFFFFF), dbAddr, 0);
    // Update WQ tail
    auto curWQTailAddr = wqCtxEntry->tailAddr;
    st_dev(curTail, (__gm__ uint32_t*)curWQTailAddr, 0);
}

ACLSHMEM_DEVICE void aclshmemi_dump_wqe(__gm__ uint8_t* wqeAddr)
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
    AscendC::printf("WQE: sqe_bb_idx: %x flag: %x rsv0: %x nf: %x token_en: %x rmt_jetty_type: %x\n", sqeBbIdx, flag,
        rsv0, nf, tokenEn, rmtJettyType);
    auto owner = sqeCtx->owner;
    auto targetHint = sqeCtx->targetHint;
    auto opcode = sqeCtx->opcode;
    auto rsv1 = sqeCtx->rsv1;
    auto inlineMsgLen = sqeCtx->inlineMsgLen;
    auto tpId = sqeCtx->tpId;
    AscendC::printf("WQE: owner: %x target_hint: %x opcode: %x rsv1: %x inline_msg_len: %x tp_id: %x\n", 
        owner, targetHint, opcode, rsv1, inlineMsgLen, tpId);
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
    AscendC::printf("WQE: rmt_token_value: %x udf_type: %x reduce_data_type: %x reduce_opcode: %x\n", 
        rmtTokenValue, udfType, reduceDataType, reduceOpcode);
    AscendC::printf("WQE: rmt_addr_l_or_token_id: %x rmt_addr_h_or_token_value: %x\n", 
        rmtAddrLOrTokenId, rmtAddrHOrTokenValue);
    aclshmemi_dump_sge(wqeAddr, sgeNum);
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
        AscendC::printf("SGE: sge idx: %d, va: %p sge_len: %d\n",i, sgeRmtAddr, sgeLen);
        sgeCtx++;
    }
}

ACLSHMEM_DEVICE void aclshmemi_udma_post_send(__gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr,
                                        uint32_t destRankId, uint32_t qpIdx,
                                        UDMAOPCODE opcode, uint64_t messageLen)
{
    __gm__ ACLSHMEMAIVUDMAInfo* udmaInfo = aclshmemi_udma_qp_info_fetch();

    uint32_t qpNum = udmaInfo->qpNum;
    __gm__ ACLSHMEMUDMAWQCtx* qpCtxEntry = (__gm__ ACLSHMEMUDMAWQCtx*)(udmaInfo->sqPtr
        + (destRankId * qpNum + qpIdx) * sizeof(ACLSHMEMUDMAWQCtx));
    auto sqBaseAddr = qpCtxEntry->bufAddr;
    auto wqeSize = 1 << qpCtxEntry->baseBkShift; // basebk_shift
    
    auto curHardwareHeadAddr = qpCtxEntry->headAddr;
    uint32_t curHead = ld_dev((__gm__ uint32_t*)(curHardwareHeadAddr), 0);
    auto curHardwareTailAddr = qpCtxEntry->tailAddr;
    auto depth = qpCtxEntry->depth;
    // Poll CQ if send queue is full
    constexpr uint32_t pollCqTreshold = 10;
    uint32_t curTail = ld_dev((__gm__ uint32_t*)(curHardwareTailAddr), 0);
    if ((curHead + pollCqTreshold) % depth == (curTail) % depth) {
        (void)aclshmemi_udma_poll_cq(destRankId, qpIdx, curTail + ACLSHMEM_NUM_CQE_PER_POLL_CQ);
    }

    // Get memory info
    auto myPE = aclshmemi_get_my_pe();
    __gm__ ACLSHMEMUBmemInfo* localMemInfo = (__gm__ ACLSHMEMUBmemInfo*)(udmaInfo->memPtr + sizeof(ACLSHMEMUBmemInfo) * myPE);
    __gm__ ACLSHMEMUBmemInfo* remoteMemInfo = (__gm__ ACLSHMEMUBmemInfo*)(udmaInfo->memPtr + sizeof(ACLSHMEMUBmemInfo) * destRankId);

    // Write SQE to HBM
    __gm__ uint8_t* wqeAddr = (__gm__ uint8_t*)(sqBaseAddr + wqeSize * (curHead % depth));
    __gm__ ACLSHMEMSqeCtx* sqeCtx = (__gm__ ACLSHMEMSqeCtx*)wqeAddr;
    
    // Fill SQE control information (reference udma_fill_write_sqe logic)
    sqeCtx->opcode = static_cast<uint32_t>(opcode);
    sqeCtx->flag = 0b00100010;
    sqeCtx->nf = 0; // Need fence
    sqeCtx->tokenEn = remoteMemInfo->tokenValueValid;
    sqeCtx->rmtJettyType = remoteMemInfo->rmtJettyType;
    constexpr uint32_t MAX_SGE_NUM_SHIFT = 2;
    sqeCtx->owner = (curHead & (depth << MAX_SGE_NUM_SHIFT)) == 0 ? 1 : 0;
    sqeCtx->targetHint = remoteMemInfo->targetHint;
    sqeCtx->inlineMsgLen = 0; // No inline data
    sqeCtx->tpId = remoteMemInfo->tpn;
    sqeCtx->sgeNum = 1; // Single SGE
    sqeCtx->rmtJettyOrSegId = remoteMemInfo->tid;
    sqeCtx->rmtTokenValue = remoteMemInfo->rmtTokenValue;
    
    // Set remote address (reference udma_fill_write_sqe logic)
    uint64_t remoteAddrValue = reinterpret_cast<uint64_t>(remoteAddr);
    sqeCtx->rmtAddrLOrTokenId = remoteAddrValue & 0xFFFFFFFF;
    sqeCtx->rmtAddrHOrTokenValue = (remoteAddrValue >> 32) & 0xFFFFFFFF;
    auto rmtEid = (__gm__ uint64_t*)(remoteMemInfo->eidAddr);
    sqeCtx->rmtEidL = rmtEid[0];
    sqeCtx->rmtEidH = rmtEid[1];
    // Write SGE to HBM
    __gm__ ACLSHMEMSgeCtx* sgeCtx = (__gm__ ACLSHMEMSgeCtx*)(wqeAddr + sizeof(ACLSHMEMSqeCtx));
    sgeCtx->len = messageLen;
    sgeCtx->va = reinterpret_cast<uint64_t>(localAddr);
    // WQE & SGE cache flush
    dcci_cachelines(wqeAddr, sizeof(ACLSHMEMSqeCtx) + sizeof(ACLSHMEMSgeCtx));
    curHead++;
    aclshmemi_udma_post_send_update_info(curHead, qpCtxEntry);
    ACLSHMEM_DEBUG_FUNC(aclshmemi_dump_wqe, wqeAddr);
}

ACLSHMEM_DEVICE void aclshmemi_udma_post_send_update_info(uint32_t curHead, __gm__ ACLSHMEMUDMAWQCtx *&qpCtxEntry)
{
    // Ring SQ Doorbell (reference udma_update_sq_db in UDMA)
    // Note: db address is 64-bit, but we only update 32-bit value
    __gm__ uint32_t* doorBellAddr = (__gm__ uint32_t*)qpCtxEntry->dbAddr;
    st_dev(curHead, doorBellAddr, 0);
    auto curHardwareHeadAddr = qpCtxEntry->headAddr;
    st_dev(curHead, (__gm__ uint32_t*)curHardwareHeadAddr, 0);
    return;
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_write(__gm__ T* destDmaAddr, __gm__ T* srcDmaAddr, uint32_t destRankId,
                                    uint32_t qpIdx, uint64_t messageLen)
{
    aclshmemi_udma_post_send(reinterpret_cast<__gm__ uint8_t*>(destDmaAddr),
                           reinterpret_cast<__gm__ uint8_t*>(srcDmaAddr),
                           destRankId, qpIdx, UDMAOPCODE::UDMA_OP_WRITE, messageLen);
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_read(__gm__ T* destDmaAddr, __gm__ T* srcDmaAddr, uint32_t srcRankId,
                                   uint32_t qpIdx, uint64_t messageLen)
{
    aclshmemi_udma_post_send(reinterpret_cast<__gm__ uint8_t*>(srcDmaAddr),
                           reinterpret_cast<__gm__ uint8_t*>(destDmaAddr),
                           srcRankId, qpIdx, UDMAOPCODE::UDMA_OP_READ, messageLen);
}

ACLSHMEM_DEVICE void aclshmemx_udma_quiet(int pe)
{
    __gm__ ACLSHMEMAIVUDMAInfo* udmaInfo = aclshmemi_udma_qp_info_fetch();
    uint32_t qpNum = udmaInfo->qpNum;
    // Only support one qp, multi-qp will be support later.
    __gm__ ACLSHMEMUDMAWQCtx* qpCtxEntry = (__gm__ ACLSHMEMUDMAWQCtx*)(udmaInfo->sqPtr
        + (pe * qpNum + 0) * sizeof(ACLSHMEMUDMAWQCtx));
    auto curHardwareHeadAddr = qpCtxEntry->headAddr;
    uint32_t curHead = ld_dev((__gm__ uint32_t*)(curHardwareHeadAddr), 0);
    aclshmemi_udma_poll_cq(pe, 0, curHead);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_get_nbi(__gm__ T* dst, __gm__ T* src, uint32_t elem_size, int pe)
{
    if constexpr(ACLSHMEM_UDMA_SUPPORTED) {
        auto ptr = aclshmem_ptr(src, pe);
        aclshmemi_udma_read((__gm__ uint8_t*)dst, (__gm__ uint8_t*)ptr, pe, 0, elem_size * sizeof(T));
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on NPU_ARCH 3510\n");
    }
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_get_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe)
{
    (void)buf;
    aclshmemi_udma_get_nbi(dst, src, elem_size, pe);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_get_nbi(const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src,
                                            const AscendC::LocalTensor<T>& buf, uint32_t elem_size, int pe)
{
    (void)buf;
    aclshmemi_udma_get_nbi((__gm__ T*)dst.GetPhyAddr(), (__gm__ T *)src.GetPhyAddr(), elem_size, pe);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_put_nbi(__gm__ T* dst, __gm__ T* src, uint32_t elem_size, int pe)
{
    if constexpr(ACLSHMEM_UDMA_SUPPORTED) {
        auto ptr = aclshmem_ptr(dst, pe);
        aclshmemi_udma_write((__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, pe, 0, elem_size * sizeof(T));
    } else {
        ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "UDMA is supported only on NPU_ARCH 3510\n");
    }
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_put_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe)
{
    (void)buf;
    aclshmemi_udma_put_nbi(dst, src, elem_size, pe);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_put_nbi(const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src,
                                            const AscendC::LocalTensor<T>& buf,
                                            uint32_t elem_size, int pe, uint32_t sync_id)
{
    (void)buf;
    (void)sync_id;
    auto dstPhyAddr = (__gm__ T *)dst.GetPhyAddr();
    auto srcPhyAddr = (__gm__ T *)src.GetPhyAddr();
    aclshmemi_udma_put_nbi(dstPhyAddr, srcPhyAddr, elem_size, pe);
}

#endif
