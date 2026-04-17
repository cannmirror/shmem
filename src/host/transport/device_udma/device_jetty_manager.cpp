/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "device_jetty_manager.h"

#include <acl/acl.h>
#include <cmath>
#include <vector>

#include "shmemi_host_common.h"
#include "host/shmem_host_def.h"
#include "runtime/mem.h"

constexpr uint32_t CQ_DEPTH_DEFAULT = 16384;
constexpr uint32_t SQ_DEPTH_DEFAULT = 4096;
constexpr uint32_t RQ_DEPTH_DEFAULT = 256;
constexpr uint32_t MAX_SQE_BB_NUM = 4;
constexpr uint32_t SQ_BASKBLK_CNT = SQ_DEPTH_DEFAULT * MAX_SQE_BB_NUM;
constexpr uint8_t RNR_RETRY_COUNT_DEFAULT = 7;

namespace shm {
namespace transport {
namespace device {
DeviceJettyManager::DeviceJettyManager(uint32_t deviceId, uint32_t rankId, uint32_t rankCount) noexcept
    : deviceId_{deviceId}, rankId_{rankId}, rankCount_{rankCount}, 
    wqInfoList_(rankCount_),
    cqInfoList_(rankCount_),
    ubMemInfoList_(rankCount_),
    hccpEidList_(rankCount_),
    tpnList_(rankCount_),
    qpKeyList_(rankCount_),
    allQpImportInfoT_(rankCount_),
    remoteQpHandleList_(rankCount_)
{}

DeviceJettyManager::~DeviceJettyManager() noexcept
{
    Shutdown();
}

Result DeviceJettyManager::Shutdown() noexcept
{
    if (cqPiAddr_ != nullptr) {
        aclrtFree(cqPiAddr_);
        cqPiAddr_ = nullptr;
    }
    if (cqCiAddr_ != nullptr) {
        aclrtFree(cqCiAddr_);
        cqCiAddr_ = nullptr;
    }
    if (sqPiAddr_ != nullptr) {
        aclrtFree(sqPiAddr_);
        sqPiAddr_ = nullptr;
    }
    if (sqCiAddr_ != nullptr) {
        aclrtFree(sqCiAddr_);
        sqCiAddr_ = nullptr;
    }
    if (wqeCntAddr_ != nullptr) {
        aclrtFree(wqeCntAddr_);
        wqeCntAddr_ = nullptr;
    }
    if (amoAddr_ != nullptr) {
        aclrtFree(amoAddr_);
        amoAddr_ = nullptr;
    }
    if (udmaInfo_ != nullptr) {
        aclrtFree(udmaInfo_);
        udmaInfo_ = nullptr;
    }
    if (hccpEidDevice_ != nullptr) {
        aclrtFree(hccpEidDevice_);
        hccpEidDevice_ = nullptr;
    }
    int ret = 0;
    if (transportMode_ != TransportModeT::CONN_RM && qpHandle_ != nullptr) {
        ret = DlHccpV2Api::RaCtxQpUnbind(qpHandle_);
        if (ret != 0) {
            SHM_LOG_ERROR("Qp unbind failed, ret = " << ret);
            return ACLSHMEM_INNER_ERROR;
        }
    }

    if (remoteQpHandleList_.size() == rankCount_) {
        for (uint32_t i = 0; i < rankCount_; ++i) {
            if (i == rankId_ || remoteQpHandleList_[i] == nullptr) {
                continue;
            }
            ret = DlHccpV2Api::RaCtxQpUnimport(ctxHandle_, remoteQpHandleList_[i]);
            if (ret != 0) {
                SHM_LOG_ERROR("Qp unimport failed, rankId: " << i << " ret: " << ret);
                return ACLSHMEM_INNER_ERROR;
            }
        }
        remoteQpHandleList_.clear();
    }

    if (qpHandle_ != nullptr) {
        ret = DlHccpV2Api::RaCtxQpDestroy(qpHandle_);
        if (ret != 0) {
            SHM_LOG_ERROR("Qp destroy failed, ret = " << ret);
            return ACLSHMEM_INNER_ERROR;
        }
        qpHandle_ = nullptr;
    }

    if (cqHandle_ != nullptr) {
        ret = DlHccpV2Api::RaCtxCqDestroy(ctxHandle_, cqHandle_);
        if (ret != 0) {
            SHM_LOG_ERROR("Cq destroy failed, ret = " << ret);
            return ACLSHMEM_INNER_ERROR;
        }
        cqHandle_ = nullptr;
    }

    if (chanHandle_ != nullptr) {
        ret = DlHccpV2Api::RaCtxChanDestroy(ctxHandle_, chanHandle_);
        if (ret != 0) {
            SHM_LOG_ERROR("Channel destroy failed, ret = " << ret);
            return ACLSHMEM_INNER_ERROR;
        }
        chanHandle_ = nullptr;
    }
    return ACLSHMEM_SUCCESS;
}

Result DeviceJettyManager::SetLocalMemInfo(const ACLSHMEMUBmemInfo &localMemInfo) noexcept
{
    localMemInfo_ = localMemInfo;
    return ACLSHMEM_SUCCESS;
}

Result DeviceJettyManager::SetSwapEid(const HccpEid &hccpEid) noexcept
{
    // after import jetty, eid should be __builtin_bswap64
    uint64_t eidL, eidH;
    memcpy(&eidL, hccpEid.raw, sizeof(uint64_t));
    memcpy(&eidH, hccpEid.raw + sizeof(uint64_t), sizeof(uint64_t));
    eidL = __builtin_bswap64(eidL);
    eidH = __builtin_bswap64(eidH);
    memcpy(localHccpEid_.raw, &eidH, sizeof(uint64_t));
    memcpy(localHccpEid_.raw + sizeof(uint64_t), &eidL, sizeof(uint64_t));
    return ACLSHMEM_SUCCESS;
}

Result DeviceJettyManager::SetTokenIdHandle(void *tokenIdHandle) noexcept
{
    tokenIdHandle_ = tokenIdHandle;
    return ACLSHMEM_SUCCESS;
}

bool DeviceJettyManager::ReserveUdmaInfoSpace() noexcept
{
    if (udmaInfo_ != nullptr) {
        return true;
    }

    constexpr int32_t qpNum = 1;
    auto wqSize = sizeof(ACLSHMEMUDMAWQCtx) * qpNum;
    auto cqSize = sizeof(ACLSHMEMUDMACqCtx) * qpNum;
    auto oneQpSize = 2U * (wqSize + cqSize) + sizeof(ACLSHMEMUBmemInfo) * qpNum; // 2 means (sq + rq) (scq + rcq)
    udmaInfoSize_ = sizeof(ACLSHMEMAIVUDMAInfo) + oneQpSize * rankCount_;

    SHM_VALIDATE_RETURN(aclrtMalloc(&udmaInfo_, udmaInfoSize_, ACL_MEM_MALLOC_HUGE_FIRST) == 0,
        "Allocate device size: " << udmaInfoSize_ << " for udmaInfo failed", false);
    SHM_VALIDATE_RETURN(aclrtMalloc(&hccpEidDevice_, rankCount_ * sizeof(HccpEid), ACL_MEM_MALLOC_HUGE_FIRST) == 0,
        "Allocate device size: " << rankCount_ * sizeof(HccpEid) << " for eid failed", false);
    return true;
}

Result DeviceJettyManager::JFCCreate() noexcept
{
    ChanInfoT chanInfo = {0};
    chanInfo.in.dataPlaneFlag.bs.poolCqCstm = 1; // default 0:hccp poll cq; 1: caller poll cq
    auto ret = DlHccpV2Api::RaCtxChanCreate(ctxHandle_, &chanInfo, &chanHandle_);
    if (ret != 0) {
        SHM_LOG_ERROR("Create udma channel faild: " << ret);
        return ACLSHMEM_INNER_ERROR;
    }

    cqInfo_.in.chanHandle = chanHandle_;
    cqInfo_.in.depth = CQ_DEPTH_DEFAULT;           // optional, normal mode default 16384
    cqInfo_.in.ub.userCtx = 0;                     // optional, default 0
    cqInfo_.in.ub.mode = JFC_MODE_USER_CTL_NORMAL; // corresponding with jetty mode : JETTY_MODE_USER_CTL_NORMAL
    cqInfo_.in.ub.ceqn = 0;                        // optional, default 0
    cqInfo_.in.ub.flag.bs.lockFree = 0;            // optional, default 0
    cqInfo_.in.ub.flag.bs.jfcInline = 0;           // optional, default 0
    ret = DlHccpV2Api::RaCtxCqCreate(ctxHandle_, &cqInfo_, &cqHandle_);
    if (ret != 0) {
        SHM_LOG_ERROR("Create udma jfc create failed, ret = " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    cqVa_ = cqInfo_.out.va;

    // save & allgather cq related info
    ACLSHMEMUDMACqCtx localCq;
    localCq.cqn = 0;
    localCq.bufAddr = cqInfo_.out.bufAddr;
    localCq.cqeShiftSize = log2(cqInfo_.out.cqeSize); // cqeSize = 64 = 2^6, cqeShiftSize此处取6
    localCq.depth = cqInfo_.in.depth;
    aclrtMalloc(&cqPiAddr_, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(cqPiAddr_, sizeof(uint32_t), 0, sizeof(uint32_t));
    localCq.headAddr = reinterpret_cast<uintptr_t>(cqPiAddr_);
    aclrtMalloc(&cqCiAddr_, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(cqCiAddr_, sizeof(uint32_t), 0, sizeof(uint32_t));
    localCq.tailAddr = reinterpret_cast<uintptr_t>(cqCiAddr_);
    localCq.dbMode = ACLSHMEMUDMADBMode::SW_DB;
    localCq.dbAddr = cqInfo_.out.swdbAddr;
    g_boot_handle.allgather((void *)&localCq, cqInfoList_.data(), sizeof(ACLSHMEMUDMACqCtx), &g_boot_handle);

    SHM_LOG_INFO("Cq create success.");
    return ACLSHMEM_SUCCESS;
}

Result DeviceJettyManager::JettyCreate() noexcept
{
    QpCreateAttr qpCreateAttr = {0};
    qpCreateAttr.scqHandle = cqHandle_;
    qpCreateAttr.rcqHandle = cqHandle_;
    qpCreateAttr.srqHandle = cqHandle_;
    qpCreateAttr.sqDepth = SQ_DEPTH_DEFAULT; // optional, default 4096
    qpCreateAttr.rqDepth = RQ_DEPTH_DEFAULT; // optional, default 256
    qpCreateAttr.transportMode = transportMode_;

    qpCreateAttr.ub.mode = JettyMode::JETTY_MODE_USER_CTL_NORMAL;
    qpCreateAttr.ub.jettyId = 0;       // 0 means not specified
    qpCreateAttr.ub.flag.value = 1;    // URMA_SHARE_JFR
    qpCreateAttr.ub.jfsFlag.value = 2; // 0b10
                                       /* default as 0, lock protected */
                                       /*  1: error suspend */
    qpCreateAttr.ub.tokenValue = TOKEN_VALUE;
    qpCreateAttr.ub.priority = 0;
    qpCreateAttr.ub.rnrRetry = RNR_RETRY_COUNT_DEFAULT;
    qpCreateAttr.ub.errTimeout = 0;

    qpCreateAttr.ub.extMode.piType = 0;             // optional, default 0 op mode
    qpCreateAttr.ub.extMode.cstmFlag.bs.sqCstm = 0; // optional, USER_CTL_NORMAL default is 0, sqbuff no need, others default 1
    qpCreateAttr.ub.extMode.sqebbNum = SQ_DEPTH_DEFAULT;
    qpCreateAttr.ub.tokenIdHandle = tokenIdHandle_;

    int ret = DlHccpV2Api::RaCtxQpCreate(ctxHandle_, &qpCreateAttr, &qpCreateInfo_, &qpHandle_);
    if (ret != 0) {
        SHM_LOG_ERROR("Qp create failed, ret = " << ret);
        return ACLSHMEM_INNER_ERROR;
    }

    // save & allgather wq related info
    ACLSHMEMUDMAWQCtx localWq;
    localWq.wqn = 0;
    localWq.bufAddr = qpCreateInfo_.ub.sqBuffVa;
    localWq.wqeShiftSize = log2(qpCreateInfo_.ub.wqebbSize); // wqeSize = 64 = 2^6, wqeShiftSize此处取6
    localWq.depth = SQ_BASKBLK_CNT;
    aclrtMalloc(&sqPiAddr_, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(sqPiAddr_, sizeof(uint32_t), 0, sizeof(uint32_t));
    localWq.headAddr = reinterpret_cast<uintptr_t>(sqPiAddr_);
    aclrtMalloc(&sqCiAddr_, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(sqCiAddr_, sizeof(uint32_t), 0, sizeof(uint32_t));
    localWq.tailAddr = reinterpret_cast<uintptr_t>(sqCiAddr_);
    localWq.dbMode = ACLSHMEMUDMADBMode::SW_DB;
    localWq.dbAddr = qpCreateInfo_.ub.dbAddr;
    localWq.sl = 0;
    aclrtMalloc(&wqeCntAddr_, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(wqeCntAddr_, sizeof(uint32_t), 0, sizeof(uint32_t));
    localWq.wqeCntAddr = reinterpret_cast<uintptr_t>(wqeCntAddr_);
    aclrtMalloc(&amoAddr_, sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(amoAddr_, sizeof(uint64_t), 0, sizeof(uint64_t));
    localWq.amoAddr = reinterpret_cast<uintptr_t>(amoAddr_);
    g_boot_handle.allgather((void *)&localWq, wqInfoList_.data(), sizeof(ACLSHMEMUDMAWQCtx), &g_boot_handle);

    SHM_LOG_INFO("Qp create success.");
    return ACLSHMEM_SUCCESS;
}

Result DeviceJettyManager::JettyImport() noexcept
{
    QpImportInfoT qpImportInfoT = {0};
    qpImportInfoT.in.ub.mode = JettyImportMode::JETTY_IMPORT_MODE_NORMAL;
    qpImportInfoT.in.ub.tokenValue = TOKEN_VALUE; // same as qpCreateattr.ub.tokenValue
    qpImportInfoT.in.ub.policy = JettyGrpPolicy::JETTY_GRP_POLICY_RR;
    qpImportInfoT.in.ub.type = TargetType::TARGET_TYPE_JETTY;
    qpImportInfoT.in.ub.flag.bs.tokenPolicy = TokenPolicy::TOKEN_POLICY_PLAIN_TEXT;
    qpImportInfoT.in.ub.tpType = 1; // mode ctp
    g_boot_handle.allgather((void *)&qpImportInfoT, allQpImportInfoT_.data(), sizeof(QpImportInfoT), &g_boot_handle);
    g_boot_handle.allgather((void *)&qpCreateInfo_.key, qpKeyList_.data(), sizeof(QpKeyT), &g_boot_handle);

    for (uint32_t i = 0; i < rankCount_; i++) {
        if (i == rankId_) {
            continue;
        }
        allQpImportInfoT_[i].in.key = qpKeyList_[i];
        int ret = DlHccpV2Api::RaCtxQpImport(ctxHandle_, &(allQpImportInfoT_[i]), &remoteQpHandleList_[i]);
        if (ret != 0) {
            SHM_LOG_ERROR("Qp import failed, rankId: " << i << " ret: " << ret);
            return ACLSHMEM_INNER_ERROR;
        }
        tpnList_[i] = allQpImportInfoT_[i].out.ub.tpn;
    }
    SHM_LOG_INFO("Qp import success");
    return ACLSHMEM_SUCCESS;
}

Result DeviceJettyManager::JettyBind() noexcept
{
    if (transportMode_ == TransportModeT::CONN_RM) {
        return ACLSHMEM_SUCCESS; // no need to bind in RM mode
    }
    for (uint32_t i = 0; i < rankCount_; ++i) {
        if (i == rankId_) {
            continue;
        }
        int ret = DlHccpV2Api::RaCtxQpBind(qpHandle_, remoteQpHandleList_[i]);
        if (ret != 0) {
            SHM_LOG_ERROR("Qp bind failed, rankId: " << i << " ret: " << ret);
            return ACLSHMEM_INNER_ERROR;
        }
    }
    SHM_LOG_INFO("Qp bind success.");
    return ACLSHMEM_SUCCESS;
}

Result DeviceJettyManager::Startup(void *ctx_handle) noexcept
{
    ctxHandle_ = ctx_handle;
    if (!ReserveUdmaInfoSpace()) {
        SHM_LOG_ERROR("Reserve UDMA info space failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    SHM_VALIDATE_RETURN(JFCCreate() == 0, "Create JFC failed.", ACLSHMEM_INNER_ERROR);
    SHM_VALIDATE_RETURN(JettyCreate() == 0, "Create Jetty failed.", ACLSHMEM_INNER_ERROR);
    SHM_VALIDATE_RETURN(JettyImport() == 0, "Jetty import failed.", ACLSHMEM_INNER_ERROR);
    SHM_VALIDATE_RETURN(JettyBind() == 0, "Jetty bind failed.", ACLSHMEM_INNER_ERROR);
    SHM_VALIDATE_RETURN(FillUdmaInfo() == ACLSHMEM_SUCCESS, "Fill udma info failed.", ACLSHMEM_INNER_ERROR);

    return ACLSHMEM_SUCCESS;
}

void *DeviceJettyManager::GetJettyInfoAddress() noexcept
{
    return udmaInfo_;
}

uint64_t DeviceJettyManager::GetJFCInfoAddress() const noexcept
{
    return cqVa_;
}

void DeviceJettyManager::FillUdmaWq(ACLSHMEMUDMAWQCtx &srcWq, ACLSHMEMUDMAWQCtx &dstWq) const
{
    dstWq.wqn = srcWq.wqn;
    dstWq.bufAddr = srcWq.bufAddr;
    dstWq.wqeShiftSize = srcWq.wqeShiftSize;
    dstWq.depth = srcWq.depth;
    dstWq.headAddr = srcWq.headAddr;
    dstWq.tailAddr = srcWq.tailAddr;
    dstWq.dbMode = srcWq.dbMode;
    dstWq.dbAddr = srcWq.dbAddr;
    dstWq.sl = srcWq.sl;
    dstWq.wqeCntAddr = srcWq.wqeCntAddr;
    dstWq.amoAddr = srcWq.amoAddr;
}

void DeviceJettyManager::FillUdmaCq(ACLSHMEMUDMACqCtx &srcCq, ACLSHMEMUDMACqCtx &dstCq) const
{
    dstCq.cqn = srcCq.cqn;
    dstCq.bufAddr = srcCq.bufAddr;
    dstCq.cqeShiftSize = srcCq.cqeShiftSize;
    dstCq.depth = srcCq.depth;
    dstCq.headAddr = srcCq.headAddr;
    dstCq.tailAddr = srcCq.tailAddr;
    dstCq.dbMode = srcCq.dbMode;
    dstCq.dbAddr = srcCq.dbAddr;
}

void DeviceJettyManager::FillUdmaMem(ACLSHMEMUBmemInfo &srcMem, ACLSHMEMUBmemInfo &dstMem) const
{
    dstMem.token_value_valid = srcMem.token_value_valid;
    dstMem.rmt_jetty_type = srcMem.rmt_jetty_type;
    dstMem.target_hint = srcMem.target_hint;
    dstMem.tpn = srcMem.tpn;
    dstMem.tid = srcMem.tid;
    dstMem.rmt_token_value = srcMem.rmt_token_value;
    dstMem.len = srcMem.len;
    dstMem.addr = srcMem.addr;
}

void DeviceJettyManager::PrintHostInfo(ACLSHMEMAIVUDMAInfo &hostInfo) const
{
    SHM_LOG_DEBUG("=======================rank [" << rankId_ << "] host info====================");
    auto tempWQCtx = ((ACLSHMEMUDMAWQCtx *)hostInfo.sqPtr)[rankId_];
    SHM_LOG_DEBUG("rank[" << rankId_ << "] WQCtx.wqn: " << tempWQCtx.wqn);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] WQCtx.bufAddr: " << tempWQCtx.bufAddr);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] WQCtx.wqeShiftSize: " << tempWQCtx.wqeShiftSize);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] WQCtx.depth: " << tempWQCtx.depth);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] WQCtx.headAddr: " << tempWQCtx.headAddr);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] WQCtx.tailAddr: " << tempWQCtx.tailAddr);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] WQCtx.dbMode: " << static_cast<int>(tempWQCtx.dbMode));
    SHM_LOG_DEBUG("rank[" << rankId_ << "] WQCtx.dbAddr: " << tempWQCtx.dbAddr);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] WQCtx.sl: " << tempWQCtx.sl);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] WQCtx.wqeCntAddr: " << tempWQCtx.wqeCntAddr);

    auto tempCQCtx = ((ACLSHMEMUDMACqCtx *)hostInfo.scqPtr)[rankId_];
    SHM_LOG_DEBUG("rank[" << rankId_ << "] CQCtx.cqn: " << tempCQCtx.cqn);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] CQCtx.bufAddr: " << tempCQCtx.bufAddr);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] CQCtx.cqeShiftSize: " << tempCQCtx.cqeShiftSize);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] CQCtx.depth: " << tempCQCtx.depth);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] CQCtx.headAddr: " << tempCQCtx.headAddr);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] CQCtx.tailAddr: " << tempCQCtx.tailAddr);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] CQCtx.dbMode: " << static_cast<int>(tempCQCtx.dbMode));
    SHM_LOG_DEBUG("rank[" << rankId_ << "] CQCtx.dbAddr: " << tempCQCtx.dbAddr);

    auto tempMemInfo = ((ACLSHMEMUBmemInfo *)hostInfo.memPtr)[rankId_];
    SHM_LOG_DEBUG("rank[" << rankId_ << "] MemInfo.token_value_valid: " << tempMemInfo.token_value_valid);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] MemInfo.rmt_jetty_type: " << tempMemInfo.rmt_jetty_type);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] MemInfo.target_hint: " << tempMemInfo.target_hint);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] MemInfo.tpn: " << tempMemInfo.tpn);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] MemInfo.tid: " << tempMemInfo.tid);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] MemInfo.rmt_token_value: " << tempMemInfo.rmt_token_value);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] MemInfo.len: " << tempMemInfo.len);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] MemInfo.addr: " << tempMemInfo.addr);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] MemInfo.eidAddr: " << tempMemInfo.eidAddr);

    auto tempHccpEid = hccpEidList_[rankId_];
    for (auto idx = 0; idx < sizeof(HccpEid); idx++) {
        SHM_LOG_DEBUG("rank[" << rankId_ << "] eid.raw[" << idx << "]: " << static_cast<uint32_t>(tempHccpEid.raw[idx]));
    }
}

Result DeviceJettyManager::FillUdmaInfo() noexcept
{
    g_boot_handle.allgather((void *)&localMemInfo_, ubMemInfoList_.data(), sizeof(ACLSHMEMUBmemInfo), &g_boot_handle);
    g_boot_handle.allgather((void *)&localHccpEid_, hccpEidList_.data(), sizeof(HccpEid), &g_boot_handle);
    g_boot_handle.barrier(&g_boot_handle);
    auto ret = aclrtMemcpy(hccpEidDevice_, rankCount_ * sizeof(HccpEid), hccpEidList_.data(), rankCount_ * sizeof(HccpEid),
        ACL_MEMCPY_HOST_TO_DEVICE);

    // construct udma info in host
    constexpr uint32_t qpNum = 1;
    std::vector<uint8_t> udmaInfoBuffer(udmaInfoSize_, 0);
    auto copyInfo = reinterpret_cast<ACLSHMEMAIVUDMAInfo *>(udmaInfoBuffer.data());
    copyInfo->qpNum = qpNum;
    copyInfo->sqPtr = (uint64_t)(copyInfo + 1);
    copyInfo->rqPtr = (uint64_t)((ACLSHMEMUDMAWQCtx *)copyInfo->sqPtr + rankCount_ * qpNum);
    copyInfo->scqPtr = (uint64_t)((ACLSHMEMUDMAWQCtx *)copyInfo->rqPtr + rankCount_ * qpNum);
    copyInfo->rcqPtr = (uint64_t)((ACLSHMEMUDMACqCtx *)copyInfo->scqPtr + rankCount_ * qpNum);
    copyInfo->memPtr = (uint64_t)((ACLSHMEMUDMACqCtx *)copyInfo->rcqPtr + rankCount_ * qpNum);

    for (size_t rank = 0; rank < rankCount_; rank++) {
        ubMemInfoList_[rank].tpn = tpnList_[rank];
        FillUdmaWq(wqInfoList_[rankId_], ((ACLSHMEMUDMAWQCtx *)copyInfo->sqPtr)[rank]);
        FillUdmaWq(wqInfoList_[rankId_], ((ACLSHMEMUDMAWQCtx *)copyInfo->rqPtr)[rank]);

        FillUdmaCq(cqInfoList_[rankId_], ((ACLSHMEMUDMACqCtx *)copyInfo->scqPtr)[rank]);
        FillUdmaCq(cqInfoList_[rankId_], ((ACLSHMEMUDMACqCtx *)copyInfo->rcqPtr)[rank]);

        FillUdmaMem(ubMemInfoList_[rank], ((ACLSHMEMUBmemInfo *)copyInfo->memPtr)[rank]);
        ((ACLSHMEMUBmemInfo *)(copyInfo->memPtr))[rank].eidAddr = (uint64_t)((HccpEid *)hccpEidDevice_ + rank);
    }
    PrintHostInfo(*copyInfo);

    // link position in device
    copyInfo->sqPtr = (uint64_t)((ACLSHMEMAIVUDMAInfo *)udmaInfo_ + 1);
    copyInfo->rqPtr = (uint64_t)((ACLSHMEMUDMAWQCtx *)copyInfo->sqPtr + rankCount_ * qpNum);
    copyInfo->scqPtr = (uint64_t)((ACLSHMEMUDMAWQCtx *)copyInfo->rqPtr + rankCount_ * qpNum);
    copyInfo->rcqPtr = (uint64_t)((ACLSHMEMUDMACqCtx *)copyInfo->scqPtr + rankCount_ * qpNum);
    copyInfo->memPtr = (uint64_t)((ACLSHMEMUDMACqCtx *)copyInfo->rcqPtr + rankCount_ * qpNum);
    ret = aclrtMemcpy(udmaInfo_, udmaInfoSize_, copyInfo, udmaInfoSize_, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != 0) {
        SHM_LOG_ERROR("Copy udma info to device failed: " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    SHM_LOG_INFO("Copy udma info success");

    return ACLSHMEM_SUCCESS;
}

} // namespace device
} // namespace transport
} // namespace shm
