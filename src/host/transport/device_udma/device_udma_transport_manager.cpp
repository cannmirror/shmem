/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>

#include "dl_acl_api.h"
#include "dl_hccp_v2_api.h"
#include "mem_entity_def.h"
#include "shmemi_logger.h"
#include "utils/shmemi_host_types.h"
#include "device_udma_transport_manager.h"

namespace shm {
namespace transport {
namespace device {
bool UdmaTransportManager::tsdOpened_ = false;
bool UdmaTransportManager::raInitialized_ = false;
void *UdmaTransportManager::storedCtxHandle_ = nullptr;
int UdmaTransportManager::subPid_ = 0;

UdmaTransportManager::UdmaTransportManager() noexcept {}

UdmaTransportManager::~UdmaTransportManager() noexcept
{
    CleanupResources();
    tsdOpened_ = false;
    raInitialized_ = false;
    storedCtxHandle_ = nullptr;
    subPid_ = 0;
}

Result UdmaTransportManager::OpenDevice(const TransportOptions &options)
{
    int32_t userId = -1;
    int32_t logicId = -1;

    SHM_LOG_DEBUG(rankId_ << " begin to open device with " << options);
    auto ret = DlAclApi::AclrtGetDevice(&userId);
    SHM_ASSERT_LOG_AND_RETURN(ret == 0 && userId >= 0,
        "AclrtGetDevice() return=" << ret << ", output deviceId=" << userId, ACLSHMEM_DL_FUNC_FAILED);

    ret = DlAclApi::RtGetLogicDevIdByUserDevId(userId, &logicId);
    SHM_ASSERT_LOG_AND_RETURN(ret == 0 && logicId >= 0,
        "RtGetLogicDevIdByUserDevId() return=" << ret << ", output deviceId=" << logicId, ACLSHMEM_DL_FUNC_FAILED);

    deviceId_ = static_cast<uint32_t>(logicId);
    rankId_ = options.rankId;
    rankCount_ = options.rankCount;
    role_ = options.role;

    if (!PrepareOpenDevice(deviceId_, rankCount_, ctxHandle_, logicId)) {
        SHM_LOG_ERROR("PrepareOpenDevice failed.");
        return ACLSHMEM_INNER_ERROR;
    }

    deviceJettyManager_ = new DeviceJettyManager(deviceId_, rankId_, rankCount_);
    deviceJettyManager_->SetTokenIdHandle(tokenIdHandle_);
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::CloseDevice()
{
    if (deviceJettyManager_ != nullptr) {
        deviceJettyManager_->Shutdown();
        delete deviceJettyManager_;
        deviceJettyManager_ = nullptr;
    }
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::RegisterMemoryRegion(const TransportMemoryRegion &mr)
{
    struct MrRegInfoT mrInfo = {};
    mrInfo.in.mem.addr = mr.addr;
    mrInfo.in.mem.size = mr.size;
    mrInfo.in.ub.tokenValue = mr.tokenValue;
    mrInfo.in.ub.tokenIdHandle = tokenIdHandle_;
    mrInfo.in.ub.flags.bs.access = MEM_SEG_ACCESS_DEFAULT;
    mrInfo.in.ub.flags.bs.cacheable = mr.cacheable;
    mrInfo.in.ub.flags.bs.tokenIdValid = mr.tokenIdValid;
    mrInfo.in.ub.flags.bs.nonPin = 0;
    mrInfo.in.ub.flags.bs.userIova = 0;
    mrInfo.in.ub.flags.bs.tokenPolicy = URMA_TOKEN_PLAIN_TEXT;

    auto ret = shm::DlHccpV2Api::RaCtxLmemRegister(ctxHandle_, &mrInfo, &lmemHandle_);
    if (ret != 0) {
        SHM_LOG_ERROR("Register MR = " << mr << " failed, ret = " << ret);
        return ACLSHMEM_INNER_ERROR;
    }

    RegMemResultInfo regResult{mr.addr, mr.size, lmemHandle_, mrInfo.out.key, mrInfo.out.ub.tokenId,
        mrInfo.in.ub.tokenValue, mrInfo.out.ub.targetSegHandle, tokenIdHandle_, mr.cacheable, mr.access};
    localMR_ = regResult;
    SHM_LOG_INFO("Register MR success, regResult = " << regResult);
    ACLSHMEMUBmemInfo localMemInfo{};
    localMemInfo.token_value_valid = mr.tokenIdValid;
    localMemInfo.rmt_jetty_type = 1;
    localMemInfo.target_hint = 0;
    localMemInfo.tpn = 0;
    localMemInfo.tid = mrInfo.out.ub.tokenId >> 8;
    localMemInfo.rmt_token_value = mr.tokenValue;
    localMemInfo.len = mr.size;
    localMemInfo.addr = mr.addr;
    ret = deviceJettyManager_->SetLocalMemInfo(localMemInfo);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Jetty manager set mr failed, ret = " << ret);
        return ret;
    }
    ret = deviceJettyManager_->SetSwapEid(hccpEid_);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Jetty manager set swap eid failed, ret = " << ret);
        return ret;
    }
    registerMRS_.emplace(mr.addr, regResult);
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::UnregisterMemoryRegion(uint64_t addr)
{
    auto pos = registerMRS_.find(addr);
    if (pos == registerMRS_.end()) {
        SHM_LOG_ERROR("Input address not register!");
        return ACLSHMEM_INVALID_PARAM;
    }

    auto ret = shm::DlHccpV2Api::RaCtxLmemUnregister(ctxHandle_, pos->second.lmemHandle);
    if (ret != 0) {
        SHM_LOG_ERROR("Unregister MR failed, ret = " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    registerMRS_.erase(pos);
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::Prepare(const HybmTransPrepareOptions &options)
{
    SHM_LOG_DEBUG("UdmaTransportManager Prepare with options: " << options);
    int ret;
    if ((ret = CheckPrepareOptions(options)) != 0) {
        return ret;
    }

    ret = deviceJettyManager_->Startup(ctxHandle_);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Jetty manager startup failed, ret = " << ret);
        return ret;
    }

    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::Connect()
{
    auto ret = AsyncConnect();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Async connect failed, ret = " << ret);
        return ret;
    }
    SHM_LOG_INFO("Async connect success");

    return ACLSHMEM_SUCCESS;
}

const void *UdmaTransportManager::GetQpInfo() const
{
    return deviceJettyManager_->GetJettyInfoAddress();
}

Result UdmaTransportManager::AsyncConnect()
{
    std::vector<RegMemResultInfo> mrList(rankCount_);
    g_boot_handle.allgather(&localMR_, mrList.data(), sizeof(RegMemResultInfo), &g_boot_handle);
    memoryHandleList_ = (void **)calloc(rankCount_, sizeof(void *));
    if (memoryHandleList_ == nullptr) {
        SHM_LOG_ERROR("Calloc memoryHandleList failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    struct MrImportInfoT mrImportInfo = {};
    void *rmemHandle = nullptr;
    int idx = 0;
    for (auto &mr : mrList) {
        if (mr.lmemHandle == lmemHandle_) {
            memoryHandleList_[idx++] = lmemHandle_;
            continue;
        }
        mrImportInfo.in.key = mr.key;
        mrImportInfo.in.ub.tokenValue = mr.tokenValue;
        mrImportInfo.in.ub.flags.bs.cacheable = mr.cacheable;
        mrImportInfo.in.ub.flags.bs.access = mr.access;
        auto ret = shm::DlHccpV2Api::RaCtxRmemImport(ctxHandle_, &mrImportInfo, &rmemHandle);
        if (ret != 0) {
            SHM_LOG_ERROR("RaCtxRmemImport failed, ret =" << ret);
            return ACLSHMEM_INNER_ERROR;
        }
        memoryHandleList_[idx++] = rmemHandle;
    }
    SHM_LOG_INFO("Import remote MR success.");
    return ACLSHMEM_SUCCESS;
}

bool UdmaTransportManager::PrepareOpenDevice(
    uint32_t deviceId, uint32_t rankCount, void *&ctxHandle, uint32_t logicDeviceId)
{
    if (!OpenTsd(deviceId, rankCount)) {
        SHM_LOG_ERROR("Open tsd failed.");
        return false;
    }

    if (!RaInit(deviceId)) {
        SHM_LOG_ERROR("RaInit failed.");
        return false;
    }

    if (!RaCtxInit(deviceId, ctxHandle)) {
        SHM_LOG_ERROR("RaRdevInit failed.");
        return false;
    }

    return true;
}

bool UdmaTransportManager::OpenTsd(uint32_t deviceId, uint32_t rankCount)
{
    if (tsdOpened_) {
        SHM_LOG_INFO("Tsd already opened.");
        return true;
    }
    struct ProcOpenArgs args = {};
    args.procType = TSD_SUB_PROC_HCCP;
    args.filePath = nullptr;
    char paramInfo0[20] = "--hdcType=18";
    ProcExtParam extParam;
    extParam.paramInfo = paramInfo0;
    extParam.paramLen = sizeof("--hdcType=18");
    args.extParamList = &extParam;
    args.extParamCnt = 1UL;
    args.pathLen = 0UL;
    int subPid = 0;
    args.subPid = &subPid;
    auto ret = shm::DlHccpV2Api::TsdProcessOpen(deviceId, &args);
    if (ret != 0) {
        SHM_LOG_ERROR("TsdProcessOpen for (deviceId = " << deviceId << ", rankCount = " << rankCount
                                                        << ") failed, ret = " << ret);
        return false;
    }
    subPid_ = subPid;
    SHM_LOG_DEBUG("Open tsd for device id: " << deviceId << ", rank count: " << rankCount << " success.");
    tsdOpened_ = true;
    return true;
}

bool UdmaTransportManager::RaInit(uint32_t deviceId)
{
    if (raInitialized_) {
        SHM_LOG_INFO("Ra already initialized.");
        return true;
    }

    RaInitConfig initConfig{};
    initConfig.phyId = deviceId;
    initConfig.nicPosition = NETWORK_OFFLINE;
    initConfig.hdcType = HDC_SERVICE_TYPE_RDMA_V2;
    initConfig.enableHdcAsync = 1;

    SHM_LOG_DEBUG("RaInit = " << initConfig);
    auto ret = shm::DlHccpV2Api::RaInit(&initConfig);
    if (ret != 0) {
        SHM_LOG_ERROR("RaInit failed, ret = " << ret << ", device id: " << deviceId);
        return false;
    }

    SHM_LOG_DEBUG("RaInit for device id: " << deviceId << " success.");
    raInitialized_ = true;
    return true;
}

bool UdmaTransportManager::RaGetDevEidInfoNum(uint32_t deviceId, unsigned int &num)
{
    struct RaInfo info = {NETWORK_PEER_ONLINE, 0};
    info.phyId = deviceId;
    info.mode = NETWORK_OFFLINE;
    SHM_LOG_DEBUG("RaInfo=" << info);
    auto ret = shm::DlHccpV2Api::RaGetDevEidInfoNum(info, &num);
    if (ret != 0) {
        SHM_LOG_ERROR("RaGetDevEidInfoNum failed, ret = " << ret);
        return false;
    }
    SHM_LOG_DEBUG("RaGetDevEidInfoNum = " << num);
    return true;
}

bool UdmaTransportManager::RaGetDevEidInfoList(uint32_t deviceId, unsigned int eidNum, struct CtxInitAttr *attr)
{
    struct RaInfo info = {NETWORK_PEER_ONLINE, 0};
    info.phyId = deviceId;
    info.mode = NETWORK_OFFLINE;
    SHM_LOG_DEBUG("RaInfo=" << info);
    unsigned int infoListNum = eidNum;
    struct DevEidInfo *infoList = (struct DevEidInfo *)calloc(eidNum, sizeof(struct DevEidInfo));
    if (infoList == nullptr) {
        SHM_LOG_ERROR("DevEidInfo malloc failed.");
        return false;
    }
    int ret = shm::DlHccpV2Api::RaGetDevEidInfoList(info, infoList, &infoListNum);
    if (ret != 0 || infoListNum != eidNum) {
        SHM_LOG_ERROR("RaGetDevEidInfoList failed, info_list_num =" << infoListNum);
        aclrtFree(infoList);
        return false;
    }
    SHM_LOG_INFO("RaGetDevEidInfoList success.");
    attr->ub.eid = infoList[0].eid;
    attr->ub.eidIndex = infoList[0].eidIndex;

    free(infoList);
    return true;
}

bool UdmaTransportManager::RaCtxTokenId(void *ctxHandle, void *&tokenIdHandle)
{
    struct HccpTokenId tokenId = {0};
    auto ret = shm::DlHccpV2Api::RaCtxTokenIdAlloc(ctxHandle, &tokenId, &tokenIdHandle);
    if (ret != 0) {
        SHM_LOG_ERROR("RaCtxTokenIdAlloc failed, ret = " << ret);
        return false;
    }
    tokenId_ = tokenId.tokenId;
    SHM_LOG_DEBUG("RaCtxTokenIdAlloc success, tokenId = " << tokenId.tokenId);
    return true;
}

bool UdmaTransportManager::RaCtxInit(uint32_t deviceId, void *&ctxHandle)
{
    if (storedCtxHandle_ != nullptr) {
        SHM_LOG_INFO("CtxHandle already initialized.");
        ctxHandle = storedCtxHandle_;
        return true;
    }

    unsigned int eidNum = 0;
    if (!RaGetDevEidInfoNum(deviceId, eidNum)) {
        return false;
    }
    struct CtxInitAttr attr = {};
    attr.phyId = deviceId;
    struct CtxInitCfg cfg = {};
    cfg.mode = NETWORK_OFFLINE;
    if (!RaGetDevEidInfoList(deviceId, eidNum, &attr)) {
        return false;
    }
    hccpEid_ = attr.ub.eid;
    SHM_LOG_DEBUG("CtxInitAttr = " << attr);
    auto ret = shm::DlHccpV2Api::RaCtxInit(&cfg, &attr, &ctxHandle);
    if (ret != 0) {
        SHM_LOG_ERROR("RaCtxInit failed,  ret = " << ret);
        return false;
    }
    if (!RaCtxTokenId(ctxHandle, tokenIdHandle_)) {
        return false;
    }
    storedCtxHandle_ = ctxHandle;
    SHM_LOG_INFO("RaCtxInit success.");
    return true;
}

Result UdmaTransportManager::CheckPrepareOptions(const HybmTransPrepareOptions &options)
{
    if (role_ != HYBM_ROLE_PEER) {
        SHM_LOG_INFO("Transport role: " << role_ << " check options passed.");
        return ACLSHMEM_SUCCESS;
    }

    if (options.options.size() > rankCount_) {
        SHM_LOG_ERROR("Options size() is " << options.options.size() << " larger than rank count: " << rankCount_);
        return ACLSHMEM_INVALID_PARAM;
    }

    if (options.options.find(rankId_) == options.options.end()) {
        SHM_LOG_ERROR("Options do not contain self rankId: " << rankId_);
        return ACLSHMEM_INVALID_PARAM;
    }

    for (auto it = options.options.begin(); it != options.options.end(); ++it) {
        if (it->first >= rankCount_) {
            SHM_LOG_ERROR("RankId: " << it->first << " is out of range [0, " << rankCount_ << ")");
            return ACLSHMEM_INVALID_PARAM;
        }
    }

    return ACLSHMEM_SUCCESS;
}

void UdmaTransportManager::CleanupResources()
{
    if (memoryHandleList_ != nullptr) {
        for (int i = 0; i < rankCount_; ++i) {
            if (i == rankId_) {
                continue;
            }
            auto ret = shm::DlHccpV2Api::RaCtxRmemUnimport(ctxHandle_, memoryHandleList_[i]);
            if (ret != 0) {
                SHM_LOG_ERROR("Rmem unimport failed, rankId = " << i << ", ret = " << ret);
                return;
            }
        }
        free(memoryHandleList_);
        memoryHandleList_ = nullptr;
        SHM_LOG_INFO("Unimport remote handle success.");
    }

    if (lmemHandle_ != nullptr) {
        auto ret = shm::DlHccpV2Api::RaCtxLmemUnregister(ctxHandle_, lmemHandle_);
        if (ret != 0) {
            SHM_LOG_ERROR("Unregister failed, ret = " << ret << ", lmemHandle = " << lmemHandle_);
            return;
        }
        lmemHandle_ = nullptr;
        SHM_LOG_INFO("Unregister memory success.");
    }

    if (tokenIdHandle_ != nullptr) {
        auto ret = shm::DlHccpV2Api::RaCtxTokenIdFree(ctxHandle_, tokenIdHandle_);
        if (ret != 0) {
            SHM_LOG_ERROR("RaCtxTokenIdFree failed, ret = " << ret);
            return;
        }
        tokenIdHandle_ = nullptr;
        SHM_LOG_INFO("RaCtxTokenIdFree success.");
    }

    if (ctxHandle_ != nullptr) {
        auto ret = shm::DlHccpV2Api::RaCtxDeinit(ctxHandle_);
        if (ret != 0) {
            SHM_LOG_ERROR("RaCtxDeinit failed, ret = " << ret);
            return;
        }
        ctxHandle_ = nullptr;
        SHM_LOG_INFO("RaCtxDeinit success.");
    }

    RaInitConfig config{};
    config.phyId = deviceId_;
    config.nicPosition = NETWORK_OFFLINE;
    config.hdcType = HDC_SERVICE_TYPE_RDMA_V2;
    config.enableHdcAsync = 1;
    auto ret = shm::DlHccpV2Api::RaDeinit(&config);
    if (ret != 0) {
        SHM_LOG_ERROR("Hccp Deinit RA failed, ret = " << ret);
        return;
    }
    SHM_LOG_INFO("RaDeinit success.");

    ret = shm::DlHccpV2Api::TsdProcessClose(deviceId_, subPid_);
    if (ret != 0) {
        SHM_LOG_ERROR("TsdClose for deviceId = " << deviceId_ << " failed, ret = " << ret);
        return;
    }
    SHM_LOG_INFO("TsdProcessClose success.");
}

} // namespace device
} // namespace transport
} // namespace shm
