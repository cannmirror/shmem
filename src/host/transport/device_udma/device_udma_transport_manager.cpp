/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

#include "securec.h"
#include "dl_acl_api.h"
#include "device_udma_def.h"
#include "hcomm/hcomm_res.h"
#include "hcomm/hcomm_res_entity_defs.h"
#include "hybm_mem_segment.h"
#include "shmemi_host_common.h"
#include "../host_device/shmemi_host_device_constant.h"
#include "device_udma_transport_manager.h"

namespace shm {
namespace transport {
namespace device {

namespace {

struct ExchangedEndpointDescInfo {
    uint32_t eidIndex{0};
    uint32_t valid{0};
    EndpointDesc desc{};
};

constexpr int32_t INVALID_EID_INDEX = -1;
constexpr const char ACLSHMEM_HCOMM_MEM_TAG[] = "aclshmem_udma";
// Number of HCOMM channels created per peer. One channel maps to one QP.
constexpr uint32_t ACLSHMEM_HCOMM_CHANNEL_NUM_PER_PEER = 1;
constexpr uint32_t ACLSHMEM_HCOMM_DEFAULT_QOS = 4;

Result AllocateAndClearDeviceBuffer(void*& ptr, uint64_t size, const char* name)
{
    if (ptr != nullptr) {
        return ACLSHMEM_SUCCESS;
    }
    auto ret = DlAclApi::AclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACLSHMEM_SUCCESS || ptr == nullptr) {
        SHM_LOG_ERROR("Allocate " << name << " failed, ret = " << ret << ", size = " << size);
        ptr = nullptr;
        return ACLSHMEM_INNER_ERROR;
    }
    ret = DlAclApi::AclrtMemset(ptr, size, 0, size);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Clear " << name << " failed, ret = " << ret << ", size = " << size);
        (void)DlAclApi::AclrtFree(ptr);
        ptr = nullptr;
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

Result CopyHcommChannelEntityFromDevice(void* dst, uint64_t size, uint64_t devAddr, const char* name, uint32_t peer)
{
    if (dst == nullptr || size == 0 || devAddr == 0) {
        SHM_LOG_ERROR(
            "Invalid Hcomm " << name << " input for peer " << peer << ", size = " << size << ", devAddr = " << devAddr);
        return ACLSHMEM_INNER_ERROR;
    }

    auto ret = DlAclApi::AclrtMemcpy(
        dst, size, reinterpret_cast<const void*>(static_cast<uintptr_t>(devAddr)), size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Copy Hcomm " << name << " from device failed for peer " << peer << ", ret = " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

// Holds the per-peer HCOMM channel contexts read back from device memory. These
// are translated into the legacy udmaInfo layout (WQCtx / CqCtx / UBmemInfo) that
// the unmodified data plane consumes.
struct PeerChannelContexts {
    SqContext sqContext{};
    CqContext cqContext{};
    RegedBufferEntity remoteBuffer{};
};

Result ReadPeerChannelContexts(uint64_t channelPtr, uint32_t peer, PeerChannelContexts& out)
{
    ChannelEntity channel{};
    auto ret = CopyHcommChannelEntityFromDevice(&channel, sizeof(channel), channelPtr, "channel entity", peer);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }

    if (channel.sqNum == 0 || channel.cqNum == 0 || channel.remoteBufferNum == 0 ||
        channel.sqContextAddr == nullptr || channel.cqContextAddr == nullptr || channel.remoteBufferAddr == nullptr) {
        SHM_LOG_ERROR(
            "Invalid Hcomm channel entity for peer " << peer << ", sqNum = " << channel.sqNum
                                                     << ", cqNum = " << channel.cqNum
                                                     << ", remoteBufferNum = " << channel.remoteBufferNum);
        return ACLSHMEM_INNER_ERROR;
    }

    uint64_t sqContextAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(channel.sqContextAddr));
    uint64_t cqContextAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(channel.cqContextAddr));
    uint64_t remoteBufferAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(channel.remoteBufferAddr));

    ret = CopyHcommChannelEntityFromDevice(&out.sqContext, sizeof(out.sqContext), sqContextAddr, "SQ context", peer);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }
    if (out.sqContext.contextInfo.ubJfs.sqDepth != shm::UDMA_SQ_BASKBLK_CNT) {
        SHM_LOG_ERROR(
            "Unexpected Hcomm SQ depth for peer " << peer << ", sqDepth = " << out.sqContext.contextInfo.ubJfs.sqDepth
                                                  << ", expected = " << shm::UDMA_SQ_BASKBLK_CNT);
        return ACLSHMEM_INNER_ERROR;
    }

    ret = CopyHcommChannelEntityFromDevice(&out.cqContext, sizeof(out.cqContext), cqContextAddr, "CQ context", peer);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }
    if (out.cqContext.contextInfo.ubJfc.cqDepth != shm::UDMA_CQ_DEPTH_DEFAULT) {
        SHM_LOG_ERROR(
            "Unexpected Hcomm CQ depth for peer " << peer << ", cqDepth = " << out.cqContext.contextInfo.ubJfc.cqDepth
                                                  << ", expected = " << shm::UDMA_CQ_DEPTH_DEFAULT);
        return ACLSHMEM_INNER_ERROR;
    }

    ret = CopyHcommChannelEntityFromDevice(
        &out.remoteBuffer, sizeof(out.remoteBuffer), remoteBufferAddr, "remote buffer", peer);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }
    return ACLSHMEM_SUCCESS;
}

void SetUdmaInfoSectionPtrs(ACLSHMEMAIVUDMAInfo& info, void* infoBase, uint32_t rankCount, uint32_t qpNum)
{
    info.sqPtr = reinterpret_cast<uint64_t>(static_cast<ACLSHMEMAIVUDMAInfo*>(infoBase) + 1);
    info.rqPtr = reinterpret_cast<uint64_t>(
        reinterpret_cast<ACLSHMEMUDMAWQCtx*>(info.sqPtr) + static_cast<uint64_t>(rankCount) * qpNum);
    info.scqPtr = reinterpret_cast<uint64_t>(
        reinterpret_cast<ACLSHMEMUDMAWQCtx*>(info.rqPtr) + static_cast<uint64_t>(rankCount) * qpNum);
    info.rcqPtr = reinterpret_cast<uint64_t>(
        reinterpret_cast<ACLSHMEMUDMACqCtx*>(info.scqPtr) + static_cast<uint64_t>(rankCount) * qpNum);
    info.memPtr = reinterpret_cast<uint64_t>(
        reinterpret_cast<ACLSHMEMUDMACqCtx*>(info.rcqPtr) + static_cast<uint64_t>(rankCount) * qpNum);
}

} // namespace

UdmaTransportManager::UdmaTransportManager() noexcept {}

UdmaTransportManager::~UdmaTransportManager() noexcept { CleanupResources(); }

Result UdmaTransportManager::OpenDevice(const TransportOptions& options)
{
    int32_t userId = -1;
    int32_t logicId = -1;

    auto ret = DlAclApi::AclrtGetDevice(&userId);
    SHM_ASSERT_LOG_AND_RETURN(
        ret == 0 && userId >= 0, "AclrtGetDevice() return=" << ret << ", output deviceId=" << userId,
        ACLSHMEM_DL_FUNC_FAILED);

    ret = DlAclApi::RtGetLogicDevIdByUserDevId(userId, &logicId);
    SHM_ASSERT_LOG_AND_RETURN(
        ret == 0 && logicId >= 0, "RtGetLogicDevIdByUserDevId() return=" << ret << ", output deviceId=" << logicId,
        ACLSHMEM_DL_FUNC_FAILED);

    const uint32_t deviceId = static_cast<uint32_t>(logicId);
    rankId_ = options.rankId;
    rankCount_ = options.rankCount;
    role_ = options.role;

    if (!PrepareOpenDevice(deviceId, rankCount_)) {
        SHM_LOG_ERROR("PrepareOpenDevice failed.");
        return ACLSHMEM_INNER_ERROR;
    }

    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::CloseDevice()
{
    CleanupResources();
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::RegisterMemoryRegion(const TransportMemoryRegion& mr)
{
    auto addrRecordIt = memRecordMap_.find(mr.addr);
    if (addrRecordIt != memRecordMap_.end()) {
        SHM_LOG_INFO("Hcomm MR address has been registered, address = " << mr.addr);
        return ACLSHMEM_SUCCESS;
    }
    if (connected_) {
        SHM_LOG_ERROR("Register Hcomm MR after connected is not supported, address = " << mr.addr);
        return ACLSHMEM_INVALID_PARAM;
    }

    bool isHbm = (mr.flags & REG_MR_FLAG_HBM) != 0;
    bool isDram = (mr.flags & REG_MR_FLAG_DRAM) != 0;
    if (isHbm == isDram) {
        SHM_LOG_ERROR("Invalid memory region flags for Hcomm MR, flags = " << mr.flags);
        return ACLSHMEM_INVALID_PARAM;
    }
    CommMemType memType = isHbm ? COMM_MEM_TYPE_DEVICE : COMM_MEM_TYPE_HOST;

    std::map<uint32_t, HcommMemHandle> hcommHandles;
    auto rollbackRegisteredHandles = [this, &hcommHandles]() {
        for (const auto& registered : hcommHandles) {
            auto registeredEndpointIt = endpointHandleMap_.find(registered.first);
            if (registeredEndpointIt == endpointHandleMap_.end() || registeredEndpointIt->second == nullptr ||
                registered.second == nullptr) {
                continue;
            }
            auto hcommRet = HcommMemUnreg(registeredEndpointIt->second, registered.second);
            if (hcommRet != 0) {
                SHM_LOG_WARN(
                    "Rollback Hcomm memory registration failed for EID index " << registered.first
                                                                               << ", ret = " << hcommRet);
            }
        }
    };
    for (const auto& endpointEntry : endpointHandleMap_) {
        const uint32_t eidIndex = endpointEntry.first;
        EndpointHandle endpointHandle = endpointEntry.second;
        if (endpointHandle == nullptr) {
            SHM_LOG_ERROR("Invalid hcomm endpoint for EID index " << eidIndex);
            rollbackRegisteredHandles();
            return ACLSHMEM_INNER_ERROR;
        }

        CommMem mem{};
        mem.type = memType;
        mem.addr = reinterpret_cast<void*>(mr.addr);
        mem.size = mr.size;

        HcommMemHandle hcommMemHandle = nullptr;
        auto hcommRet = HcommMemReg(endpointHandle, ACLSHMEM_HCOMM_MEM_TAG, &mem, &hcommMemHandle);
        if (hcommRet != 0 || hcommMemHandle == nullptr) {
            SHM_LOG_ERROR("Failed to register hcomm memory for EID index " << eidIndex << ", ret = " << hcommRet);
            rollbackRegisteredHandles();
            return ACLSHMEM_INNER_ERROR;
        }
        hcommHandles[eidIndex] = hcommMemHandle;
    }

    memRecordMap_[mr.addr] = hcommHandles;

    SHM_LOG_INFO("Register MR success.");
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::UnregisterMemoryRegion(uint64_t addr)
{
    auto hcommPos = memRecordMap_.find(addr);
    if (hcommPos == memRecordMap_.end()) {
        SHM_LOG_ERROR("Input address is not registered, address = " << addr);
        return ACLSHMEM_INVALID_PARAM;
    }

    auto& hcommHandles = hcommPos->second;
    if (!channelHandles_.empty() || udmaInfoDev_ != nullptr) {
        DestroyChannels();
        FreeDeviceInfo();
        connected_ = false;
    }
    Result result = ACLSHMEM_SUCCESS;
    for (const auto& hcommMemEntry : hcommHandles) {
        auto endpointIt = endpointHandleMap_.find(hcommMemEntry.first);
        if (endpointIt == endpointHandleMap_.end() || endpointIt->second == nullptr ||
            hcommMemEntry.second == nullptr) {
            continue;
        }
        auto hcommRet = HcommMemUnreg(endpointIt->second, hcommMemEntry.second);
        if (hcommRet != 0) {
            SHM_LOG_WARN(
                "Failed to unregister hcomm memory for EID index " << hcommMemEntry.first << ", ret = " << hcommRet);
            result = ACLSHMEM_INNER_ERROR;
        }
    }
    memRecordMap_.erase(hcommPos);
    return result;
}

Result UdmaTransportManager::Prepare(const HybmTransPrepareOptions& options)
{
    SHM_LOG_DEBUG("UdmaTransportManager Prepare with options: " << options);
    return CheckPrepareOptions(options);
}

Result UdmaTransportManager::Connect()
{
    if (connected_) {
        return ACLSHMEM_SUCCESS;
    }
    auto ret = AsyncConnect();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Async connect failed, ret = " << ret);
        return ret;
    }
    SHM_LOG_INFO("Async connect success");
    return ACLSHMEM_SUCCESS;
}

const void* UdmaTransportManager::GetQpInfo() const { return udmaInfoDev_; }

Result UdmaTransportManager::BuildUdmaInfo(
    const std::vector<uint64_t>& channelPtrs, const std::vector<uint32_t>& channelPeers)
{
    if (channelPtrs.empty() || channelPeers.size() != channelPtrs.size()) {
        SHM_LOG_ERROR(
            "Invalid hcomm channel ptr input, channelPtrs = " << channelPtrs.size()
                                                              << ", channelPeers = " << channelPeers.size());
        return ACLSHMEM_INVALID_PARAM;
    }

    std::vector<SqContext> sqContextsByPeer(rankCount_);
    std::vector<CqContext> cqContextsByPeer(rankCount_);
    std::vector<RegedBufferEntity> remoteBuffersByPeer(rankCount_);
    std::vector<bool> peerValid(rankCount_, false);
    auto ret = ReadChannelContexts(
        channelPtrs, channelPeers, sqContextsByPeer, cqContextsByPeer, remoteBuffersByPeer, peerValid);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }

    constexpr uint32_t qpNum = 1;
    std::vector<uint8_t> eidTableHost;
    ret = PrepareUdmaInfoBuffers(eidTableHost);
    if (ret != ACLSHMEM_SUCCESS) {
        FreeDeviceInfo();
        return ret;
    }

    std::vector<uint8_t> udmaInfoBuffer;
    ACLSHMEMAIVUDMAInfo* copyInfo = nullptr;
    InitHostUdmaInfo(qpNum, udmaInfoBuffer, copyInfo);
    FillHostUdmaInfo(
        sqContextsByPeer, cqContextsByPeer, remoteBuffersByPeer, peerValid, eidTableHost, *copyInfo);

    ret = CopyEidTableToDevice(eidTableHost);
    if (ret != ACLSHMEM_SUCCESS) {
        FreeDeviceInfo();
        return ret;
    }

    // Print the filled host-side udmaInfo (section pointers still reference the local
    // buffer here) for maintenance/debugging before rebasing to device addresses.
    PrintHostUdmaInfo(*copyInfo);

    ret = CopyUdmaInfoToDevice(qpNum, udmaInfoBuffer, *copyInfo);
    if (ret != ACLSHMEM_SUCCESS) {
        FreeDeviceInfo();
        return ret;
    }

    SHM_LOG_INFO(
        "Build UDMA info success, rankCount = " << rankCount_ << ", totalChannelNum = " << channelHandles_.size()
                                                << ", qpNum = " << qpNum);
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::ReadChannelContexts(const std::vector<uint64_t>& channelPtrs,
    const std::vector<uint32_t>& channelPeers, std::vector<SqContext>& sqContextsByPeer,
    std::vector<CqContext>& cqContextsByPeer, std::vector<RegedBufferEntity>& remoteBuffersByPeer,
    std::vector<bool>& peerValid) const
{
    for (size_t idx = 0; idx < channelPtrs.size(); ++idx) {
        const uint32_t peer = channelPeers[idx];
        if (peer >= rankCount_ || peer == rankId_ || channelPtrs[idx] == 0) {
            SHM_LOG_ERROR("Invalid hcomm channel entity for peer " << peer << ", index = " << idx);
            return ACLSHMEM_INVALID_PARAM;
        }

        PeerChannelContexts peerContexts{};
        auto readRet = ReadPeerChannelContexts(channelPtrs[idx], peer, peerContexts);
        if (readRet != ACLSHMEM_SUCCESS) {
            return readRet;
        }

        sqContextsByPeer[peer] = peerContexts.sqContext;
        cqContextsByPeer[peer] = peerContexts.cqContext;
        remoteBuffersByPeer[peer] = peerContexts.remoteBuffer;
        peerValid[peer] = true;
    }
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::PrepareUdmaInfoBuffers(std::vector<uint8_t>& eidTableHost)
{
    // Reserve the per-peer AMO scratch buffers the same way the legacy
    // DeviceJettyManager did: one malloc per peer, referenced from WQCtx.
    auto ret = ReserveScratchBuffers();
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }

    // Allocate the device-side remote EID table: uint8_t[rankCount_][URMA_EID_RAW_SIZE].
    const uint64_t eidTableSize = static_cast<uint64_t>(rankCount_) * URMA_EID_RAW_SIZE;
    ret = AllocateAndClearDeviceBuffer(eidDev_, eidTableSize, "udma remote eid table");
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }
    eidTableHost.assign(eidTableSize, 0);
    return ACLSHMEM_SUCCESS;
}

void UdmaTransportManager::InitHostUdmaInfo(
    uint32_t qpNum, std::vector<uint8_t>& udmaInfoBuffer, ACLSHMEMAIVUDMAInfo*& copyInfo)
{
    // Build the contiguous udmaInfo blob in host memory using the legacy layout:
    //   [ACLSHMEMAIVUDMAInfo][WQCtx * rankCount_][WQCtx(rq) * rankCount_]
    //   [CqCtx(scq) * rankCount_][CqCtx(rcq) * rankCount_][UBmemInfo * rankCount_]
    const uint64_t wqSize = sizeof(ACLSHMEMUDMAWQCtx) * qpNum;
    const uint64_t cqSize = sizeof(ACLSHMEMUDMACqCtx) * qpNum;
    const uint64_t oneQpSize = 2U * (wqSize + cqSize) + sizeof(ACLSHMEMUBmemInfo) * qpNum;
    udmaInfoSize_ = sizeof(ACLSHMEMAIVUDMAInfo) + oneQpSize * rankCount_;

    udmaInfoBuffer.assign(udmaInfoSize_, 0);
    copyInfo = reinterpret_cast<ACLSHMEMAIVUDMAInfo*>(udmaInfoBuffer.data());
    copyInfo->qpNum = qpNum;
    // Host-side section pointers used only while filling the local buffer.
    SetUdmaInfoSectionPtrs(*copyInfo, copyInfo, rankCount_, qpNum);
}

void UdmaTransportManager::FillHostUdmaInfo(const std::vector<SqContext>& sqContextsByPeer,
    const std::vector<CqContext>& cqContextsByPeer, const std::vector<RegedBufferEntity>& remoteBuffersByPeer,
    const std::vector<bool>& peerValid, std::vector<uint8_t>& eidTableHost, ACLSHMEMAIVUDMAInfo& copyInfo)
{
    auto* wqArray = reinterpret_cast<ACLSHMEMUDMAWQCtx*>(copyInfo.sqPtr);
    auto* rqArray = reinterpret_cast<ACLSHMEMUDMAWQCtx*>(copyInfo.rqPtr);
    auto* scqArray = reinterpret_cast<ACLSHMEMUDMACqCtx*>(copyInfo.scqPtr);
    auto* rcqArray = reinterpret_cast<ACLSHMEMUDMACqCtx*>(copyInfo.rcqPtr);
    auto* memArray = reinterpret_cast<ACLSHMEMUBmemInfo*>(copyInfo.memPtr);

    for (uint32_t peer = 0; peer < rankCount_; ++peer) {
        if (peer == rankId_ || !peerValid[peer]) {
            // Self entry and any unconnected peer stay zero-initialized; the data
            // plane never issues a self-send and asserts against self pe.
            continue;
        }
        FillWqCtx(sqContextsByPeer[peer], peer, wqArray[peer]);
        rqArray[peer] = wqArray[peer];
        FillCqCtx(cqContextsByPeer[peer], scqArray[peer]);
        rcqArray[peer] = scqArray[peer];
        FillMemInfo(sqContextsByPeer[peer], remoteBuffersByPeer[peer], memArray[peer]);

        // Stage the remote EID raw bytes (from the SQ context) into the device EID table
        // and point memArray[peer].eidAddr at the device-resident copy, matching the
        // legacy behavior where the data plane reads rmtEid[0..1] from eidAddr.
        uint8_t* eidSlot = eidTableHost.data() + static_cast<uint64_t>(peer) * URMA_EID_RAW_SIZE;
        (void)memcpy_s(eidSlot, URMA_EID_RAW_SIZE, sqContextsByPeer[peer].contextInfo.ubJfs.remoteEID,
            URMA_EID_RAW_SIZE);
        memArray[peer].eidAddr =
            reinterpret_cast<uint64_t>(
                static_cast<uint8_t*>(eidDev_) + static_cast<uint64_t>(peer) * URMA_EID_RAW_SIZE);
    }
}

Result UdmaTransportManager::CopyEidTableToDevice(const std::vector<uint8_t>& eidTableHost)
{
    auto ret = DlAclApi::AclrtMemcpy(
        eidDev_, eidTableHost.size(), eidTableHost.data(), eidTableHost.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Copy udma remote eid table to device failed, ret = " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::CopyUdmaInfoToDevice(
    uint32_t qpNum, std::vector<uint8_t>& udmaInfoBuffer, ACLSHMEMAIVUDMAInfo& copyInfo)
{
    // Allocate the device blob and rebase the section pointers to device addresses.
    auto allocRet = AllocateAndClearDeviceBuffer(udmaInfoDev_, udmaInfoSize_, "udma info");
    if (allocRet != ACLSHMEM_SUCCESS) {
        return allocRet;
    }
    SetUdmaInfoSectionPtrs(copyInfo, udmaInfoDev_, rankCount_, qpNum);

    auto ret = DlAclApi::AclrtMemcpy(
        udmaInfoDev_, udmaInfoSize_, udmaInfoBuffer.data(), udmaInfoSize_, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Copy udma info to device failed, ret = " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::ReserveScratchBuffers()
{
    amoDevList_.assign(rankCount_, nullptr);
    for (uint32_t peer = 0; peer < rankCount_; ++peer) {
        if (peer == rankId_) {
            continue;
        }
        auto ret = AllocateAndClearDeviceBuffer(amoDevList_[peer], sizeof(uint64_t), "udma amo scratch");
        if (ret != ACLSHMEM_SUCCESS) {
            return ret;
        }
    }
    return ACLSHMEM_SUCCESS;
}

void UdmaTransportManager::FillWqCtx(const SqContext& sqContext, uint32_t peer, ACLSHMEMUDMAWQCtx& dstWq) const
{
    const auto& ubJfs = sqContext.contextInfo.ubJfs;
    dstWq.wqn = ubJfs.jfsID;
    dstWq.bufAddr = ubJfs.sqVa;
    // HCOMM exposes the raw WQE byte size; the data plane consumes it directly.
    dstWq.wqeSize = ubJfs.wqeSize;
    dstWq.depth = shm::UDMA_SQ_BASKBLK_CNT;
    dstWq.head = 0;
    dstWq.tail = 0;
    dstWq.dbMode = ACLSHMEMUDMADBMode::SW_DB;
    dstWq.dbAddr = ubJfs.dbVa;
    dstWq.sl = 0;
    dstWq.wqeCnt = 0;
    dstWq.amoAddr = reinterpret_cast<uint64_t>(amoDevList_[peer]);
}

void UdmaTransportManager::FillCqCtx(const CqContext& cqContext, ACLSHMEMUDMACqCtx& dstCq) const
{
    const auto& ubJfc = cqContext.contextInfo.ubJfc;
    dstCq.cqn = ubJfc.jfcID;
    dstCq.bufAddr = ubJfc.scqVa;
    dstCq.cqeSize = ubJfc.cqeSize;
    dstCq.depth = shm::UDMA_CQ_DEPTH_DEFAULT;
    dstCq.head = 0;
    dstCq.tail = 0;
    dstCq.dbMode = ACLSHMEMUDMADBMode::SW_DB;
    dstCq.dbAddr = ubJfc.dbVa;
}

void UdmaTransportManager::FillMemInfo(
    const SqContext& sqContext, const RegedBufferEntity& remoteBuffer, ACLSHMEMUBmemInfo& dstMem) const
{
    const auto& ub = remoteBuffer.bufferInfo.rma.protectionInfo.memInfo.ub;
    dstMem.tokenValueValid = true; // token-based access control enabled (data plane sets tokenEn = 1)
    dstMem.rmtJettyType = 1;       // remote jetty type: 1 = jetty (peer-to-peer)
    dstMem.targetHint = 0;         // no target selection preference
    dstMem.tpn = sqContext.contextInfo.ubJfs.tpID;
    dstMem.tid = ub.tokenId;
    dstMem.rmtTokenValue = ub.tokenValue;
    dstMem.len = static_cast<uint32_t>(remoteBuffer.bufferInfo.rma.size);
    dstMem.addr = remoteBuffer.bufferInfo.rma.addr;
    dstMem.eidAddr = 0; // filled by the caller after the device EID table is staged
}

void UdmaTransportManager::PrintHostUdmaInfo(const ACLSHMEMAIVUDMAInfo& hostInfo) const
{
    SHM_LOG_DEBUG("=======================rank [" << rankId_ << "] udma host info====================");
    SHM_LOG_DEBUG("rank[" << rankId_ << "] udmaInfo.qpNum: " << hostInfo.qpNum);

    const auto* wqArray = reinterpret_cast<const ACLSHMEMUDMAWQCtx*>(hostInfo.sqPtr);
    const auto* cqArray = reinterpret_cast<const ACLSHMEMUDMACqCtx*>(hostInfo.scqPtr);
    const auto* memArray = reinterpret_cast<const ACLSHMEMUBmemInfo*>(hostInfo.memPtr);
    if (wqArray == nullptr || cqArray == nullptr || memArray == nullptr) {
        SHM_LOG_WARN("rank[" << rankId_ << "] udma host info section pointer is null, skip printing.");
        return;
    }

    for (uint32_t pe = 0; pe < rankCount_; ++pe) {
        const auto& wq = wqArray[pe];
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] WQCtx.wqn: " << wq.wqn);
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] WQCtx.bufAddr: " << wq.bufAddr);
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] WQCtx.wqeSize: " << wq.wqeSize);
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] WQCtx.depth: " << wq.depth);
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] WQCtx.head: " << wq.head);
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] WQCtx.tail: " << wq.tail);
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] WQCtx.dbMode: " << static_cast<int>(wq.dbMode));
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] WQCtx.dbAddr: " << wq.dbAddr);
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] WQCtx.sl: " << wq.sl);
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] WQCtx.wqeCnt: " << wq.wqeCnt);
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] WQCtx.amoAddr: " << wq.amoAddr);

        const auto& cq = cqArray[pe];
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] CQCtx.cqn: " << cq.cqn);
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] CQCtx.bufAddr: " << cq.bufAddr);
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] CQCtx.cqeSize: " << cq.cqeSize);
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] CQCtx.depth: " << cq.depth);
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] CQCtx.head: " << cq.head);
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] CQCtx.tail: " << cq.tail);
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] CQCtx.dbMode: " << static_cast<int>(cq.dbMode));
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] CQCtx.dbAddr: " << cq.dbAddr);

        const auto& mem = memArray[pe];
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] MemInfo.tokenValueValid: " << mem.tokenValueValid);
        SHM_LOG_DEBUG(
            "rank[" << rankId_ << "] peer[" << pe << "] MemInfo.rmtJettyType: " << mem.rmtJettyType);
        SHM_LOG_DEBUG(
            "rank[" << rankId_ << "] peer[" << pe << "] MemInfo.targetHint: " << static_cast<int>(mem.targetHint));
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] MemInfo.tpn: " << mem.tpn);
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] MemInfo.tid: " << mem.tid);
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] MemInfo.rmtTokenValue: " << mem.rmtTokenValue);
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] MemInfo.len: " << mem.len);
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] MemInfo.addr: " << mem.addr);
        SHM_LOG_DEBUG("rank[" << rankId_ << "] peer[" << pe << "] MemInfo.eidAddr: " << mem.eidAddr);
    }
}

std::vector<HcommMemHandle> UdmaTransportManager::CollectChannelMemHandles(uint32_t eidIndex) const
{
    std::vector<HcommMemHandle> memHandles;
    for (const auto& addrEntry : memRecordMap_) {
        const auto& hcommHandles = addrEntry.second;
        auto handleIt = hcommHandles.find(eidIndex);
        if (handleIt != hcommHandles.end() && handleIt->second != nullptr) {
            memHandles.push_back(handleIt->second);
        }
    }
    return memHandles;
}

Result UdmaTransportManager::AsyncConnect()
{
    const uint32_t localEndpointCount = static_cast<uint32_t>(endpointDescMap_.size());
    std::vector<uint32_t> allEndpointCounts(rankCount_);
    g_boot_handle.allgather(&localEndpointCount, allEndpointCounts.data(), sizeof(uint32_t), &g_boot_handle);

    const auto maxEndpointCountIt = std::max_element(allEndpointCounts.begin(), allEndpointCounts.end());
    const uint32_t maxEndpointCount = (maxEndpointCountIt == allEndpointCounts.end()) ? 0 : *maxEndpointCountIt;
    if (maxEndpointCount == 0) {
        SHM_LOG_ERROR("No local hcomm endpoint descriptor was exchanged.");
        return ACLSHMEM_INNER_ERROR;
    }

    std::vector<ExchangedEndpointDescInfo> localEndpoints(maxEndpointCount);
    uint32_t packedIndex = 0;
    for (const auto& endpointEntry : endpointDescMap_) {
        ExchangedEndpointDescInfo& packed = localEndpoints[packedIndex];
        packed.eidIndex = endpointEntry.first;
        packed.valid = 1;
        packed.desc = endpointEntry.second;
        ++packedIndex;
    }

    std::vector<ExchangedEndpointDescInfo> allEndpointInfo(rankCount_ * maxEndpointCount);
    g_boot_handle.allgather(
        localEndpoints.data(), allEndpointInfo.data(),
        static_cast<uint64_t>(sizeof(ExchangedEndpointDescInfo) * maxEndpointCount), &g_boot_handle);

    std::vector<ChannelHandle> createdChannels;
    std::vector<uint32_t> channelPeers;
    auto destroyCreatedResources = [&createdChannels]() {
        if (!createdChannels.empty()) {
            (void)HcommChannelDestroy(createdChannels.data(), static_cast<uint32_t>(createdChannels.size()));
            createdChannels.clear();
        }
    };
    for (uint32_t peer = 0; peer < rankCount_; ++peer) {
        if (peer == rankId_) {
            continue;
        }

        auto localEidIt = peerEidIndexMap_.find(peer);
        auto remoteEidIt = peerRemoteEidIndexMap_.find(peer);
        if (localEidIt == peerEidIndexMap_.end() || remoteEidIt == peerRemoteEidIndexMap_.end()) {
            SHM_LOG_ERROR("Failed to find EID route for peer " << peer);
            destroyCreatedResources();
            return ACLSHMEM_INNER_ERROR;
        }

        const uint32_t localEidIndex = localEidIt->second;
        const uint32_t remoteEidIndex = remoteEidIt->second;
        auto endpointIt = endpointHandleMap_.find(localEidIndex);
        if (endpointIt == endpointHandleMap_.end() || endpointIt->second == nullptr) {
            SHM_LOG_ERROR("Failed to find hcomm endpoint for peer " << peer << ", localEidIndex = " << localEidIndex);
            destroyCreatedResources();
            return ACLSHMEM_INNER_ERROR;
        }

        const ExchangedEndpointDescInfo* remoteEndpointInfo = nullptr;
        const uint32_t peerEndpointCount = allEndpointCounts[peer];
        for (uint32_t idx = 0; idx < peerEndpointCount; ++idx) {
            const ExchangedEndpointDescInfo& candidate = allEndpointInfo[peer * maxEndpointCount + idx];
            if (candidate.valid != 0 && candidate.eidIndex == remoteEidIndex) {
                remoteEndpointInfo = &candidate;
                break;
            }
        }
        if (remoteEndpointInfo == nullptr) {
            SHM_LOG_ERROR(
                "No remote hcomm endpoint descriptor was exchanged for peer " << peer << " on remote EID index "
                                                                              << remoteEidIndex);
            destroyCreatedResources();
            return ACLSHMEM_INNER_ERROR;
        }

        std::vector<HcommMemHandle> memHandles = CollectChannelMemHandles(localEidIndex);
        if (memHandles.empty()) {
            SHM_LOG_ERROR(
                "No active Hcomm mem handle for local EID index " << localEidIndex << " when creating channel for peer "
                                                                  << peer);
            destroyCreatedResources();
            return ACLSHMEM_INNER_ERROR;
        }
        HcommChannelDesc channelDesc{};
        auto descInitRet = HcommChannelDescInit(&channelDesc, ACLSHMEM_HCOMM_CHANNEL_NUM_PER_PEER);
        if (descInitRet != 0) {
            SHM_LOG_ERROR("HcommChannelDescInit failed for peer " << peer << ", ret = " << descInitRet);
            destroyCreatedResources();
            return ACLSHMEM_INNER_ERROR;
        }
        channelDesc.remoteEndpoint = remoteEndpointInfo->desc;
        channelDesc.notifyNum = 0;
        channelDesc.exchangeAllMems = false;
        channelDesc.memHandles = memHandles.data();
        channelDesc.memHandleNum = static_cast<uint32_t>(memHandles.size());
        channelDesc.qos = ACLSHMEM_HCOMM_DEFAULT_QOS;

        ChannelHandle channelHandle = 0;
        auto hcommRet = HcommChannelCreate(
            endpointIt->second, COMM_ENGINE_AIV, &channelDesc, ACLSHMEM_HCOMM_CHANNEL_NUM_PER_PEER, &channelHandle);
        if (hcommRet != 0 || channelHandle == 0) {
            SHM_LOG_ERROR("HcommChannelCreate failed for peer " << peer << ", ret = " << hcommRet);
            if (channelHandle != 0) {
                (void)HcommChannelDestroy(&channelHandle, ACLSHMEM_HCOMM_CHANNEL_NUM_PER_PEER);
            }
            destroyCreatedResources();
            return ACLSHMEM_INNER_ERROR;
        }

        createdChannels.push_back(channelHandle);
        channelPeers.push_back(peer);
    }

    channelHandles_ = std::move(createdChannels);
    channelPeers_ = channelPeers;
    auto buildRet = BuildUdmaInfo(channelHandles_, channelPeers);
    if (buildRet != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("BuildUdmaInfo failed, ret = " << buildRet);
        DestroyChannels();
        FreeDeviceInfo();
        return buildRet;
    }

    SHM_LOG_INFO(
        "Create hcomm channels success, totalChannelNum = " << channelHandles_.size() << ", channelNumPerPeer = "
                                                            << ACLSHMEM_HCOMM_CHANNEL_NUM_PER_PEER);
    connected_ = true;
    return ACLSHMEM_SUCCESS;
}

bool UdmaTransportManager::PrepareOpenDevice(uint32_t deviceId, uint32_t rankCount)
{
    RootInfo rootInfo;
    TopoInfo topoInfo;
    uint32_t localId = 0;
    uint32_t eidCount = 0;
    int32_t phyId = -1;

    auto ret = DlAclApi::AclrtGetPhyDevIdByLogicDevId(static_cast<int32_t>(deviceId), &phyId);
    SHM_ASSERT_LOG_AND_RETURN(
        ret == 0 && phyId >= 0,
        "AclrtGetPhyDevIdByLogicDevId() return=" << ret << ", input deviceId=" << deviceId
                                                 << ", output phyId=" << phyId,
        false);

    if (!TopoReader::ParseRootInfo(phyId, rootInfo)) {
        SHM_LOG_ERROR("Failed to parse the rootinfo file.");
        return false;
    }
    phyId_ = static_cast<uint32_t>(phyId);
    SHM_LOG_INFO("Resolved phy id from current device mapping, deviceId=" << deviceId << ", phyId=" << phyId);
    if (!TopoReader::ParseTopoInfo(rootInfo.topo_file_path, topoInfo)) {
        SHM_LOG_ERROR("Failed to parse the topology file at path " << rootInfo.topo_file_path);
        return false;
    }

    if (!TopoReader::GetLocalId(rootInfo, phyId_, localId)) {
        SHM_LOG_ERROR("Failed to find localId for phyId: " << phyId_);
        return false;
    }
    if (!TopoReader::GetEidCount(rootInfo, eidCount)) {
        SHM_LOG_ERROR("Failed to find eid count from rootinfo.");
        return false;
    }

    std::vector<uint32_t> eidCountList(rankCount);
    g_boot_handle.allgather(&eidCount, eidCountList.data(), sizeof(uint32_t), &g_boot_handle);
    const auto maxEidCountIt = std::max_element(eidCountList.begin(), eidCountList.end());
    const uint32_t maxEidCount = (maxEidCountIt == eidCountList.end()) ? 0 : *maxEidCountIt;
    if (maxEidCount == 0) {
        SHM_LOG_ERROR("Invalid eidSlotCount resolved from rootinfo rank_addr_list.");
        return false;
    }

    std::vector<uint32_t> localIdList(rankCount);
    g_boot_handle.allgather(&localId, localIdList.data(), sizeof(uint32_t), &g_boot_handle);
    std::vector<int32_t> localRouteByPeer(rankCount, INVALID_EID_INDEX);

    for (uint32_t peer = 0; peer < rankCount; ++peer) {
        if (peer == rankId_) {
            continue;
        }
        uint32_t eidIndex = 0;
        std::array<uint8_t, URMA_EID_RAW_SIZE> eidRaw{};
        uint32_t peerLocalId = localIdList[peer];
        if (!TopoReader::GetLocalEidRouteForPeer(rootInfo, topoInfo, localId, peerLocalId, eidIndex, eidRaw)) {
            SHM_LOG_ERROR(
                "Failed to resolve the local EID route for peer rank " << peer << ". The localId was " << localId
                                                                       << " and the peer localId was " << peerLocalId);
            return false;
        }
        peerEidIndexMap_[peer] = eidIndex;
        localRouteByPeer[peer] = static_cast<int32_t>(eidIndex);

        if (!CreateEndpoint(eidIndex, eidRaw)) {
            SHM_LOG_ERROR("CreateEndpoint failed for peer " << peer << " with EID index " << eidIndex);
            return false;
        }
    }

    std::vector<int32_t> allRouteByPeer(rankCount * rankCount, INVALID_EID_INDEX);
    g_boot_handle.allgather(
        localRouteByPeer.data(), allRouteByPeer.data(), sizeof(int32_t) * rankCount, &g_boot_handle);

    for (uint32_t peer = 0; peer < rankCount; ++peer) {
        if (peer == rankId_) {
            continue;
        }
        int32_t remoteRoute = allRouteByPeer[peer * rankCount + rankId_];
        if (remoteRoute < 0 || static_cast<uint32_t>(remoteRoute) >= maxEidCount) {
            SHM_LOG_ERROR(
                "Invalid remote EID route for peer rank "
                << peer << ", remoteRoute = " << remoteRoute << ", local rank = " << rankId_ << ", localId = "
                << localId << ", peerLocalId = " << localIdList[peer] << ", eidSlotCount = " << maxEidCount);
            return false;
        }
        peerRemoteEidIndexMap_[peer] = static_cast<uint32_t>(remoteRoute);
    }

    return true;
}

bool UdmaTransportManager::CreateEndpoint(uint32_t eidIndex, const std::array<uint8_t, URMA_EID_RAW_SIZE>& targetEidRaw)
{
    auto endpointIt = endpointHandleMap_.find(eidIndex);
    if (endpointIt != endpointHandleMap_.end() && endpointIt->second != nullptr) {
        return true;
    }

    EndpointDesc endpointDesc{};
    auto descInitRet = EndpointDescInit(&endpointDesc, 1);
    if (descInitRet != 0) {
        SHM_LOG_ERROR("EndpointDescInit failed for EID index " << eidIndex << ", ret = " << descInitRet);
        return false;
    }

    uint32_t sdId = 0;
    uint32_t serverId = 0;
    uint32_t superPodId = 0;
    auto deviceInfoRet = shm::MemSegment::GetDeviceInfo(sdId, serverId, superPodId);
    if (deviceInfoRet != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Get local device info for HCOMM endpoint failed, ret = " << deviceInfoRet);
        return false;
    }

    endpointDesc.protocol = COMM_PROTOCOL_UBC_CTP;
    endpointDesc.commAddr.type = COMM_ADDR_TYPE_EID;
    int copyRet = memcpy_s(
        endpointDesc.commAddr.eid, sizeof(endpointDesc.commAddr.eid), targetEidRaw.data(), targetEidRaw.size());
    if (copyRet != EOK) {
        SHM_LOG_ERROR("Copy target EID to HCOMM endpoint desc failed, ret = " << copyRet);
        return false;
    }
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    endpointDesc.loc.device.devPhyId = phyId_;
    endpointDesc.loc.device.superDevId = sdId;
    endpointDesc.loc.device.serverIdx = serverId;
    endpointDesc.loc.device.superPodIdx = superPodId;

    EndpointHandle endpointHandle = nullptr;
    auto ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    if (ret != 0 || endpointHandle == nullptr) {
        SHM_LOG_ERROR("HcommEndpointCreate failed for EID index " << eidIndex << ", ret = " << ret);
        return false;
    }
    endpointDescMap_[eidIndex] = endpointDesc;
    endpointHandleMap_[eidIndex] = endpointHandle;
    return true;
}

Result UdmaTransportManager::CheckPrepareOptions(const HybmTransPrepareOptions& options)
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

void UdmaTransportManager::FreeDeviceInfo()
{
    if (udmaInfoDev_ != nullptr) {
        (void)DlAclApi::AclrtFree(udmaInfoDev_);
        udmaInfoDev_ = nullptr;
    }
    if (eidDev_ != nullptr) {
        (void)DlAclApi::AclrtFree(eidDev_);
        eidDev_ = nullptr;
    }
    for (auto& amoDev : amoDevList_) {
        if (amoDev != nullptr) {
            (void)DlAclApi::AclrtFree(amoDev);
            amoDev = nullptr;
        }
    }
    amoDevList_.clear();
    udmaInfoSize_ = 0;
}

void UdmaTransportManager::DestroyChannels()
{
    if (!channelHandles_.empty()) {
        auto hcommRet = HcommChannelDestroy(channelHandles_.data(), static_cast<uint32_t>(channelHandles_.size()));
        if (hcommRet != 0) {
            SHM_LOG_WARN("HcommChannelDestroy failed, ret = " << hcommRet);
        }
    }
    channelHandles_.clear();
    channelPeers_.clear();
    SHM_LOG_INFO("Destroy hcomm channels success.");
}

void UdmaTransportManager::CleanupResources()
{
    DestroyChannels();
    FreeDeviceInfo();

    for (const auto& memByAddr : memRecordMap_) {
        for (const auto& memEntry : memByAddr.second) {
            auto endpointIt = endpointHandleMap_.find(memEntry.first);
            if (endpointIt == endpointHandleMap_.end() || endpointIt->second == nullptr || memEntry.second == nullptr) {
                continue;
            }
            auto hcommRet = HcommMemUnreg(endpointIt->second, memEntry.second);
            if (hcommRet != 0) {
                SHM_LOG_WARN("HcommMemUnreg failed for EID index " << memEntry.first << ", ret = " << hcommRet);
            }
        }
    }
    memRecordMap_.clear();
    SHM_LOG_INFO("Unregister memory success.");

    for (const auto& endpointEntry : endpointHandleMap_) {
        if (endpointEntry.second == nullptr) {
            continue;
        }
        auto ret = HcommEndpointDestroy(endpointEntry.second);
        if (ret != 0) {
            SHM_LOG_WARN("HcommEndpointDestroy failed for EID index " << endpointEntry.first << ", ret = " << ret);
        }
    }
    endpointHandleMap_.clear();
    peerEidIndexMap_.clear();
    peerRemoteEidIndexMap_.clear();
    endpointDescMap_.clear();
    channelHandles_.clear();
    channelPeers_.clear();
    connected_ = false;
}

} // namespace device
} // namespace transport
} // namespace shm
