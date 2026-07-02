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

struct exchanged_endpoint_desc_info_t {
    uint32_t eid_index{0};
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
    SqContext sq_context{};
    CqContext cq_context{};
    RegedBufferEntity remote_buffer{};
};

Result ReadPeerChannelContexts(uint64_t channel_ptr, uint32_t peer, PeerChannelContexts& out)
{
    ChannelEntity channel{};
    auto ret = CopyHcommChannelEntityFromDevice(&channel, sizeof(channel), channel_ptr, "channel entity", peer);
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

    ret = CopyHcommChannelEntityFromDevice(&out.sq_context, sizeof(out.sq_context), sqContextAddr, "SQ context", peer);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }
    if (out.sq_context.contextInfo.ubJfs.sqDepth != shm::UDMA_SQ_BASKBLK_CNT) {
        SHM_LOG_ERROR(
            "Unexpected Hcomm SQ depth for peer " << peer << ", sqDepth = " << out.sq_context.contextInfo.ubJfs.sqDepth
                                                  << ", expected = " << shm::UDMA_SQ_BASKBLK_CNT);
        return ACLSHMEM_INNER_ERROR;
    }

    ret = CopyHcommChannelEntityFromDevice(&out.cq_context, sizeof(out.cq_context), cqContextAddr, "CQ context", peer);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }
    if (out.cq_context.contextInfo.ubJfc.cqDepth != shm::UDMA_CQ_DEPTH_DEFAULT) {
        SHM_LOG_ERROR(
            "Unexpected Hcomm CQ depth for peer " << peer << ", cqDepth = " << out.cq_context.contextInfo.ubJfc.cqDepth
                                                  << ", expected = " << shm::UDMA_CQ_DEPTH_DEFAULT);
        return ACLSHMEM_INNER_ERROR;
    }

    ret = CopyHcommChannelEntityFromDevice(
        &out.remote_buffer, sizeof(out.remote_buffer), remoteBufferAddr, "remote buffer", peer);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }
    return ACLSHMEM_SUCCESS;
}

void SetUdmaInfoSectionPtrs(aclshmemi_aiv_udma_info_t& info, void* infoBase, uint32_t rank_count, uint32_t qp_num)
{
    info.sq_ptr = reinterpret_cast<uint64_t>(static_cast<aclshmemi_aiv_udma_info_t*>(infoBase) + 1);
    info.rq_ptr = reinterpret_cast<uint64_t>(
        reinterpret_cast<aclshmemi_udma_wq_ctx_t*>(info.sq_ptr) + static_cast<uint64_t>(rank_count) * qp_num);
    info.scq_ptr = reinterpret_cast<uint64_t>(
        reinterpret_cast<aclshmemi_udma_wq_ctx_t*>(info.rq_ptr) + static_cast<uint64_t>(rank_count) * qp_num);
    info.rcq_ptr = reinterpret_cast<uint64_t>(
        reinterpret_cast<aclshmemi_udma_cq_ctx_t*>(info.scq_ptr) + static_cast<uint64_t>(rank_count) * qp_num);
    info.mem_ptr = reinterpret_cast<uint64_t>(
        reinterpret_cast<aclshmemi_udma_cq_ctx_t*>(info.rcq_ptr) + static_cast<uint64_t>(rank_count) * qp_num);
}

} // namespace

UdmaTransportManager::UdmaTransportManager() noexcept {}

UdmaTransportManager::~UdmaTransportManager() noexcept { CleanupResources(); }

Result UdmaTransportManager::OpenDevice(const TransportOptions& options)
{
    int32_t user_id = -1;
    int32_t logic_id = -1;

    auto ret = DlAclApi::AclrtGetDevice(&user_id);
    SHM_ASSERT_LOG_AND_RETURN(
        ret == 0 && user_id >= 0, "AclrtGetDevice() return=" << ret << ", output device_id=" << user_id,
        ACLSHMEM_DL_FUNC_FAILED);

    ret = DlAclApi::RtGetLogicDevIdByUserDevId(user_id, &logic_id);
    SHM_ASSERT_LOG_AND_RETURN(
        ret == 0 && logic_id >= 0, "RtGetLogicDevIdByUserDevId() return=" << ret << ", output device_id=" << logic_id,
        ACLSHMEM_DL_FUNC_FAILED);

    const uint32_t device_id = static_cast<uint32_t>(logic_id);
    rank_id_ = options.rankId;
    rank_count_ = options.rankCount;
    role_ = options.role;

    if (!PrepareOpenDevice(device_id, rank_count_)) {
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
    auto addrRecordIt = mem_record_map_.find(mr.addr);
    if (addrRecordIt != mem_record_map_.end()) {
        SHM_LOG_INFO("Hcomm MR address has been registered, address = " << mr.addr);
        return ACLSHMEM_SUCCESS;
    }
    if (connected_) {
        SHM_LOG_ERROR("Register Hcomm MR after connected is not supported, address = " << mr.addr);
        return ACLSHMEM_INVALID_PARAM;
    }

    bool is_hbm = (mr.flags & REG_MR_FLAG_HBM) != 0;
    bool is_dram = (mr.flags & REG_MR_FLAG_DRAM) != 0;
    if (is_hbm == is_dram) {
        SHM_LOG_ERROR("Invalid memory region flags for Hcomm MR, flags = " << mr.flags);
        return ACLSHMEM_INVALID_PARAM;
    }
    CommMemType mem_type = is_hbm ? COMM_MEM_TYPE_DEVICE : COMM_MEM_TYPE_HOST;

    std::map<uint32_t, HcommMemHandle> hcomm_handles;
    auto rollback_registered_handles = [this, &hcomm_handles]() {
        for (const auto& registered : hcomm_handles) {
            auto registered_endpoint_it = endpoint_handle_map_.find(registered.first);
            if (registered_endpoint_it == endpoint_handle_map_.end() || registered_endpoint_it->second == nullptr ||
                registered.second == nullptr) {
                continue;
            }
            auto hcomm_ret = HcommMemUnreg(registered_endpoint_it->second, registered.second);
            if (hcomm_ret != 0) {
                SHM_LOG_WARN(
                    "Rollback Hcomm memory registration failed for EID index " << registered.first
                                                                               << ", ret = " << hcomm_ret);
            }
        }
    };
    for (const auto& endpoint_entry : endpoint_handle_map_) {
        const uint32_t eid_index = endpoint_entry.first;
        EndpointHandle endpoint_handle = endpoint_entry.second;
        if (endpoint_handle == nullptr) {
            SHM_LOG_ERROR("Invalid hcomm endpoint for EID index " << eid_index);
            rollback_registered_handles();
            return ACLSHMEM_INNER_ERROR;
        }

        CommMem mem{};
        mem.type = mem_type;
        mem.addr = reinterpret_cast<void*>(mr.addr);
        mem.size = mr.size;

        HcommMemHandle hcomm_mem_handle = nullptr;
        auto hcomm_ret = HcommMemReg(endpoint_handle, ACLSHMEM_HCOMM_MEM_TAG, &mem, &hcomm_mem_handle);
        if (hcomm_ret != 0 || hcomm_mem_handle == nullptr) {
            SHM_LOG_ERROR("Failed to register hcomm memory for EID index " << eid_index << ", ret = " << hcomm_ret);
            rollback_registered_handles();
            return ACLSHMEM_INNER_ERROR;
        }
        hcomm_handles[eid_index] = hcomm_mem_handle;
    }

    mem_record_map_[mr.addr] = hcomm_handles;

    SHM_LOG_INFO("Register MR success.");
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::UnregisterMemoryRegion(uint64_t addr)
{
    auto hcomm_pos = mem_record_map_.find(addr);
    if (hcomm_pos == mem_record_map_.end()) {
        SHM_LOG_ERROR("Input address is not registered, address = " << addr);
        return ACLSHMEM_INVALID_PARAM;
    }

    auto& hcomm_handles = hcomm_pos->second;
    if (!channel_handles_.empty() || udma_info_dev_ != nullptr) {
        DestroyChannels();
        FreeDeviceInfo();
        connected_ = false;
    }
    Result result = ACLSHMEM_SUCCESS;
    for (const auto& hcomm_mem_entry : hcomm_handles) {
        auto endpoint_it = endpoint_handle_map_.find(hcomm_mem_entry.first);
        if (endpoint_it == endpoint_handle_map_.end() || endpoint_it->second == nullptr ||
            hcomm_mem_entry.second == nullptr) {
            continue;
        }
        auto hcomm_ret = HcommMemUnreg(endpoint_it->second, hcomm_mem_entry.second);
        if (hcomm_ret != 0) {
            SHM_LOG_WARN(
                "Failed to unregister hcomm memory for EID index " << hcomm_mem_entry.first << ", ret = " << hcomm_ret);
            result = ACLSHMEM_INNER_ERROR;
        }
    }
    mem_record_map_.erase(hcomm_pos);
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

const void* UdmaTransportManager::GetQpInfo() const { return udma_info_dev_; }

Result UdmaTransportManager::BuildUdmaInfo(
    const std::vector<uint64_t>& channel_ptrs, const std::vector<uint32_t>& channel_peers)
{
    if (channel_ptrs.empty() || channel_peers.size() != channel_ptrs.size()) {
        SHM_LOG_ERROR(
            "Invalid hcomm channel ptr input, channel_ptrs = " << channel_ptrs.size()
                                                              << ", channel_peers = " << channel_peers.size());
        return ACLSHMEM_INVALID_PARAM;
    }

    std::vector<SqContext> sq_contexts_by_peer(rank_count_);
    std::vector<CqContext> cq_contexts_by_peer(rank_count_);
    std::vector<RegedBufferEntity> remote_buffers_by_peer(rank_count_);
    std::vector<bool> peer_valid(rank_count_, false);
    auto ret = ReadChannelContexts(
        channel_ptrs, channel_peers, sq_contexts_by_peer, cq_contexts_by_peer, remote_buffers_by_peer, peer_valid);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }

    constexpr uint32_t qp_num = 1;
    std::vector<uint8_t> eid_table_host;
    ret = PrepareUdmaInfoBuffers(eid_table_host);
    if (ret != ACLSHMEM_SUCCESS) {
        FreeDeviceInfo();
        return ret;
    }

    std::vector<uint8_t> udma_info_buffer;
    aclshmemi_aiv_udma_info_t* copy_info = nullptr;
    InitHostUdmaInfo(qp_num, udma_info_buffer, copy_info);
    FillHostUdmaInfo(
        sq_contexts_by_peer, cq_contexts_by_peer, remote_buffers_by_peer, peer_valid, eid_table_host, *copy_info);

    ret = CopyEidTableToDevice(eid_table_host);
    if (ret != ACLSHMEM_SUCCESS) {
        FreeDeviceInfo();
        return ret;
    }

    // Print the filled host-side udmaInfo (section pointers still reference the local
    // buffer here) for maintenance/debugging before rebasing to device addresses.
    PrintHostUdmaInfo(*copy_info);

    ret = CopyUdmaInfoToDevice(qp_num, udma_info_buffer, *copy_info);
    if (ret != ACLSHMEM_SUCCESS) {
        FreeDeviceInfo();
        return ret;
    }

    SHM_LOG_INFO(
        "Build UDMA info success, rank_count = " << rank_count_ << ", totalChannelNum = " << channel_handles_.size()
                                                << ", qp_num = " << qp_num);
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::ReadChannelContexts(const std::vector<uint64_t>& channel_ptrs,
    const std::vector<uint32_t>& channel_peers, std::vector<SqContext>& sq_contexts_by_peer,
    std::vector<CqContext>& cq_contexts_by_peer, std::vector<RegedBufferEntity>& remote_buffers_by_peer,
    std::vector<bool>& peer_valid) const
{
    for (size_t idx = 0; idx < channel_ptrs.size(); ++idx) {
        const uint32_t peer = channel_peers[idx];
        if (peer >= rank_count_ || peer == rank_id_ || channel_ptrs[idx] == 0) {
            SHM_LOG_ERROR("Invalid hcomm channel entity for peer " << peer << ", index = " << idx);
            return ACLSHMEM_INVALID_PARAM;
        }

        PeerChannelContexts peer_contexts{};
        auto read_ret = ReadPeerChannelContexts(channel_ptrs[idx], peer, peer_contexts);
        if (read_ret != ACLSHMEM_SUCCESS) {
            return read_ret;
        }

        sq_contexts_by_peer[peer] = peer_contexts.sq_context;
        cq_contexts_by_peer[peer] = peer_contexts.cq_context;
        remote_buffers_by_peer[peer] = peer_contexts.remote_buffer;
        peer_valid[peer] = true;
    }
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::PrepareUdmaInfoBuffers(std::vector<uint8_t>& eid_table_host)
{
    // Reserve the per-peer AMO scratch buffers the same way the legacy
    // DeviceJettyManager did: one malloc per peer, referenced from WQCtx.
    auto ret = ReserveScratchBuffers();
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }

    // Allocate the device-side remote EID table: uint8_t[rank_count_][URMA_EID_RAW_SIZE].
    const uint64_t eid_table_size = static_cast<uint64_t>(rank_count_) * URMA_EID_RAW_SIZE;
    ret = AllocateAndClearDeviceBuffer(eid_dev_, eid_table_size, "udma remote eid table");
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }
    eid_table_host.assign(eid_table_size, 0);
    return ACLSHMEM_SUCCESS;
}

void UdmaTransportManager::InitHostUdmaInfo(
    uint32_t qp_num, std::vector<uint8_t>& udma_info_buffer, aclshmemi_aiv_udma_info_t*& copy_info)
{
    // Build the contiguous udmaInfo blob in host memory using the legacy layout:
    //   [aclshmemi_aiv_udma_info_t][WQCtx * rank_count_][WQCtx(rq) * rank_count_]
    //   [CqCtx(scq) * rank_count_][CqCtx(rcq) * rank_count_][UBmemInfo * rank_count_]
    const uint64_t wq_size = sizeof(aclshmemi_udma_wq_ctx_t) * qp_num;
    const uint64_t cq_size = sizeof(aclshmemi_udma_cq_ctx_t) * qp_num;
    const uint64_t one_qp_size = 2U * (wq_size + cq_size) + sizeof(aclshmemi_ubmem_info_t) * qp_num;
    udma_info_size_ = sizeof(aclshmemi_aiv_udma_info_t) + one_qp_size * rank_count_;

    udma_info_buffer.assign(udma_info_size_, 0);
    copy_info = reinterpret_cast<aclshmemi_aiv_udma_info_t*>(udma_info_buffer.data());
    copy_info->qp_num = qp_num;
    // Host-side section pointers used only while filling the local buffer.
    SetUdmaInfoSectionPtrs(*copy_info, copy_info, rank_count_, qp_num);
}

void UdmaTransportManager::FillHostUdmaInfo(const std::vector<SqContext>& sq_contexts_by_peer,
    const std::vector<CqContext>& cq_contexts_by_peer, const std::vector<RegedBufferEntity>& remote_buffers_by_peer,
    const std::vector<bool>& peer_valid, std::vector<uint8_t>& eid_table_host, aclshmemi_aiv_udma_info_t& copy_info)
{
    auto* wq_array = reinterpret_cast<aclshmemi_udma_wq_ctx_t*>(copy_info.sq_ptr);
    auto* rq_array = reinterpret_cast<aclshmemi_udma_wq_ctx_t*>(copy_info.rq_ptr);
    auto* scq_array = reinterpret_cast<aclshmemi_udma_cq_ctx_t*>(copy_info.scq_ptr);
    auto* rcq_array = reinterpret_cast<aclshmemi_udma_cq_ctx_t*>(copy_info.rcq_ptr);
    auto* mem_array = reinterpret_cast<aclshmemi_ubmem_info_t*>(copy_info.mem_ptr);

    for (uint32_t peer = 0; peer < rank_count_; ++peer) {
        if (peer == rank_id_ || !peer_valid[peer]) {
            // Self entry and any unconnected peer stay zero-initialized; the data
            // plane never issues a self-send and asserts against self pe.
            continue;
        }
        FillWqCtx(sq_contexts_by_peer[peer], peer, wq_array[peer]);
        rq_array[peer] = wq_array[peer];
        FillCqCtx(cq_contexts_by_peer[peer], scq_array[peer]);
        rcq_array[peer] = scq_array[peer];
        FillMemInfo(sq_contexts_by_peer[peer], remote_buffers_by_peer[peer], mem_array[peer]);

        // Stage the remote EID raw bytes (from the SQ context) into the device EID table
        // and point mem_array[peer].eid_addr at the device-resident copy, matching the
        // legacy behavior where the data plane reads rmtEid[0..1] from eid_addr.
        uint8_t* eid_slot = eid_table_host.data() + static_cast<uint64_t>(peer) * URMA_EID_RAW_SIZE;
        (void)memcpy_s(eid_slot, URMA_EID_RAW_SIZE, sq_contexts_by_peer[peer].contextInfo.ubJfs.remoteEID,
            URMA_EID_RAW_SIZE);
        mem_array[peer].eid_addr =
            reinterpret_cast<uint64_t>(
                static_cast<uint8_t*>(eid_dev_) + static_cast<uint64_t>(peer) * URMA_EID_RAW_SIZE);
    }
}

Result UdmaTransportManager::CopyEidTableToDevice(const std::vector<uint8_t>& eid_table_host)
{
    auto ret = DlAclApi::AclrtMemcpy(
        eid_dev_, eid_table_host.size(), eid_table_host.data(), eid_table_host.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Copy udma remote eid table to device failed, ret = " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::CopyUdmaInfoToDevice(
    uint32_t qp_num, std::vector<uint8_t>& udma_info_buffer, aclshmemi_aiv_udma_info_t& copy_info)
{
    // Allocate the device blob and rebase the section pointers to device addresses.
    auto alloc_ret = AllocateAndClearDeviceBuffer(udma_info_dev_, udma_info_size_, "udma info");
    if (alloc_ret != ACLSHMEM_SUCCESS) {
        return alloc_ret;
    }
    SetUdmaInfoSectionPtrs(copy_info, udma_info_dev_, rank_count_, qp_num);

    auto ret = DlAclApi::AclrtMemcpy(
        udma_info_dev_, udma_info_size_, udma_info_buffer.data(), udma_info_size_, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Copy udma info to device failed, ret = " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::ReserveScratchBuffers()
{
    amo_dev_list_.assign(rank_count_, nullptr);
    for (uint32_t peer = 0; peer < rank_count_; ++peer) {
        if (peer == rank_id_) {
            continue;
        }
        auto ret = AllocateAndClearDeviceBuffer(amo_dev_list_[peer], sizeof(uint64_t), "udma amo scratch");
        if (ret != ACLSHMEM_SUCCESS) {
            return ret;
        }
    }
    return ACLSHMEM_SUCCESS;
}

void UdmaTransportManager::FillWqCtx(const SqContext& sq_context, uint32_t peer, aclshmemi_udma_wq_ctx_t& dst_wq) const
{
    const auto& ubJfs = sq_context.contextInfo.ubJfs;
    dst_wq.wqn = ubJfs.jfsID;
    dst_wq.buf_addr = ubJfs.sqVa;
    // HCOMM exposes the raw WQE byte size; the data plane consumes it directly.
    dst_wq.wqe_size = ubJfs.wqeSize;
    dst_wq.depth = shm::UDMA_SQ_BASKBLK_CNT;
    dst_wq.head = 0;
    dst_wq.tail = 0;
    dst_wq.db_mode = aclshmemi_udma_db_mode_t::SW_DB;
    dst_wq.db_addr = ubJfs.dbVa;
    dst_wq.sl = 0;
    dst_wq.wqe_cnt = 0;
    dst_wq.amo_addr = reinterpret_cast<uint64_t>(amo_dev_list_[peer]);
}

void UdmaTransportManager::FillCqCtx(const CqContext& cq_context, aclshmemi_udma_cq_ctx_t& dst_cq) const
{
    const auto& ubJfc = cq_context.contextInfo.ubJfc;
    dst_cq.cqn = ubJfc.jfcID;
    dst_cq.buf_addr = ubJfc.scqVa;
    dst_cq.cqe_size = ubJfc.cqeSize;
    dst_cq.depth = shm::UDMA_CQ_DEPTH_DEFAULT;
    dst_cq.head = 0;
    dst_cq.tail = 0;
    dst_cq.db_mode = aclshmemi_udma_db_mode_t::SW_DB;
    dst_cq.db_addr = ubJfc.dbVa;
}

void UdmaTransportManager::FillMemInfo(
    const SqContext& sq_context, const RegedBufferEntity& remote_buffer, aclshmemi_ubmem_info_t& dst_mem) const
{
    const auto& ub = remote_buffer.bufferInfo.rma.protectionInfo.memInfo.ub;
    dst_mem.token_value_valid = true; // token-based access control enabled (data plane sets tokenEn = 1)
    dst_mem.rmt_jetty_type = 1;       // remote jetty type: 1 = jetty (peer-to-peer)
    dst_mem.target_hint = 0;         // no target selection preference
    dst_mem.tpn = sq_context.contextInfo.ubJfs.tpID;
    dst_mem.tid = ub.tokenId;
    dst_mem.rmt_token_value = ub.tokenValue;
    dst_mem.len = static_cast<uint32_t>(remote_buffer.bufferInfo.rma.size);
    dst_mem.addr = remote_buffer.bufferInfo.rma.addr;
    dst_mem.eid_addr = 0; // filled by the caller after the device EID table is staged
}

void UdmaTransportManager::PrintHostUdmaInfo(const aclshmemi_aiv_udma_info_t& host_info) const
{
    SHM_LOG_DEBUG("=======================rank [" << rank_id_ << "] udma host info====================");
    SHM_LOG_DEBUG("rank[" << rank_id_ << "] udmaInfo.qp_num: " << host_info.qp_num);

    const auto* wq_array = reinterpret_cast<const aclshmemi_udma_wq_ctx_t*>(host_info.sq_ptr);
    const auto* cq_array = reinterpret_cast<const aclshmemi_udma_cq_ctx_t*>(host_info.scq_ptr);
    const auto* mem_array = reinterpret_cast<const aclshmemi_ubmem_info_t*>(host_info.mem_ptr);
    if (wq_array == nullptr || cq_array == nullptr || mem_array == nullptr) {
        SHM_LOG_WARN("rank[" << rank_id_ << "] udma host info section pointer is null, skip printing.");
        return;
    }

    for (uint32_t pe = 0; pe < rank_count_; ++pe) {
        const auto& wq = wq_array[pe];
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] WQCtx.wqn: " << wq.wqn);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] WQCtx.buf_addr: " << wq.buf_addr);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] WQCtx.wqe_size: " << wq.wqe_size);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] WQCtx.depth: " << wq.depth);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] WQCtx.head: " << wq.head);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] WQCtx.tail: " << wq.tail);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] WQCtx.db_mode: " << static_cast<int>(wq.db_mode));
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] WQCtx.db_addr: " << wq.db_addr);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] WQCtx.sl: " << wq.sl);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] WQCtx.wqe_cnt: " << wq.wqe_cnt);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] WQCtx.amo_addr: " << wq.amo_addr);

        const auto& cq = cq_array[pe];
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] CQCtx.cqn: " << cq.cqn);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] CQCtx.buf_addr: " << cq.buf_addr);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] CQCtx.cqe_size: " << cq.cqe_size);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] CQCtx.depth: " << cq.depth);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] CQCtx.head: " << cq.head);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] CQCtx.tail: " << cq.tail);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] CQCtx.db_mode: " << static_cast<int>(cq.db_mode));
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] CQCtx.db_addr: " << cq.db_addr);

        const auto& mem = mem_array[pe];
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] MemInfo.token_value_valid: " << mem.token_value_valid);
        SHM_LOG_DEBUG(
            "rank[" << rank_id_ << "] peer[" << pe << "] MemInfo.rmt_jetty_type: " << mem.rmt_jetty_type);
        SHM_LOG_DEBUG(
            "rank[" << rank_id_ << "] peer[" << pe << "] MemInfo.target_hint: " << static_cast<int>(mem.target_hint));
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] MemInfo.tpn: " << mem.tpn);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] MemInfo.tid: " << mem.tid);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] MemInfo.rmt_token_value: " << mem.rmt_token_value);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] MemInfo.len: " << mem.len);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] MemInfo.addr: " << mem.addr);
        SHM_LOG_DEBUG("rank[" << rank_id_ << "] peer[" << pe << "] MemInfo.eid_addr: " << mem.eid_addr);
    }
}

std::vector<HcommMemHandle> UdmaTransportManager::CollectChannelMemHandles(uint32_t eid_index) const
{
    std::vector<HcommMemHandle> mem_handles;
    for (const auto& addrEntry : mem_record_map_) {
        const auto& hcomm_handles = addrEntry.second;
        auto handleIt = hcomm_handles.find(eid_index);
        if (handleIt != hcomm_handles.end() && handleIt->second != nullptr) {
            mem_handles.push_back(handleIt->second);
        }
    }
    return mem_handles;
}

Result UdmaTransportManager::AsyncConnect()
{
    const uint32_t local_endpoint_count = static_cast<uint32_t>(endpoint_desc_map_.size());
    std::vector<uint32_t> all_endpoint_counts(rank_count_);
    g_boot_handle.allgather(&local_endpoint_count, all_endpoint_counts.data(), sizeof(uint32_t), &g_boot_handle);

    const auto max_endpoint_count_it = std::max_element(all_endpoint_counts.begin(), all_endpoint_counts.end());
    const uint32_t max_endpoint_count = (max_endpoint_count_it == all_endpoint_counts.end()) ? 0 : *max_endpoint_count_it;
    if (max_endpoint_count == 0) {
        SHM_LOG_ERROR("No local hcomm endpoint descriptor was exchanged.");
        return ACLSHMEM_INNER_ERROR;
    }

    std::vector<exchanged_endpoint_desc_info_t> local_endpoints(max_endpoint_count);
    uint32_t packed_index = 0;
    for (const auto& endpoint_entry : endpoint_desc_map_) {
        exchanged_endpoint_desc_info_t& packed = local_endpoints[packed_index];
        packed.eid_index = endpoint_entry.first;
        packed.valid = 1;
        packed.desc = endpoint_entry.second;
        ++packed_index;
    }

    std::vector<exchanged_endpoint_desc_info_t> all_endpoint_info(rank_count_ * max_endpoint_count);
    g_boot_handle.allgather(
        local_endpoints.data(), all_endpoint_info.data(),
        static_cast<uint64_t>(sizeof(exchanged_endpoint_desc_info_t) * max_endpoint_count), &g_boot_handle);

    std::vector<ChannelHandle> created_channels;
    std::vector<uint32_t> channel_peers;
    auto destroyCreatedResources = [&created_channels]() {
        if (!created_channels.empty()) {
            (void)HcommChannelDestroy(created_channels.data(), static_cast<uint32_t>(created_channels.size()));
            created_channels.clear();
        }
    };
    for (uint32_t peer = 0; peer < rank_count_; ++peer) {
        if (peer == rank_id_) {
            continue;
        }

        auto local_eid_it = peer_eid_index_map_.find(peer);
        auto remote_eid_it = peer_remote_eid_index_map_.find(peer);
        if (local_eid_it == peer_eid_index_map_.end() || remote_eid_it == peer_remote_eid_index_map_.end()) {
            SHM_LOG_ERROR("Failed to find EID route for peer " << peer);
            destroyCreatedResources();
            return ACLSHMEM_INNER_ERROR;
        }

        const uint32_t local_eid_index = local_eid_it->second;
        const uint32_t remote_eid_index = remote_eid_it->second;
        auto endpoint_it = endpoint_handle_map_.find(local_eid_index);
        if (endpoint_it == endpoint_handle_map_.end() || endpoint_it->second == nullptr) {
            SHM_LOG_ERROR("Failed to find hcomm endpoint for peer " << peer << ", local_eid_index = " << local_eid_index);
            destroyCreatedResources();
            return ACLSHMEM_INNER_ERROR;
        }

        const exchanged_endpoint_desc_info_t* remote_endpoint_info = nullptr;
        const uint32_t peer_endpoint_count = all_endpoint_counts[peer];
        for (uint32_t idx = 0; idx < peer_endpoint_count; ++idx) {
            const exchanged_endpoint_desc_info_t& candidate = all_endpoint_info[peer * max_endpoint_count + idx];
            if (candidate.valid != 0 && candidate.eid_index == remote_eid_index) {
                remote_endpoint_info = &candidate;
                break;
            }
        }
        if (remote_endpoint_info == nullptr) {
            SHM_LOG_ERROR(
                "No remote hcomm endpoint descriptor was exchanged for peer " << peer << " on remote EID index "
                                                                              << remote_eid_index);
            destroyCreatedResources();
            return ACLSHMEM_INNER_ERROR;
        }

        std::vector<HcommMemHandle> mem_handles = CollectChannelMemHandles(local_eid_index);
        if (mem_handles.empty()) {
            SHM_LOG_ERROR(
                "No active Hcomm mem handle for local EID index " << local_eid_index << " when creating channel for peer "
                                                                  << peer);
            destroyCreatedResources();
            return ACLSHMEM_INNER_ERROR;
        }
        HcommChannelDesc channel_desc{};
        auto desc_init_ret = HcommChannelDescInit(&channel_desc, ACLSHMEM_HCOMM_CHANNEL_NUM_PER_PEER);
        if (desc_init_ret != 0) {
            SHM_LOG_ERROR("HcommChannelDescInit failed for peer " << peer << ", ret = " << desc_init_ret);
            destroyCreatedResources();
            return ACLSHMEM_INNER_ERROR;
        }
        channel_desc.remoteEndpoint = remote_endpoint_info->desc;
        channel_desc.notifyNum = 0;
        channel_desc.exchangeAllMems = false;
        channel_desc.memHandles = mem_handles.data();
        channel_desc.memHandleNum = static_cast<uint32_t>(mem_handles.size());
        channel_desc.qos = ACLSHMEM_HCOMM_DEFAULT_QOS;

        ChannelHandle channel_handle = 0;
        auto hcomm_ret = HcommChannelCreate(
            endpoint_it->second, COMM_ENGINE_AIV, &channel_desc, ACLSHMEM_HCOMM_CHANNEL_NUM_PER_PEER, &channel_handle);
        if (hcomm_ret != 0 || channel_handle == 0) {
            SHM_LOG_ERROR("HcommChannelCreate failed for peer " << peer << ", ret = " << hcomm_ret);
            if (channel_handle != 0) {
                (void)HcommChannelDestroy(&channel_handle, ACLSHMEM_HCOMM_CHANNEL_NUM_PER_PEER);
            }
            destroyCreatedResources();
            return ACLSHMEM_INNER_ERROR;
        }

        created_channels.push_back(channel_handle);
        channel_peers.push_back(peer);
    }

    channel_handles_ = std::move(created_channels);
    channel_peers_ = channel_peers;
    auto build_ret = BuildUdmaInfo(channel_handles_, channel_peers);
    if (build_ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("BuildUdmaInfo failed, ret = " << build_ret);
        DestroyChannels();
        FreeDeviceInfo();
        return build_ret;
    }

    SHM_LOG_INFO(
        "Create hcomm channels success, totalChannelNum = " << channel_handles_.size() << ", channelNumPerPeer = "
                                                            << ACLSHMEM_HCOMM_CHANNEL_NUM_PER_PEER);
    connected_ = true;
    return ACLSHMEM_SUCCESS;
}

bool UdmaTransportManager::PrepareOpenDevice(uint32_t device_id, uint32_t rank_count)
{
    RootInfo root_info;
    TopoInfo topo_info;
    uint32_t local_id = 0;
    uint32_t eid_count = 0;
    int32_t phy_id = -1;

    auto ret = DlAclApi::AclrtGetPhyDevIdByLogicDevId(static_cast<int32_t>(device_id), &phy_id);
    SHM_ASSERT_LOG_AND_RETURN(
        ret == 0 && phy_id >= 0,
        "AclrtGetPhyDevIdByLogicDevId() return=" << ret << ", input device_id=" << device_id
                                                 << ", output phy_id=" << phy_id,
        false);

    if (!TopoReader::ParseRootInfo(phy_id, root_info)) {
        SHM_LOG_ERROR("Failed to parse the rootinfo file.");
        return false;
    }
    phy_id_ = static_cast<uint32_t>(phy_id);
    SHM_LOG_INFO("Resolved phy id from current device mapping, device_id=" << device_id << ", phy_id=" << phy_id);
    if (!TopoReader::ParseTopoInfo(root_info.topo_file_path, topo_info)) {
        SHM_LOG_ERROR("Failed to parse the topology file at path " << root_info.topo_file_path);
        return false;
    }

    if (!TopoReader::GetLocalId(root_info, phy_id_, local_id)) {
        SHM_LOG_ERROR("Failed to find local_id for phy_id: " << phy_id_);
        return false;
    }
    if (!TopoReader::GetEidCount(root_info, eid_count)) {
        SHM_LOG_ERROR("Failed to find eid count from rootinfo.");
        return false;
    }

    std::vector<uint32_t> eid_count_list(rank_count);
    g_boot_handle.allgather(&eid_count, eid_count_list.data(), sizeof(uint32_t), &g_boot_handle);
    const auto max_eid_count_it = std::max_element(eid_count_list.begin(), eid_count_list.end());
    const uint32_t max_eid_count = (max_eid_count_it == eid_count_list.end()) ? 0 : *max_eid_count_it;
    if (max_eid_count == 0) {
        SHM_LOG_ERROR("Invalid eidSlotCount resolved from rootinfo rank_addr_list.");
        return false;
    }

    std::vector<uint32_t> local_id_list(rank_count);
    g_boot_handle.allgather(&local_id, local_id_list.data(), sizeof(uint32_t), &g_boot_handle);
    std::vector<int32_t> local_route_by_peer(rank_count, INVALID_EID_INDEX);

    for (uint32_t peer = 0; peer < rank_count; ++peer) {
        if (peer == rank_id_) {
            continue;
        }
        uint32_t eid_index = 0;
        std::array<uint8_t, URMA_EID_RAW_SIZE> eid_raw{};
        uint32_t peer_local_id = local_id_list[peer];
        if (!TopoReader::GetLocalEidRouteForPeer(root_info, topo_info, local_id, peer_local_id, eid_index, eid_raw)) {
            SHM_LOG_ERROR(
                "Failed to resolve the local EID route for peer rank " << peer << ". The local_id was " << local_id
                                                                       << " and the peer local_id was " << peer_local_id);
            return false;
        }
        peer_eid_index_map_[peer] = eid_index;
        local_route_by_peer[peer] = static_cast<int32_t>(eid_index);

        if (!CreateEndpoint(eid_index, eid_raw)) {
            SHM_LOG_ERROR("CreateEndpoint failed for peer " << peer << " with EID index " << eid_index);
            return false;
        }
    }

    std::vector<int32_t> all_route_by_peer(rank_count * rank_count, INVALID_EID_INDEX);
    g_boot_handle.allgather(
        local_route_by_peer.data(), all_route_by_peer.data(), sizeof(int32_t) * rank_count, &g_boot_handle);

    for (uint32_t peer = 0; peer < rank_count; ++peer) {
        if (peer == rank_id_) {
            continue;
        }
        int32_t remote_route = all_route_by_peer[peer * rank_count + rank_id_];
        if (remote_route < 0 || static_cast<uint32_t>(remote_route) >= max_eid_count) {
            SHM_LOG_ERROR(
                "Invalid remote EID route for peer rank "
                << peer << ", remote_route = " << remote_route << ", local rank = " << rank_id_ << ", local_id = "
                << local_id << ", peer_local_id = " << local_id_list[peer] << ", eidSlotCount = " << max_eid_count);
            return false;
        }
        peer_remote_eid_index_map_[peer] = static_cast<uint32_t>(remote_route);
    }

    return true;
}

bool UdmaTransportManager::CreateEndpoint(uint32_t eid_index, const std::array<uint8_t, URMA_EID_RAW_SIZE>& target_eid_raw)
{
    auto endpoint_it = endpoint_handle_map_.find(eid_index);
    if (endpoint_it != endpoint_handle_map_.end() && endpoint_it->second != nullptr) {
        return true;
    }

    EndpointDesc endpoint_desc{};
    auto desc_init_ret = EndpointDescInit(&endpoint_desc, 1);
    if (desc_init_ret != 0) {
        SHM_LOG_ERROR("EndpointDescInit failed for EID index " << eid_index << ", ret = " << desc_init_ret);
        return false;
    }

    uint32_t sd_id = 0;
    uint32_t server_id = 0;
    uint32_t super_pod_id = 0;
    auto device_info_ret = shm::MemSegment::GetDeviceInfo(sd_id, server_id, super_pod_id);
    if (device_info_ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Get local device info for HCOMM endpoint failed, ret = " << device_info_ret);
        return false;
    }

    endpoint_desc.protocol = COMM_PROTOCOL_UBC_CTP;
    endpoint_desc.commAddr.type = COMM_ADDR_TYPE_EID;
    int copyRet = memcpy_s(
        endpoint_desc.commAddr.eid, sizeof(endpoint_desc.commAddr.eid), target_eid_raw.data(), target_eid_raw.size());
    if (copyRet != EOK) {
        SHM_LOG_ERROR("Copy target EID to HCOMM endpoint desc failed, ret = " << copyRet);
        return false;
    }
    endpoint_desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    endpoint_desc.loc.device.devPhyId = phy_id_;
    endpoint_desc.loc.device.superDevId = sd_id;
    endpoint_desc.loc.device.serverIdx = server_id;
    endpoint_desc.loc.device.superPodIdx = super_pod_id;

    EndpointHandle endpoint_handle = nullptr;
    auto ret = HcommEndpointCreate(&endpoint_desc, &endpoint_handle);
    if (ret != 0 || endpoint_handle == nullptr) {
        SHM_LOG_ERROR("HcommEndpointCreate failed for EID index " << eid_index << ", ret = " << ret);
        return false;
    }
    endpoint_desc_map_[eid_index] = endpoint_desc;
    endpoint_handle_map_[eid_index] = endpoint_handle;
    return true;
}

Result UdmaTransportManager::CheckPrepareOptions(const HybmTransPrepareOptions& options)
{
    if (role_ != HYBM_ROLE_PEER) {
        SHM_LOG_INFO("Transport role: " << role_ << " check options passed.");
        return ACLSHMEM_SUCCESS;
    }

    if (options.options.size() > rank_count_) {
        SHM_LOG_ERROR("Options size() is " << options.options.size() << " larger than rank count: " << rank_count_);
        return ACLSHMEM_INVALID_PARAM;
    }

    if (options.options.find(rank_id_) == options.options.end()) {
        SHM_LOG_ERROR("Options do not contain self rankId: " << rank_id_);
        return ACLSHMEM_INVALID_PARAM;
    }

    for (auto it = options.options.begin(); it != options.options.end(); ++it) {
        if (it->first >= rank_count_) {
            SHM_LOG_ERROR("RankId: " << it->first << " is out of range [0, " << rank_count_ << ")");
            return ACLSHMEM_INVALID_PARAM;
        }
    }

    return ACLSHMEM_SUCCESS;
}

void UdmaTransportManager::FreeDeviceInfo()
{
    if (udma_info_dev_ != nullptr) {
        (void)DlAclApi::AclrtFree(udma_info_dev_);
        udma_info_dev_ = nullptr;
    }
    if (eid_dev_ != nullptr) {
        (void)DlAclApi::AclrtFree(eid_dev_);
        eid_dev_ = nullptr;
    }
    for (auto& amo_dev : amo_dev_list_) {
        if (amo_dev != nullptr) {
            (void)DlAclApi::AclrtFree(amo_dev);
            amo_dev = nullptr;
        }
    }
    amo_dev_list_.clear();
    udma_info_size_ = 0;
}

void UdmaTransportManager::DestroyChannels()
{
    if (!channel_handles_.empty()) {
        auto hcomm_ret = HcommChannelDestroy(channel_handles_.data(), static_cast<uint32_t>(channel_handles_.size()));
        if (hcomm_ret != 0) {
            SHM_LOG_WARN("HcommChannelDestroy failed, ret = " << hcomm_ret);
        }
    }
    channel_handles_.clear();
    channel_peers_.clear();
    SHM_LOG_INFO("Destroy hcomm channels success.");
}

void UdmaTransportManager::CleanupResources()
{
    DestroyChannels();
    FreeDeviceInfo();

    for (const auto& mem_by_addr : mem_record_map_) {
        for (const auto& mem_entry : mem_by_addr.second) {
            auto endpoint_it = endpoint_handle_map_.find(mem_entry.first);
            if (endpoint_it == endpoint_handle_map_.end() || endpoint_it->second == nullptr || mem_entry.second == nullptr) {
                continue;
            }
            auto hcomm_ret = HcommMemUnreg(endpoint_it->second, mem_entry.second);
            if (hcomm_ret != 0) {
                SHM_LOG_WARN("HcommMemUnreg failed for EID index " << mem_entry.first << ", ret = " << hcomm_ret);
            }
        }
    }
    mem_record_map_.clear();
    SHM_LOG_INFO("Unregister memory success.");

    for (const auto& endpoint_entry : endpoint_handle_map_) {
        if (endpoint_entry.second == nullptr) {
            continue;
        }
        auto ret = HcommEndpointDestroy(endpoint_entry.second);
        if (ret != 0) {
            SHM_LOG_WARN("HcommEndpointDestroy failed for EID index " << endpoint_entry.first << ", ret = " << ret);
        }
    }
    endpoint_handle_map_.clear();
    peer_eid_index_map_.clear();
    peer_remote_eid_index_map_.clear();
    endpoint_desc_map_.clear();
    channel_handles_.clear();
    channel_peers_.clear();
    connected_ = false;
}

} // namespace device
} // namespace transport
} // namespace shm
