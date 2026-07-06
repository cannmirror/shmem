/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBRID_DEVICE_UDMA_TRANSPORT_MANAGER_H
#define MF_HYBRID_DEVICE_UDMA_TRANSPORT_MANAGER_H

#include <map>
#include <array>
#include <string>
#include <vector>
#include <cstdint>

#include "transport_manager.h"
#include "transport/topo/topo_reader.h"
#include "hcomm/hcomm_res_entity_defs.h"
#include "device_udma_def.h"

namespace shm {
namespace transport {
namespace device {

class UdmaTransportManager : public TransportManager {
public:
    UdmaTransportManager() noexcept;
    ~UdmaTransportManager() noexcept;
    Result OpenDevice(const TransportOptions& options) override;
    Result CloseDevice() override;
    Result RegisterMemoryRegion(const TransportMemoryRegion& mr) override;
    Result UnregisterMemoryRegion(uint64_t addr) override;
    Result Prepare(const HybmTransPrepareOptions& options) override;
    Result Connect() override;
    const void* GetQpInfo() const override;
    Result QueryMemoryKey(uint64_t /*addr*/, TransportMemoryKey& /*key*/) override { return ACLSHMEM_SUCCESS; }
    Result ParseMemoryKey(const TransportMemoryKey& /*key*/, uint64_t& /*addr*/, uint64_t& /*size*/) override
    {
        return ACLSHMEM_SUCCESS;
    }
    Result AsyncConnect() override;
    Result WaitForConnected(int64_t /*timeoutNs*/) override { return ACLSHMEM_SUCCESS; }
    Result UpdateRankOptions(const HybmTransPrepareOptions& /*options*/) override { return ACLSHMEM_SUCCESS; }
    const std::string& GetNic() const override
    {
        static const std::string empty_nic;
        return empty_nic;
    }

private:
    bool CreateEndpoint(uint32_t eid_index, const std::array<uint8_t, 16>& target_eid_raw);
    bool PrepareOpenDevice(uint32_t device_id, uint32_t rank_count);
    Result BuildUdmaInfo(const std::vector<uint64_t>& channel_ptrs, const std::vector<uint32_t>& channel_peers);
    Result ReadChannelContexts(const std::vector<uint64_t>& channel_ptrs, const std::vector<uint32_t>& channel_peers,
        std::vector<SqContext>& sq_contexts_by_peer, std::vector<CqContext>& cq_contexts_by_peer,
        std::vector<RegedBufferEntity>& remote_buffers_by_peer, std::vector<bool>& peer_valid) const;
    Result PrepareUdmaInfoBuffers(std::vector<uint8_t>& eid_table_host);
    void InitHostUdmaInfo(uint32_t qp_num, std::vector<uint8_t>& udma_info_buffer, aclshmemi_aiv_udma_info_t*& copy_info);
    void FillHostUdmaInfo(const std::vector<SqContext>& sq_contexts_by_peer,
        const std::vector<CqContext>& cq_contexts_by_peer, const std::vector<RegedBufferEntity>& remote_buffers_by_peer,
        const std::vector<bool>& peer_valid, std::vector<uint8_t>& eid_table_host, aclshmemi_aiv_udma_info_t& copy_info);
    Result CopyEidTableToDevice(const std::vector<uint8_t>& eid_table_host);
    Result CopyUdmaInfoToDevice(
        uint32_t qp_num, std::vector<uint8_t>& udma_info_buffer, aclshmemi_aiv_udma_info_t& copy_info);
    Result ReserveScratchBuffers();
    void FillWqCtx(const SqContext& sq_context, uint32_t peer, aclshmemi_udma_wq_ctx_t& dst_wq) const;
    void FillCqCtx(const CqContext& cq_context, aclshmemi_udma_cq_ctx_t& dst_cq) const;
    void FillMemInfo(
        const SqContext& sq_context, const RegedBufferEntity& remote_buffer, aclshmemi_ubmem_info_t& dst_mem) const;
    void PrintHostUdmaInfo(const aclshmemi_aiv_udma_info_t& host_info) const;
    std::vector<HcommMemHandle> CollectChannelMemHandles(uint32_t eid_index) const;
    void FreeDeviceInfo();
    void DestroyChannels();
    Result CheckPrepareOptions(const HybmTransPrepareOptions& options);
    void CleanupResources();

private:
    uint32_t rank_id_{0};
    uint32_t rank_count_{1};
    uint32_t user_id_{0};
    uint32_t phy_id_{0};
    hybm_role_type role_{HYBM_ROLE_PEER};
    std::map<uint32_t, uint32_t> peer_eid_index_map_;                        // peerRankId -> local eid_index
    std::map<uint32_t, uint32_t> peer_remote_eid_index_map_;                  // peerRankId -> remote eid_index
    std::map<uint32_t, EndpointDesc> endpoint_desc_map_;                    // eid_index -> local hcomm endpoint desc
    std::map<uint32_t, EndpointHandle> endpoint_handle_map_;                // eid_index -> hcomm endpoint handle
    std::map<uint64_t, std::map<uint32_t, HcommMemHandle>> mem_record_map_; // addr -> eid_index -> hcomm mem handle
    std::vector<ChannelHandle> channel_handles_;
    std::vector<uint32_t> channel_peers_;
    // The control plane fills a contiguous aclshmemi_aiv_udma_info_t blob using the legacy
    // (jetty-manager) layout so the data plane consumes it unchanged. The per-peer
    // amo / remote-EID scratch buffers are allocated separately, mirroring
    // the original DeviceJettyManager allocation scheme.
    void* udma_info_dev_{nullptr};            // device pointer to the contiguous aclshmemi_aiv_udma_info_t blob
    uint64_t udma_info_size_{0};              // byte size of the contiguous blob
    void* eid_dev_{nullptr};                 // device pointer to uint8_t[rank_count_][16] remote EID raw, indexed by pe
    std::vector<void*> amo_dev_list_;         // per-peer uint64_t AMO scratch device buffers, indexed by pe
};
} // namespace device
} // namespace transport
} // namespace shm

#endif // MF_HYBRID_DEVICE_UDMA_TRANSPORT_MANAGER_H
