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
    bool CreateEndpoint(uint32_t eidIndex, const std::array<uint8_t, 16>& targetEidRaw);
    bool PrepareOpenDevice(uint32_t deviceId, uint32_t rankCount);
    Result BuildUdmaInfo(const std::vector<uint64_t>& channelPtrs, const std::vector<uint32_t>& channelPeers);
    Result ReadChannelContexts(const std::vector<uint64_t>& channelPtrs, const std::vector<uint32_t>& channelPeers,
        std::vector<SqContext>& sqContextsByPeer, std::vector<CqContext>& cqContextsByPeer,
        std::vector<RegedBufferEntity>& remoteBuffersByPeer, std::vector<bool>& peerValid) const;
    Result PrepareUdmaInfoBuffers(std::vector<uint8_t>& eidTableHost);
    void InitHostUdmaInfo(uint32_t qpNum, std::vector<uint8_t>& udmaInfoBuffer, ACLSHMEMAIVUDMAInfo*& copyInfo);
    void FillHostUdmaInfo(const std::vector<SqContext>& sqContextsByPeer,
        const std::vector<CqContext>& cqContextsByPeer, const std::vector<RegedBufferEntity>& remoteBuffersByPeer,
        const std::vector<bool>& peerValid, std::vector<uint8_t>& eidTableHost, ACLSHMEMAIVUDMAInfo& copyInfo);
    Result CopyEidTableToDevice(const std::vector<uint8_t>& eidTableHost);
    Result CopyUdmaInfoToDevice(
        uint32_t qpNum, std::vector<uint8_t>& udmaInfoBuffer, ACLSHMEMAIVUDMAInfo& copyInfo);
    Result ReserveScratchBuffers();
    void FillWqCtx(const SqContext& sqContext, uint32_t peer, ACLSHMEMUDMAWQCtx& dstWq) const;
    void FillCqCtx(const CqContext& cqContext, ACLSHMEMUDMACqCtx& dstCq) const;
    void FillMemInfo(
        const SqContext& sqContext, const RegedBufferEntity& remoteBuffer, ACLSHMEMUBmemInfo& dstMem) const;
    void PrintHostUdmaInfo(const ACLSHMEMAIVUDMAInfo& hostInfo) const;
    std::vector<HcommMemHandle> CollectChannelMemHandles(uint32_t eidIndex) const;
    void FreeDeviceInfo();
    void DestroyChannels();
    Result CheckPrepareOptions(const HybmTransPrepareOptions& options);
    void CleanupResources();

private:
    uint32_t rankId_{0};
    uint32_t rankCount_{1};
    uint32_t phyId_{0};
    hybm_role_type role_{HYBM_ROLE_PEER};
    std::map<uint32_t, uint32_t> peerEidIndexMap_;                        // peerRankId -> local eidIndex
    std::map<uint32_t, uint32_t> peerRemoteEidIndexMap_;                  // peerRankId -> remote eidIndex
    std::map<uint32_t, EndpointDesc> endpointDescMap_;                    // eidIndex -> local hcomm endpoint desc
    std::map<uint32_t, EndpointHandle> endpointHandleMap_;                // eidIndex -> hcomm endpoint handle
    std::map<uint64_t, std::map<uint32_t, HcommMemHandle>> memRecordMap_; // addr -> eidIndex -> hcomm mem handle
    std::vector<ChannelHandle> channelHandles_;
    std::vector<uint32_t> channelPeers_;
    // The control plane fills a contiguous ACLSHMEMAIVUDMAInfo blob using the legacy
    // (jetty-manager) layout so the data plane consumes it unchanged. The per-peer
    // wqeCnt / amo / remote-EID scratch buffers are allocated separately, mirroring
    // the original DeviceJettyManager allocation scheme.
    void* udmaInfoDev_{nullptr};            // device pointer to the contiguous ACLSHMEMAIVUDMAInfo blob
    uint64_t udmaInfoSize_{0};              // byte size of the contiguous blob
    void* eidDev_{nullptr};                 // device pointer to uint8_t[rankCount_][16] remote EID raw, indexed by pe
    std::vector<void*> wqeCntDevList_;      // per-peer uint32_t wqe counter device buffers, indexed by pe
    std::vector<void*> amoDevList_;         // per-peer uint64_t AMO scratch device buffers, indexed by pe
};
} // namespace device
} // namespace transport
} // namespace shm

#endif // MF_HYBRID_DEVICE_UDMA_TRANSPORT_MANAGER_H
