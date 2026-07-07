/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBRID_DEVICE_JETTY_MANAGER_H
#define MF_HYBRID_DEVICE_JETTY_MANAGER_H

#include <netinet/in.h>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "hybm_def.h"
#include "dl_hccp_v2_api.h"
#include "dl_hccp_v2_def.h"
#include "device_udma_def.h"

namespace shm {
namespace transport {
namespace device {

struct PerEidJettyState {
    uint32_t eidIndex{0};
    void* ctxHandle{nullptr};
    void* tokenIdHandle{nullptr};
    void* chanHandle{nullptr};
    uint64_t cqVa{0};
    CqInfoT cqInfo = {0};
    void* cqHandle{nullptr};
    void* qpHandle{nullptr};
    QpCreateInfo qpCreateInfo_;
    std::vector<void*> remoteQpHandleList;
    std::vector<uint32_t> tpnList;
    void* amoAddr{nullptr};
    ACLSHMEMUDMAWQCtx localWq{};
    ACLSHMEMUDMACqCtx localCq{};
};

class DeviceJettyManager {
public:
    DeviceJettyManager(uint32_t deviceId, uint32_t rankId, uint32_t rankCount, uint32_t eidSlotCount) noexcept;
    ~DeviceJettyManager() noexcept;

    Result SetCtxHandles(const std::map<uint32_t, void*>& ctxHandleMap) noexcept;
    Result SetLocalMemInfos(const std::map<uint32_t, ACLSHMEMUBmemInfo>& localMemInfoMap) noexcept;
    Result SetEids(const std::map<uint32_t, HccpEid>& hccpEidMap) noexcept;
    Result SetTokenIdHandles(const std::map<uint32_t, void*>& tokenIdHandleMap) noexcept;
    Result SetPeerRoutes(
        const std::map<uint32_t, uint32_t>& peerLocalEidMap,
        const std::map<uint32_t, uint32_t>& peerRemoteEidMap) noexcept;
#if defined(ACLSHMEM_RELAY_SUPPORT)
    // Full N x N routing matrix from PrepareOpenDevice's allgather.
    // entry[rank * N + peer] is the local-port eidIndex that `rank` uses to reach `peer`.
    // Used to compute target EIDs for (actual_pe, relay_pe) slot in relay mode.
    Result SetGlobalRoutes(const std::vector<int32_t>& globalRoutes) noexcept;
#endif
    Result Startup() noexcept;
    Result Shutdown() noexcept;
    void* GetJettyInfoAddress() noexcept;
    uint64_t GetJFCInfoAddress() const noexcept;

private:
    Result JFCCreate(PerEidJettyState& state) noexcept;
    Result JettyCreate(PerEidJettyState& state) noexcept;
    Result JettyImport() noexcept;
    Result JettyBind() noexcept;
    bool ReserveUdmaInfoSpace() noexcept;
    std::vector<uint32_t> CollectUsedLocalEids() const noexcept;
    bool BuildLocalQpPublishByEid(
        std::vector<QpImportInfoT>& qpImportByEid, std::vector<QpKeyT>& qpKeyByEid) const noexcept;
    uint32_t GetFallbackLocalEid() const noexcept;
    HccpEid ToImportedEid(const HccpEid& hccpEid) const noexcept;

    void FillUdmaWq(ACLSHMEMUDMAWQCtx& srcWq, ACLSHMEMUDMAWQCtx& dstWq) const;
    void FillUdmaCq(ACLSHMEMUDMACqCtx& srcCq, ACLSHMEMUDMACqCtx& dstCq) const;
    void FillUdmaMem(ACLSHMEMUBmemInfo& srcMem, ACLSHMEMUBmemInfo& dstMem) const;
    // Write one resolved (localEid, remoteEid) tuple into the UDMA info slot `slot` for target
    // `actualRank`. `remoteTpn` is the imported tpn for this slot, resolved by the caller (peer-
    // indexed for direct, slot-indexed for relay). Shared by the direct and relay fill loops.
    Result FillOneUdmaSlot(
        ACLSHMEMAIVUDMAInfo* copyInfo, const std::vector<ACLSHMEMUBmemInfo>& allMemByEid, uint32_t fallbackLocalEid,
        uint32_t slot, uint32_t actualRank, uint32_t localEid, uint32_t remoteEid, uint32_t remoteTpn) noexcept;
    Result FillUdmaInfo() noexcept;
    void PrintHostInfo(ACLSHMEMAIVUDMAInfo& hostInfo) const;

#if defined(ACLSHMEM_RELAY_SUPPORT)
    // Resolve the (local egress EID, remote target EID) pair for one device slot (actualRank,
    // relayRank). Shared by ImportRelayQps and FillRelayUdmaSlots so import and fill stay in lock
    // step. Sets `skip` for the meaningless (actual==relay, actual!=self) diagonal.
    Result ResolveRelaySlotRoute(
        uint32_t actualRank, uint32_t relayRank, uint32_t fallbackLocalEid, bool& skip, uint32_t& localEid,
        uint32_t& remoteEid) noexcept;
    // Relay-mode QP import: import one remote QP per device slot (actualPe * N + relayPe). Slot
    // indexing (instead of the lossy eidIndex->relay inversion) lets multiple relays that share one
    // local egress EID each keep their own remote QP / tpn.
    Result ImportRelayQps(
        const std::vector<QpImportInfoT>& allQpImportByEid, const std::vector<QpKeyT>& allQpKeyByEid) noexcept;
    // Relay-mode slot fill: N*N (actual_pe, relay_pe) slots resolved from globalRoutes_.
    Result FillRelayUdmaSlots(
        ACLSHMEMAIVUDMAInfo* copyInfo, const std::vector<ACLSHMEMUBmemInfo>& allMemByEid,
        uint32_t fallbackLocalEid) noexcept;
#else
    // Direct-mode QP import (v1.5.0 equivalent): each bucket imports only the peers whose local
    // egress EID equals that bucket's eidIndex.
    Result ImportDirectQps(
        const std::vector<QpImportInfoT>& allQpImportByEid, const std::vector<QpKeyT>& allQpKeyByEid) noexcept;
    // Direct-mode slot fill (v1.5.0 equivalent): N slots, one per target peer (slot == peer).
    Result FillDirectUdmaSlots(
        ACLSHMEMAIVUDMAInfo* copyInfo, const std::vector<ACLSHMEMUBmemInfo>& allMemByEid,
        uint32_t fallbackLocalEid) noexcept;
#endif

    const uint32_t deviceId_;
    const uint32_t rankId_;
    const uint32_t rankCount_;
    const uint32_t eidCount_;
    std::map<uint32_t, void*> ctxHandleMap_;                // eidIndex -> ctxHandle
    std::map<uint32_t, ACLSHMEMUBmemInfo> localMemInfoMap_; // eidIndex -> local UDMA mem info
    std::map<uint32_t, HccpEid> localHccpEidMap_;           // eidIndex -> local HCCP EID
    std::map<uint32_t, void*> tokenIdHandleMap_;            // eidIndex -> tokenIdHandle
    std::map<uint32_t, uint32_t> peerLocalEidMap_;          // peerRankId -> local eidIndex
    std::map<uint32_t, uint32_t> peerRemoteEidMap_;         // peerRankId -> remote eidIndex
#if defined(ACLSHMEM_RELAY_SUPPORT)
    std::vector<int32_t> globalRoutes_;                     // [rank * rankCount + peer] -> rank's local eidIndex toward peer
    // Slot-indexed relay remote-QP / tpn storage, sized N*N and addressed by (actualPe * N +
    // relayPe). Unlike the per-bucket peer-indexed PerEidJettyState::{remoteQpHandleList,tpnList},
    // this keeps a distinct handle/tpn for every (actual, relay) slot even when several relays
    // share one local egress EID (e.g. cross-node peers behind the same NIC port).
    std::vector<void*> relayRemoteQpBySlot_;                // [actualPe * rankCount + relayPe] -> imported remote QP handle
    std::vector<void*> relayRemoteQpCtxBySlot_;             // [actualPe * rankCount + relayPe] -> ctxHandle used for import
    std::vector<uint32_t> relayTpnBySlot_;                  // [actualPe * rankCount + relayPe] -> imported tpn
#endif
    std::map<uint32_t, PerEidJettyState> jettyStateMap_;    // eidIndex -> per-EID jetty state
    TransportModeT transportMode_ = TransportModeT::CONN_RM;

    // device
    void* udmaInfo_{nullptr};
    void* hccpEidDevice_{nullptr};

    // host
    uint32_t udmaInfoSize_{0};
};
} // namespace device
} // namespace transport
} // namespace shm

#endif // MF_HYBRID_DEVICE_JETTY_MANAGER_H
