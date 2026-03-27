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
#include <memory>
#include <vector>

#include "hybm_def.h"
#include "dl_hccp_v2_api.h"
#include "dl_hccp_v2_def.h"
#include "device_udma_def.h"

namespace shm {
namespace transport {
namespace device {

class DeviceJettyManager {
public:
    DeviceJettyManager(uint32_t deviceId, uint32_t rankId, uint32_t rankCount) noexcept;
    ~DeviceJettyManager() noexcept;

    Result SetLocalMemInfo(const ACLSHMEMUBmemInfo &localMemInfo) noexcept;
    Result SetSwapEid(const HccpEid &hccpEid) noexcept;
    Result SetTokenIdHandle(void *tokenIdHandle) noexcept;
    Result Startup(void *ctx_handle) noexcept;
    Result Shutdown() noexcept;
    void *GetJettyInfoAddress() noexcept;
    uint64_t GetJFCInfoAddress() const noexcept;

private:
    Result JFCCreate() noexcept;
    Result JettyCreate() noexcept;
    Result JettyImport() noexcept;
    Result JettyBind() noexcept;
    bool ReserveUdmaInfoSpace() noexcept;

    void FillUdmaWq(ACLSHMEMUDMAWQCtx &srcWq, ACLSHMEMUDMAWQCtx &dstWq) const;
    void FillUdmaCq(ACLSHMEMUDMACqCtx &srcCq, ACLSHMEMUDMACqCtx &dstCq) const;
    void FillUdmaMem(ACLSHMEMUBmemInfo &srcMem, ACLSHMEMUBmemInfo &dstMem) const;
    Result FillUdmaInfo() noexcept;
    void PrintHostInfo(ACLSHMEMAIVUDMAInfo &hostInfo) const;

    const uint32_t deviceId_;
    const uint32_t rankId_;
    const uint32_t rankCount_;
    void *ctxHandle_{nullptr};
    ACLSHMEMUBmemInfo localMemInfo_;
    HccpEid localHccpEid_;
    void *tokenIdHandle_{nullptr};

    void *chanHandle_{nullptr};
    uint64_t cqVa_;
    CqInfoT cqInfo_ = {0};
    void *cqHandle_{nullptr};
    void *qpHandle_{nullptr};
    QpCreateInfo qpCreateInfo_;
    std::vector<void*> remoteQpHandleList_;
    TransportModeT transportMode_ = TransportModeT::CONN_RM;

    // device
    void *udmaInfo_{nullptr};
    void *hccpEidDevice_{nullptr};
    void *cqPiAddr_{nullptr};
    void *cqCiAddr_{nullptr};
    void *sqPiAddr_{nullptr};
    void *sqCiAddr_{nullptr};

    // host
    std::vector<ACLSHMEMUDMAWQCtx> wqInfoList_;
    std::vector<ACLSHMEMUDMACqCtx> cqInfoList_;
    uint32_t udmaInfoSize_{0};
    std::vector<ACLSHMEMUBmemInfo> ubMemInfoList_;
    std::vector<HccpEid> hccpEidList_;
    std::vector<uint32_t> tpnList_;
    std::vector<QpKeyT> qpKeyList_;
    std::vector<QpImportInfoT> allQpImportInfoT_;
};
} // namespace device
} // namespace transport
} // namespace shm

#endif // MF_HYBRID_DEVICE_JETTY_MANAGER_H
