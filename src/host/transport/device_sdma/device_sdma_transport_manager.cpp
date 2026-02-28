/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "device_sdma_transport_manager.h"

#include <dlfcn.h>
#include <functional>
#include <numeric>
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "init/shmemi_init.h"
#include "shmemi_logger.h"
#include "dl_acl_api.h"
#include "dl_opapi_api.h"
#include "dl_rt_api.h"
#include "shmemi_scope_guard.h"

#ifndef ACL_STREAM_DEVICE_USE_ONLY
#define ACL_STREAM_DEVICE_USE_ONLY 0x00000020U
#endif

namespace shm {
namespace transport {
namespace device {
sdma_op_res_info_t SdmaTransportManager::op_res_info_ = {};
void *SdmaTransportManager::op_res_info_device_ptr_ = nullptr;

Result SdmaTransportManager::OpenDevice(const TransportOptions &options)
{
    mype_ = options.rankId;
    npes_ = options.rankCount;

    SHM_LOG_DEBUG(mype_ << " start to init sdma with " << options);

    dlerror(); // 清除历史错误

    // 创建host测stream
    ACLSHMEM_CHECK_RET(CreateStarsStreams(ACLSHMEM_SDMA_MAX_CHAN));

    // 申请AICPU和AIV的共享内存
    constexpr size_t workspace_size = 16 * 1024; // 16KB
    ACLSHMEM_CHECK_RET(MallocSdmaWorkspace(workspace_size));

    // 申请的资源H2D
    ACLSHMEM_CHECK_RET(CopyHostOpResToDevice());

    // 调用内置aicpu算子
    ACLSHMEM_CHECK_RET(LaunchSdmaAicpuKernel(reinterpret_cast<uint64_t>(op_res_info_device_ptr_),
        op_res_info_.workspace_addr));

    SHM_LOG_DEBUG(mype_ << " init sdma success.");
    inited_ = true;
    return ACLSHMEM_SUCCESS;
}

Result SdmaTransportManager::CreateStarsStreams(int32_t channel_num)
{
    int32_t device_id = -1;
    ACLSHMEM_CHECK_RET(aclrtGetDevice(&device_id));

    int64_t die_id = -1;
    constexpr int info_type_phy_die_id = 19;
    ACLSHMEM_CHECK_RET(DlAclApi::RtGetDeviceInfo(device_id, 0, info_type_phy_die_id, &die_id));
    SHM_LOG_DEBUG(mype_ << " get device_id: " << device_id << ", die_id: " << die_id);

    for (int32_t i = 0; i < channel_num; i++) {
        op_res_info_.streams[i].stream_ = 0;

        void *stream = nullptr;
        ACLSHMEM_CHECK_RET(aclrtCreateStreamWithConfig(&stream, 0, ACL_STREAM_DEVICE_USE_ONLY));
        op_res_info_.streams[i].stream_ = reinterpret_cast<uint64_t>(stream);

        int32_t stream_id = 0;
        int ret = aclrtStreamGetId(stream, &stream_id);
        if (ret != 0) {
            SHM_LOG_ERROR(mype_ << " get stream id " << i << " failed");
            ACLSHMEM_CHECK_RET(aclrtDestroyStream(stream));
            return ACLSHMEM_INNER_ERROR;
        }
        uint32_t sq_id = 0;
        ret = DlRtApi::RtStreamGetSqid(stream, &sq_id);
        if (ret != 0) {
            SHM_LOG_ERROR(mype_ << " get sqid " << i << " failed");
            ACLSHMEM_CHECK_RET(aclrtDestroyStream(stream));
            return ACLSHMEM_INNER_ERROR;
        }
        uint32_t cq_id = 0;
        uint32_t logic_cq_id = 0;
        ret = DlRtApi::RtStreamGetCqid(stream, &cq_id, &logic_cq_id);
        if (ret != 0) {
            SHM_LOG_ERROR(mype_ << " get cqid " << i << " failed");
            ACLSHMEM_CHECK_RET(aclrtDestroyStream(stream));
            return ACLSHMEM_INNER_ERROR;
        }
        void *ctx = nullptr;
        ret = aclrtGetCurrentContext(&ctx);
        if (ret != 0) {
            SHM_LOG_ERROR(mype_ << " get context " << i << " failed");
            ACLSHMEM_CHECK_RET(aclrtDestroyStream(stream));
            return ACLSHMEM_INNER_ERROR;
        }
        op_res_info_.streams[i].ctx_ = reinterpret_cast<uint64_t>(ctx);
        op_res_info_.streams[i].stream_id = stream_id;
        op_res_info_.streams[i].sq_id = sq_id;
        op_res_info_.streams[i].cq_id = cq_id;
        op_res_info_.streams[i].logic_cq_id = logic_cq_id;
        op_res_info_.streams[i].dev_id = die_id;
        SHM_LOG_DEBUG(mype_ << "create stream " << i << "," << op_res_info_.streams[i].stream_ << "," <<
            op_res_info_.streams[i].dev_id << "," << stream_id << "," << sq_id << "," << cq_id << "," <<
            logic_cq_id << "," << op_res_info_.streams[i].ctx_);
    }
    SHM_LOG_INFO(mype_ << " create " << channel_num << " stars streams success.");
    return ACLSHMEM_SUCCESS;
}

Result SdmaTransportManager::MallocSdmaWorkspace(size_t workspace_size)
{
    void *sdma_workspace;
    ACLSHMEM_CHECK_RET(aclrtMalloc(&sdma_workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLSHMEM_CHECK_RET(aclrtMemset(sdma_workspace, workspace_size, 0, workspace_size));
    g_state.sdma_workspace_addr = reinterpret_cast<uint64_t>(sdma_workspace);
    SHM_LOG_INFO(mype_ << " malloc sdma_workspace success.");
    return ACLSHMEM_SUCCESS;
}

Result SdmaTransportManager::CopyHostOpResToDevice()
{
    op_res_info_.workspace_addr = g_state.sdma_workspace_addr;
    size_t size = sizeof(op_res_info_);
    ACLSHMEM_CHECK_RET(aclrtMalloc(&op_res_info_device_ptr_, size, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLSHMEM_CHECK_RET(aclrtMemset(op_res_info_device_ptr_, size, 0, size));
    ACLSHMEM_CHECK_RET(aclrtMemcpy(op_res_info_device_ptr_, size, &op_res_info_, size, ACL_MEMCPY_HOST_TO_DEVICE));
    SHM_LOG_INFO(mype_ << " copy op res to device success.");
    return ACLSHMEM_SUCCESS;
}

Result SdmaTransportManager::CreateAclTensor(const std::vector<uint64_t> &host_data, const std::vector<int64_t> &shape,
                                             void **device_addr, aclDataType data_type, aclTensor **tensor)
{
    *tensor = nullptr;
    *device_addr = nullptr;
    auto size = std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1),
        [](int64_t a, int64_t b) { return a * b; }) * sizeof(uint64_t);
    ACLSHMEM_CHECK_RET(aclrtMalloc(device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST));
    auto dev_guard = utils::make_scope_guard(*device_addr, [](void *addr) { aclrtFree(addr); }); // 托管device_addr
    ACLSHMEM_CHECK_RET(aclrtMemcpy(*device_addr, size, host_data.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), data_type, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), *device_addr);
    if (*tensor == nullptr) {
        SHM_LOG_ERROR(mype_ << " create acl tensor failed");
        return ACLSHMEM_INNER_ERROR;
    }
    auto tensor_guard = utils::make_scope_guard(*tensor, [](aclTensor *t) { aclDestroyTensor(t); });

    // 放弃所有权，交给外部接管
    dev_guard.release();
    tensor_guard.release();
    return ACLSHMEM_SUCCESS;
}

Result SdmaTransportManager::LaunchSdmaAicpuKernel(uint64_t streams_addr, uint64_t sdma_workspace_addr)
{
    void *aicpu_stream = nullptr;
    ACLSHMEM_CHECK_RET(aclrtCreateStreamWithConfig(&aicpu_stream, 0, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC));
    auto stream_guard = utils::make_scope_guard(aicpu_stream, [](void *stm) { aclrtDestroyStream(stm); });
    aclrtStreamAttrValue value;
    value.failureMode = 1; // 遇错即停
    ACLSHMEM_CHECK_RET(aclrtSetStreamAttribute(aicpu_stream, ACL_STREAM_ATTR_FAILURE_MODE, &value));

    // 构造输入输出
    std::vector<int64_t> in_shape = {2}; // 2: 输入个数
    std::vector<int64_t> out_shape = {1};
    std::vector<uint64_t> in_host_data = {streams_addr, sdma_workspace_addr};
    std::vector<uint64_t> out_host_data = {0};

    // 创建输入aclTensor
    void *in_device_addr = nullptr;
    aclTensor *input = nullptr;
    ACLSHMEM_CHECK_RET(CreateAclTensor(in_host_data, in_shape, &in_device_addr, aclDataType::ACL_UINT64, &input));
    auto input_dev_guard = utils::make_scope_guard(in_device_addr, [](void *addr) { aclrtFree(addr); });
    auto input_tensor_guard = utils::make_scope_guard(input, [](aclTensor *t) { aclDestroyTensor(t); });

    // 创建输出aclTensor
    void *out_device_addr = nullptr;
    aclTensor *output = nullptr;
    ACLSHMEM_CHECK_RET(CreateAclTensor(out_host_data, out_shape, &out_device_addr, aclDataType::ACL_UINT64, &output));
    auto output_dev_guard = utils::make_scope_guard(out_device_addr, [](void *addr) { aclrtFree(addr); });
    auto output_tensor_guard = utils::make_scope_guard(output, [](aclTensor *t) { aclDestroyTensor(t); });

    // 调用aclnn二段式接口
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    ACLSHMEM_CHECK_RET(DlOpapiApi::AclnnShmemSdmaStarsQueryGetWorkspaceSize(input, output, &workspace_size, &executor));

    void *workspace = nullptr;
    auto workspace_guard = utils::make_scope_guard(static_cast<void *>(nullptr), [](void *) {});
    if (workspace_size > 0) {
        ACLSHMEM_CHECK_RET(aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
        workspace_guard = utils::make_scope_guard(workspace, [](void *addr) { aclrtFree(addr); });
    }
    ACLSHMEM_CHECK_RET(DlOpapiApi::AclnnShmemSdmaStarsQuery(workspace, workspace_size, executor, aicpu_stream));
    ACLSHMEM_CHECK_RET(aclrtSynchronizeStream(aicpu_stream));
    return ACLSHMEM_SUCCESS;
}

Result SdmaTransportManager::CloseDevice()
{
    if (!inited_) {
        SHM_LOG_WARN(mype_ << " sdma not initialized");
        return ACLSHMEM_SUCCESS;
    }
    if (op_res_info_device_ptr_) {
        (void)aclrtFree(op_res_info_device_ptr_);
        op_res_info_device_ptr_ = nullptr;
    }
    if (op_res_info_.workspace_addr) {
        (void)aclrtFree(reinterpret_cast<void *>(op_res_info_.workspace_addr));
    }
    for (size_t i = 0; i < ACLSHMEM_SDMA_MAX_CHAN; i++) {
        if (op_res_info_.streams[i].stream_) {
            (void)aclrtDestroyStream(reinterpret_cast<void *>(op_res_info_.streams[i].stream_));
        }
    }
    op_res_info_ = {};
    inited_ = false;
    SHM_LOG_INFO(mype_ << " sdma transport finalize.");
    return ACLSHMEM_SUCCESS;
}
} // namespace device
} // namespace transport
} // namespace shm
