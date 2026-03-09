/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <random>
#include <fstream>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include "shmemi_init_backend.h"
#include "shmemi_host_common.h"
#include "host/shmem_host_def.h"
#include "utils/shmemi_logger.h"
#include "store_factory.h"
#include "mem_entity_entry.h"
#include "sotre_net.h"
#include "store_net_common.h"
#include "mem_entity_def.h"

constexpr int DEFAULT_ID = 0;

aclshmemi_init_backend::aclshmemi_init_backend(aclshmemx_init_attr_t *attr, aclshmem_device_host_state_t *global_state, aclshmemi_bootstrap_handle_t *handle)
{
    auto status = aclrtGetDevice(&device_id);
    if (status != 0) {
        SHM_LOG_ERROR("Get Device_id error");
    }
    
    attributes = attr;
    boot_handle_ = handle;
    npes = attr->n_pes;
    host_state_ = global_state;
    mstx_reg_ptr_ = shm::create_mstx_mem_register_instance();
    if (mstx_reg_ptr_ == nullptr) {
        SHM_LOG_ERROR("create mstx_reg_ptr_ returned nullptr!");
    }
}

aclshmemi_init_backend::~aclshmemi_init_backend()
{
    if (mstx_reg_ptr_ != nullptr) {
        delete mstx_reg_ptr_;
        mstx_reg_ptr_ = nullptr;
    }
    host_state_->is_aclshmem_created = false;
}

int aclshmemi_init_backend::init_device_state()
{
    int32_t status = ACLSHMEM_SUCCESS;
    // 初始化hybm，分配meta存储空间
    auto ret = hybm_init(device_id, 0);
    if (ret != 0) {
        SHM_LOG_ERROR("init hybm failed, result: " << ret);
        return ACLSHMEM_SMEM_ERROR;
    }
    inited_ = true;
    mstx_reg_ptr_->add_mem_regions((void *)shm::HYBM_DEVICE_META_ADDR, shm::HYBM_DEVICE_INFO_SIZE, MSTX_GROUP_FOR_STATE);
    mstx_reg_ptr_->mstx_mem_regions_register(MSTX_GROUP_FOR_STATE);
    ret = aclshmemi_control_barrier_all();  // 保证所有rank都初始化了
    if (ret != 0) {
        SHM_LOG_ERROR("aclshmemi_control_barrier_allfailed, result: " << ret);
        return ACLSHMEM_SMEM_ERROR;
    }

    SHM_LOG_INFO("init_device_state success. world_size: " << attributes->n_pes);
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::create_entity(aclshmem_mem_type_t mem_type)
{
    // 创建entity
    hybm_options options;
    options.bmType = HYBM_TYPE_AI_CORE_INITIATE;
    options.bmDataOpType = static_cast<hybm_data_op_type>(HYBM_DOP_TYPE_MTE);
    if (attributes->option_attr.data_op_engine_type & ACLSHMEM_DATA_OP_ROCE) {
        auto temp = static_cast<uint32_t>(options.bmDataOpType) | HYBM_DOP_TYPE_DEVICE_RDMA;
        options.bmDataOpType = static_cast<hybm_data_op_type>(temp);
    }
    if (attributes->option_attr.data_op_engine_type & ACLSHMEM_DATA_OP_SDMA) {
        auto temp = static_cast<uint32_t>(options.bmDataOpType) | HYBM_DOP_TYPE_DEVICE_SDMA;
        options.bmDataOpType = static_cast<hybm_data_op_type>(temp);
    }
    options.bmScope = HYBM_SCOPE_CROSS_NODE;
    options.rankCount = attributes->n_pes;
    options.rankId = attributes->my_pe;
    if (mem_type == HOST_SIDE) {
        options.memType = HYBM_MEM_TYPE_HOST;
        options.deviceVASpace = 0;
        options.hostVASpace = host_state_->heap_size;
    } else {
        options.memType = HYBM_MEM_TYPE_DEVICE;
        options.deviceVASpace = host_state_->heap_size;
        options.hostVASpace = 0;
    }
    options.preferredGVA = 0;
    options.role = HYBM_ROLE_PEER;
    std::string defaultNic = "10002";
    std::copy_n(defaultNic.c_str(), defaultNic.size() + 1, options.nic);
    auto entity_id = DEFAULT_ID << 1;
    if (mem_type == HOST_SIDE) {
        dram_entity_ = hybm_create_entity(entity_id + 1, &options, 0);
        if (dram_entity_ == nullptr) {
            SHM_LOG_ERROR("create host dram entity failed");
            return ACLSHMEM_SMEM_ERROR;
        }
        return ACLSHMEM_SUCCESS;
    }
    device_state_ = (aclshmem_device_host_state_t *)std::calloc(1, sizeof(aclshmem_device_host_state_t));
    hbm_entity_ = hybm_create_entity(entity_id, &options, 0);
    if (hbm_entity_ == nullptr) {
        SHM_LOG_ERROR("create device hbm entity failed");
        return ACLSHMEM_SMEM_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::update_device_state(void* host_ptr, size_t size)
{
    int32_t ptr_size = npes * sizeof(void *);
    ACLSHMEM_CHECK_RET(aclrtMemcpy(device_state_->p2p_device_heap_base, ptr_size, host_state_->p2p_device_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLSHMEM_CHECK_RET(aclrtMemcpy(device_state_->rdma_device_heap_base, ptr_size, host_state_->rdma_device_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLSHMEM_CHECK_RET(aclrtMemcpy(device_state_->sdma_device_heap_base, ptr_size, host_state_->sdma_device_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));
    if (host_state_->host_heap_base != nullptr) {
        ACLSHMEM_CHECK_RET(aclrtMemcpy(device_state_->p2p_host_heap_base, ptr_size,
            host_state_->p2p_host_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));
        ACLSHMEM_CHECK_RET(aclrtMemcpy(device_state_->rdma_host_heap_base, ptr_size,
            host_state_->rdma_host_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));
        ACLSHMEM_CHECK_RET(aclrtMemcpy(device_state_->sdma_host_heap_base, ptr_size,
            host_state_->sdma_host_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    auto backup_p2p = device_state_->p2p_device_heap_base;
    auto backup_rdma = device_state_->rdma_device_heap_base;
    auto backup_sdma = device_state_->sdma_device_heap_base;
    auto backup_host_p2p = device_state_->p2p_host_heap_base;
    auto backup_host_rdma = device_state_->rdma_host_heap_base;
    auto backup_host_sdma = device_state_->sdma_host_heap_base;
    std::memcpy(device_state_, host_ptr, size);
    device_state_->p2p_device_heap_base = backup_p2p;
    device_state_->rdma_device_heap_base = backup_rdma;
    device_state_->sdma_device_heap_base = backup_sdma;
    device_state_->p2p_host_heap_base = backup_host_p2p;
    device_state_->rdma_host_heap_base = backup_host_rdma;
    device_state_->sdma_host_heap_base = backup_host_sdma;

    auto ret = hybm_set_extra_context(hbm_entity_, device_state_, size);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("hybm_set_extra_context failed, ret: " << ret);
        return ret;
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::finalize_device_state()
{
    if (!inited_) {
        SHM_LOG_INFO("shm hybm backend not initialized yet, skip finalize device state.");
        return ACLSHMEM_SUCCESS;
    }
    if (device_state_ != nullptr) {
        std::free(device_state_);
        device_state_ = nullptr;
    }
    hybm_uninit();
    inited_ = false;
    boot_handle_ = nullptr;
    SHM_LOG_INFO("shmemi uninit finished");
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::reserve_heap(aclshmem_mem_type_t mem_type)
{
    auto ret = create_entity(mem_type);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("create entity failed, result: " << ret << ", mem_type: " << mem_type << ".");
        return ret;
    }
    auto entity = mem_type == HOST_SIDE ? dram_entity_ : hbm_entity_;
    void *gva = nullptr;
    ret = hybm_reserve_mem_space(entity, 0, &gva);
    auto aligned = ALIGN_UP(host_state_->heap_size, ACLSHMEM_HEAP_ALIGNMENT_SIZE);
    if (ret != 0 || gva == nullptr) {
        SHM_LOG_ERROR("reserve mem failed, result: " << ret << ", mem_type: " << mem_type << ".");
        return ACLSHMEM_SMEM_ERROR;
    }
    if (mem_type == HOST_SIDE) {
        dram_gva_ = gva;
        mstx_reg_ptr_->add_mem_regions_multi_pe_align(dram_gva_, host_state_->heap_size, aligned, host_state_->npes, MSTX_GROUP_FOR_HOST_HEAP);
        mstx_reg_ptr_->mstx_mem_regions_register(MSTX_GROUP_FOR_HOST_HEAP);
    } else {
        hbm_gva_ = gva;
        mstx_reg_ptr_->add_mem_regions_multi_pe_align(hbm_gva_, host_state_->heap_size, aligned, host_state_->npes, MSTX_GROUP_FOR_DEVICE_HEAP);
        mstx_reg_ptr_->mstx_mem_regions_register(MSTX_GROUP_FOR_DEVICE_HEAP);
    }
    SHM_LOG_INFO("reserve_heap success, mem_type: " << mem_type << ".");
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::reach_info_init(void *&gva)
{
    auto aligned = ALIGN_UP(host_state_->heap_size, ACLSHMEM_HEAP_ALIGNMENT_SIZE);
    for (int32_t i = 0; i < host_state_->npes; i++) {
        hybm_data_op_type reaches_types;
        auto ret = hybm_entity_reach_types(hbm_entity_, i, reaches_types, 0);
        if (ret != 0) {
            SHM_LOG_ERROR("hybm_entity_reach_types failed: " << ret);
            return ACLSHMEM_SMEM_ERROR;
        }
        host_state_->p2p_device_heap_base[i] = (void *)((uintptr_t)gva + aligned * static_cast<uint32_t>(i));
        SHM_LOG_DEBUG("rank " << i << " p2p_heap_base: " << host_state_->p2p_device_heap_base[i]);
        if (reaches_types & HYBM_DOP_TYPE_MTE) {
            host_state_->topo_list[i] |= ACLSHMEM_TRANSPORT_MTE;
        }
        if (reaches_types & HYBM_DOP_TYPE_DEVICE_RDMA) {
            host_state_->topo_list[i] |= ACLSHMEM_TRANSPORT_ROCE;
        }
        if (reaches_types & HYBM_DOP_TYPE_DEVICE_SDMA) {
            host_state_->topo_list[i] |= ACLSHMEM_TRANSPORT_SDMA;
        }
        host_state_->sdma_device_heap_base[i] = nullptr;
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::exchange_slice(aclshmem_mem_type_t mem_type)
{
    // export memory
    hybm_exchange_info ex_info;
    bzero(&ex_info, sizeof(ex_info));
    auto entity = mem_type == HOST_SIDE ? dram_entity_ : hbm_entity_;
    auto slice = mem_type == HOST_SIDE ? dram_slice_ : hbm_slice_;
    auto ret = hybm_export(entity, slice, 0, &ex_info);
    if (ret != 0) {
        SHM_LOG_ERROR("hybm export slice failed, result: " << ret);
        return ret;
    }

    hybm_exchange_info all_ex_info[host_state_->npes];
    ret = boot_handle_->allgather((void *)&ex_info, (void *)all_ex_info, sizeof(hybm_exchange_info), boot_handle_);
    if (ret != 0) {
        SHM_LOG_ERROR("hybm gather export slice failed, result: " << ret);
        return ret;
    }

    // import memory
    ret = hybm_import(entity, all_ex_info, host_state_->npes, nullptr, 0);
    if (ret != 0) {
        SHM_LOG_ERROR("hybm import failed, result: " << ret);
        return ret;
    }

    ret = aclshmemi_control_barrier_all();
    if (ret != 0) {
        SHM_LOG_ERROR("hybm barrier for slice failed, result: " << ret);
        return ret;
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::exchange_entity(aclshmem_mem_type_t mem_type)
{
    // export entity
    auto entity = mem_type == HOST_SIDE ? dram_entity_ : hbm_entity_;
    hybm_exchange_info ex_info;
    bzero(&ex_info, sizeof(ex_info));
    auto ret = hybm_export(entity, nullptr, 0, &ex_info);
    if (ret != 0) {
        SHM_LOG_ERROR("hybm export entity failed, result: " << ret);
        return ret;
    }

    if (ex_info.descLen == 0) {
        return ACLSHMEM_SUCCESS;
    }

    hybm_exchange_info all_ex_info[host_state_->npes];
    ret = boot_handle_->allgather((void *)&ex_info, (void *)all_ex_info, sizeof(hybm_exchange_info), boot_handle_);
    if (ret != 0) {
        SHM_LOG_ERROR("hybm gather export entity failed, result: " << ret);
        return ret;
    }
    // import entity
    ret = hybm_import(entity, all_ex_info, host_state_->npes, nullptr, 0);
    if (ret != 0) {
        SHM_LOG_ERROR("hybm import entity failed, result: " << ret);
        return ret;
    }

    ret = aclshmemi_control_barrier_all();
    if (ret != 0) {
        SHM_LOG_ERROR("hybm barrier for slice failed, result: " << ret);
        return ret;
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::setup_heap(aclshmem_mem_type_t mem_type)
{
    auto entity = mem_type == HOST_SIDE ? dram_entity_ : hbm_entity_;
    auto mType = mem_type == HOST_SIDE ? HYBM_MEM_TYPE_HOST : HYBM_MEM_TYPE_DEVICE;
    // alloc memory
    auto slice = hybm_alloc_local_memory(entity, mType, host_state_->heap_size, 0);
    if (slice == nullptr) {
        SHM_LOG_ERROR("alloc local mem failed, size: " << host_state_->heap_size);
        return ACLSHMEM_SMEM_ERROR;
    }
    if (mem_type == HOST_SIDE) {
        dram_slice_ = slice;
    } else {
        hbm_slice_= slice;
    }
    auto ret = exchange_slice(mem_type);
    if (ret != 0) {
        SHM_LOG_ERROR("exchange slice failed, result: " << ret);
        return ret;
    }
    ret = exchange_entity(mem_type);
    if (ret != 0) {
        SHM_LOG_ERROR("exchange entity failed, result: " << ret);
        return ret;
    }

    // mmap memory
    ret = hybm_mmap(entity, 0);
    if (ret != 0) {
        SHM_LOG_ERROR("hybm mmap failed, result: " << ret);
        return ret;
    }
    if (mem_type == HOST_SIDE) {
        auto aligned = ALIGN_UP(host_state_->heap_size, ACLSHMEM_HEAP_ALIGNMENT_SIZE);
        host_state_->host_heap_base = (void *)((uintptr_t)dram_gva_ + aligned * static_cast<uint32_t>(attributes->my_pe));
        ACLSHMEM_CHECK_RET(aclrtMallocHost((void **)&host_state_->p2p_host_heap_base, host_state_->npes * sizeof(void *)));
        ACLSHMEM_CHECK_RET(aclrtMallocHost((void **)&host_state_->rdma_host_heap_base, host_state_->npes * sizeof(void *)));
        ACLSHMEM_CHECK_RET(aclrtMallocHost((void **)&host_state_->sdma_host_heap_base, host_state_->npes * sizeof(void *)));

        ACLSHMEM_CHECK_RET(aclrtMalloc((void **)&device_state_->p2p_host_heap_base, host_state_->npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
        ACLSHMEM_CHECK_RET(aclrtMalloc((void **)&device_state_->rdma_host_heap_base, host_state_->npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
        ACLSHMEM_CHECK_RET(aclrtMalloc((void **)&device_state_->sdma_host_heap_base, host_state_->npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
        for (int32_t i = 0; i < host_state_->npes; i++) {
            host_state_->p2p_host_heap_base[i] = (void *)((uintptr_t)dram_gva_ + aligned * static_cast<uint32_t>(i));
        }
        SHM_LOG_DEBUG("host side dram heap setup successed, host_state_->host_heap_base: " << (void *)(uintptr_t)host_state_->host_heap_base);
        return ACLSHMEM_SUCCESS;
    }
    // device side heap info save
    auto aligned = ALIGN_UP(host_state_->heap_size, ACLSHMEM_HEAP_ALIGNMENT_SIZE);
    host_state_->heap_base = (void *)((uintptr_t)hbm_gva_ + aligned * static_cast<uint32_t>(attributes->my_pe));
    ACLSHMEM_CHECK_RET(aclrtMallocHost((void **)&host_state_->p2p_device_heap_base, host_state_->npes * sizeof(void *)));
    ACLSHMEM_CHECK_RET(aclrtMallocHost((void **)&host_state_->rdma_device_heap_base, host_state_->npes * sizeof(void *)));
    ACLSHMEM_CHECK_RET(aclrtMallocHost((void **)&host_state_->sdma_device_heap_base, host_state_->npes * sizeof(void *)));

    ACLSHMEM_CHECK_RET(aclrtMalloc((void **)&device_state_->p2p_device_heap_base, host_state_->npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
    ACLSHMEM_CHECK_RET(aclrtMalloc((void **)&device_state_->rdma_device_heap_base, host_state_->npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
    ACLSHMEM_CHECK_RET(aclrtMalloc((void **)&device_state_->sdma_device_heap_base, host_state_->npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
    SHM_LOG_DEBUG("device side hbm heap setup successed, host_state_->heap_base: " << (void *)(uintptr_t)host_state_->heap_base << " p2p_device_heap_base: " << (void *)(uintptr_t)device_state_->p2p_device_heap_base);
    ret = reach_info_init(hbm_gva_);
    if (ret != 0) {
        SHM_LOG_ERROR("reach_info_init failed, result: " << ret);
        return ret;
    }

    host_state_->is_aclshmem_created = true;
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::remove_heap(aclshmem_mem_type_t mem_type)
{
    auto entity = mem_type == HOST_SIDE ? dram_entity_ : hbm_entity_;
    if (entity == nullptr) {
        SHM_LOG_DEBUG("entity not create, skip unmap.");
        return ACLSHMEM_SUCCESS;
    }
    hybm_unmap(entity, 0);
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::release_heap(aclshmem_mem_type_t mem_type)
{
    if (host_state_->p2p_device_heap_base != nullptr) {
        ACLSHMEM_CHECK_RET(aclrtFreeHost(host_state_->p2p_device_heap_base));
        ACLSHMEM_CHECK_RET(aclrtFree(device_state_->p2p_device_heap_base));
        host_state_->p2p_device_heap_base = nullptr;
    }
    if (host_state_->rdma_device_heap_base != nullptr) {
        ACLSHMEM_CHECK_RET(aclrtFreeHost(host_state_->rdma_device_heap_base));
        ACLSHMEM_CHECK_RET(aclrtFree(device_state_->rdma_device_heap_base));
        host_state_->rdma_device_heap_base = nullptr;
    }
    if (host_state_->sdma_device_heap_base != nullptr) {
        ACLSHMEM_CHECK_RET(aclrtFreeHost(host_state_->sdma_device_heap_base));
        ACLSHMEM_CHECK_RET(aclrtFree(device_state_->sdma_device_heap_base));
        host_state_->sdma_device_heap_base = nullptr;
    }
    if (host_state_->p2p_host_heap_base != nullptr) {
        ACLSHMEM_CHECK_RET(aclrtFreeHost(host_state_->p2p_host_heap_base));
        ACLSHMEM_CHECK_RET(aclrtFree(device_state_->p2p_host_heap_base));
        host_state_->p2p_host_heap_base = nullptr;
    }
    if (host_state_->rdma_host_heap_base != nullptr) {
        ACLSHMEM_CHECK_RET(aclrtFreeHost(host_state_->rdma_host_heap_base));
        ACLSHMEM_CHECK_RET(aclrtFree(device_state_->rdma_host_heap_base));
        host_state_->rdma_host_heap_base = nullptr;
    }
    if (host_state_->sdma_host_heap_base != nullptr) {
        ACLSHMEM_CHECK_RET(aclrtFreeHost(host_state_->sdma_host_heap_base));
        ACLSHMEM_CHECK_RET(aclrtFree(device_state_->sdma_host_heap_base));
        host_state_->sdma_host_heap_base = nullptr;
    }

    auto entity = mem_type == HOST_SIDE ? dram_entity_ : hbm_entity_;
    auto reserved_mem = mem_type == HOST_SIDE ? dram_gva_ : hbm_gva_;
    if (entity == nullptr) {
        SHM_LOG_DEBUG("entity not create, skip release.");
        return ACLSHMEM_SUCCESS;
    }
    auto ret = hybm_unreserve_mem_space(entity, 0, reserved_mem);
    if (ret != 0) {
        SHM_LOG_INFO("unreserve mem space failed: " << ret);
    }
    hybm_destroy_entity(entity, 0);
    if (mem_type == HOST_SIDE) {
        dram_slice_ = nullptr;
        dram_gva_ = nullptr;
        dram_entity_ = nullptr;
    }
    else {
        hbm_slice_ = nullptr;
        hbm_gva_ = nullptr;
        hbm_entity_ = nullptr;
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::aclshmemi_control_barrier_all()
{
    return boot_handle_->barrier(boot_handle_);
}

int aclshmemi_init_backend::is_alloc_size_symmetric(size_t size)
{
    std::vector<size_t> all_size(host_state_->npes, 0);

    int ret = 0;
    ret = boot_handle_->allgather(&size, all_size.data(), static_cast<int>(sizeof(size_t)), boot_handle_);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("bootstrap allgather failed, ret: " <<ret);
        return ret;
    }

    for (int i = 0; i < host_state_->npes; ++i) {
        SHM_LOG_INFO("alloc_size[pe " << i << "]=" << all_size[i]);
    }

    size_t ref = all_size[0];

    auto it = std::find_if(all_size.begin() + 1, all_size.end(), [ref](size_t alloc_size) {
        return alloc_size != ref;
    });

    if (it != all_size.end()) {
        int bad_pe = std::distance(all_size.begin(), it);
        size_t bad_val = *it;

        SHM_LOG_ERROR("Asymmetric alloc size detected, ref(pe0) = " << ref << ", first detected bad(pe" << bad_pe << ")= "  
                                        << bad_val << ", cur(pe" << host_state_->mype << ")= " << all_size[host_state_->mype]);
        return ACLSHMEM_INVALID_PARAM;
    }
    return ACLSHMEM_SUCCESS;
}