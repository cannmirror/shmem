/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PCIE_NIC_MATCHER_H
#define PCIE_NIC_MATCHER_H

#include <cstddef>
#include <cstring>

namespace shm {
namespace topo {

inline size_t get_common_prefix_len(const char* left, const char* right)
{
    if (left == nullptr || right == nullptr) {
        return 0;
    }
    size_t len = 0;
    while (left[len] == right[len] && left[len] != '\0' && right[len] != '\0') {
        len++;
    }
    return len;
}

inline const char* get_path_basename(const char* path)
{
    if (path == nullptr) {
        return "";
    }
    const char* base = strrchr(path, '/');
    return (base == nullptr) ? path : base + 1;
}

inline int select_closest_nic_path_index(const char* npu_pcie_path, const char* const* nic_pcie_paths, size_t nic_len,
                                         size_t min_prefix_len)
{
    int max_pos = -1;
    size_t max_len = min_prefix_len;
    for (size_t i = 0; i < nic_len; ++i) {
        if (nic_pcie_paths == nullptr || nic_pcie_paths[i] == nullptr || nic_pcie_paths[i][0] == '\0') {
            continue;
        }
        size_t common_prefix_len = get_common_prefix_len(npu_pcie_path, nic_pcie_paths[i]);
        if (common_prefix_len > max_len) {
            max_pos = static_cast<int>(i);
            max_len = common_prefix_len;
        }
    }
    return max_pos;
}

} // namespace topo
} // namespace shm

#endif // PCIE_NIC_MATCHER_H
