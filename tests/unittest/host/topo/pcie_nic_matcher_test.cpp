/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <iterator>

#include "pcie_nic_matcher.h"

namespace shm {
namespace topo {

class PcieNicMatcherTest : public testing::Test {};

TEST_F(PcieNicMatcherTest, SelectClosestNicAllowsSharedNic)
{
    constexpr size_t min_prefix_len = 23;
    const char* nic_pcie_paths[] = {
        "/sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0/net/enp1s0f0",
        "/sys/devices/pci0000:00/0000:00:02.0/0000:02:00.0/net/enp2s0f0",
        "/sys/devices/pci0000:00/0000:00:03.0/0000:03:00.0/net/enp3s0f0",
        "/sys/devices/pci0000:00/0000:00:04.0/0000:04:00.0/net/enp4s0f0",
    };
    const char* first_npu_path = "/sys/devices/pci0000:00/0000:00:01.0/0000:01:00.1";
    const char* second_npu_path = "/sys/devices/pci0000:00/0000:00:01.0/0000:01:00.2";

    EXPECT_EQ(select_closest_nic_path_index(first_npu_path, nic_pcie_paths, std::size(nic_pcie_paths), min_prefix_len),
              0);
    EXPECT_EQ(select_closest_nic_path_index(second_npu_path, nic_pcie_paths, std::size(nic_pcie_paths), min_prefix_len),
              0);
}

TEST_F(PcieNicMatcherTest, SelectClosestNicUsesLongestPrefix)
{
    constexpr size_t min_prefix_len = 23;
    const char* nic_pcie_paths[] = {
        "/sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0/net/enp1s0f0",
        "/sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0/0000:05:00.0/net/enp5s0f0",
    };
    const char* npu_path = "/sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0/0000:05:00.1";

    EXPECT_EQ(select_closest_nic_path_index(npu_path, nic_pcie_paths, std::size(nic_pcie_paths), min_prefix_len), 1);
}

TEST_F(PcieNicMatcherTest, SelectClosestNicIgnoresInvalidPath)
{
    constexpr size_t min_prefix_len = 23;
    const char* nic_pcie_paths[] = {
        nullptr,
        "",
        "/sys/devices/pci0000:00/0000:00:09.0/0000:09:00.0/net/enp9s0f0",
    };
    const char* npu_path = "/sys/devices/pci0000:00/0000:00:01.0/0000:01:00.1";

    EXPECT_EQ(select_closest_nic_path_index(npu_path, nic_pcie_paths, std::size(nic_pcie_paths), min_prefix_len), -1);
}

TEST_F(PcieNicMatcherTest, GetPathBasenameHandlesConstPath)
{
    EXPECT_STREQ(get_path_basename("/sys/class/net/enp1s0f0"), "enp1s0f0");
    EXPECT_STREQ(get_path_basename("enp1s0f0"), "enp1s0f0");
    EXPECT_STREQ(get_path_basename(nullptr), "");
}

} // namespace topo
} // namespace shm
