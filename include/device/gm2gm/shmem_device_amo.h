/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef SHMEM_DEVICE_ATOMIC_H
#define SHMEM_DEVICE_ATOMIC_H

#include "kernel_operator.h"
#include "device/gm2gm/engine/shmem_device_mte.h"

/**
 * @brief Standard Atomic Add Types and Names
 *
 * |NAME       | TYPE      |
 * |-----------|-----------|
 * |int8       | int8      |
 * |int16      | int16     |
 * |int32      | int32     |
 * |uint32     | uint32    |
 * |int64      | int64     |
 * |uint64     | uint64    |
 * |half       | half      |
 * |bfloat16   | bfloat16  |
 * |float      | float     |
 */
#define ACLSHMEM_TYPE_FUNC_ATOMIC_ADD(FUNC) \
    FUNC(int8, int8_t);                     \
    FUNC(int16, int16_t);                   \
    FUNC(int32, int32_t);                   \
    FUNC(uint32, uint32_t);                 \
    FUNC(int64, int64_t);                   \
    FUNC(uint64, uint64_t);                 \
    FUNC(half, half);                       \
    FUNC(bfloat16, bfloat16_t);             \
    FUNC(float, float)

/**
 * @brief  Automatically generates aclshmem atomic add functions for different data types
 *         (MTE supports int8, int16, int32, float, half, bfloat16; UDMA supports int32, uint32, int64, uint64, float).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_atomic_add(\_\_gm\_\_ TYPE *dst, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Asynchronous interface. Perform contiguous data atomic add operation on
 * symmetric memory from the specified PE to address on the local PE.
 *
 * @par Parameters
 * - **dst**    - [in] Pointer on local device of the destination data.
 * - **value**  - [in] Value atomic add to destination.
 * - **pe**     - [in] PE number of the remote PE.
 */
#define ACLSHMEM_ATOMIC_ADD_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_add(__gm__ TYPE *dst, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC_ADD(ACLSHMEM_ATOMIC_ADD_TYPENAME);
/** \endcond */
#define shmem_int8_atomic_add aclshmem_int8_atomic_add
#define shmem_int16_atomic_add aclshmem_int16_atomic_add
#define shmem_int32_atomic_add aclshmem_int32_atomic_add
#define shmem_half_atomic_add aclshmem_half_atomic_add
#define shmem_bfloat16_atomic_add aclshmem_bfloat16_atomic_add
#define shmem_float_atomic_add aclshmem_float_atomic_add

/**
 * @brief Standard Atomic Fetch Types and Names (supported types for fetch_add and compare swap)
 *
 * |NAME       | TYPE      |
 * |-----------|-----------|
 * |int32      | int32     |
 * |uint32     | uint32    |
 * |int64      | int64     |
 * |uint64     | uint64    |
 */
#define ACLSHMEM_TYPE_FUNC_ATOMIC_FETCH(FUNC) \
    FUNC(int32, int32_t);                     \
    FUNC(uint32, uint32_t);                   \
    FUNC(int64, int64_t);                     \
    FUNC(uint64, uint64_t)

/**
 * @brief  Automatically generates aclshmem atomic fetch add functions for different data types
 *         (e.g., int32, uint32, int64, uint64).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE TYPE aclshmem_NAME_atomic_fetch_add(\_\_gm\_\_ TYPE *dst, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Add value to dst (remote symmetric address) on the specified PE pe,
 * and return the previous content of dst.
 *
 * @par Parameters
 * - **dst**    - [in] Pointer on local device of the destination data.
 * - **value**  - [in] Value atomic add to destination.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @par Returns
 *      Return the previous content of dst.
 */
#define ACLSHMEM_ATOMIC_FETCH_ADD_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_fetch_add(__gm__ TYPE *dst, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC_FETCH(ACLSHMEM_ATOMIC_FETCH_ADD_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic compare swap functions for different data types
 *        (e.g., int32, uint32, int64, uint64).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE TYPE aclshmem_NAME_atomic_compare_swap(\_\_gm\_\_ TYPE *dst, TYPE cond, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Conditionally update dst (remote symmetric address) on the specified PE pe
 * and return the previous content of dst. If cond and the remote dst value are equal,
 * then value is swapped into the remote dst; otherwise, the remote dst is unchanged. In either case, the old
 * value of the remote dest is returned.
 *
 * @par Parameters
 * - **dst**    - [in] Pointer on local device of the destination data.
 * - **cond**   - [in] Condition compared to the remote dst value.
 * - **value**  - [in] Value atomic swap to destination.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @par Returns
 *      Return the previous content of dst.
 */
#define ACLSHMEM_ATOMIC_COMPARE_SWAP_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_compare_swap(__gm__ TYPE *dst, TYPE cond, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC_FETCH(ACLSHMEM_ATOMIC_COMPARE_SWAP_TYPENAME);
/** \endcond */

#include "gm2gm/shmem_device_amo.hpp"
#endif