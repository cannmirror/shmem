/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v2_mrgsort_out.h
 * \brief
 */
#ifndef INNER_MOE_V2_MRGSORT_OUT_H
#define INNER_MOE_V2_MRGSORT_OUT_H

#include "moe_v2_mrgsort.h"
#include "kernel_operator.h"

namespace MoeInitRoutingQuantV2 {
class MoeV2MrgsortOut {
public:
    __aicore__ inline MoeV2MrgsortOut()
    {};

    __aicore__ inline void Init(MoeV2MrgsortParam *param, AscendC::TPipe *tPipe);

    __aicore__ inline void Process();

    __aicore__ inline void SetInput(AscendC::GlobalTensor<float> &gmInput, AscendC::LocalTensor<float> &ubInput);

    __aicore__ inline void SetOutput(AscendC::GlobalTensor<int32_t> &gmOutput1, AscendC::GlobalTensor<int32_t> &gmOutput2,
                                     AscendC::LocalTensor<float> &ubOutput1, AscendC::LocalTensor<float> &ubOutput2);

    __aicore__ inline void SetBuffer(AscendC::LocalTensor<float> &tempBuffer);

private:
    __aicore__ inline void CopyIn();

    __aicore__ inline void UpdateMrgParam();

    __aicore__ inline void MrgsortCompute();

    __aicore__ inline void UpdateSortInfo();

    __aicore__ inline void Extract();

    __aicore__ inline void CopyOut();

    __aicore__ inline void ClearCache();

    MoeV2MrgsortParam *param = nullptr;

    AscendC::GlobalTensor<float> gmInputs[4];
    AscendC::GlobalTensor<int32_t> gmOutput1;
    AscendC::GlobalTensor<int32_t> gmOutput2;

    AscendC::LocalTensor<float> ubInputs[4];
    AscendC::LocalTensor<float> tempBuffer;

    // for extract
    AscendC::LocalTensor<float> ubOutput1;
    AscendC::LocalTensor<uint32_t> ubOutput2;

    // for copy out
    AscendC::LocalTensor<int32_t> ubOutputInt1;
    AscendC::LocalTensor<int32_t> ubOutputInt2;

    int64_t listNum{0};
    int64_t remainListNum{0};
    int64_t outOffset{0};
    int64_t offsets[4];
    int64_t listRemainElements[4];
    int64_t lengths[4];
    int64_t allRemainElements{0};
    int64_t curLoopSortedNum{0};

    // for MrgSort
    uint16_t validBitTail;
    uint16_t elementCountListTail[4];
    uint32_t listSortedNums[4];
    AscendC::LocalTensor<float> tmpUbInputs[4];
};

__aicore__ inline void MoeV2MrgsortOut::ClearCache()
{
    this->listNum = 0;
    this->allRemainElements = 0;
    this->outOffset = 0;
}

__aicore__ inline void MoeV2MrgsortOut::SetInput(AscendC::GlobalTensor<float> &gmInput, AscendC::LocalTensor<float> &ubInput)
{
    this->gmInputs[listNum] = gmInput;
    this->ubInputs[listNum] = ubInput;
    this->listNum += 1;
}

__aicore__ inline void MoeV2MrgsortOut::SetOutput(AscendC::GlobalTensor<int32_t> &gmOutput1, AscendC::GlobalTensor<int32_t> &gmOutput2,
                                                  AscendC::LocalTensor<float> &ubOutput1, AscendC::LocalTensor<float> &ubOutput2)
{
    this->gmOutput1 = gmOutput1;
    this->ubOutput1 = ubOutput1;
    this->ubOutputInt1 = ubOutput1.ReinterpretCast<int32_t>();

    this->gmOutput2 = gmOutput2;
    this->ubOutput2 = ubOutput2.ReinterpretCast<uint32_t>();
    this->ubOutputInt2 = ubOutput2.ReinterpretCast<int32_t>();
}

__aicore__ inline void MoeV2MrgsortOut::SetBuffer(AscendC::LocalTensor<float> &tempBuffer)
{
    this->tempBuffer = tempBuffer;
}

__aicore__ inline void MoeV2MrgsortOut::UpdateMrgParam()
{
    if (this->remainListNum == MERGE_LIST_TWO) {
        elementCountListTail[MERGE_LIST_IDX_TWO] = 0;
        elementCountListTail[MERGE_LIST_IDX_THREE] = 0;
        validBitTail = 0b0011;
    } else if (this->remainListNum == MERGE_LIST_THREE) {
        elementCountListTail[MERGE_LIST_IDX_THREE] = 0;
        validBitTail = 0b0111;
    } else if (this->remainListNum == MERGE_LIST_FOUR) {
        validBitTail = 0b1111;
    } else {
        validBitTail = 0b0001;
    }
}

__aicore__ inline void MoeV2MrgsortOut::CopyIn()
{
    this->remainListNum = 0;
    SetWaitFlag<AscendC::HardEvent::MTE3_MTE2>(AscendC::HardEvent::MTE3_MTE2);
    for (int64_t i = 0, j = 0; i < listNum; i++) {
        lengths[i] = Min(param->oneLoopMaxElements, listRemainElements[i]);
        if (lengths[i] > 0) {
            AscendC::DataCopy(this->ubInputs[i], this->gmInputs[i][offsets[i]],
                     Align(AscendC::GetSortLen<float>(lengths[i]), sizeof(float)));
            tmpUbInputs[j] = this->ubInputs[i];
            elementCountListTail[j] = lengths[i];
            this->remainListNum += 1;
            j++;
        }
    }
}

__aicore__ inline void MoeV2MrgsortOut::MrgsortCompute()
{
    SetWaitFlag<AscendC::HardEvent::MTE2_V>(AscendC::HardEvent::MTE2_V);
    if (this->remainListNum == MERGE_LIST_TWO) {
        AscendC::MrgSortSrcList sortListTail = AscendC::MrgSortSrcList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[0], tmpUbInputs[0]);
        AscendC::MrgSort<float, true>(this->tempBuffer, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);
    } else if (this->remainListNum == MERGE_LIST_THREE) {
        AscendC::MrgSortSrcList sortListTail =
                AscendC::MrgSortSrcList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[MERGE_LIST_IDX_TWO], tmpUbInputs[0]);
        AscendC::MrgSort<float, true>(this->tempBuffer, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);
    } else if (this->remainListNum == MERGE_LIST_FOUR) {
        AscendC::MrgSortSrcList sortListTail = AscendC::MrgSortSrcList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[MERGE_LIST_IDX_TWO],
                                                     tmpUbInputs[MERGE_LIST_IDX_THREE]);
        AscendC::MrgSort<float, true>(this->tempBuffer, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);
    } else {
        AscendC::DataCopy(this->tempBuffer, this->tmpUbInputs[0],
                 Align(AscendC::GetSortLen<float>(elementCountListTail[0]), sizeof(float)));
        listSortedNums[0] = elementCountListTail[0];
    }
}

__aicore__ inline void MoeV2MrgsortOut::UpdateSortInfo()
{
    curLoopSortedNum = 0;
    for (int64_t i = 0, j = 0; i < listNum; i++) {
        if (lengths[i] > 0) {
            // update remain size
            listRemainElements[i] -= listSortedNums[j];
            allRemainElements -= listSortedNums[j];
            // update offset
            offsets[i] += AscendC::GetSortOffset<float>(listSortedNums[j]);
            // update current loop sorted nums
            curLoopSortedNum += listSortedNums[j];
            j += 1;
        }
    }
}

__aicore__ inline void MoeV2MrgsortOut::Extract()
{
    AscendC::Extract(this->ubOutput1, this->ubOutput2, this->tempBuffer, Ceil(curLoopSortedNum, ONE_REPEAT_SORT_NUM));
    pipe_barrier(PIPE_V);
    AscendC::Muls(this->ubOutput1, this->ubOutput1, (float) -1, Align(curLoopSortedNum, sizeof(float)));
    pipe_barrier(PIPE_V);
    AscendC::Cast(this->ubOutputInt1, this->ubOutput1, AscendC::RoundMode::CAST_ROUND, Align(curLoopSortedNum, sizeof(float)));
}

__aicore__ inline void MoeV2MrgsortOut::CopyOut()
{
    AscendC::DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = curLoopSortedNum * sizeof(int32_t);
    SetWaitFlag<AscendC::HardEvent::V_MTE3>(AscendC::HardEvent::V_MTE3);
    AscendC::DataCopyPad(this->gmOutput1[outOffset], this->ubOutputInt1, intriParams);
    AscendC::DataCopyPad(this->gmOutput2[outOffset], this->ubOutputInt2, intriParams);
    outOffset += curLoopSortedNum;
}

__aicore__ inline void MoeV2MrgsortOut::Init(MoeV2MrgsortParam *param, AscendC::TPipe *tPipe)
{
    this->param = param;
    this->allRemainElements = 0;
    for (int64_t i = 0; i < listNum; i++) {
        offsets[i] = AscendC::GetSortOffset<float>(param->perListElements * i);
        if (i == listNum - 1) {
            listRemainElements[i] = param->lastListElements;
        } else {
            listRemainElements[i] = param->perListElements;
        }
        allRemainElements += listRemainElements[i];
    }
}

__aicore__ inline void MoeV2MrgsortOut::Process()
{
    for (; allRemainElements > 0;) {
        CopyIn();
        UpdateMrgParam();
        MrgsortCompute();
        UpdateSortInfo();
        Extract();
        CopyOut();
    }
    ClearCache();
}
}  // namespace MoeInitRoutingQuantV2
#endif  // INNER_MOE_V2_MRGSORT_OUT_H