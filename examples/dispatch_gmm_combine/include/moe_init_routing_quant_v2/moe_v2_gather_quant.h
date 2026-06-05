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
 * \file moe_v2_gather_quant.h
 * \brief
 */
#ifndef MOE_V2_GATHER_QUANT_H
#define MOE_V2_GATHER_QUANT_H

#include "moe_v2_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingQuantV2 {
constexpr int64_t BUFFER_NUM = 2;

template<typename T>
class MoeV2GatherQuant {
public:
    __aicore__ inline MoeV2GatherQuant()
    {};

    __aicore__ inline void
    Init(GM_ADDR inputX, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
         GM_ADDR workspace, const optiling::MoeInitRoutingQuantV2TilingData *tilingData, AscendC::TPipe *tPipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInIndices(int64_t progress);

    __aicore__ inline void Compute();

    __aicore__ inline void CopyOut(int64_t progress);

    AscendC::TPipe *pipe;
    AscendC::TQue <AscendC::QuePosition::VECIN, BUFFER_NUM> inputXCopyInQueue;
    AscendC::TQue <AscendC::QuePosition::VECIN, BUFFER_NUM> expandRowIdxCopyInQueue;
    AscendC::TQue <AscendC::QuePosition::VECOUT, BUFFER_NUM> inputXCopyOutQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> floatQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> halfQueue;

    AscendC::GlobalTensor<T> inputXGm;
    AscendC::GlobalTensor<int8_t> expandedXGm;
    AscendC::GlobalTensor<int32_t> expandedRowIdxGm;
    AscendC::GlobalTensor<float> scaleGm;
    AscendC::GlobalTensor<float> offsetGm;

    const optiling::InnerMoeV2GatherOutComputeTilingData *gatherOutTilingData;

    int64_t needCoreNum;
    int64_t blockIdx;
    int64_t cols;
    int64_t n;
    int64_t k;
    int64_t activateRows;
    int64_t currentLoopRows;
    int64_t coreRows;
    int64_t perLoopRows;
    int64_t lastLoopRows;
    int64_t rowLoops;
    int64_t colsTileLength;
    int64_t perLoopCols;
    int64_t lastLoopCols;
    int64_t colLoops;
    int64_t dropPadMode;
    float scale;
    float offset;

    int64_t indicesOffset;
    int64_t inputOffset;
    int64_t outOffset;
};

template<typename T>
__aicore__ inline void MoeV2GatherQuant<T>::CopyInIndices(int64_t progress) {
    this->indicesOffset = progress * this->perLoopRows;
    AscendC::LocalTensor<int32_t> indicesLocal = expandRowIdxCopyInQueue.AllocTensor<int32_t>();
   AscendC::DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(this->currentLoopRows * sizeof(int32_t)), 0, 0, 0};
   AscendC::DataCopyPadExtParams <int32_t> dataCopyPadParams{false, 0, 0, 0};
    AscendC::DataCopyPad(indicesLocal, expandedRowIdxGm[indicesOffset], dataCopyParams, dataCopyPadParams);
    expandRowIdxCopyInQueue.
    EnQue<int32_t>(indicesLocal);
}

template<typename T>
__aicore__ inline void MoeV2GatherQuant<T>::Compute()
{
    AscendC::LocalTensor<T> inLocal = inputXCopyInQueue.DeQue<T>();
    AscendC::LocalTensor<int8_t> outLocal = inputXCopyOutQueue.AllocTensor<int8_t>();
    AscendC::LocalTensor<float> floatLocal = floatQueue.AllocTensor<float>();
    AscendC::LocalTensor<half> halfLocal = halfQueue.AllocTensor<half>();
    uint32_t elements = Align(this->colsTileLength, sizeof(T));
    if constexpr(AscendC::IsSameType<T, bfloat16_t>::value) {
        AscendC::Cast(floatLocal, inLocal, AscendC::RoundMode::CAST_NONE, elements);
        pipe_barrier(PIPE_V);
        AscendC::Cast(halfLocal, floatLocal, AscendC::RoundMode::CAST_NONE, elements);
        pipe_barrier(PIPE_V);
        AscendC::Muls(halfLocal, halfLocal, static_cast<half>(this->scale), elements);
        pipe_barrier(PIPE_V);
        AscendC::Adds(halfLocal, halfLocal, static_cast<half>(this->offset), elements);
        pipe_barrier(PIPE_V);
        AscendC::LocalTensor<int32_t> intLocal = floatLocal.ReinterpretCast<int32_t>();
        AscendC::Cast(intLocal, halfLocal, AscendC::RoundMode::CAST_RINT, elements);
        pipe_barrier(PIPE_V);
        AscendC::SetDeqScale((half) 1.000000e+00f);
        pipe_barrier(PIPE_V);
        AscendC::Cast(halfLocal, intLocal, AscendC::RoundMode::CAST_RINT, elements);
        pipe_barrier(PIPE_V);
        AscendC::Cast(outLocal, halfLocal, AscendC::RoundMode::CAST_RINT, elements);
    } else if constexpr(AscendC::IsSameType<T, float>::value) {
        AscendC::Cast(halfLocal, inLocal, AscendC::RoundMode::CAST_NONE, elements);
        pipe_barrier(PIPE_V);
        AscendC::Muls(halfLocal, halfLocal, static_cast<half>(this->scale), elements);
        pipe_barrier(PIPE_V);
        AscendC::Adds(halfLocal, halfLocal, static_cast<half>(this->offset), elements);
        pipe_barrier(PIPE_V);
        AscendC::Cast(outLocal, halfLocal, AscendC::RoundMode::CAST_RINT, elements);
    } else {
        AscendC::Muls(inLocal, inLocal, static_cast<T>(this->scale), elements);
        pipe_barrier(PIPE_V);
        AscendC::Adds(inLocal, inLocal, static_cast<T>(this->offset), elements);
        pipe_barrier(PIPE_V);
        AscendC::Cast(outLocal, inLocal, AscendC::RoundMode::CAST_RINT, elements);
    }
    inputXCopyOutQueue.EnQue(outLocal);
    floatQueue.FreeTensor(floatLocal);
    halfQueue.FreeTensor(halfLocal);
}

template<typename T>
__aicore__ inline void MoeV2GatherQuant<T>::CopyOut(int64_t progress) {
    AscendC::LocalTensor<int32_t> indicesLocal = expandRowIdxCopyInQueue.DeQue<int32_t>();
    SetWaitFlag<AscendC::HardEvent::MTE2_S>(AscendC::HardEvent::MTE2_S);
    colsTileLength = this->perLoopCols;
    for (int64_t colsLoop = 0; colsLoop < this->colLoops; colsLoop++) {
        int64_t initialRow = this->gatherOutTilingData->perCoreRows * this->blockIdx + this->perLoopRows * progress;
        int64_t curLoopRow = 0;
        if (colsLoop == this->colLoops - 1) {
            colsTileLength = this->lastLoopCols;
        }
        int64_t currentLoopStartRow = initialRow / this->k;
        int64_t currentLoopLastRow = (initialRow + this->currentLoopRows - 1) / this->k;
        for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
            AscendC::LocalTensor<T> inLocal = inputXCopyInQueue.AllocTensor<T>();
            // input row position
            inputOffset = row * this->cols + colsLoop * this->perLoopCols;
           AscendC::DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(this->colsTileLength * sizeof(T)), 0, 0, 0};
           AscendC::DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
            AscendC::DataCopyPad(inLocal, inputXGm[inputOffset], dataCopyParams, dataCopyPadParams);
            inputXCopyInQueue.
            EnQue<T>(inLocal);

            Compute();

            AscendC::LocalTensor<int8_t> outLocal = inputXCopyOutQueue.DeQue<int8_t>();
           AscendC::DataCopyExtParams intriParams{1, static_cast<uint32_t>(this->colsTileLength * sizeof(int8_t)), 0, 0, 0};
            while (curLoopRow < this->currentLoopRows &&initialRow / this->k == row) {
                int32_t outIndex = indicesLocal.GetValue(curLoopRow);
                curLoopRow++;
                initialRow++;
                if (outIndex == -1 || (this->dropPadMode == DROPLESS_MODE &&outIndex >= this->activateRows)) {
                    continue;
                }
                outOffset = outIndex * cols + colsLoop * this->perLoopCols;
                AscendC::DataCopyPad(expandedXGm[outOffset], outLocal, intriParams);
            }
            inputXCopyInQueue.
            FreeTensor(inLocal);
            inputXCopyOutQueue.
            FreeTensor(outLocal);
        }
    }

    expandRowIdxCopyInQueue.
    FreeTensor(indicesLocal);
}

template<typename T>
__aicore__ inline void MoeV2GatherQuant<T>::Init(GM_ADDR inputX, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedRowIdx,
                                                 GM_ADDR expandedX, GM_ADDR workspace,
                                                 const optiling::MoeInitRoutingQuantV2TilingData *tilingData, AscendC::TPipe *tPipe)
{
    this->pipe = tPipe;
    this->blockIdx = get_block_idx() + get_subblockid() * get_block_num();
    this->gatherOutTilingData = &(tilingData->gatherOutComputeParamsOp);

    this->needCoreNum = this->gatherOutTilingData->needCoreNum;
    this->activateRows = this->gatherOutTilingData->activateRows;
    this->cols = tilingData->cols;
    this->n = tilingData->n;
    this->k = tilingData->k;
    this->dropPadMode = tilingData->dropPadMode;

    if (this->blockIdx == this->gatherOutTilingData->needCoreNum - 1) {
        this->coreRows = this->gatherOutTilingData->lastCoreRows;
        this->perLoopRows = this->gatherOutTilingData->lastCorePerLoopRows;
        this->lastLoopRows = this->gatherOutTilingData->lastCoreLastLoopRows;
        this->rowLoops = this->gatherOutTilingData->lastCoreLoops;
    } else {
        this->coreRows = this->gatherOutTilingData->perCoreRows;
        this->perLoopRows = this->gatherOutTilingData->perCorePerLoopRows;
        this->lastLoopRows = this->gatherOutTilingData->perCoreLastLoopRows;
        this->rowLoops = this->gatherOutTilingData->perCoreLoops;
    }
    this->perLoopCols = this->gatherOutTilingData->perLoopCols;
    this->lastLoopCols = this->gatherOutTilingData->lastLoopCols;
    this->colLoops = this->gatherOutTilingData->colLoops;

    inputXGm.SetGlobalBuffer((__gm__ T*)inputX);
    expandedXGm.SetGlobalBuffer((__gm__ int8_t *)expandedX);
    expandedRowIdxGm.SetGlobalBuffer(
            (__gm__ int32_t *)expandedRowIdx + this->blockIdx * this->gatherOutTilingData->perCoreRows,
            Align(this->coreRows, sizeof(int32_t)));
    scaleGm.SetGlobalBuffer((__gm__ float*)scale, 1);
    offsetGm.SetGlobalBuffer((__gm__ float*)offset, 1);
    this->scale = scaleGm.GetValue(0);
    this->offset = offsetGm.GetValue(0);

    pipe->InitBuffer(inputXCopyInQueue, BUFFER_NUM, AlignBytes(this->perLoopCols, sizeof(T)));
    pipe->InitBuffer(inputXCopyOutQueue, BUFFER_NUM, AlignBytes(this->perLoopCols, sizeof(int8_t)));
    pipe->InitBuffer(expandRowIdxCopyInQueue, BUFFER_NUM, AlignBytes(this->perLoopRows, sizeof(int32_t)));
    pipe->InitBuffer(floatQueue, 1, AlignBytes(this->perLoopCols, sizeof(float)));
    pipe->InitBuffer(halfQueue, 1, AlignBytes(this->perLoopCols, sizeof(half)));
}

template<typename T>
__aicore__ inline void MoeV2GatherQuant<T>::Process()
{
    if (this->blockIdx < this->needCoreNum) {
        currentLoopRows = perLoopRows;
        for (int64_t loop = 0; loop < this->rowLoops; loop++) {
            if (loop == this->rowLoops - 1) {
                currentLoopRows = lastLoopRows;
            }
            CopyInIndices(loop);
            CopyOut(loop);
        }
    }
}
}  // namespace MoeInitRoutingQuantV2
#endif  // MOE_V2_GATHER_QUANT_H
