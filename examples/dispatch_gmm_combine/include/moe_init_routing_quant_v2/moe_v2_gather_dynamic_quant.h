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
 * \file moe_v2_gather_dynamic_quant.h
 * \brief
 */
#ifndef MOE_V2_GATHER_DYNAMIC_QUANT_H
#define MOE_V2_GATHER_DYNAMIC_QUANT_H

#include "moe_v2_common.h"

namespace MoeInitRoutingQuantV2 {
template<typename T>
class MoeV2GatherDynamicQuant {
public:
    __aicore__ inline MoeV2GatherDynamicQuant() {};

    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR quantSmooth, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                GM_ADDR dynamicQuantScale, GM_ADDR workspace,
                                const optiling::MoeInitRoutingQuantV2TilingData *tilingData, AscendC::TPipe *tPipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInExpandedRowIdx(int64_t progress);

    __aicore__ inline void CopyInExpandedExpertIdx(int64_t progress);

    __aicore__ inline void CopyOutXQuant1H(int64_t progress);

    __aicore__ inline void CopyOutXQuantEH(int64_t progress);

    __aicore__ inline void Compute(AscendC::LocalTensor<float> &smoothLocal);

    __aicore__ inline void CopyOutPartialXQuantEH(int64_t progress);

    __aicore__ inline void CopyOutPartialXQuant1H(int64_t progress);

    __aicore__ inline float ComputeMax(AscendC::LocalTensor<float> &inLocal, AscendC::LocalTensor<float> &tempLocal,
                                       AscendC::LocalTensor<float> &dynamicQuantLocal, int32_t srcIdx, int32_t expertIdx,
                                       int64_t j);

    __aicore__ inline void ComputeScale(AscendC::LocalTensor<float> &inLocal, AscendC::LocalTensor<float> &tempLocal, float scaleTemp,
                                        int64_t dstIndex, int64_t j);

private:
    AscendC::TPipe *pipe;
    AscendC::TQue <AscendC::QuePosition::VECIN, BUFFER_NUM> inputXInQueue;
    AscendC::TQue <AscendC::QuePosition::VECIN, BUFFER_NUM> smoothInQueue;
    AscendC::TQue <AscendC::QuePosition::VECIN, BUFFER_NUM> expandRowIdxInQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> calcQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> inputXOutQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> scaleOutQueue;

    AscendC::GlobalTensor <T> inputXGm;
    AscendC::GlobalTensor <int8_t> expandedXGm;
    AscendC::GlobalTensor <int32_t> expandedRowIdxGm;
    AscendC::GlobalTensor<float> quantSmoothGm;
    AscendC::GlobalTensor<float> dynamicQuantScaleGm;
    AscendC::GlobalTensor<float> quantSrcGm;
    AscendC::GlobalTensor <int32_t> expandedExpertIdxGm;
    AscendC::GlobalTensor <int32_t> sortedRowIdxGm;

    const optiling::InnerMoeV2GatherOutComputeTilingData *gatherOutTilingData;

    int64_t needCoreNum;
    int64_t blockIdx;
    int64_t cols;
    int64_t n;
    int64_t k;
    int64_t totalLength;
    int64_t activateRows;
    int64_t currentLoopRows;
    int64_t currentLoopRowsAlign;
    int64_t coreRows;
    int64_t perLoopRows;
    int64_t lastLoopRows;
    int64_t rowLoops;
    int64_t colsTileLength;
    int64_t perLoopCols;
    int64_t perLoopColsAlign;
    int64_t lastLoopCols;
    int64_t colLoops;
    int64_t dropPadMode;
    int64_t smoothType;

    int64_t indicesOffset;
    int64_t inputOffset;
    int64_t outOffset;
};

template<typename T>
__aicore__ inline void MoeV2GatherDynamicQuant<T>::CopyInExpandedRowIdx(int64_t progress)
{
    this->indicesOffset = progress * this->perLoopRows;
    AscendC::LocalTensor <int32_t> indicesLocal = expandRowIdxInQueue.AllocTensor<int32_t>();
   AscendC::DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(this->currentLoopRows * sizeof(int32_t)), 0, 0, 0};
   AscendC::DataCopyPadExtParams <int32_t> dataCopyPadParams{false, 0, 0, 0};
    AscendC::DataCopyPad(indicesLocal, expandedRowIdxGm[indicesOffset], dataCopyParams, dataCopyPadParams);
    expandRowIdxInQueue.EnQue<int32_t>(indicesLocal);
}

template<typename T>
__aicore__ inline void MoeV2GatherDynamicQuant<T>::CopyInExpandedExpertIdx(int64_t progress)
{
    this->indicesOffset = progress * this->perLoopRows;
    AscendC::LocalTensor <int32_t> indicesLocal = expandRowIdxInQueue.AllocTensor<int32_t>();
   AscendC::DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(this->currentLoopRows * sizeof(int32_t)), 0, 0, 0};
   AscendC::DataCopyPadExtParams <int32_t> dataCopyPadParams{false, 0, 0, 0};
    AscendC::DataCopyPad(indicesLocal, sortedRowIdxGm[indicesOffset], dataCopyParams, dataCopyPadParams);
    AscendC::DataCopyPad(indicesLocal[currentLoopRowsAlign], expandedExpertIdxGm[indicesOffset], dataCopyParams,
                dataCopyPadParams);
    expandRowIdxInQueue.EnQue<int32_t>(indicesLocal);
}

template<typename T>
__aicore__ inline void MoeV2GatherDynamicQuant<T>::Compute(AscendC::LocalTensor<float> &smoothLocal)
{
    AscendC::LocalTensor<float> inLocal = inputXInQueue.DeQue<float>();

    AscendC::LocalTensor<float> tempLocal = calcQueue.AllocTensor<float>();
    AscendC::LocalTensor <int8_t> outLocal = inputXOutQueue.AllocTensor<int8_t>();
    AscendC::LocalTensor<float> dynamicQuantLocal = scaleOutQueue.AllocTensor<float>();

    if constexpr(!AscendC::IsSameType<T, float>::value) {
        AscendC::Cast(inLocal, inLocal.ReinterpretCast<T>()[perLoopColsAlign], AscendC::RoundMode::CAST_NONE, this->cols);
        pipe_barrier(PIPE_V);
    }

    if (smoothType != 0) {
        AscendC::Mul(inLocal, inLocal, smoothLocal, this->cols);
        pipe_barrier(PIPE_V);
    }

    Abs(tempLocal, inLocal, this->cols);
    pipe_barrier(PIPE_V);

    AscendC::ReduceMax(dynamicQuantLocal, tempLocal, tempLocal, this->cols);
    pipe_barrier(PIPE_V);

    float maxValue = dynamicQuantLocal.GetValue(0) / 127.0f;

    AscendC::Duplicate<float>(dynamicQuantLocal, maxValue, 8);
    AscendC::Duplicate<float>(tempLocal, maxValue, this->cols);
    pipe_barrier(PIPE_V);

    Div(tempLocal, inLocal, tempLocal, this->cols);
    pipe_barrier(PIPE_V);

    AscendC::Cast(tempLocal.ReinterpretCast<half>(), tempLocal, AscendC::RoundMode::CAST_TRUNC, this->cols);
    pipe_barrier(PIPE_V);

    AscendC::Cast(outLocal, tempLocal.ReinterpretCast<half>(), AscendC::RoundMode::CAST_ROUND, this->cols);

    calcQueue.FreeTensor(tempLocal);
    inputXOutQueue.EnQue(outLocal);
    scaleOutQueue.EnQue(dynamicQuantLocal);
}

template<typename T>
__aicore__ inline void MoeV2GatherDynamicQuant<T>::CopyOutXQuant1H(int64_t progress)
{
    AscendC::LocalTensor <int32_t> indicesLocal = expandRowIdxInQueue.DeQue<int32_t>();

    int64_t initialRow = this->gatherOutTilingData->perCoreRows * this->blockIdx + this->perLoopRows * progress;
    int64_t curLoopRow = 0;
    int64_t currentLoopStartRow = initialRow / this->k;
    int64_t currentLoopLastRow = (initialRow + this->currentLoopRows - 1) / this->k;
   AscendC::DataCopyExtParams copyInParams{1, static_cast<uint32_t>(this->cols * sizeof(T)), 0, 0, 0};
   AscendC::DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(this->cols * sizeof(int8_t)), 0, 0, 0};
   AscendC::DataCopyExtParams smoothParams{1, static_cast<uint32_t>(this->cols * sizeof(float)), 0, 0, 0};

    AscendC::LocalTensor<float> smoothLocal;
    if (smoothType == 1) {
        smoothLocal = smoothInQueue.AllocTensor<float>();
        AscendC::DataCopyPad(smoothLocal, quantSmoothGm, smoothParams, {false, 0, 0, 0});
        smoothInQueue.EnQue(smoothLocal);
        smoothLocal = smoothInQueue.DeQue<float>();
    }

    for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
        AscendC::LocalTensor <T> inLocal = inputXInQueue.AllocTensor<T>();
        if constexpr(AscendC::IsSameType<T, float>::value)
        {
            AscendC::DataCopyPad(inLocal, inputXGm[row * this->cols], copyInParams, {false, 0, 0, 0});
        } else {
            AscendC::DataCopyPad(inLocal[perLoopColsAlign], inputXGm[row * this->cols], copyInParams, {false, 0, 0, 0});
        }

        inputXInQueue.EnQue<T>(inLocal);

        // 计算quant
        Compute(smoothLocal);

        AscendC::LocalTensor<float> quantScaleLocal = scaleOutQueue.DeQue<float>();
        AscendC::LocalTensor <int8_t> outLocal = inputXOutQueue.DeQue<int8_t>();

        while (curLoopRow < this->currentLoopRows && initialRow / this->k == row) {
            int32_t outIndex = indicesLocal.GetValue(curLoopRow);
            curLoopRow++;
            initialRow++;
            if (outIndex == -1 || (this->dropPadMode == DROPLESS_MODE && outIndex >= this->activateRows)) {
                continue;
            }
            AscendC::DataCopyPad(expandedXGm[outIndex * cols], outLocal, copyOutParams);
            AscendC::DataCopyPad(dynamicQuantScaleGm[outIndex], quantScaleLocal, {1, 4, 0, 0, 0});
        }
        inputXInQueue.FreeTensor(inLocal);
        inputXOutQueue.FreeTensor(outLocal);
        scaleOutQueue.FreeTensor(quantScaleLocal);
    }
    if (smoothType == 1) {
        smoothInQueue.FreeTensor(smoothLocal);
    }
    expandRowIdxInQueue.FreeTensor(indicesLocal);
}

template<typename T>
__aicore__ inline void MoeV2GatherDynamicQuant<T>::CopyOutXQuantEH(int64_t progress)
{
    AscendC::LocalTensor <int32_t> indicesLocal = expandRowIdxInQueue.DeQue<int32_t>();
    SetWaitFlag<AscendC::HardEvent::MTE2_S>(AscendC::HardEvent::MTE2_S);

   AscendC::DataCopyExtParams copyInParams{1, static_cast<uint32_t>(this->perLoopCols * sizeof(T)), 0, 0, 0};
   AscendC::DataCopyExtParams smoothParams{1, static_cast<uint32_t>(this->perLoopCols * sizeof(float)), 0, 0, 0};
   AscendC::DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(this->perLoopCols * sizeof(int8_t)), 0, 0, 0};

    int32_t lastExpertIdx = -1;
    AscendC::LocalTensor <T> inLocal = inputXInQueue.AllocTensor<T>();
    AscendC::LocalTensor<float> smoothLocal = smoothInQueue.AllocTensor<float>();
    for (int64_t i = 0; i < this->currentLoopRows; i++) {
        int64_t rowOffset = this->gatherOutTilingData->perCoreRows * this->blockIdx + this->perLoopRows * progress;
        if (this->dropPadMode == DROPLESS_MODE && rowOffset + i >= this->activateRows) {
            break;
        }
        int32_t srcIdx = indicesLocal.GetValue(i);
        int32_t expertIdx = indicesLocal.GetValue(currentLoopRowsAlign + i);

        if constexpr(AscendC::IsSameType<T, float>::value)
        {
            AscendC::DataCopyPad(inLocal, inputXGm[srcIdx / this->k * this->cols], copyInParams, {false, 0, 0, 0});
        } else {
            AscendC::DataCopyPad(inLocal[perLoopColsAlign], inputXGm[srcIdx / this->k * this->cols], copyInParams,
                        {false, 0, 0, 0});
        }
        inputXInQueue.EnQue<T>(inLocal);

        if (expertIdx != lastExpertIdx) {
            AscendC::DataCopyPad(smoothLocal, quantSmoothGm[expertIdx * this->cols], smoothParams, {false, 0, 0, 0});
            smoothInQueue.EnQue(smoothLocal);
            smoothLocal = smoothInQueue.DeQue<float>();
            lastExpertIdx = expertIdx;
        }

        Compute(smoothLocal);

        AscendC::LocalTensor<float> quantScaleLocal = scaleOutQueue.DeQue<float>();
        AscendC::DataCopyPad(dynamicQuantScaleGm[(rowOffset + i)], quantScaleLocal, {1, 4, 0, 0, 0});

        AscendC::LocalTensor <int8_t> outLocal = inputXOutQueue.DeQue<int8_t>();
        AscendC::DataCopyPad(expandedXGm[(rowOffset + i) * this->cols], outLocal, copyOutParams);

        inputXOutQueue.FreeTensor(outLocal);
        scaleOutQueue.FreeTensor(quantScaleLocal);
    }

    inputXInQueue.FreeTensor(inLocal);
    smoothInQueue.FreeTensor(smoothLocal);
    expandRowIdxInQueue.FreeTensor(indicesLocal);
}

template<typename T>
__aicore__ inline float MoeV2GatherDynamicQuant<T>::ComputeMax(AscendC::LocalTensor<float> &inLocal,
                                                               AscendC::LocalTensor<float> &tempLocal,
                                                               AscendC::LocalTensor<float> &dynamicQuantLocal, int32_t srcIdx,
                                                               int32_t expertIdx, int64_t j)
{
    AscendC::LocalTensor<float> smoothLocal = smoothInQueue.AllocTensor<float>();

   AscendC::DataCopyExtParams intriParamsT{1, static_cast<uint32_t>(colsTileLength * sizeof(T)), 0, 0, 0};
   AscendC::DataCopyExtParams intriParamsFp32{1, static_cast<uint32_t>(colsTileLength * sizeof(float)), 0, 0, 0};

    if constexpr(!AscendC::IsSameType<T, float>::value)
    {
        AscendC::DataCopyPad(inLocal.ReinterpretCast<T>()[perLoopColsAlign],
                    inputXGm[srcIdx * this->cols + j * this->perLoopCols],
                    intriParamsT, {false, 0, 0, 0});
    } else {
        AscendC::DataCopyPad(inLocal, inputXGm[srcIdx * this->cols + j * this->perLoopCols], intriParamsT, {false, 0, 0, 0});
    }

    inputXInQueue.EnQue<float>(inLocal);
    inLocal = inputXInQueue.DeQue<float>();

    if (smoothType != 0) {
        AscendC::DataCopyPad(smoothLocal, quantSmoothGm[expertIdx * this->cols + j * this->perLoopCols], intriParamsFp32,
                    {false, 0, 0, 0});
        smoothInQueue.EnQue(smoothLocal);
        smoothLocal = smoothInQueue.DeQue<float>();
    }

    if constexpr(!AscendC::IsSameType<T, float>::value)
    {
        AscendC::Cast(inLocal, inLocal.ReinterpretCast<T>()[perLoopColsAlign], AscendC::RoundMode::CAST_NONE, colsTileLength);
        pipe_barrier(PIPE_V);
    }

    if (smoothType != 0) {
        AscendC::Mul(inLocal, inLocal, smoothLocal, colsTileLength);
        pipe_barrier(PIPE_V);
    }

    Abs(tempLocal, inLocal, colsTileLength);
    pipe_barrier(PIPE_V);

    AscendC::ReduceMax(dynamicQuantLocal[8], tempLocal, tempLocal, colsTileLength);

    AscendC::DataCopyPad(quantSrcGm[j * this->perLoopCols], inLocal, intriParamsFp32);
    smoothInQueue.FreeTensor(smoothLocal);
    SetWaitFlag<AscendC::HardEvent::MTE3_MTE2>(AscendC::HardEvent::MTE3_MTE2);

    return dynamicQuantLocal.GetValue(8);
}

template<typename T>
__aicore__ inline void MoeV2GatherDynamicQuant<T>::ComputeScale(AscendC::LocalTensor<float> &inLocal,
                                                                AscendC::LocalTensor<float> &tempLocal, float scaleTemp,
                                                                int64_t dstIndex, int64_t j)
{
   AscendC::DataCopyExtParams copyInParams{1, static_cast<uint32_t>(colsTileLength * sizeof(float)), 0, 0, 0};
   AscendC::DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(colsTileLength * sizeof(int8_t)), 0, 0, 0};

    AscendC::LocalTensor <int8_t> outLocal = inputXOutQueue.AllocTensor<int8_t>();

    AscendC::DataCopyPad(inLocal, quantSrcGm[j * this->perLoopCols], copyInParams, {false, 0, 0, 0});
    inputXInQueue.EnQue<float>(inLocal);
    inLocal = inputXInQueue.DeQue<float>();

    AscendC::Duplicate<float>(tempLocal, scaleTemp, colsTileLength);
    pipe_barrier(PIPE_V);

    Div(tempLocal, inLocal, tempLocal, colsTileLength);
    pipe_barrier(PIPE_V);

    AscendC::Cast(tempLocal.ReinterpretCast<half>(), tempLocal, AscendC::RoundMode::CAST_TRUNC, colsTileLength);
    pipe_barrier(PIPE_V);

    AscendC::Cast(outLocal, tempLocal.ReinterpretCast<half>(), AscendC::RoundMode::CAST_ROUND, colsTileLength);

    inputXOutQueue.EnQue(outLocal);
    outLocal = inputXOutQueue.DeQue<int8_t>();
    AscendC::DataCopyPad(expandedXGm[dstIndex * this->cols + j * this->perLoopCols], outLocal, copyOutParams);

    inputXOutQueue.FreeTensor(outLocal);
    SetWaitFlag<AscendC::HardEvent::MTE3_MTE2>(AscendC::HardEvent::MTE3_MTE2);
}

template<typename T>
__aicore__ inline void MoeV2GatherDynamicQuant<T>::CopyOutPartialXQuantEH(int64_t progress)
{
    AscendC::LocalTensor <int32_t> indicesLocal = expandRowIdxInQueue.DeQue<int32_t>();
    SetWaitFlag<AscendC::HardEvent::MTE2_S>(AscendC::HardEvent::MTE2_S);

    for (int64_t i = 0; i < this->currentLoopRows; i++) {
        int64_t rowOffset = this->gatherOutTilingData->perCoreRows * this->blockIdx + this->perLoopRows * progress;
        if (this->dropPadMode == DROPLESS_MODE && rowOffset + i >= this->activateRows) {
            break;
        }
        int32_t srcIdx = indicesLocal.GetValue(i);
        int32_t expertIdx = indicesLocal.GetValue(currentLoopRowsAlign + i);

        AscendC::LocalTensor<float> inLocal = inputXInQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> tempLocal = calcQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> quantScaleLocal = scaleOutQueue.AllocTensor<float>();

        uint32_t tmp = 0xFF7FFFFF;
        float reduceMax = *((float *) &tmp);
        for (int64_t j = 0; j < this->colLoops; j++) {
            colsTileLength = this->perLoopCols;
            if (j == this->colLoops - 1) {
                colsTileLength = this->lastLoopCols;
            }
            float tileMax = ComputeMax(inLocal, tempLocal, quantScaleLocal, srcIdx / this->k, expertIdx, j);
            reduceMax = (reduceMax > tileMax) ? reduceMax : tileMax;
        }

        float scaleTemp = reduceMax / 127.0f;
        AscendC::Duplicate<float>(quantScaleLocal, scaleTemp, 8);
        scaleOutQueue.EnQue(quantScaleLocal);
        quantScaleLocal = scaleOutQueue.DeQue<float>();

        AscendC::DataCopyPad(dynamicQuantScaleGm[(rowOffset + i)], quantScaleLocal, {1, 4, 0, 0, 0});

        for (int64_t j = 0; j < this->colLoops; j++) {
            colsTileLength = this->perLoopCols;
            if (j == this->colLoops - 1) {
                colsTileLength = this->lastLoopCols;
            }

            ComputeScale(inLocal, tempLocal, scaleTemp, rowOffset + i, j);
        }

        inputXInQueue.FreeTensor(inLocal);
        calcQueue.FreeTensor(tempLocal);
        scaleOutQueue.FreeTensor(quantScaleLocal);
    }

    expandRowIdxInQueue.FreeTensor(indicesLocal);
}

template<typename T>
__aicore__ inline void MoeV2GatherDynamicQuant<T>::CopyOutPartialXQuant1H(int64_t progress)
{
    AscendC::LocalTensor <int32_t> indicesLocal = expandRowIdxInQueue.DeQue<int32_t>();

    int64_t initialRow = this->gatherOutTilingData->perCoreRows * this->blockIdx + this->perLoopRows * progress;
    int64_t curLoopRow = 0;

    int64_t currentLoopStartRow = initialRow / this->k;
    int64_t currentLoopLastRow = (initialRow + this->currentLoopRows - 1) / this->k;

    for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
        AscendC::LocalTensor<float> inLocal = inputXInQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> tempLocal = calcQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> quantScaleLocal = scaleOutQueue.AllocTensor<float>();

        uint32_t tmp = 0xFF7FFFFF;
        float reduceMax = *((float *) &tmp);
        for (int64_t j = 0; j < this->colLoops; j++) {
            colsTileLength = this->perLoopCols;
            if (j == this->colLoops - 1) {
                colsTileLength = this->lastLoopCols;
            }

            float tileMax = ComputeMax(inLocal, tempLocal, quantScaleLocal, row, 0, j);
            reduceMax = (reduceMax > tileMax) ? reduceMax : tileMax;
        }

        float scaleTemp = reduceMax / 127.0f;
        AscendC::Duplicate<float>(quantScaleLocal, scaleTemp, 8);
        scaleOutQueue.EnQue(quantScaleLocal);
        quantScaleLocal = scaleOutQueue.DeQue<float>();

        while (curLoopRow < this->currentLoopRows && initialRow / this->k == row) {
            int32_t outIndex = indicesLocal.GetValue(curLoopRow);
            curLoopRow++;
            initialRow++;
            if (outIndex == -1 || (this->dropPadMode == DROPLESS_MODE && outIndex >= this->activateRows)) {
                continue;
            }
            AscendC::DataCopyPad(dynamicQuantScaleGm[outIndex], quantScaleLocal, {1, 4, 0, 0, 0});
            for (int64_t j = 0; j < this->colLoops; j++) {
                colsTileLength = this->perLoopCols;
                if (j == this->colLoops - 1) {
                    colsTileLength = this->lastLoopCols;
                }

                ComputeScale(inLocal, tempLocal, scaleTemp, outIndex, j);
            }
        }
        inputXInQueue.FreeTensor(inLocal);
        calcQueue.FreeTensor(tempLocal);
        scaleOutQueue.FreeTensor(quantScaleLocal);
    }

    expandRowIdxInQueue.FreeTensor(indicesLocal);
}

template<typename T>
__aicore__ inline void MoeV2GatherDynamicQuant<T>::Init(GM_ADDR inputX, GM_ADDR quantSmooth, GM_ADDR expandedRowIdx,
                                                        GM_ADDR expandedX, GM_ADDR dynamicQuantScale, GM_ADDR workspace,
                                                        const optiling::MoeInitRoutingQuantV2TilingData *tilingData,
                                                        AscendC::TPipe *tPipe)
{
    this->pipe = tPipe;
    this->blockIdx = get_block_idx() + get_subblockid() * get_block_num();
    this->gatherOutTilingData = &(tilingData->gatherOutComputeParamsOp);

    this->needCoreNum = this->gatherOutTilingData->needCoreNum;
    this->activateRows = this->gatherOutTilingData->activateRows;
    this->cols = tilingData->cols;
    this->n = tilingData->n;
    this->k = tilingData->k;
    this->totalLength = tilingData->n * tilingData->k;
    this->dropPadMode = tilingData->dropPadMode;
    this->smoothType = tilingData->smoothType;

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
    this->perLoopColsAlign = Align(this->perLoopCols, sizeof(T));

    inputXGm.SetGlobalBuffer((__gm__ T*)inputX);
    expandedXGm.SetGlobalBuffer((__gm__ int8_t *)expandedX);

    expandedRowIdxGm.SetGlobalBuffer(
            (__gm__ int32_t *)expandedRowIdx + this->blockIdx * this->gatherOutTilingData->perCoreRows,
            Align(this->coreRows, sizeof(int32_t)));

    quantSmoothGm.SetGlobalBuffer((__gm__ float *) quantSmooth);
    dynamicQuantScaleGm.SetGlobalBuffer((__gm__ float *) dynamicQuantScale);

    expandedExpertIdxGm.SetGlobalBuffer(
            (__gm__ int32_t *)workspace + this->blockIdx * this->gatherOutTilingData->perCoreRows,
            Align(this->coreRows, sizeof(int32_t)));
    sortedRowIdxGm.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(this->totalLength, sizeof(int32_t)) +
              this->blockIdx * this->gatherOutTilingData->perCoreRows,
            Align(this->coreRows, sizeof(int32_t)));
    if (this->cols > 1) {
        quantSrcGm.SetGlobalBuffer(
                (__gm__ float *) workspace + Align(this->totalLength, sizeof(int32_t)) * 2 +
                this->blockIdx * this->cols,
                this->cols * sizeof(float));
    }

    this->currentLoopRowsAlign = Align(this->perLoopRows, sizeof(int32_t));

    int64_t perLoopColsAlignBytes = AlignBytes(this->perLoopCols, sizeof(T));
    perLoopColsAlignBytes =
            Max(int64_t(perLoopColsAlignBytes * sizeof(float) / sizeof(T)), int64_t(BLOCK_BYTES + BLOCK_BYTES));

    pipe->InitBuffer(expandRowIdxInQueue, BUFFER_NUM, 2 * AlignBytes(this->perLoopRows, sizeof(int32_t)));
    pipe->InitBuffer(inputXInQueue, BUFFER_NUM, perLoopColsAlignBytes);
    pipe->InitBuffer(smoothInQueue, BUFFER_NUM, AlignBytes(this->perLoopCols, sizeof(float)));
    pipe->InitBuffer(calcQueue, 1, AlignBytes(this->perLoopCols, sizeof(float)));
    pipe->InitBuffer(inputXOutQueue, 1, AlignBytes(this->perLoopCols, sizeof(int8_t)));
    pipe->InitBuffer(scaleOutQueue, 1, BLOCK_BYTES + BLOCK_BYTES);
}

template<typename T>
__aicore__ inline void MoeV2GatherDynamicQuant<T>::Process()
{
    if (this->blockIdx < this->needCoreNum) {
        currentLoopRows = perLoopRows;
        if (colLoops > 1) {  // 一行无法全载，需要workspace
            if (smoothType == 2) {
                for (int64_t loop = 0; loop < this->rowLoops - 1; loop++) {
                    CopyInExpandedExpertIdx(loop);
                    CopyOutPartialXQuantEH(loop);
                }
                currentLoopRows = lastLoopRows;
                CopyInExpandedExpertIdx(this->rowLoops - 1);
                CopyOutPartialXQuantEH(this->rowLoops - 1);
            } else {
                for (int64_t loop = 0; loop < this->rowLoops - 1; loop++) {
                    CopyInExpandedRowIdx(loop);
                    CopyOutPartialXQuant1H(loop);
                }
                currentLoopRows = lastLoopRows;
                CopyInExpandedRowIdx(this->rowLoops - 1);
                CopyOutPartialXQuant1H(this->rowLoops - 1);
            }
        } else {  // 一行可以全载
            if (smoothType == 2) {
                for (int64_t loop = 0; loop < this->rowLoops - 1; loop++) {
                    CopyInExpandedExpertIdx(loop);
                    CopyOutXQuantEH(loop);
                }
                currentLoopRows = lastLoopRows;
                CopyInExpandedExpertIdx(this->rowLoops - 1);
                CopyOutXQuantEH(this->rowLoops - 1);
            } else {
                for (int64_t loop = 0; loop < this->rowLoops - 1; loop++) {
                    CopyInExpandedRowIdx(loop);
                    CopyOutXQuant1H(loop);
                }
                currentLoopRows = lastLoopRows;
                CopyInExpandedRowIdx(this->rowLoops - 1);
                CopyOutXQuant1H(this->rowLoops - 1);
            }
        }
    }
}
}  // namespace MoeInitRoutingQuantV2
#endif  // MOE_V2_GATHER_DYNAMIC_QUANT_H
