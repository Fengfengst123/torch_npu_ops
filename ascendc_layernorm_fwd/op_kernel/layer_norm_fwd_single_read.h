/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

#ifndef XLLM_LAYER_NORM_FWD_SINGLE_READ_H
#define XLLM_LAYER_NORM_FWD_SINGLE_READ_H

#include <type_traits>

#include "../op_host/layer_norm_fwd_tiling.h"
#include "layer_norm_fwd_common.h"

namespace LayerNormFwd {

template <typename Tfm, typename Tweight>
class LayerNormFwdSingleRead {
public:
    __aicore__ inline LayerNormFwdSingleRead() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR z, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace,
        const optiling::LayerNormFwdTilingData* __restrict tilingData)
    {
        (void)z;
        (void)workspace;
        numBlocks = tilingData->numBlocks;
        colSize = tilingData->colSize;
        rowSize = tilingData->rowSize;
        nRow = tilingData->nRow;
        tileLength = tilingData->tileLength;
        blockLength = tilingData->blockLength;
        tailNRow = tilingData->tailNRow;
        loopCount = tilingData->loopCount;
        tailLoop = tilingData->tailLoop;
        rowAlign = tilingData->rowAlign;
        nullptrGamma = tilingData->nullptrGamma;
        nullptrBeta = tilingData->nullptrBeta;
        isRmsNorm = tilingData->isRmsNorm;
        eps = tilingData->eps;
        coefficient = tilingData->coefficient;
        inputRows = tilingData->inputRows;
        groupCount = tilingData->groupCount;

        if (GetBlockIdx() < tailLoop) {
            baseLogicalRow = (loopCount + 1) * nRow * GetBlockIdx();
        } else {
            baseLogicalRow = loopCount * nRow * GetBlockIdx() + nRow * tailLoop;
        }

        xGm.SetGlobalBuffer((__gm__ Tfm*)x, colSize * rowSize);
        yGm.SetGlobalBuffer((__gm__ Tfm*)y, colSize * rowSize);
        if (!isRmsNorm) {
            meanGm.SetGlobalBuffer((__gm__ float*)mean, colSize);
        }
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd, colSize);
        if (!nullptrGamma) {
            gammaGm.SetGlobalBuffer((__gm__ Tweight*)gamma, groupCount * rowSize);
        }
        if (!nullptrBeta) {
            betaGm.SetGlobalBuffer((__gm__ Tweight*)beta, groupCount * rowSize);
        }

        pipe.InitBuffer(inQueueX, 1, tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, 1, tileLength * sizeof(float));
        pipe.InitBuffer(outQueueMean, 1, kBlockBytes);
        pipe.InitBuffer(outQueueRstd, 1, kBlockBytes);
    }

    __aicore__ inline void Process()
    {
        uint32_t count = loopCount;
        if (GetBlockIdx() < tailLoop) {
            count += 1;
        }
        for (uint32_t loopIdx = 0; loopIdx < count; ++loopIdx) {
            ProcessBasicBlock(nRow, baseLogicalRow + loopIdx * nRow);
        }
        if (tailNRow > 0 && GetBlockIdx() == numBlocks - 1) {
            ProcessBasicBlock(tailNRow, baseLogicalRow + count * nRow);
        }
    }

protected:
    __aicore__ inline void LoadInput(LocalTensor<float>& xLocal, uint32_t currentLogicalRow, uint32_t currentNRow)
    {
        DataCopyExtParams dataCopyParams;
        DataCopyPadExtParams<Tfm> padParams;
        dataCopyParams.blockCount = currentNRow;
        dataCopyParams.blockLen = rowSize * sizeof(Tfm);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        padParams.isPad = false;
        DataCopyPad(
            xLocal.ReinterpretCast<Tfm>()[(sizeof(Tfm) == kHalfBytes) * tileLength],
            xGm[currentLogicalRow * rowSize],
            dataCopyParams,
            padParams);
        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.DeQue<float>();
        if (sizeof(Tfm) == kHalfBytes) {
            Cast(xLocal, xLocal.ReinterpretCast<Tfm>()[tileLength], RoundMode::CAST_NONE, tileLength);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void StoreScalar(GlobalTensor<float> gm, uint32_t gmOffset, LocalTensor<float> local)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        SetEvtFlag<HardEvent::S_MTE3>();
        DataCopyPad(gm[gmOffset], local, copyParams);
    }

    __aicore__ inline void LoadWeight(
        LocalTensor<float> dst, GlobalTensor<Tweight> src, uint32_t groupOffset)
    {
        DataCopyExtParams copyParams;
        DataCopyPadExtParams<Tweight> padParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = rowSize * sizeof(Tweight);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        padParams.isPad = false;
        SetEvtFlag<HardEvent::V_MTE2>();
        DataCopyPad(
            dst.ReinterpretCast<Tweight>()[(sizeof(Tweight) == kHalfBytes) * rowAlign],
            src[groupOffset],
            copyParams,
            padParams);
        SetEvtFlag<HardEvent::MTE2_V>();
        if (sizeof(Tweight) == kHalfBytes) {
            Cast(dst, dst.ReinterpretCast<Tweight>()[rowAlign], RoundMode::CAST_NONE, rowAlign);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ApplyAffine(
        LocalTensor<float> xLocal, LocalTensor<float> yLocal, uint32_t currentLogicalRow, uint32_t currentNRow)
    {
        if (groupCount <= 1) {
            if (!nullptrGamma) {
                LoadWeight(xLocal, gammaGm, 0);
                for (uint32_t rowIdx = 0; rowIdx < currentNRow; ++rowIdx) {
                    const uint32_t rowOffset = rowIdx * rowAlign;
                    Mul(yLocal[rowOffset], yLocal[rowOffset], xLocal, rowSize);
                    PipeBarrier<PIPE_V>();
                }
            }
            if (!nullptrBeta) {
                LoadWeight(xLocal, betaGm, 0);
                for (uint32_t rowIdx = 0; rowIdx < currentNRow; ++rowIdx) {
                    const uint32_t rowOffset = rowIdx * rowAlign;
                    Add(yLocal[rowOffset], yLocal[rowOffset], xLocal, rowSize);
                    PipeBarrier<PIPE_V>();
                }
            }
            return;
        }

        for (uint32_t rowIdx = 0; rowIdx < currentNRow; ++rowIdx) {
            const uint32_t logicalRow = currentLogicalRow + rowIdx;
            const uint32_t rowOffset = rowIdx * rowAlign;
            const uint32_t groupOffset = GetGroupOffset(logicalRow, groupCount, rowSize);
            if (!nullptrGamma) {
                LoadWeight(xLocal, gammaGm, groupOffset);
                Mul(yLocal[rowOffset], yLocal[rowOffset], xLocal, rowSize);
                PipeBarrier<PIPE_V>();
            }
            if (!nullptrBeta) {
                LoadWeight(xLocal, betaGm, groupOffset);
                Add(yLocal[rowOffset], yLocal[rowOffset], xLocal, rowSize);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void CastAndStoreOutput(
        LocalTensor<float> yLocal, uint32_t currentLogicalRow, uint32_t currentNRow)
    {
        if (sizeof(Tfm) == kHalfBytes) {
            if (std::is_same<Tfm, bfloat16_t>::value) {
                Cast(yLocal.ReinterpretCast<Tfm>(), yLocal, RoundMode::CAST_ROUND, tileLength);
            }
            PipeBarrier<PIPE_V>();
        }

        DataCopyExtParams yCopyParams;
        yCopyParams.blockCount = currentNRow;
        yCopyParams.blockLen = rowSize * sizeof(Tfm);
        yCopyParams.srcStride = 0;
        yCopyParams.dstStride = 0;
        DataCopyPad(yGm[currentLogicalRow * rowSize], yLocal.ReinterpretCast<Tfm>(), yCopyParams);
    }

    __aicore__ inline void ProcessBasicBlock(uint32_t currentNRow, uint32_t currentLogicalRow)
    {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<float> meanLocal = outQueueMean.AllocTensor<float>();
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();

        LoadInput(xLocal, currentLogicalRow, currentNRow);
        SetMaskNorm();

        if (isRmsNorm) {
            Mul(yLocal, xLocal, xLocal, tileLength);
            PipeBarrier<PIPE_V>();
            Muls(yLocal, yLocal, coefficient, tileLength);
            PipeBarrier<PIPE_V>();
            for (uint32_t rowIdx = 0; rowIdx < currentNRow; ++rowIdx) {
                const uint32_t rowOffset = rowIdx * rowAlign;
                ReduceSum(yLocal[rowOffset], yLocal[rowOffset], yLocal[rowOffset], rowSize);
                accVal = GetAccVal();
                value = *reinterpret_cast<float*>(&accVal);
                const float rstdValue = 1.0f / sqrt(value + eps);
                rstdLocal.SetValue(0, rstdValue);
                StoreScalar(rstdGm, GetParamOffset(currentLogicalRow + rowIdx, inputRows, groupCount), rstdLocal);
                Muls(yLocal[rowOffset], xLocal[rowOffset], rstdValue, rowSize);
                PipeBarrier<PIPE_V>();
            }
        } else {
            Muls(yLocal, xLocal, coefficient, tileLength);
            PipeBarrier<PIPE_V>();
            for (uint32_t rowIdx = 0; rowIdx < currentNRow; ++rowIdx) {
                const uint32_t rowOffset = rowIdx * rowAlign;
                ReduceSum(yLocal[rowOffset], yLocal[rowOffset], yLocal[rowOffset], rowSize);
                accVal = GetAccVal();
                value = *reinterpret_cast<float*>(&accVal);
                meanLocal.SetValue(0, value);
                StoreScalar(meanGm, GetParamOffset(currentLogicalRow + rowIdx, inputRows, groupCount), meanLocal);
                Adds(yLocal[rowOffset], xLocal[rowOffset], -value, rowSize);
                PipeBarrier<PIPE_V>();
            }

            Mul(xLocal, yLocal, yLocal, tileLength);
            PipeBarrier<PIPE_V>();
            Muls(xLocal, xLocal, coefficient, tileLength);
            PipeBarrier<PIPE_V>();
            for (uint32_t rowIdx = 0; rowIdx < currentNRow; ++rowIdx) {
                const uint32_t rowOffset = rowIdx * rowAlign;
                ReduceSum(xLocal[rowOffset], xLocal[rowOffset], xLocal[rowOffset], rowSize);
                accVal = GetAccVal();
                value = *reinterpret_cast<float*>(&accVal);
                const float rstdValue = 1.0f / sqrt(value + eps);
                rstdLocal.SetValue(0, rstdValue);
                StoreScalar(rstdGm, GetParamOffset(currentLogicalRow + rowIdx, inputRows, groupCount), rstdLocal);
                Muls(yLocal[rowOffset], yLocal[rowOffset], rstdValue, rowSize);
                PipeBarrier<PIPE_V>();
            }
        }

        ApplyAffine(xLocal, yLocal, currentLogicalRow, currentNRow);
        inQueueX.FreeTensor(xLocal);
        outQueueMean.FreeTensor(meanLocal);
        outQueueRstd.FreeTensor(rstdLocal);
        CastAndStoreOutput(yLocal, currentLogicalRow, currentNRow);
        outQueueY.FreeTensor(yLocal);
    }

protected:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TQue<QuePosition::VECOUT, 1> outQueueMean;
    TQue<QuePosition::VECOUT, 1> outQueueRstd;

    GlobalTensor<Tfm> xGm;
    GlobalTensor<Tfm> yGm;
    GlobalTensor<Tweight> gammaGm;
    GlobalTensor<Tweight> betaGm;
    GlobalTensor<float> meanGm;
    GlobalTensor<float> rstdGm;

    float value = 0.0f;
    uint64_t accVal = 0;
    uint32_t numBlocks = 0;
    uint32_t colSize = 0;
    uint32_t rowSize = 0;
    uint32_t nRow = 1;
    uint32_t tileLength = 0;
    uint32_t blockLength = 0;
    uint32_t tailNRow = 0;
    uint32_t loopCount = 0;
    uint32_t tailLoop = 0;
    uint32_t rowAlign = 0;
    uint32_t nullptrGamma = 0;
    uint32_t nullptrBeta = 0;
    uint32_t isRmsNorm = 0;
    uint32_t inputRows = 0;
    uint32_t groupCount = 1;
    uint32_t baseLogicalRow = 0;
    float eps = 0.0f;
    float coefficient = 0.0f;
};

} // namespace LayerNormFwd

#endif // XLLM_LAYER_NORM_FWD_SINGLE_READ_H
