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

#ifndef XLLM_LAYER_NORM_FWD_COMMON_H
#define XLLM_LAYER_NORM_FWD_COMMON_H

#include "kernel_operator.h"

namespace LayerNormFwd {
using namespace AscendC;

constexpr uint32_t kBlockBytes = 32;
constexpr uint32_t kHalfBytes = 2;
constexpr uint32_t kFloatBytes = 4;

struct DataCopyContiguous {
    uint32_t blockCount = 0;
    uint32_t blockLen = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
    bool isPad = false;
    uint8_t leftPad = 0;
    uint8_t rightPad = 0;
};

template <typename T>
__aicore__ inline void DataCopyInContiguous(
    LocalTensor<T> dst, GlobalTensor<T> src, const DataCopyContiguous& copy, uint32_t castOffset)
{
    DataCopyExtParams dataCopyParams;
    DataCopyPadExtParams<T> padParams{copy.isPad, copy.leftPad, copy.rightPad, 0};
    dataCopyParams.blockCount = copy.blockCount;
    dataCopyParams.blockLen = copy.blockLen;
    dataCopyParams.srcStride = copy.srcStride;
    dataCopyParams.dstStride = copy.dstStride;
    DataCopyPad(dst[(sizeof(T) == kHalfBytes) * castOffset], src, dataCopyParams, padParams);
}

template <typename T>
__aicore__ inline void DataCopyOutContiguous(
    GlobalTensor<T> dst, LocalTensor<T> src, const DataCopyContiguous& copy)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = copy.blockCount;
    dataCopyParams.blockLen = copy.blockLen;
    dataCopyParams.srcStride = copy.srcStride;
    dataCopyParams.dstStride = copy.dstStride;
    DataCopyPad(dst, src, dataCopyParams);
}

template <HardEvent evt>
__aicore__ inline void SetEvtFlag()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<evt>(eventId);
    WaitFlag<evt>(eventId);
}

__aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return (value + align - 1) / align * align;
}

__aicore__ inline uint32_t GetParamOffset(uint32_t logicalRow, uint32_t inputRows, uint32_t groupCount)
{
    if (groupCount <= 1) {
        return logicalRow;
    }
    const uint32_t row = logicalRow / groupCount;
    const uint32_t group = logicalRow - row * groupCount;
    return group * inputRows + row;
}

__aicore__ inline uint32_t GetGroupOffset(uint32_t logicalRow, uint32_t groupCount, uint32_t rowSize)
{
    if (groupCount <= 1) {
        return 0;
    }
    const uint32_t row = logicalRow / groupCount;
    const uint32_t group = logicalRow - row * groupCount;
    return group * rowSize;
}

__aicore__ inline void ComputeSigmoid(LocalTensor<float> out, LocalTensor<float> in, LocalTensor<float> tmp, uint32_t count)
{
#if defined(ASCENDC_LAYER_NORM_FWD_USE_SIGMOID_API)
    Sigmoid(out, in, count);
    PipeBarrier<PIPE_V>();
#elif defined(ASCENDC_LAYER_NORM_FWD_USE_DIV_SIGMOID)
    Neg(out, in, count);
    PipeBarrier<PIPE_V>();
    Exp(out, out, count);
    PipeBarrier<PIPE_V>();
    Adds(out, out, 1.0f, count);
    PipeBarrier<PIPE_V>();
    Muls(tmp, in, 0.0f, count);
    PipeBarrier<PIPE_V>();
    Adds(tmp, tmp, 1.0f, count);
    PipeBarrier<PIPE_V>();
    Div(out, tmp, out, count);
    PipeBarrier<PIPE_V>();
#else
    Neg(out, in, count);
    PipeBarrier<PIPE_V>();
    Exp(out, out, count);
    PipeBarrier<PIPE_V>();
    Adds(out, out, 1.0f, count);
    PipeBarrier<PIPE_V>();
    Reciprocal(out, out, count);
    PipeBarrier<PIPE_V>();
#endif
}

} // namespace LayerNormFwd

#endif // XLLM_LAYER_NORM_FWD_COMMON_H
