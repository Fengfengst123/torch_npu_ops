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

#include "layer_norm_fwd_tiling.h"

#include "log/log.h"
#include "platform/platform_info.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"
#include "util/math_util.h"

namespace {
constexpr size_t kXIndex = 0;
constexpr size_t kGammaIndex = 1;
constexpr size_t kBetaIndex = 2;
constexpr size_t kZIndex = 3;
constexpr size_t kEpsAttrIndex = 0;
constexpr size_t kGroupSizeAttrIndex = 1;
constexpr size_t kNormBeforeGateAttrIndex = 2;
constexpr size_t kIsRmsNormAttrIndex = 3;
constexpr uint32_t kBlockBytes = 32;
constexpr uint32_t kFp32Bytes = 4;
constexpr uint64_t kSingleReadKey = 100;
constexpr uint64_t kSingleReadZKey = 200;
constexpr uint64_t kNomaskKey = 300;
constexpr uint64_t kFp32DtypeKey = 0;
constexpr uint64_t kBf16DtypeKey = 20;
} // namespace

namespace optiling {

struct LayerNormFwdCompileInfo {
};

static uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return (value + align - 1) / align * align;
}

static uint64_t GetDtypeKey(ge::DataType dtype)
{
    if (dtype == ge::DT_BF16) {
        return kBf16DtypeKey;
    }
    return kFp32DtypeKey;
}

static ge::graphStatus TilingLayerNormFwd(gert::TilingContext* context)
{
    const auto xShape = context->GetInputShape(kXIndex)->GetStorageShape();
    OP_CHECK_IF(xShape.GetDimNum() != 2,
        OP_LOGE(context, "LayerNormFwd expects x to be a 2D tensor after xllm wrapper reshape."),
        return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t inputRows = xShape.GetDim(0);
    const int64_t inputCols = xShape.GetDim(1);
    int64_t groupSize = *attrs->GetAttrPointer<int64_t>(kGroupSizeAttrIndex);
    if (groupSize <= 0) {
        groupSize = inputCols;
    }
    OP_CHECK_IF(groupSize <= 0 || inputCols % groupSize != 0,
        OP_LOGE(context, "LayerNormFwd requires inputCols %% group_size == 0."),
        return ge::GRAPH_FAILED);

    const uint32_t groupCount = static_cast<uint32_t>(inputCols / groupSize);
    const uint32_t logicalRows = static_cast<uint32_t>(inputRows * groupCount);
    const uint32_t rowSize = static_cast<uint32_t>(groupSize);
    const uint32_t rowAlign = AlignUp(rowSize, kBlockBytes / kFp32Bytes);
    const uint32_t nRow = 1;

    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (coreNum == 0) {
        coreNum = 1;
    }
    uint32_t numBlocks = logicalRows < coreNum ? logicalRows : coreNum;
    if (numBlocks == 0) {
        numBlocks = 1;
    }

    const uint32_t blockCount = logicalRows / nRow;
    const uint32_t tailNRow = logicalRows - blockCount * nRow;
    const uint32_t loopCount = blockCount / numBlocks;
    const uint32_t tailLoop = blockCount - loopCount * numBlocks;

    const bool hasZ = context->GetOptionalInputDesc(kZIndex) != nullptr;
    const bool hasGamma = context->GetOptionalInputDesc(kGammaIndex) != nullptr;
    const bool hasBeta = context->GetOptionalInputDesc(kBetaIndex) != nullptr;
    const bool isRmsNorm = *attrs->GetAttrPointer<bool>(kIsRmsNormAttrIndex);
    const bool normBeforeGate = *attrs->GetAttrPointer<bool>(kNormBeforeGateAttrIndex);
    const bool useNomask = (!hasZ && rowSize == 128 && inputRows >= 1024);

    LayerNormFwdTilingData tiling;
    tiling.set_numBlocks(numBlocks);
    tiling.set_colSize(logicalRows);
    tiling.set_rowSize(rowSize);
    tiling.set_nRow(nRow);
    tiling.set_tileLength(nRow * rowAlign);
    tiling.set_blockLength(nRow * rowSize);
    tiling.set_tailNRow(tailNRow);
    tiling.set_loopCount(loopCount);
    tiling.set_tailLoop(tailLoop);
    tiling.set_rowAlign(rowAlign);
    tiling.set_nullptrGamma(hasGamma ? 0 : 1);
    tiling.set_nullptrBeta(hasBeta ? 0 : 1);
    tiling.set_hasZ(hasZ ? 1 : 0);
    tiling.set_normBeforeGate(normBeforeGate ? 1 : 0);
    tiling.set_isRmsNorm(isRmsNorm ? 1 : 0);
    tiling.set_useNomask(useNomask ? 1 : 0);
    tiling.set_eps(*attrs->GetAttrPointer<float>(kEpsAttrIndex));
    tiling.set_coefficient(1.0f / static_cast<float>(rowSize));
    tiling.set_inputRows(static_cast<uint32_t>(inputRows));
    tiling.set_groupCount(groupCount);

    uint64_t keyBase = hasZ ? kSingleReadZKey : (useNomask ? kNomaskKey : kSingleReadKey);
    const auto xDtype = context->GetInputDesc(kXIndex)->GetDataType();
    const auto gammaDesc = context->GetOptionalInputDesc(kGammaIndex);
    const auto gammaDtype = gammaDesc == nullptr ? ge::DT_FLOAT : gammaDesc->GetDataType();
    uint64_t tilingKey = keyBase + GetDtypeKey(xDtype);
    if (xDtype == ge::DT_BF16 && gammaDtype == ge::DT_BF16) {
        tilingKey += 2;
    }

    context->SetBlockDim(numBlocks);
    context->SetTilingKey(tilingKey);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = platform.GetLibApiWorkSpaceSize();
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareLayerNormFwd(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LayerNormFwd)
    .Tiling(TilingLayerNormFwd)
    .TilingParse<LayerNormFwdCompileInfo>(TilingPrepareLayerNormFwd);

} // namespace optiling
