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

#include "error_util.h"
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"

namespace {
constexpr size_t kXIndex = 0;
constexpr size_t kYIndex = 0;
constexpr size_t kMeanIndex = 1;
constexpr size_t kRstdIndex = 2;
constexpr size_t kGroupSizeAttrIndex = 1;
constexpr size_t kIsRmsNormAttrIndex = 3;
} // namespace

namespace ops {

static ge::graphStatus InferShapeLayerNormFwd(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(kXIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(kYIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape* rstdShape = context->GetOutputShape(kRstdIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context, rstdShape);

    *yShape = *xShape;
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    int64_t groupSize = *attrs->GetAttrPointer<int64_t>(kGroupSizeAttrIndex);
    const bool isRmsNorm = *attrs->GetAttrPointer<bool>(kIsRmsNormAttrIndex);
    const int64_t inputRows = xShape->GetDim(0);
    const int64_t inputCols = xShape->GetDim(1);
    if (groupSize <= 0) {
        groupSize = inputCols;
    }
    const int64_t groupCount = inputCols / groupSize;
    const int64_t paramRows = inputRows * groupCount;

    rstdShape->SetDimNum(1);
    rstdShape->SetDim(0, paramRows);
    gert::Shape* meanShape = context->GetOutputShape(kMeanIndex);
    if (meanShape != nullptr) {
        meanShape->SetDimNum(1);
        meanShape->SetDim(0, isRmsNorm ? 0 : paramRows);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeLayerNormFwd(gert::InferDataTypeContext* context)
{
    const auto xDtype = context->GetInputDataType(kXIndex);
    context->SetOutputDataType(kYIndex, xDtype);
    context->SetOutputDataType(kMeanIndex, ge::DT_FLOAT);
    context->SetOutputDataType(kRstdIndex, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LayerNormFwd)
    .InferShape(InferShapeLayerNormFwd)
    .InferDataType(InferDtypeLayerNormFwd);

} // namespace ops
