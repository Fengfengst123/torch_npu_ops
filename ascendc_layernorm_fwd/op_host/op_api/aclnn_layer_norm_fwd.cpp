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

#include "aclnn_layer_norm_fwd.h"

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "layer_norm_fwd.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* x, aclTensor* y, aclTensor* rstd)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(rstd, return false);
    return true;
}

static bool CheckDtype(const aclTensor* x, const aclTensor* gamma, const aclTensor* beta, const aclTensor* z, aclTensor* y)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SAME(x, y, return false);
    if (z != nullptr) {
        OP_CHECK_DTYPE_NOT_SAME(x, z, return false);
    }
    if (gamma != nullptr && gamma->GetDataType() != DataType::DT_FLOAT) {
        OP_CHECK_DTYPE_NOT_SAME(x, gamma, return false);
    }
    if (beta != nullptr && beta->GetDataType() != DataType::DT_FLOAT) {
        OP_CHECK_DTYPE_NOT_SAME(x, beta, return false);
    }
    if (gamma != nullptr && beta != nullptr) {
        OP_CHECK_DTYPE_NOT_SAME(gamma, beta, return false);
    }
    return true;
}

static bool CheckShape(
    const aclTensor* x,
    const aclTensor* gamma,
    const aclTensor* beta,
    const aclTensor* z,
    const aclTensor* y,
    const aclTensor* mean,
    const aclTensor* rstd,
    int64_t groupSize,
    bool isRmsNorm)
{
    OP_CHECK_WRONG_DIMENSION(x, 2, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(x, y, return false);
    const auto xShape = x->GetViewShape();
    const int64_t rows = xShape.GetDim(0);
    const int64_t cols = xShape.GetDim(1);
    if (groupSize <= 0) {
        groupSize = cols;
    }
    if (groupSize <= 0 || cols % groupSize != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groupSize must divide the last x dimension.");
        return false;
    }
    const int64_t paramRows = rows * (cols / groupSize);
    if (gamma != nullptr && gamma->GetViewShape().GetDim(0) != cols) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gamma shape must be [x.size(1)].");
        return false;
    }
    if (beta != nullptr && beta->GetViewShape().GetDim(0) != cols) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "beta shape must be [x.size(1)].");
        return false;
    }
    if (z != nullptr) {
        OP_CHECK_SHAPE_NOT_EQUAL(x, z, return false);
    }
    if (!isRmsNorm) {
        OP_CHECK_NULL(mean, return false);
        if (mean->GetViewShape().GetDim(0) != paramRows) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mean shape must be [M * group_count].");
            return false;
        }
    }
    if (rstd->GetViewShape().GetDim(0) != paramRows) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "rstd shape must be [M * group_count].");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* x,
    const aclTensor* gamma,
    const aclTensor* beta,
    const aclTensor* z,
    aclTensor* y,
    aclTensor* mean,
    aclTensor* rstd,
    int64_t groupSize,
    bool isRmsNorm)
{
    CHECK_RET(CheckNotNull(x, y, rstd), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtype(x, gamma, beta, z, y), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(x, gamma, beta, z, y, mean, rstd, groupSize, isRmsNorm), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnLayerNormFwdGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* gamma,
    const aclTensor* beta,
    const aclTensor* z,
    aclTensor* y,
    aclTensor* mean,
    aclTensor* rstd,
    double eps,
    int64_t groupSize,
    bool normBeforeGate,
    bool isRmsNorm,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnLayerNormFwd,
        DFX_IN(x, gamma, beta, z, eps, groupSize, normBeforeGate, isRmsNorm),
        DFX_OUT(y, mean, rstd));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(x, gamma, beta, z, y, mean, rstd, groupSize, isRmsNorm);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor* xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* gammaContiguous = gamma == nullptr ? nullptr : l0op::Contiguous(gamma, uniqueExecutor.get());
    if (gamma != nullptr) {
        CHECK_RET(gammaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor* betaContiguous = beta == nullptr ? nullptr : l0op::Contiguous(beta, uniqueExecutor.get());
    if (beta != nullptr) {
        CHECK_RET(betaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor* zContiguous = z == nullptr ? nullptr : l0op::Contiguous(z, uniqueExecutor.get());
    if (z != nullptr) {
        CHECK_RET(zContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    l0op::LayerNormFwd(
        xContiguous,
        gammaContiguous,
        betaContiguous,
        zContiguous,
        y,
        mean,
        rstd,
        eps,
        groupSize,
        normBeforeGate,
        isRmsNorm,
        uniqueExecutor.get());

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLayerNormFwd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLayerNormFwd);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
