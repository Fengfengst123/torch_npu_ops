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

#ifndef XLLM_ACLNN_LAYER_NORM_FWD_H
#define XLLM_ACLNN_LAYER_NORM_FWD_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

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
    aclOpExecutor** executor);

aclnnStatus aclnnLayerNormFwd(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // XLLM_ACLNN_LAYER_NORM_FWD_H
