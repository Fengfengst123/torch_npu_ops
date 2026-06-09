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

#include "layer_norm_fwd.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(LayerNormFwd);

std::array<aclTensor*, LAYER_NORM_FWD_OUT_NUM> LayerNormFwd(
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
    aclOpExecutor* executor)
{
    L0_DFX(LayerNormFwd, x, gamma, beta, z, y, mean, rstd, eps, groupSize, normBeforeGate, isRmsNorm);
    ADD_TO_LAUNCHER_LIST_AICORE(
        LayerNormFwd,
        OP_INPUT(x, gamma, beta, z),
        OP_OUTPUT(y, mean, rstd),
        OP_ATTR(static_cast<float>(eps), groupSize, normBeforeGate, isRmsNorm));
    return {y, mean, rstd};
}

} // namespace l0op
