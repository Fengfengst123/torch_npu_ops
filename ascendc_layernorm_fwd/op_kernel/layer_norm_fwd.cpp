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

#include "kernel_operator.h"
#include "layer_norm_fwd_single_read.h"
#include "layer_norm_fwd_single_read_nomask.h"
#include "layer_norm_fwd_single_read_z.h"

using namespace LayerNormFwd;
using namespace optiling;

#define INVOKE_LAYER_NORM_FWD_SINGLE_READ(Tfm, Tweight)                  \
    do {                                                                 \
        GET_TILING_DATA_WITH_STRUCT(LayerNormFwdTilingData, tilingData, tiling); \
        LayerNormFwdSingleRead<Tfm, Tweight> op;                         \
        op.Init(x, gamma, beta, z, y, mean, rstd, workspace, &tilingData); \
        op.Process();                                                    \
    } while (0)

#define INVOKE_LAYER_NORM_FWD_SINGLE_READ_Z(Tfm, Tweight)                \
    do {                                                                 \
        GET_TILING_DATA_WITH_STRUCT(LayerNormFwdTilingData, tilingData, tiling); \
        LayerNormFwdSingleReadZ<Tfm, Tweight> op;                        \
        op.Init(x, gamma, beta, z, y, mean, rstd, workspace, &tilingData); \
        op.Process();                                                    \
    } while (0)

#define INVOKE_LAYER_NORM_FWD_NOMASK(Tfm, Tweight)                       \
    do {                                                                 \
        GET_TILING_DATA_WITH_STRUCT(LayerNormFwdTilingData, tilingData, tiling); \
        LayerNormFwdSingleReadNomask<Tfm, Tweight> op;                   \
        op.Init(x, gamma, beta, z, y, mean, rstd, workspace, &tilingData); \
        op.Process();                                                    \
    } while (0)

extern "C" __global__ __aicore__ void layer_norm_fwd(
    GM_ADDR x,
    GM_ADDR gamma,
    GM_ADDR beta,
    GM_ADDR z,
    GM_ADDR y,
    GM_ADDR mean,
    GM_ADDR rstd,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    if (g_coreType == AIC) {
        return;
    }
#endif
    if (TILING_KEY_IS(100)) {
        INVOKE_LAYER_NORM_FWD_SINGLE_READ(float, float);
        return;
    }
    if (TILING_KEY_IS(120)) {
        INVOKE_LAYER_NORM_FWD_SINGLE_READ(bfloat16_t, float);
        return;
    }
    if (TILING_KEY_IS(122)) {
        INVOKE_LAYER_NORM_FWD_SINGLE_READ(bfloat16_t, bfloat16_t);
        return;
    }
    if (TILING_KEY_IS(200)) {
        INVOKE_LAYER_NORM_FWD_SINGLE_READ_Z(float, float);
        return;
    }
    if (TILING_KEY_IS(220)) {
        INVOKE_LAYER_NORM_FWD_SINGLE_READ_Z(bfloat16_t, float);
        return;
    }
    if (TILING_KEY_IS(222)) {
        INVOKE_LAYER_NORM_FWD_SINGLE_READ_Z(bfloat16_t, bfloat16_t);
        return;
    }
    if (TILING_KEY_IS(300)) {
        INVOKE_LAYER_NORM_FWD_NOMASK(float, float);
        return;
    }
    if (TILING_KEY_IS(320)) {
        INVOKE_LAYER_NORM_FWD_NOMASK(bfloat16_t, float);
        return;
    }
    if (TILING_KEY_IS(322)) {
        INVOKE_LAYER_NORM_FWD_NOMASK(bfloat16_t, bfloat16_t);
        return;
    }
}
