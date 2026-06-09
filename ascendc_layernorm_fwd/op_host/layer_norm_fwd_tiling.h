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

#ifndef XLLM_LAYER_NORM_FWD_TILING_H
#define XLLM_LAYER_NORM_FWD_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(LayerNormFwdTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numBlocks);
TILING_DATA_FIELD_DEF(uint32_t, colSize);
TILING_DATA_FIELD_DEF(uint32_t, rowSize);
TILING_DATA_FIELD_DEF(uint32_t, nRow);
TILING_DATA_FIELD_DEF(uint32_t, tileLength);
TILING_DATA_FIELD_DEF(uint32_t, blockLength);
TILING_DATA_FIELD_DEF(uint32_t, tailNRow);
TILING_DATA_FIELD_DEF(uint32_t, loopCount);
TILING_DATA_FIELD_DEF(uint32_t, tailLoop);
TILING_DATA_FIELD_DEF(uint32_t, rowAlign);
TILING_DATA_FIELD_DEF(uint32_t, nullptrGamma);
TILING_DATA_FIELD_DEF(uint32_t, nullptrBeta);
TILING_DATA_FIELD_DEF(uint32_t, hasZ);
TILING_DATA_FIELD_DEF(uint32_t, normBeforeGate);
TILING_DATA_FIELD_DEF(uint32_t, isRmsNorm);
TILING_DATA_FIELD_DEF(uint32_t, useNomask);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(float, coefficient);
TILING_DATA_FIELD_DEF(uint32_t, inputRows);
TILING_DATA_FIELD_DEF(uint32_t, groupCount);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormFwd_100, LayerNormFwdTilingData)
REGISTER_TILING_DATA_CLASS(LayerNormFwd_120, LayerNormFwdTilingData)
REGISTER_TILING_DATA_CLASS(LayerNormFwd_122, LayerNormFwdTilingData)
REGISTER_TILING_DATA_CLASS(LayerNormFwd_200, LayerNormFwdTilingData)
REGISTER_TILING_DATA_CLASS(LayerNormFwd_220, LayerNormFwdTilingData)
REGISTER_TILING_DATA_CLASS(LayerNormFwd_222, LayerNormFwdTilingData)
REGISTER_TILING_DATA_CLASS(LayerNormFwd_300, LayerNormFwdTilingData)
REGISTER_TILING_DATA_CLASS(LayerNormFwd_320, LayerNormFwdTilingData)
REGISTER_TILING_DATA_CLASS(LayerNormFwd_322, LayerNormFwdTilingData)

} // namespace optiling

#endif // XLLM_LAYER_NORM_FWD_TILING_H
