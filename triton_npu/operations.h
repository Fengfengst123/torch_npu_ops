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

#pragma once

#include <string>

#include "operation_base.h"

namespace xllm::kernel::npu {

class L2NormFwdOp final : public OperationBase {
 public:
  L2NormFwdOp() : OperationBase("l2norm_fwd_kernel2_loop") {}
};

class RopeInplaceOp final : public OperationBase {
 public:
  RopeInplaceOp() : OperationBase("rope_inplace_kernel") {}
};

class FusedGdnGatingOp final : public OperationBase {
 public:
  FusedGdnGatingOp() : OperationBase("fused_gdn_gating_decode_kernel") {}
};

class LayerNormFwdOp final : public OperationBase {
 public:
  LayerNormFwdOp() : OperationBase("layer_norm_fwd_kernel") {}
};

class RecurrentGatedDeltaRuleFwdOp final : public OperationBase {
 public:
  RecurrentGatedDeltaRuleFwdOp()
      : OperationBase("fused_recurrent_gated_delta_rule_fwd_kernel") {}
};

class ChunkGatedDeltaRuleFwdHOp final : public OperationBase {
 public:
  ChunkGatedDeltaRuleFwdHOp()
      : OperationBase("chunk_gated_delta_rule_fwd_kernel_h_blockdim64") {}
};

class ChunkLocalCumsumScalarOp final : public OperationBase {
 public:
  ChunkLocalCumsumScalarOp()
      : OperationBase("chunk_local_cumsum_scalar_kernel") {}
};

class ChunkScaledDotKktFwdOp final : public OperationBase {
 public:
  ChunkScaledDotKktFwdOp()
      : OperationBase("chunk_scaled_dot_kkt_fwd_kernel") {}
};

class SolveTril16x16Op final : public OperationBase {
 public:
  SolveTril16x16Op() : OperationBase("solve_tril_16x16_kernel") {}
};

class Merge16x16To64x64InverseOp final : public OperationBase {
 public:
  Merge16x16To64x64InverseOp()
      : OperationBase("merge_16x16_to_64x64_inverse_kernel") {}
};

class RecomputeWUForwardOp final : public OperationBase {
 public:
  RecomputeWUForwardOp() : OperationBase("recompute_w_u_fwd_kernel") {}
};

class ChunkForwardOOp final : public OperationBase {
 public:
  ChunkForwardOOp() : OperationBase("chunk_fwd_kernel_o") {}
};

class CausalConv1dUpdateNoCacheNoMtpOp final : public OperationBase {
 public:
  CausalConv1dUpdateNoCacheNoMtpOp()
      : OperationBase("_causal_conv1d_update_kernel_no_cache_len_no_mtp") {}
};

class CausalConv1dUpdateQwenDecodeOp final : public OperationBase {
 public:
  CausalConv1dUpdateQwenDecodeOp()
      : OperationBase("_causal_conv1d_update_qwen_decode_kernel") {}
};

class CausalConv1dUpdateTiledV2Op final : public OperationBase {
 public:
  CausalConv1dUpdateTiledV2Op()
      : OperationBase("_causal_conv1d_update_kernel_npu_tiled_v2") {}
};

class FusedQkvzbaSplitReshapeOp final : public OperationBase {
 public:
  FusedQkvzbaSplitReshapeOp()
      : OperationBase("fused_qkvzba_split_reshape_cat_kernel") {}
};

}  // namespace xllm::kernel::npu
