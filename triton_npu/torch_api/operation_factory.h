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

#include <memory>
#include <type_traits>
#include <unordered_map>

#include "operations.h"

namespace xllm::kernel::npu {

class OperationFactory final {
 public:
  static OperationFactory& instance() {
    static OperationFactory inst;
    return inst;
  }

  L2NormFwdOp& l2norm_fwd() {
    return get_or_create<L2NormFwdOp>("l2norm_fwd_kernel2_loop");
  }

  RopeInplaceOp& rope_inplace() {
    // input param is key for OperationFactory
    return get_or_create<RopeInplaceOp>("rope_inplace_kernel");
  }

  FusedGdnGatingOp& fused_gdn_gating() {
    return get_or_create<FusedGdnGatingOp>("fused_gdn_gating_decode_kernel");
  }

  LayerNormFwdOp& layer_norm_fwd() {
    return get_or_create<LayerNormFwdOp>("layer_norm_fwd_kernel");
  }

  LayerNormFwdFastOp& layer_norm_fwd_fast() {
    return get_or_create<LayerNormFwdFastOp>("layer_norm_fwd_kernel_fast");
  }

  LayerNormFwdFastRMSBiasOp& layer_norm_fwd_fast_rms_bias() {
    return get_or_create<LayerNormFwdFastRMSBiasOp>(
        "layer_norm_fwd_kernel_fast_rms_bias");
  }

  LayerNormFwdFastRMSNoBiasOp& layer_norm_fwd_fast_rms_nobias() {
    return get_or_create<LayerNormFwdFastRMSNoBiasOp>(
        "layer_norm_fwd_kernel_fast_rms_nobias");
  }

  LayerNormFwdFastBF16Op& layer_norm_fwd_fast_bf16() {
    return get_or_create<LayerNormFwdFastBF16Op>("layer_norm_fwd_kernel_fast_bf16");
  }

  LayerNormFwdFastRMSBF16BiasOp& layer_norm_fwd_fast_rms_bf16_bias() {
    return get_or_create<LayerNormFwdFastRMSBF16BiasOp>(
        "layer_norm_fwd_kernel_fast_rms_bf16_bias");
  }

  LayerNormFwdFastRMSBF16NoBiasOp& layer_norm_fwd_fast_rms_bf16_nobias() {
    return get_or_create<LayerNormFwdFastRMSBF16NoBiasOp>(
        "layer_norm_fwd_kernel_fast_rms_bf16_nobias");
  }

  LayerNormFwdFastZOp& layer_norm_fwd_fast_z() {
    return get_or_create<LayerNormFwdFastZOp>("layer_norm_fwd_kernel_fast_z");
  }

  LayerNormFwdFastRMSZBiasOp& layer_norm_fwd_fast_rms_z_bias() {
    return get_or_create<LayerNormFwdFastRMSZBiasOp>(
        "layer_norm_fwd_kernel_fast_rms_z_bias");
  }

  LayerNormFwdFastRMSZNoBiasOp& layer_norm_fwd_fast_rms_z_nobias() {
    return get_or_create<LayerNormFwdFastRMSZNoBiasOp>(
        "layer_norm_fwd_kernel_fast_rms_z_nobias");
  }

  LayerNormFwdFastBF16ZOp& layer_norm_fwd_fast_bf16_z() {
    return get_or_create<LayerNormFwdFastBF16ZOp>("layer_norm_fwd_kernel_fast_bf16_z");
  }

  LayerNormFwdFastRMSBF16ZBiasOp& layer_norm_fwd_fast_rms_bf16_z_bias() {
    return get_or_create<LayerNormFwdFastRMSBF16ZBiasOp>(
        "layer_norm_fwd_kernel_fast_rms_bf16_z_bias");
  }

  LayerNormFwdFastRMSBF16ZNoBiasOp& layer_norm_fwd_fast_rms_bf16_z_nobias() {
    return get_or_create<LayerNormFwdFastRMSBF16ZNoBiasOp>(
        "layer_norm_fwd_kernel_fast_rms_bf16_z_nobias");
  }

  RecurrentGatedDeltaRuleFwdOp& recurrent_gated_delta_rule_fwd() {
    return get_or_create<RecurrentGatedDeltaRuleFwdOp>(
        "fused_recurrent_gated_delta_rule_fwd_kernel");
  }

  RecurrentGatedDeltaRuleSpecFwdOp& recurrent_gated_delta_rule_spec_fwd() {
    return get_or_create<RecurrentGatedDeltaRuleSpecFwdOp>(
        "fused_recurrent_gated_delta_rule_spec_fwd_kernel");
  }

  FusedSigmoidGatingDeltaRuleUpdateOp& fused_sigmoid_gating_delta_rule_update() {
    return get_or_create<FusedSigmoidGatingDeltaRuleUpdateOp>(
        "fused_sigmoid_gating_delta_rule_update_kernel");
  }

  ChunkGatedDeltaRuleFwdHOp& chunk_gated_delta_rule_fwd_h() {
    return get_or_create<ChunkGatedDeltaRuleFwdHOp>(
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64");
  }

  ChunkLocalCumsumScalarOp& chunk_local_cumsum_scalar() {
    return get_or_create<ChunkLocalCumsumScalarOp>(
        "chunk_local_cumsum_scalar_kernel");
  }

  ChunkScaledDotKktFwdOp& chunk_scaled_dot_kkt_fwd() {
    return get_or_create<ChunkScaledDotKktFwdOp>(
        "chunk_scaled_dot_kkt_fwd_kernel");
  }

  SolveTril16x16Op& solve_tril_16x16() {
    return get_or_create<SolveTril16x16Op>("solve_tril_16x16_kernel");
  }

  Merge16x16To64x64InverseOp& merge_16x16_to_64x64_inverse() {
    return get_or_create<Merge16x16To64x64InverseOp>(
        "merge_16x16_to_64x64_inverse_kernel");
  }

  RecomputeWUForwardOp& recompute_w_u_fwd() {
    return get_or_create<RecomputeWUForwardOp>("recompute_w_u_fwd_kernel");
  }

  ChunkForwardOOp& chunk_fwd_o() {
    return get_or_create<ChunkForwardOOp>("chunk_fwd_kernel_o");
  }

  CausalConv1dUpdateNoCacheNoMtpOp& causal_conv1d_update_no_cache_no_mtp() {
    return get_or_create<CausalConv1dUpdateNoCacheNoMtpOp>(
        "_causal_conv1d_update_kernel_no_cache_len_no_mtp");
  }

  CausalConv1dUpdateQwenDecodeOp& causal_conv1d_update_qwen_decode() {
    return get_or_create<CausalConv1dUpdateQwenDecodeOp>(
        "_causal_conv1d_update_qwen_decode_kernel");
  }

  CausalConv1dUpdateTiledV2Op& causal_conv1d_update_tiled_v2() {
    return get_or_create<CausalConv1dUpdateTiledV2Op>(
        "_causal_conv1d_update_kernel_npu_tiled_v2");
  }

  FusedQkvzbaSplitReshapeGqaR1Op& fused_qkvzba_split_reshape_gqa_r1() {
    return get_or_create<FusedQkvzbaSplitReshapeGqaR1Op>(
        "fused_qkvzba_split_reshape_cat_gqa_r1_kernel");
  }

  FusedQkvzbaSplitReshapeGqaR2Op& fused_qkvzba_split_reshape_gqa_r2() {
    return get_or_create<FusedQkvzbaSplitReshapeGqaR2Op>(
        "fused_qkvzba_split_reshape_cat_gqa_r2_kernel");
  }

  FusedQkvzbaSplitReshapeGqaR3Op& fused_qkvzba_split_reshape_gqa_r3() {
    return get_or_create<FusedQkvzbaSplitReshapeGqaR3Op>(
        "fused_qkvzba_split_reshape_cat_gqa_r3_kernel");
  }

  FusedQkvzbaSplitReshapeGqaR4Op& fused_qkvzba_split_reshape_gqa_r4() {
    return get_or_create<FusedQkvzbaSplitReshapeGqaR4Op>(
        "fused_qkvzba_split_reshape_cat_gqa_r4_kernel");
  }

 private:
  OperationFactory() = default;

  template <class T>
  T& get_or_create(const char* key) {
    static_assert(std::is_base_of_v<OperationBase, T>,
                  "T must derive from OperationBase");
    auto it = ops_.find(key);
    if (it != ops_.end()) {
      return *static_cast<T*>(it->second.get());
    }
    auto p = std::make_unique<T>();
    T* raw = p.get();
    ops_.emplace(key, std::move(p));
    return raw[0];
  }

  std::unordered_map<std::string, std::unique_ptr<OperationBase>> ops_;
};

}  // namespace xllm::kernel::npu
