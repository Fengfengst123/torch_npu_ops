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

#include <torch/torch.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace xllm::kernel::npu {

torch::Tensor npu_l2norm_last_dim(torch::Tensor& x, double eps = 1e-6);

// Chunk gated delta rule pipeline operators
torch::Tensor npu_chunk_local_cumsum(
    const torch::Tensor& g,
    int64_t chunk_size,
    const std::optional<torch::Tensor>& cu_seqlens);

torch::Tensor npu_chunk_scaled_dot_kkt_fwd(
    const torch::Tensor& k,
    const torch::Tensor& beta,
    const torch::Tensor& g_cumsum,
    int64_t chunk_size,
    const std::optional<torch::Tensor>& cu_seqlens);

torch::Tensor npu_solve_tril(const torch::Tensor& A,
                             int64_t chunk_size,
                             const std::optional<torch::Tensor>& cu_seqlens,
                             torch::ScalarType output_dtype);

std::pair<torch::Tensor, torch::Tensor> npu_recompute_w_u_fwd(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& beta,
    const torch::Tensor& g_cumsum,
    const torch::Tensor& A,
    const std::optional<torch::Tensor>& cu_seqlens);

torch::Tensor npu_chunk_fwd_o(const torch::Tensor& q,
                              const torch::Tensor& k,
                              const torch::Tensor& v,
                              const torch::Tensor& h,
                              const torch::Tensor& g_cumsum,
                              float scale,
                              int64_t chunk_size,
                              const std::optional<torch::Tensor>& cu_seqlens);

void rope_inplace(torch::Tensor& x,
                  torch::Tensor& sin,
                  torch::Tensor& cos,
                  uint32_t rope_dim = 64);

std::pair<torch::Tensor, torch::Tensor> npu_fused_gdn_gating(
    torch::Tensor& A_log,
    torch::Tensor& a,
    torch::Tensor& b,
    torch::Tensor& dt_bias,
    float beta = 1.0f,
    float threshold = 20.0f);

std::pair<torch::Tensor, torch::Tensor> npu_fused_recurrent_gated_delta_rule(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& g,
    const std::optional<torch::Tensor>& beta = std::nullopt,
    const std::optional<float>& scale = std::nullopt,
    const std::optional<torch::Tensor>& initial_state = std::nullopt,
    bool inplace_final_state = true,
    const std::optional<torch::Tensor>& cu_seqlens = std::nullopt,
    const std::optional<torch::Tensor>& ssm_state_indices = std::nullopt,
    const std::optional<torch::Tensor>& num_accepted_tokens = std::nullopt,
    bool use_qk_l2norm_in_kernel = false);

std::pair<torch::Tensor, torch::Tensor> npu_chunk_gated_delta_rule(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& g,
    torch::Tensor& beta,
    const std::optional<float>& scale = std::nullopt,
    const std::optional<torch::Tensor>& initial_state = std::nullopt,
    bool output_final_state = false,
    const std::optional<torch::Tensor>& cu_seqlens = std::nullopt,
    bool head_first = false,
    bool use_qk_l2norm_in_kernel = false);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
npu_chunk_gated_delta_rule_fwd_h(
    torch::Tensor& k,
    torch::Tensor& w,
    torch::Tensor& u,
    const std::optional<torch::Tensor>& g = std::nullopt,
    const std::optional<torch::Tensor>& initial_state = std::nullopt,
    bool output_final_state = false,
    int64_t chunk_size = 64,
    bool save_new_value = true,
    const std::optional<torch::Tensor>& cu_seqlens = std::nullopt,
    const std::optional<torch::Tensor>& chunk_offsets = std::nullopt);

torch::Tensor layer_norm_fwd(
    torch::Tensor& x,
    torch::Tensor& weight,
    torch::Tensor& bias,
    double eps,
    const std::optional<torch::Tensor>& z = std::nullopt,
    int64_t group_size = -1,
    bool norm_before_gate = true,
    bool is_rms_norm = false);

torch::Tensor npu_causal_conv1d_update(
    torch::Tensor& x,
    torch::Tensor& conv_state,
    torch::Tensor& weight,
    bool activation = true,
    const std::optional<torch::Tensor>& bias = std::nullopt,
    const std::optional<torch::Tensor>& cache_seqlens = std::nullopt,
    const std::optional<torch::Tensor>& conv_state_indices = std::nullopt,
    const std::optional<torch::Tensor>& num_accepted_tokens = std::nullopt,
    const std::optional<torch::Tensor>& query_start_loc = std::nullopt,
    int32_t max_query_len = -1,
    const std::optional<torch::Tensor>& intermediate_conv_window = std::nullopt,
    int32_t pad_slot_id = -1,
    bool validate_data = false);

torch::Tensor npu_causal_conv1d_update_v2(
    torch::Tensor& x,
    torch::Tensor& conv_state,
    torch::Tensor& weight,
    bool activation = true,
    const std::optional<torch::Tensor>& bias = std::nullopt,
    const std::optional<torch::Tensor>& conv_state_indices = std::nullopt,
    const std::optional<torch::Tensor>& query_start_loc = std::nullopt,
    int32_t max_query_len = -1,
    int32_t pad_slot_id = -1,
    const std::optional<torch::Tensor>& block_idx_last_scheduled_token =
        std::nullopt,
    const std::optional<torch::Tensor>& initial_state_idx = std::nullopt,
    bool validate_data = false,
    const std::optional<torch::Tensor>& num_accepted_tokens = std::nullopt);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
npu_fused_qkvzba_split_reshape_cat(torch::Tensor& mixed_qkvz,
                                   torch::Tensor& mixed_ba,
                                   int32_t num_heads_qk,
                                   int32_t num_heads_v,
                                   int32_t head_qk,
                                   int32_t head_v);

}  // namespace xllm::kernel::npu
