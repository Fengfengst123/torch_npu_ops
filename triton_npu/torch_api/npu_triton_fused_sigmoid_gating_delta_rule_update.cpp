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

#include "operation_factory.h"
#include "triton_ops_api.h"

#include <vector>

namespace xllm::kernel::npu {
namespace {

int64_t next_power_of_2(int64_t x) {
  if (x <= 1) return 1;
  TORCH_CHECK(x > 0 && x <= (int64_t(1) << 62),
              "x too large for next_power_of_2");
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  return x + 1;
}

int64_t cdiv(int64_t a, int64_t div) {
  TORCH_CHECK(div > 0, "cdiv divisor must be positive");
  return (a + div - 1) / div;
}

void check_npu_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
              name,
              " must be on NPU");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor npu_fused_sigmoid_gating_delta_rule_update(
    torch::Tensor& A_log,
    torch::Tensor& a,
    torch::Tensor& dt_bias,
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& b,
    torch::Tensor& initial_state_source,
    torch::Tensor& initial_state_indices,
    torch::Tensor& cu_seqlens,
    const std::optional<float>& scale,
    bool use_qk_l2norm_in_kernel,
    float softplus_beta,
    float softplus_threshold) {
  check_npu_contiguous(A_log, "A_log");
  check_npu_contiguous(a, "a");
  check_npu_contiguous(dt_bias, "dt_bias");
  check_npu_contiguous(q, "q");
  check_npu_contiguous(k, "k");
  check_npu_contiguous(v, "v");
  check_npu_contiguous(b, "b");
  check_npu_contiguous(initial_state_source, "initial_state_source");
  check_npu_contiguous(initial_state_indices, "initial_state_indices");
  check_npu_contiguous(cu_seqlens, "cu_seqlens");

  TORCH_CHECK(q.dim() == 4, "q must be 4D [B, T, H, K]");
  TORCH_CHECK(k.dim() == 4, "k must be 4D [B, T, H, K]");
  TORCH_CHECK(v.dim() == 4, "v must be 4D [B, T, HV, V]");
  TORCH_CHECK(a.dim() == 3, "a must be 3D [B, T, HV]");
  TORCH_CHECK(b.dim() == 3, "b must be 3D [B, T, HV]");
  TORCH_CHECK(A_log.dim() == 1, "A_log must be 1D [HV]");
  TORCH_CHECK(dt_bias.dim() == 1, "dt_bias must be 1D [HV]");
  TORCH_CHECK(initial_state_source.dim() == 4,
              "initial_state_source must be 4D [slots, HV, K, V]");
  TORCH_CHECK(initial_state_indices.dim() == 1,
              "initial_state_indices must be 1D [N]");
  TORCH_CHECK(cu_seqlens.dim() == 1, "cu_seqlens must be 1D [N + 1]");

  auto k_shape = k.sizes();
  auto v_shape = v.sizes();
  const int64_t batch = k_shape[0];
  const int64_t seq = k_shape[1];
  const int64_t num_k_heads = k_shape[2];
  const int64_t k_head_dim = k_shape[3];
  const int64_t num_v_heads = v_shape[2];
  const int64_t v_head_dim = v_shape[3];
  const int64_t num_sequences = cu_seqlens.numel() - 1;

  TORCH_CHECK(q.sizes() == k.sizes(), "q/k shape mismatch");
  TORCH_CHECK(v.size(0) == batch && v.size(1) == seq, "v B/T mismatch");
  TORCH_CHECK(a.size(0) == batch && a.size(1) == seq &&
                  a.size(2) == num_v_heads,
              "a shape mismatch");
  TORCH_CHECK(b.sizes() == a.sizes(), "a/b shape mismatch");
  TORCH_CHECK(A_log.numel() == num_v_heads, "A_log HV mismatch");
  TORCH_CHECK(dt_bias.numel() == num_v_heads, "dt_bias HV mismatch");
  TORCH_CHECK(initial_state_source.size(1) == num_v_heads &&
                  initial_state_source.size(2) == k_head_dim &&
                  initial_state_source.size(3) == v_head_dim,
              "initial_state_source shape mismatch");
  TORCH_CHECK(initial_state_indices.numel() == num_sequences,
              "initial_state_indices size must equal cu_seqlens size - 1");
  TORCH_CHECK(num_v_heads % num_k_heads == 0,
              "HV must be divisible by H");
  TORCH_CHECK(A_log.dtype() == torch::kFloat32, "A_log must be float32");
  TORCH_CHECK(dt_bias.dtype() == torch::kFloat32, "dt_bias must be float32");
  TORCH_CHECK(initial_state_source.dtype() == torch::kFloat32,
              "initial_state_source must be float32");
  TORCH_CHECK(initial_state_indices.dtype() == torch::kInt32,
              "initial_state_indices must be int32");
  TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32,
              "cu_seqlens must be int32");

  const float scale_value =
      scale.has_value() ? scale.value()
                        : 1.0f / std::sqrt(static_cast<float>(k_head_dim));
  const int64_t BK = next_power_of_2(k_head_dim);
  const int64_t BV = std::min(next_power_of_2(v_head_dim), static_cast<int64_t>(64));
  const int64_t NK = cdiv(k_head_dim, BK);
  const int64_t NV = cdiv(v_head_dim, BV);
  TORCH_CHECK(NK == 1, "NK > 1 is not supported yet");

  std::vector<int64_t> o_shape{NK};
  o_shape.insert(o_shape.end(), v_shape.begin(), v_shape.end());
  torch::Tensor output = torch::empty(
      o_shape, torch::TensorOptions().dtype(q.dtype()).device(q.device()));

  auto npu_stream = c10_npu::getCurrentNPUStream();
  rtStream_t stream = static_cast<rtStream_t>(npu_stream.stream());

  auto& op = OperationFactory::instance().fused_sigmoid_gating_delta_rule_update();
  auto ret = op.execute(
      stream,
      static_cast<int32_t>(NK),
      static_cast<int32_t>(NV),
      static_cast<int32_t>(num_sequences * num_v_heads),
      [&](ArgsBuilder& ab) {
        ab.constructArgs(A_log.data_ptr(),
                         a.data_ptr(),
                         dt_bias.data_ptr(),
                         softplus_beta,
                         softplus_threshold,
                         q.data_ptr(),
                         k.data_ptr(),
                         v.data_ptr(),
                         b.data_ptr(),
                         output.data_ptr(),
                         initial_state_source.data_ptr(),
                         initial_state_indices.data_ptr(),
                         cu_seqlens.data_ptr(),
                         scale_value,
                         static_cast<int64_t>(batch),
                         static_cast<int64_t>(seq),
                         static_cast<int64_t>(num_k_heads),
                         static_cast<int64_t>(num_v_heads));
      });
  if (ret != RT_ERROR_NONE) {
    LOG(ERROR) << "rtKernelLaunch failed for "
                  "'fused_sigmoid_gating_delta_rule_update_kernel': "
               << ret;
  }
  return output.squeeze(0);
}

}  // namespace xllm::kernel::npu
