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

#include <algorithm>
#include <string>

#include "operation_factory.h"
#include "triton_ops_api.h"

namespace xllm::kernel::npu {
namespace {

constexpr int32_t kVectorCoreNum = 32;

std::string format_kernel_eps(double eps) {
  std::string eps_str = std::to_string(eps);
  eps_str.erase(std::remove(eps_str.begin(), eps_str.end(), '.'),
                eps_str.end());
  eps_str.erase(std::remove(eps_str.begin(), eps_str.end(), 'e'),
                eps_str.end());
  eps_str.erase(std::remove(eps_str.begin(), eps_str.end(), '-'),
                eps_str.end());
  return eps_str;
}

std::string make_split_rmsnorm_rope_kernel_name(int64_t q_hidden_size,
                                                int64_t kv_hidden_size,
                                                int64_t head_dim,
                                                double eps,
                                                bool bias) {
  return "split_rmsnorm_rope_kernel_bias" + std::string(bias ? "1" : "0") +
         "_eps" + format_kernel_eps(eps) + "_hd" +
         std::to_string(head_dim) + "_qh" + std::to_string(q_hidden_size) +
         "_kvh" + std::to_string(kv_hidden_size);
}

bool is_power_of_two(int64_t value) {
  return value > 0 && (value & (value - 1)) == 0;
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
npu_split_rmsnorm_rope(torch::Tensor& input,
                       torch::Tensor& sin,
                       torch::Tensor& cos,
                       torch::Tensor& q_weight,
                       torch::Tensor& k_weight,
                       int64_t q_hidden_size,
                       int64_t kv_hidden_size,
                       int64_t head_dim,
                       double eps,
                       const std::optional<torch::Tensor>& q_bias,
                       const std::optional<torch::Tensor>& k_bias,
                       bool bias) {
  TORCH_CHECK(input.dim() == 2,
              "split_rmsnorm_rope input must be a 2D tensor");
  TORCH_CHECK(sin.dim() == 2 && cos.dim() == 2,
              "split_rmsnorm_rope sin and cos must be 2D tensors");
  TORCH_CHECK(q_weight.dim() == 1 && k_weight.dim() == 1,
              "split_rmsnorm_rope q_weight and k_weight must be 1D tensors");
  TORCH_CHECK(is_power_of_two(head_dim),
              "split_rmsnorm_rope head_dim must be a power of two");
  TORCH_CHECK(q_hidden_size % kv_hidden_size == 0,
              "split_rmsnorm_rope q_hidden_size must be divisible by "
              "kv_hidden_size");
  TORCH_CHECK(kv_hidden_size % head_dim == 0,
              "split_rmsnorm_rope kv_hidden_size must be divisible by "
              "head_dim");
  TORCH_CHECK(input.size(1) == q_hidden_size + 2 * kv_hidden_size,
              "split_rmsnorm_rope input hidden size mismatch");
  TORCH_CHECK(sin.size(0) == input.size(0) && cos.size(0) == input.size(0) &&
                  sin.size(1) == head_dim && cos.size(1) == head_dim,
              "split_rmsnorm_rope sin/cos shape mismatch");
  TORCH_CHECK(q_weight.size(0) == head_dim && k_weight.size(0) == head_dim,
              "split_rmsnorm_rope q/k weight shape mismatch");

  int32_t batch_size = static_cast<int32_t>(input.size(0));
  int32_t gridY = static_cast<int32_t>(kv_hidden_size / head_dim);
  TORCH_CHECK(gridY > 0, "split_rmsnorm_rope gridY must be positive");
  TORCH_CHECK(kVectorCoreNum % gridY == 0,
              "split_rmsnorm_rope gridY must divide vector core num");
  int32_t gridX = kVectorCoreNum / gridY;
  int32_t gridZ = 1;

  auto q_output = torch::empty({batch_size, q_hidden_size},
                               input.options());
  auto k_output = torch::empty({batch_size, kv_hidden_size},
                               input.options());
  auto v_output = torch::empty({batch_size, kv_hidden_size},
                               input.options());

  auto npuStream = c10_npu::getCurrentNPUStream();
  rtStream_t stream = static_cast<rtStream_t>(npuStream.stream());

  void* q_bias_ptr = nullptr;
  void* k_bias_ptr = nullptr;
  if (bias) {
    if (q_bias.has_value()) {
      q_bias_ptr = q_bias.value().data_ptr();
    }
    if (k_bias.has_value()) {
      k_bias_ptr = k_bias.value().data_ptr();
    }
  }

  const std::string kernel_name = make_split_rmsnorm_rope_kernel_name(
      q_hidden_size, kv_hidden_size, head_dim, eps, bias);
  auto& op = OperationFactory::instance().split_rmsnorm_rope(kernel_name);
  auto ret = op.execute(stream, gridX, gridY, gridZ, [&](ArgsBuilder& ab) {
    ab.constructArgs(input.data_ptr(),
                     sin.data_ptr(),
                     cos.data_ptr(),
                     q_output.data_ptr(),
                     k_output.data_ptr(),
                     v_output.data_ptr(),
                     q_weight.data_ptr(),
                     q_bias_ptr,
                     k_weight.data_ptr(),
                     k_bias_ptr,
                     batch_size);
  });

  if (ret != RT_ERROR_NONE) {
    LOG(ERROR) << "rtKernelLaunch failed for "
                  "'SplitRmsnormRopeBF16QH1024KVH128WithoutBiasKernel': "
               << ret;
  }

  return std::make_tuple(q_output, k_output, v_output);
}

}  // namespace xllm::kernel::npu
