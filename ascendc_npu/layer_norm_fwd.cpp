/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <torch/torch.h>

#include <tuple>

#include "ascendc_ops_api.h"
#include "pytorch_npu_helper.hpp"

namespace npu_ops {

namespace {

bool IsSupportedInputDtype(at::ScalarType dtype) {
  return dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat;
}

torch::Tensor MakeContiguous2d(const torch::Tensor& x) {
  const int64_t last_dim = x.size(-1);
  return x.reshape({-1, last_dim}).contiguous();
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> npu_layer_norm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    double eps,
    const c10::optional<torch::Tensor>& z,
    int64_t group_size,
    bool norm_before_gate,
    bool is_rms_norm) {
  TORCH_CHECK(x.defined(), "layer_norm_fwd: x must be defined");
  TORCH_CHECK(weight.defined(), "layer_norm_fwd: weight must be defined");
  TORCH_CHECK(x.dim() >= 1, "layer_norm_fwd: x must have at least 1 dim");
  TORCH_CHECK(IsSupportedInputDtype(x.scalar_type()),
              "layer_norm_fwd: x dtype must be fp16, bf16 or fp32, got ",
              x.scalar_type());
  TORCH_CHECK(weight.dim() == 1, "layer_norm_fwd: weight must be 1D");

  const int64_t full_n = x.size(-1);
  if (group_size < 0) {
    group_size = full_n;
  }
  TORCH_CHECK(group_size > 0, "layer_norm_fwd: group_size must be positive");
  TORCH_CHECK(full_n % group_size == 0,
              "layer_norm_fwd: last dim ",
              full_n,
              " must be divisible by group_size ",
              group_size);
  TORCH_CHECK(weight.numel() == full_n,
              "layer_norm_fwd: weight numel must equal x last dim");

  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->dim() == 1, "layer_norm_fwd: bias must be 1D");
    TORCH_CHECK(bias->numel() == full_n,
                "layer_norm_fwd: bias numel must equal x last dim");
  }
  if (z.has_value() && z->defined()) {
    TORCH_CHECK(z->sizes() == x.sizes(),
                "layer_norm_fwd: z shape must match x");
  }

  torch::Tensor x_2d = MakeContiguous2d(x);
  torch::Tensor weight_contig = weight.contiguous();
  c10::optional<torch::Tensor> bias_contig = c10::nullopt;
  if (bias.has_value() && bias->defined()) {
    bias_contig = bias->contiguous();
  }
  c10::optional<torch::Tensor> z_2d = c10::nullopt;
  if (z.has_value() && z->defined()) {
    z_2d = MakeContiguous2d(*z);
  }

  torch::Tensor y_2d = torch::empty_like(x_2d);
  const int64_t m = x_2d.size(0);
  const int64_t ngroups = full_n / group_size;
  torch::Tensor mean;
  if (is_rms_norm) {
    mean = torch::empty({0}, x.options().dtype(at::kFloat));
  } else {
    mean = torch::empty({ngroups * m}, x.options().dtype(at::kFloat));
  }
  torch::Tensor rstd = torch::empty({ngroups * m},
                                    x.options().dtype(at::kFloat));

  const float eps_f = static_cast<float>(eps);
  const int64_t group_size_arg = group_size;
  const bool norm_before_gate_arg = norm_before_gate;
  const bool is_rms_norm_arg = is_rms_norm;

  EXEC_NPU_CMD(aclnnXllmLayerNormFwd,
               x_2d,
               weight_contig,
               bias_contig,
               z_2d,
               eps_f,
               group_size_arg,
               norm_before_gate_arg,
               is_rms_norm_arg,
               y_2d,
               mean,
               rstd);

  return std::make_tuple(y_2d.reshape(x.sizes()), mean, rstd);
}

}  // namespace npu_ops
