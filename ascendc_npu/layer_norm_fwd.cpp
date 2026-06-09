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

#include <torch/torch.h>

#include "ascendc_ops_api.h"
#include "pytorch_npu_helper.hpp"

namespace npu_ops {

torch::Tensor layer_norm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    double eps,
    const c10::optional<torch::Tensor>& z,
    int64_t group_size,
    bool norm_before_gate,
    bool is_rms_norm)
{
    TORCH_CHECK(x.dim() >= 1, "layer_norm_fwd expects x.dim() >= 1");
    const int64_t last_dim = x.size(-1);
    const int64_t group_size_val = group_size > 0 ? group_size : last_dim;
    TORCH_CHECK(group_size_val > 0, "group_size must be positive");
    TORCH_CHECK(last_dim % group_size_val == 0, "x.size(-1) must be divisible by group_size");
    TORCH_CHECK(weight.defined(), "weight must be defined");
    TORCH_CHECK(weight.numel() == last_dim, "weight numel must equal x.size(-1)");

    const auto original_shape = x.sizes();
    auto x_2d = x.reshape({-1, last_dim}).contiguous();
    const int64_t M = x_2d.size(0);
    const int64_t group_count = last_dim / group_size_val;

    auto weight_contiguous = weight.contiguous();
    torch::Tensor bias_contiguous;
    if (bias.has_value() && bias.value().defined()) {
        TORCH_CHECK(bias.value().numel() == last_dim, "bias numel must equal x.size(-1)");
        bias_contiguous = bias.value().contiguous();
    }
    torch::Tensor z_2d;
    if (z.has_value() && z.value().defined()) {
        TORCH_CHECK(z.value().sizes() == x.sizes(), "z shape must match x shape");
        z_2d = z.value().reshape({-1, last_dim}).contiguous();
    }

    auto y = torch::empty_like(x_2d);
    torch::Tensor mean;
    if (!is_rms_norm) {
        mean = torch::empty({M * group_count}, x.options().dtype(torch::kFloat32));
    }
    auto rstd = torch::empty({M * group_count}, x.options().dtype(torch::kFloat32));

    EXEC_NPU_CMD(
        aclnnLayerNormFwd,
        x_2d,
        weight_contiguous,
        bias_contiguous,
        z_2d,
        y,
        mean,
        rstd,
        eps,
        group_size_val,
        norm_before_gate,
        is_rms_norm);
    return y.reshape(original_shape);
}

} // namespace npu_ops
