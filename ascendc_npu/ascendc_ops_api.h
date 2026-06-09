/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <torch/torch.h>

#include <vector>

namespace npu_ops {

void npu_gemma_rms_norm(
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    double epsilon,
    torch::Tensor &rstdOut,
    torch::Tensor &yOut);

torch::Tensor layer_norm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    double eps,
    const c10::optional<torch::Tensor>& z,
    int64_t group_size,
    bool norm_before_gate,
    bool is_rms_norm);

}  // namespace npu_ops
