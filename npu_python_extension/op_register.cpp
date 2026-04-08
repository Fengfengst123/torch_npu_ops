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

/**
 * ascendc api register
 * @file op_register.cpp
 */

#include <torch/extension.h>
#include <vector>

#include "../ascendc_npu/ascendc_ops_api.h"

std::tuple<torch::Tensor, torch::Tensor> py_npu_gemma_rms_norm(
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    double epsilon) {
    torch::Tensor rstd_out;
    torch::Tensor y_out;
    npu_ops::npu_gemma_rms_norm(x, gamma, epsilon, rstd_out, y_out);
    return std::make_tuple(rstd_out, y_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemma_rms_norm", &py_npu_gemma_rms_norm,
          "gemma_rms_norm using EXEC_NPU_CMD (aclnnGemmaRmsNorm)");
}
