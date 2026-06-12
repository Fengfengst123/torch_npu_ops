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

namespace py = pybind11;

std::tuple<torch::Tensor, torch::Tensor> py_npu_gemma_rms_norm(
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    double epsilon) {
    torch::Tensor rstd_out;
    torch::Tensor y_out;
    npu_ops::npu_gemma_rms_norm(x, gamma, epsilon, rstd_out, y_out);
    return std::make_tuple(rstd_out, y_out);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> py_npu_layer_norm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    double eps,
    const c10::optional<torch::Tensor>& z,
    int64_t group_size,
    bool norm_before_gate,
    bool is_rms_norm) {
    return npu_ops::npu_layer_norm_fwd(
        x, weight, bias, eps, z, group_size, norm_before_gate, is_rms_norm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemma_rms_norm", &py_npu_gemma_rms_norm,
          "gemma_rms_norm using EXEC_NPU_CMD (aclnnGemmaRmsNorm)");
    m.def("layer_norm_fwd", &py_npu_layer_norm_fwd,
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("eps") = 1e-6,
          py::arg("z") = py::none(),
          py::arg("group_size") = -1,
          py::arg("norm_before_gate") = true,
          py::arg("is_rms_norm") = false,
          "layer_norm_fwd using EXEC_NPU_CMD (aclnnXllmLayerNormFwd)");
}
