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



#include <torch/torch.h>

#include <vector>
#include "ascendc_ops_api.h"

#include "pytorch_npu_helper.hpp"


namespace npu_ops {

void npu_gemma_rms_norm(
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    double epsilon,
    torch::Tensor &rstdOut,
    torch::Tensor &yOut)
{
    int64_t dim_x = x.dim();
    int64_t dim_gamma = gamma.dim();
    int64_t diff = dim_x - dim_gamma;
    std::vector<int64_t> new_shape;
    at::Tensor rstd;
    if (diff > 0) {
        new_shape.reserve(dim_x);
        auto x_sizes = x.sizes();
        for (int64_t i = 0; i < diff; ++i) {
            new_shape.push_back(x_sizes[i]);
        }
        for (int64_t i = 0; i < dim_gamma; ++i) {
            new_shape.push_back(1);
        }
    } else {
        new_shape.assign(dim_x, 1);
    }
    rstd = at::empty(new_shape, x.options().dtype(at::kFloat));
    at::Tensor y = at::empty(x.sizes(), x.options());
    EXEC_NPU_CMD(aclnnGemmaRmsNorm, x, gamma, epsilon, y, rstd);
    yOut =y;
    rstdOut = rstd;
}

}  // namespace npu_ops