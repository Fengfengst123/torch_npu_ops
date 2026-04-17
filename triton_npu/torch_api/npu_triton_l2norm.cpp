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

#include <limits>

namespace xllm::kernel::npu {
namespace {

constexpr int32_t kBlockN = 128;
constexpr int32_t kMBlock = 69;

void validate_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.defined(), name, " tensor is not defined");
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
              name,
              " tensor must be on NPU device");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name,
              " tensor must be bfloat16");
}

int32_t cdiv(int32_t a, int32_t b) {
  TORCH_CHECK(b > 0, "cdiv divisor must be positive");
  return (a + b - 1) / b;
}

int32_t get_vectorcore_num() {
  int32_t device_id = 0;
  if (aclrtGetDevice(&device_id) != ACL_SUCCESS) {
    return 20;
  }
  int64_t vec_core_num = 0;
  const aclError ret = aclrtGetDeviceInfo(
      static_cast<uint32_t>(device_id), ACL_DEV_ATTR_VECTOR_CORE_NUM,
      &vec_core_num);
  if (ret == ACL_SUCCESS && vec_core_num > 0) {
    return static_cast<int32_t>(vec_core_num);
  }
  return 20;
}

}  // namespace

torch::Tensor npu_l2norm_last_dim(torch::Tensor& x, double eps) {
  validate_tensor(x, "x");
  TORCH_CHECK(x.dim() >= 1, "npu_l2norm_last_dim expects x.dim() >= 1");

  const auto original_shape = x.sizes().vec();
  const int64_t last_dim = x.size(-1);
  TORCH_CHECK(last_dim > 0, "npu_l2norm_last_dim expects hidden dim > 0");
  TORCH_CHECK(last_dim <= std::numeric_limits<int32_t>::max(),
              "npu_l2norm_last_dim hidden dim exceeds int32 range");

  auto x_contiguous = x.contiguous();
  auto x_2d = x_contiguous.view({-1, last_dim});
  auto y_2d = torch::empty_like(x_2d);

  const int32_t rows = static_cast<int32_t>(x_2d.size(0));
  TORCH_CHECK(rows > 0, "npu_l2norm_last_dim expects non-empty input");
  const int32_t cols = static_cast<int32_t>(last_dim);

  const int32_t num_vectorcore = get_vectorcore_num();
  const int32_t grid_x = std::max(1, num_vectorcore);
  const int32_t main_bs = cdiv(rows, grid_x);
  const int32_t num_row_chunks = cdiv(main_bs, kMBlock);
  const int32_t num_col_chunks = cdiv(cols, kBlockN);
  const float eps_f = static_cast<float>(eps);

  auto npu_stream = c10_npu::getCurrentNPUStream();
  rtStream_t stream = static_cast<rtStream_t>(npu_stream.stream());

  auto& op = OperationFactory::instance().l2norm_fwd();
  auto ret = op.execute(
      stream,
      grid_x,
      1,
      1,
      [&](ArgsBuilder& ab) {
        ab.constructArgs(x_2d.data_ptr(),
                         y_2d.data_ptr(),
                         eps_f,
                         rows,
                         cols,
                         num_row_chunks,
                         num_col_chunks);
      });
  if (ret != RT_ERROR_NONE) {
    LOG(ERROR) << "rtKernelLaunch failed for 'l2norm_fwd_kernel2_loop': "
               << ret;
  }

  return y_2d.view(original_shape);
}

}  // namespace xllm::kernel::npu
