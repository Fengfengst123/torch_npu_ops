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

namespace xllm::kernel::npu {

constexpr int64_t MAX_CORES = 65535;
constexpr int64_t MAX_FUSED_BYTES = 65536;

inline int64_t next_power_of_2(int64_t n) {
  if (n <= 1) return 1;
  uint64_t val = static_cast<uint64_t>(n - 1);
  return 1LL << (64 - __builtin_clzll(val ? val : 1));
}

// ---------------------------------------------------------------------------
// Helper: query available Vector Core count from the NPU where the tensor
// resides.  Uses the tensor's device index explicitly instead of relying on
// the thread's current ACL context.
// ---------------------------------------------------------------------------
static int32_t get_vectorcore_num(const torch::Tensor& x) {
  int32_t device_id = static_cast<int32_t>(x.device().index());
  int64_t vec_core_num = 0;
  const aclError ret = aclrtGetDeviceInfo(
      static_cast<uint32_t>(device_id), ACL_DEV_ATTR_VECTOR_CORE_NUM,
      &vec_core_num);
  if (ret == ACL_SUCCESS && vec_core_num > 0) {
    return static_cast<int32_t>(vec_core_num);
  }
  return 20;  // fallback for older CANN versions
}

torch::Tensor layer_norm_fwd(torch::Tensor& x,
                             torch::Tensor& weight,
                             torch::Tensor& bias,
                             double eps,
                             const std::optional<torch::Tensor>& z,
                             int64_t group_size,
                             bool norm_before_gate,
                             bool is_rms_norm) {
  c10::IntArrayRef x_shape_og = x.sizes();
  int64_t last_dim = x.size(-1);
  torch::Tensor x_2d = x.reshape({-1, last_dim});

  const auto M = x_2d.size(0);
  const auto N = x_2d.size(1);

  const int64_t group_size_val = group_size;
  const int64_t ngroups = N / group_size_val;

  torch::Tensor z_2d;
  if (z.has_value()) {
    z_2d = z->reshape({-1, last_dim});
  }

  torch::Tensor out_tensor = torch::empty_like(x_2d);
  torch::Tensor mean, rstd;
  if (!is_rms_norm) {
    mean = torch::empty({ngroups * M},
                        torch::dtype(torch::kFloat32).device(x.device()));
  }
  rstd = torch::empty({ngroups * M},
                      torch::dtype(torch::kFloat32).device(x.device()));

  auto npuStream = c10_npu::getCurrentNPUStream();
  rtStream_t stream = static_cast<rtStream_t>(npuStream.stream());

  void* x_2dPtr = x_2d.data_ptr();
  void* out_tensorPtr = out_tensor.data_ptr();
  void* weightPtr = weight.data_ptr();
  void* biasPtr = nullptr;
  if (bias.defined()) {
    biasPtr = bias.data_ptr();
  }
  void* z_2dPtr = nullptr;
  if (z_2d.defined()) {
    z_2dPtr = z_2d.data_ptr();
  }
  void* meanPtr = nullptr;
  if (mean.defined()) {
    meanPtr = mean.data_ptr();
  }
  void* rstdPtr = rstd.data_ptr();
  int32_t stride_x_row = x_2d.stride(0);
  int32_t stride_y_row = out_tensor.stride(0);
  int32_t stride_z_row = 0;
  if (z.has_value()) {
    stride_z_row = z_2d.stride(0);
  }

  // -------------------------------------------------------------------------
  // Multi-kernel dispatch (refer to npu_triton_causal_conv1d_update.cpp)
  // -------------------------------------------------------------------------
  // Fast kernel:  group_size <= 128
  // Fallback:     everything else (original implementation)
  // -------------------------------------------------------------------------
  const bool use_fast_kernel = (group_size_val <= 128);

  if (use_fast_kernel) {
    const int32_t num_vectorcore = get_vectorcore_num(x);
    const int32_t gridX =
        static_cast<int32_t>(std::max<int64_t>(1, std::min<int64_t>(num_vectorcore, M)));
    const int32_t gridY = static_cast<int32_t>(ngroups);
    const int32_t gridZ = 1;

    const bool is_bf16 = (x.scalar_type() == torch::kBFloat16);
    const bool has_z = z.has_value();
    const bool has_bias = bias.defined();

    if (has_z) {
      // Fast kernel with Z (HAS_Z=True, 7 pointer args including Z)
      OperationBase& op = [&]() -> OperationBase& {
        if (is_rms_norm) {
          if (is_bf16) {
            return has_bias
                ? static_cast<OperationBase&>(OperationFactory::instance().layer_norm_fwd_fast_rms_bf16_z_bias())
                : static_cast<OperationBase&>(OperationFactory::instance().layer_norm_fwd_fast_rms_bf16_z_nobias());
          }
          return has_bias
              ? static_cast<OperationBase&>(OperationFactory::instance().layer_norm_fwd_fast_rms_z_bias())
              : static_cast<OperationBase&>(OperationFactory::instance().layer_norm_fwd_fast_rms_z_nobias());
        } else {
          return is_bf16
              ? static_cast<OperationBase&>(OperationFactory::instance().layer_norm_fwd_fast_bf16_z())
              : static_cast<OperationBase&>(OperationFactory::instance().layer_norm_fwd_fast_z());
        }
      }();
      if (is_rms_norm) {
        rtError_t ret =
            op.execute(stream, gridX, gridY, gridZ, [&](ArgsBuilder& ab) {
              if (has_bias) {
                ab.constructArgs(x_2dPtr,
                                 out_tensorPtr,
                                 weightPtr,
                                 biasPtr,
                                 z_2dPtr,
                                 rstdPtr,
                                 stride_x_row,
                                 stride_y_row,
                                 stride_z_row,
                                 static_cast<int32_t>(M),
                                 static_cast<int32_t>(group_size_val),
                                 static_cast<float>(eps),
                                 gridX);
              } else {
                ab.constructArgs(x_2dPtr,
                                 out_tensorPtr,
                                 weightPtr,
                                 z_2dPtr,
                                 rstdPtr,
                                 stride_x_row,
                                 stride_y_row,
                                 stride_z_row,
                                 static_cast<int32_t>(M),
                                 static_cast<int32_t>(group_size_val),
                                 static_cast<float>(eps),
                                 gridX);
              }
            });
        if (ret != RT_ERROR_NONE) {
          LOG(ERROR) << "rtKernelLaunch failed for 'layer_norm_fwd_kernel_fast_rms"
                     << (is_bf16 ? "_bf16" : "") << "_z_"
                     << (has_bias ? "bias" : "nobias") << "': " << ret;
        }
      } else {
        rtError_t ret =
            op.execute(stream, gridX, gridY, gridZ, [&](ArgsBuilder& ab) {
              ab.constructArgs(x_2dPtr,
                               out_tensorPtr,
                               weightPtr,
                               biasPtr,
                               z_2dPtr,
                               meanPtr,
                               rstdPtr,
                               stride_x_row,
                               stride_y_row,
                               stride_z_row,
                               static_cast<int32_t>(M),
                               static_cast<int32_t>(group_size_val),
                               static_cast<float>(eps),
                               gridX);
            });
        if (ret != RT_ERROR_NONE) {
          LOG(ERROR) << "rtKernelLaunch failed for 'layer_norm_fwd_kernel_fast"
                     << (is_bf16 ? "_bf16" : "") << "_z': " << ret;
        }
      }
    } else {
      // Fast kernel without Z (HAS_Z=False, Z pointer optimised out)
      OperationBase& op = [&]() -> OperationBase& {
        if (is_rms_norm) {
          if (is_bf16) {
            return has_bias
                ? static_cast<OperationBase&>(OperationFactory::instance().layer_norm_fwd_fast_rms_bf16_bias())
                : static_cast<OperationBase&>(OperationFactory::instance().layer_norm_fwd_fast_rms_bf16_nobias());
          }
          return has_bias
              ? static_cast<OperationBase&>(OperationFactory::instance().layer_norm_fwd_fast_rms_bias())
              : static_cast<OperationBase&>(OperationFactory::instance().layer_norm_fwd_fast_rms_nobias());
        } else {
          return is_bf16
              ? static_cast<OperationBase&>(OperationFactory::instance().layer_norm_fwd_fast_bf16())
              : static_cast<OperationBase&>(OperationFactory::instance().layer_norm_fwd_fast());
        }
      }();
      if (is_rms_norm) {
        rtError_t ret =
            op.execute(stream, gridX, gridY, gridZ, [&](ArgsBuilder& ab) {
              if (has_bias) {
                ab.constructArgs(x_2dPtr,
                                 out_tensorPtr,
                                 weightPtr,
                                 biasPtr,
                                 rstdPtr,
                                 stride_x_row,
                                 stride_y_row,
                                 stride_z_row,
                                 static_cast<int32_t>(M),
                                 static_cast<int32_t>(group_size_val),
                                 static_cast<float>(eps),
                                 gridX);
              } else {
                ab.constructArgs(x_2dPtr,
                                 out_tensorPtr,
                                 weightPtr,
                                 rstdPtr,
                                 stride_x_row,
                                 stride_y_row,
                                 stride_z_row,
                                 static_cast<int32_t>(M),
                                 static_cast<int32_t>(group_size_val),
                                 static_cast<float>(eps),
                                 gridX);
              }
            });
        if (ret != RT_ERROR_NONE) {
          LOG(ERROR) << "rtKernelLaunch failed for 'layer_norm_fwd_kernel_fast_rms"
                     << (is_bf16 ? "_bf16" : "") << "_"
                     << (has_bias ? "bias" : "nobias") << "': " << ret;
        }
      } else {
        rtError_t ret =
            op.execute(stream, gridX, gridY, gridZ, [&](ArgsBuilder& ab) {
              // Note: the AOT binary for the fast kernel was compiled with
              // HAS_Z=False (z is always None on the fast path) and
              // HAS_BIAS=True.  Triton-Ascend optimises out the unused Z
              // pointer, so the binary signature has 6 pointer args instead
              // of 7.  We must NOT pass z_2dPtr here.
              ab.constructArgs(x_2dPtr,
                               out_tensorPtr,
                               weightPtr,
                               biasPtr,
                               meanPtr,
                               rstdPtr,
                               stride_x_row,
                               stride_y_row,
                               stride_z_row,
                               static_cast<int32_t>(M),
                               static_cast<int32_t>(group_size_val),
                               static_cast<float>(eps),
                               gridX);  // N_CORES as runtime param
            });
        if (ret != RT_ERROR_NONE) {
          LOG(ERROR) << "rtKernelLaunch failed for 'layer_norm_fwd_kernel_fast"
                     << (is_bf16 ? "_bf16" : "") << "': " << ret;
        }
      }
    }
    return out_tensor.reshape(x_shape_og);
  }

  // -------------------------------------------------------------------------
  // Fallback kernel (original implementation)
  // -------------------------------------------------------------------------
  // NOTE: the existing fallback AOT binary was compiled without biasPtr and
  // meanPtr in the argument list (those pointers were optimised out by the
  // Triton compiler because the original pytest exercised HAS_BIAS=False and
  // IS_RMS_NORM=True).  We must match that signature exactly.
  const int64_t elem_size = x.element_size();
  const int64_t MAX_FUSED_SIZE = MAX_FUSED_BYTES / elem_size;
  const int64_t BLOCK_N =
      std::min(MAX_FUSED_SIZE, next_power_of_2(group_size_val));

  int32_t gridX =
      static_cast<int32_t>(std::max<int64_t>(1, std::min<int64_t>(MAX_CORES, M)));
  int32_t gridY = static_cast<int32_t>(ngroups);
  int32_t gridZ = 1;

  auto& op = OperationFactory::instance().layer_norm_fwd();
  rtError_t ret =
      op.execute(stream, gridX, gridY, gridZ, [&](ArgsBuilder& ab) {
        ab.constructArgs(x_2dPtr,
                         out_tensorPtr,
                         weightPtr,
                         z_2dPtr,
                         rstdPtr,
                         stride_x_row,
                         stride_y_row,
                         stride_z_row,
                         static_cast<int32_t>(M),
                         static_cast<int32_t>(group_size_val),
                         static_cast<float>(eps));
      });
  if (ret != RT_ERROR_NONE) {
    LOG(ERROR) << "rtKernelLaunch failed for 'layer_norm_fwd_kernel': " << ret;
  }
  return out_tensor.reshape(x_shape_og);
}

}  // namespace xllm::kernel::npu
