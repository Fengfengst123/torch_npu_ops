/*
 * Reference host launcher for the generated xllm_layer_norm_fwd AscendC kernel.
 *
 * This file is not added to the current xLLM CMake target because the current
 * tree only builds PyTorch wrappers that call ACLNN APIs. Use this file when
 * packaging the custom OPP/ACLNN operator or when adding a direct kernel-launch
 * path to xLLM.
 */

#include <torch/torch.h>

#include "aclrtlaunch_xllm_layer_norm_fwd.h"
#include "tiling/platform/platform_ascendc.h"
#include "xllm_layer_norm_fwd_tiling.h"

namespace npu_ops {

namespace {

struct CachedPlatformInfo {
  uint64_t ub_size = 0;
  int64_t core_num = 0;
};

const CachedPlatformInfo& GetCachedPlatformInfo() {
  static const CachedPlatformInfo info = []() {
    CachedPlatformInfo result;
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,
                             result.ub_size);
    result.core_num = static_cast<int64_t>(platform->GetCoreNumAiv());
    return result;
  }();
  return info;
}

}  // namespace

void launch_xllm_layer_norm_fwd_kernel(const torch::Tensor& x2d,
                                       const torch::Tensor& weight,
                                       const c10::optional<torch::Tensor>& bias,
                                       const c10::optional<torch::Tensor>& z2d,
                                       torch::Tensor& y2d,
                                       torch::Tensor& mean,
                                       torch::Tensor& rstd,
                                       float eps,
                                       int64_t group_size,
                                       bool norm_before_gate,
                                       bool is_rms_norm,
                                       aclrtStream stream) {
  const int64_t m = x2d.size(0);
  const int64_t full_n = x2d.size(1);
  const int64_t ngroups = full_n / group_size;
  const bool has_bias = bias.has_value() && bias->defined();
  const bool has_z = z2d.has_value() && z2d->defined();

  const auto& platform_info = GetCachedPlatformInfo();
  auto cfg = xllm_layer_norm_fwd::MakeLaunchConfig(
      m,
      full_n,
      group_size,
      platform_info.core_num,
      platform_info.ub_size,
      x2d.element_size(),
      weight.element_size(),
      has_bias,
      has_z);

  void* bias_ptr = has_bias ? const_cast<void*>(bias->data_ptr()) : nullptr;
  void* z_ptr = has_z ? const_cast<void*>(z2d->data_ptr()) : nullptr;
  void* mean_ptr = is_rms_norm ? nullptr : mean.data_ptr();

  ACLRT_LAUNCH_KERNEL(xllm_layer_norm_fwd)
  (cfg.block_dim,
   stream,
   const_cast<void*>(x2d.data_ptr()),
   const_cast<void*>(weight.data_ptr()),
   bias_ptr,
   z_ptr,
   y2d.data_ptr(),
   mean_ptr,
   rstd.data_ptr(),
   static_cast<uint32_t>(m),
   static_cast<uint32_t>(full_n),
   static_cast<uint32_t>(group_size),
   static_cast<uint32_t>(ngroups),
   static_cast<uint32_t>(x2d.stride(0)),
   static_cast<uint32_t>(y2d.stride(0)),
   has_z ? static_cast<uint32_t>(z2d->stride(0)) : 0,
   cfg.group_align,
   cfg.tile_rows,
   cfg.chunk_size,
   eps,
   has_bias ? 1U : 0U,
   has_z ? 1U : 0U,
   norm_before_gate ? 1U : 0U,
   is_rms_norm ? 1U : 0U,
   static_cast<uint32_t>(cfg.mode));
}

}  // namespace npu_ops
