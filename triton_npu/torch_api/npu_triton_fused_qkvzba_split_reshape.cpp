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

// ---------------------------------------------------------------------------
// Helper: ceil integer division
// ---------------------------------------------------------------------------
static inline int32_t ceil_div(int32_t a, int32_t b) {
  return (a + b - 1) / b;
}

// ---------------------------------------------------------------------------
// Helper: next power of two (≥ 1)
// ---------------------------------------------------------------------------
static inline int32_t next_pow2(int32_t n) {
  if (n <= 1) return 1;
  --n;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

// ---------------------------------------------------------------------------
// Helper: query available Vector Core count from the current NPU.
//
// Triton kernels run on Vector Cores (not AI Cores / cube units).
// Uses aclrtGetDeviceInfo() with ACL_DEV_ATTR_VECTOR_CORE_NUM (= 201),
// defined in acl/acl_rt.h since CANN 8.x.
// ---------------------------------------------------------------------------
static int32_t get_vectorcore_num() {
  int32_t device_id = 0;
  if (aclrtGetDevice(&device_id) != ACL_SUCCESS) {
    return 20;  // no active device context
  }
  int64_t vec_core_num = 0;
  // ACL_DEV_ATTR_VECTOR_CORE_NUM = 201: number of Vector Cores
  const aclError ret = aclrtGetDeviceInfo(
      static_cast<uint32_t>(device_id), ACL_DEV_ATTR_VECTOR_CORE_NUM,
      &vec_core_num);
  if (ret == ACL_SUCCESS && vec_core_num > 0) {
    return static_cast<int32_t>(vec_core_num);
  }
  return 20;  // fallback for older CANN versions or unsupported attribute
}

/**
 * npu_fused_qkvzba_split_reshape_cat
 *
 * Inputs
 *   mixed_qkvz : [batch, num_heads_qk * (2*head_qk + 2*(num_heads_v/num_heads_qk)*head_v)]
 *   mixed_ba   : [batch, num_heads_qk * 2 * (num_heads_v/num_heads_qk)]
 *   num_heads_qk : number of Q/K heads (after TP split)
 *   num_heads_v  : number of V heads (after TP split)
 *   head_qk      : Q/K head dimension
 *   head_v       : V head dimension
 *
 * Outputs (returned as a 4-tuple)
 *   mixed_qkv : [batch, num_heads_qk*head_qk*2 + num_heads_v*head_v]
 *   z         : [batch, num_heads_v, head_v]
 *   b         : [batch, num_heads_v]
 *   a         : [batch, num_heads_v]
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
npu_fused_qkvzba_split_reshape_cat(
    torch::Tensor& mixed_qkvz,
    torch::Tensor& mixed_ba,
    int32_t num_heads_qk,
    int32_t num_heads_v,
    int32_t head_qk,
    int32_t head_v) {
  const int32_t batch = static_cast<int32_t>(mixed_qkvz.size(0));
  const int32_t total_rows = batch;  // seq_len = 1 for decode

  const int32_t v_heads_per_qk = num_heads_v / num_heads_qk;
  const int32_t v_dim_per_qk = v_heads_per_qk * head_v;
  const int32_t qkvz_dim_t = head_qk * 2 + v_dim_per_qk * 2;
  const int32_t ba_dim_t = v_heads_per_qk * 2;

  // Row strides (number of elements per row in each logical matrix)
  const int32_t qkvz_row_stride = num_heads_qk * qkvz_dim_t;
  const int32_t ba_row_stride = num_heads_qk * ba_dim_t;
  const int32_t qkv_row_stride =
      num_heads_qk * head_qk * 2 + num_heads_v * head_v;
  const int32_t z_row_stride = num_heads_v * head_v;
  const int32_t ba_out_row_stride = num_heads_v;

  // Allocate output tensors
  const int32_t qkv_dim = num_heads_qk * head_qk * 2 + num_heads_v * head_v;
  auto mixed_qkv = torch::empty(
      {total_rows, qkv_dim},
      torch::TensorOptions().dtype(mixed_qkvz.dtype()).device(mixed_qkvz.device()));

  // z is stored flat [batch, num_heads_v * head_v] by the kernel; we reshape
  // to [batch, num_heads_v, head_v] before returning.
  auto z_flat = torch::empty(
      {total_rows, num_heads_v * head_v},
      torch::TensorOptions().dtype(mixed_qkvz.dtype()).device(mixed_qkvz.device()));

  auto b = torch::empty(
      {total_rows, num_heads_v},
      torch::TensorOptions().dtype(mixed_ba.dtype()).device(mixed_ba.device()));

  auto a = torch::empty(
      {total_rows, num_heads_v},
      torch::TensorOptions().dtype(mixed_ba.dtype()).device(mixed_ba.device()));

  // -----------------------------------------------------------------------
  // Compute launch parameters (mirror Python wrapper exactly)
  // -----------------------------------------------------------------------
  const int32_t num_vectorcore = get_vectorcore_num();
  const int32_t grid_size = std::max(1, std::min(num_vectorcore, total_rows));
  const int32_t rows_per_vec = ceil_div(total_rows, grid_size);

  const int32_t rows_per_iter = 1;

  // -----------------------------------------------------------------------
  // Launch kernel
  // -----------------------------------------------------------------------
  const int32_t gridX = grid_size;
  const int32_t gridY = 1;
  const int32_t gridZ = 1;

  auto npuStream = c10_npu::getCurrentNPUStream();
  rtStream_t stream = static_cast<rtStream_t>(npuStream.stream());

  auto& op = OperationFactory::instance().fused_qkvzba_split_reshape();
  auto ret = op.execute(stream, gridX, gridY, gridZ, [&](ArgsBuilder& ab) {
    // Kernel signature (after moving tl.constexpr to the end):
    //   output ptrs:  mixed_qkv, z, b, a
    //   input ptrs:   mixed_qkvz, mixed_ba
    //   do_not_specialize: num_heads_qk, num_heads_v, total_rows, rows_per_vec,
    //                      qkvz_row_stride, ba_row_stride, qkv_row_stride,
    //                      z_row_stride, ba_out_row_stride
    //   tl.constexpr: HEAD_QK, HEAD_V, V_HEADS_PER_QK, V_DIM_PER_QK,
    //                 QKVZ_DIM_T, BA_DIM_T, ROWS_PER_ITER
    ab.constructArgs(
        mixed_qkv.data_ptr(),
        z_flat.data_ptr(),
        b.data_ptr(),
        a.data_ptr(),
        mixed_qkvz.data_ptr(),
        mixed_ba.data_ptr(),
        num_heads_qk,
        num_heads_v,
        total_rows,
        rows_per_vec,
        qkvz_row_stride,
        ba_row_stride,
        qkv_row_stride,
        z_row_stride,
        ba_out_row_stride);
  });

  if (ret != RT_ERROR_NONE) {
    LOG(ERROR) << "rtKernelLaunch failed for "
                  "'fused_qkvzba_split_reshape_cat_kernel': "
               << ret;
  }

  auto z = z_flat.view({total_rows, num_heads_v, head_v});

  return {mixed_qkv, z, b, a};
}

}  // namespace xllm::kernel::npu
