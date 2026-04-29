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

#include <tuple>
#include <vector>

namespace xllm::kernel::npu {

namespace {

torch::Tensor to_int32_contiguous(const torch::Tensor& tensor) {
  return tensor.to(torch::kInt32).contiguous();
}

std::tuple<torch::Tensor, torch::Tensor, int32_t, int32_t, bool>
prepare_v2_input(torch::Tensor& x,
                 const std::optional<torch::Tensor>& query_start_loc,
                 int32_t max_query_len) {
  if (query_start_loc.has_value()) {
    TORCH_CHECK(x.dim() == 2,
                "causal_conv1d_update_v2 expects x to be 2D [num_tokens, dim] "
                "when query_start_loc is provided.");
    const auto qsl = to_int32_contiguous(query_start_loc.value());
    TORCH_CHECK(qsl.dim() == 1 && qsl.numel() >= 2,
                "query_start_loc must be 1D with shape [batch + 1].");
    const int32_t batch = static_cast<int32_t>(qsl.numel() - 1);
    return {x.contiguous(), qsl, batch, max_query_len, false};
  }

  const bool unsqueeze = x.dim() == 2;
  auto x_work = unsqueeze ? x.unsqueeze(-1) : x;
  TORCH_CHECK(x_work.dim() == 3,
              "causal_conv1d_update_v2 expects x to have shape [B, D] or "
              "[B, D, T] when query_start_loc is not provided.");

  const int32_t batch = static_cast<int32_t>(x_work.size(0));
  const int32_t seqlen = static_cast<int32_t>(x_work.size(2));
  auto x_varlen = x_work.transpose(1, 2)
                      .contiguous()
                      .view({x_work.size(0) * x_work.size(2), x_work.size(1)});
  auto qsl = torch::arange(
      0,
      (batch + 1) * seqlen,
      seqlen,
      torch::TensorOptions().dtype(torch::kInt32).device(x.device()));
  return {x_varlen, qsl, batch, seqlen, unsqueeze};
}

torch::Tensor make_default_bias(const torch::Tensor& weight) {
  return torch::zeros({weight.size(0)},
                      torch::TensorOptions()
                          .dtype(weight.scalar_type())
                          .device(weight.device()));
}

torch::Tensor make_slot_table(
    const std::optional<torch::Tensor>& conv_state_indices,
    int32_t batch,
    const torch::Device& device) {
  if (!conv_state_indices.has_value()) {
    auto ids = torch::arange(
        batch, torch::TensorOptions().dtype(torch::kInt32).device(device));
    return torch::stack({ids, ids}, 1).contiguous();
  }

  auto indices = to_int32_contiguous(conv_state_indices.value());
  if (indices.dim() == 1) {
    return torch::stack({indices, indices}, 1).contiguous();
  }
  TORCH_CHECK(indices.dim() == 2 && indices.size(1) >= 2,
              "conv_state_indices for v2 APC path must be 1D or 2D with at "
              "least two slots per batch item.");
  return indices;
}

torch::Tensor make_zero_i32(int32_t batch, const torch::Device& device) {
  return torch::zeros(
      {batch}, torch::TensorOptions().dtype(torch::kInt32).device(device));
}

}  // namespace

torch::Tensor npu_causal_conv1d_update_v2(
    torch::Tensor& x,
    torch::Tensor& conv_state,
    torch::Tensor& weight,
    bool activation,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& conv_state_indices,
    const std::optional<torch::Tensor>& query_start_loc,
    int32_t max_query_len,
    int32_t pad_slot_id,
    const std::optional<torch::Tensor>& block_idx_last_scheduled_token,
    const std::optional<torch::Tensor>& initial_state_idx,
    bool validate_data,
    const std::optional<torch::Tensor>& num_accepted_tokens_opt) {
  TORCH_CHECK(weight.dim() == 2,
              "causal_conv1d_update_v2 expects weight shape [dim, width].");
  TORCH_CHECK(block_idx_last_scheduled_token.has_value() ==
                  initial_state_idx.has_value(),
              "causal_conv1d_update_v2 expects both APC tensors or neither.");
  TORCH_CHECK(!block_idx_last_scheduled_token.has_value() ||
                  block_idx_last_scheduled_token->size(0) ==
                      initial_state_idx->size(0),
              "APC metadata size mismatch.");
  TORCH_CHECK(conv_state.dim() == 3,
              "causal_conv1d_update_v2 expects conv_state shape "
              "[num_cache_lines, dim, state_len].");
  const auto original_x_dtype = x.dtype();

  auto [x_varlen, qsl, batch, seqlen, unsqueeze] =
      prepare_v2_input(x, query_start_loc, max_query_len);
  const int32_t dim = static_cast<int32_t>(x_varlen.size(1));
  (void)unsqueeze;
  const int32_t kernel_width = static_cast<int32_t>(weight.size(1));
  TORCH_CHECK(kernel_width >= 1 && kernel_width <= 6,
              "causal_conv1d_update_v2 expects width in [1, 6], got ",
              kernel_width);

  TORCH_CHECK(weight.size(0) == dim,
              "weight dim mismatch. Expected ",
              dim,
              ", got ",
              weight.size(0));

  auto bias_tensor = bias.has_value() ? bias.value().contiguous()
                                      : make_default_bias(weight);
  TORCH_CHECK(bias_tensor.dim() == 1 && bias_tensor.size(0) == dim,
              "bias must have shape [dim] for causal_conv1d_update_v2.");
  auto slot_table = make_slot_table(conv_state_indices, batch, x.device());
  TORCH_CHECK(slot_table.size(0) == batch,
              "conv_state_indices batch mismatch. Expected ",
              batch,
              ", got ",
              slot_table.size(0));
  auto last_idx = block_idx_last_scheduled_token.has_value()
                      ? to_int32_contiguous(block_idx_last_scheduled_token.value())
                      : make_zero_i32(batch, x.device());
  auto init_idx = initial_state_idx.has_value()
                      ? to_int32_contiguous(initial_state_idx.value())
                      : make_zero_i32(batch, x.device());
  auto num_accepted_tokens =
      num_accepted_tokens_opt.has_value()
          ? to_int32_contiguous(num_accepted_tokens_opt.value())
          : torch::ones({batch},
                        torch::TensorOptions()
                            .dtype(torch::kInt32)
                            .device(x.device()));

  auto x_kernel = x_varlen.to(conv_state.dtype()).contiguous();
  auto weight_kernel = weight.transpose(0, 1).contiguous();
  auto out_kernel = torch::empty_like(x_kernel);

  auto npu_stream = c10_npu::getCurrentNPUStream();
  rtStream_t stream = static_cast<rtStream_t>(npu_stream.stream());

  const int32_t num_cache_lines = static_cast<int32_t>(conv_state.size(0));
  const int32_t eff_state_len =
      kernel_width - 1 + (num_accepted_tokens_opt.has_value() ? seqlen - 1 : 0);
  const int32_t is_spec_decoding = num_accepted_tokens_opt.has_value() ? 1 : 0;

  const int64_t stride_x_token = x_kernel.stride(0);
  const int64_t stride_x_dim = x_kernel.stride(1);
  const int64_t stride_w_width = weight_kernel.stride(0);
  const int64_t stride_w_dim = weight_kernel.stride(1);
  const int64_t stride_state_seq = conv_state.stride(0);
  const int64_t stride_state_token = conv_state.stride(1);
  const int64_t stride_state_dim = conv_state.stride(2);
  const int64_t stride_state_indices = slot_table.stride(0);
  const int64_t stride_o_token = out_kernel.stride(0);
  const int64_t stride_o_dim = out_kernel.stride(1);

  auto& op = OperationFactory::instance().causal_conv1d_update_tiled_v2();
  auto ret = op.execute(stream,
                        batch,
                        (dim + 255) / 256,
                        1,
                        [&](ArgsBuilder& ab) {
                          ab.constructArgs(x_kernel.data_ptr(),
                                           weight_kernel.data_ptr(),
                                           bias_tensor.data_ptr(),
                                           conv_state.data_ptr(),
                                           slot_table.data_ptr(),
                                           num_accepted_tokens.data_ptr(),
                                           qsl.data_ptr(),
                                           last_idx.data_ptr(),
                                           init_idx.data_ptr(),
                                           out_kernel.data_ptr(),
                                           static_cast<int32_t>(batch),
                                           static_cast<int32_t>(dim),
                                           static_cast<int32_t>(seqlen),
                                           static_cast<int32_t>(eff_state_len),
                                           static_cast<int32_t>(num_cache_lines),
                                           static_cast<int32_t>(stride_x_token),
                                           static_cast<int32_t>(stride_w_width),
                                           static_cast<int32_t>(stride_state_seq),
                                           static_cast<int32_t>(stride_state_token),
                                           static_cast<int32_t>(stride_state_indices),
                                           static_cast<int32_t>(stride_o_token),
                                           static_cast<int32_t>(kernel_width),
                                           static_cast<int32_t>(is_spec_decoding),
                                           static_cast<int32_t>(pad_slot_id));
                        });
  if (ret != RT_ERROR_NONE) {
    LOG(ERROR) << "rtKernelLaunch failed for "
                  "'_causal_conv1d_update_kernel_npu_tiled_v2': "
               << ret;
    return torch::zeros_like(x);
  }

  auto out = activation ? torch::silu(out_kernel) : out_kernel;
  return out.to(original_x_dtype);
}

}  // namespace xllm::kernel::npu
