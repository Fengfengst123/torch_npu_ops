#pragma once

#include <algorithm>
#include <cstdint>

namespace xllm_layer_norm_fwd {

enum class KernelMode : uint32_t {
  kFullRow = 0,
  kStreamingTwoPass = 1,
};

struct LaunchConfig {
  KernelMode mode = KernelMode::kFullRow;
  uint32_t block_dim = 1;
  uint32_t group_size = 0;
  uint32_t group_align = 0;
  uint32_t chunk_size = 0;
  uint32_t tile_rows = 1;
  uint32_t full_n = 0;
  uint32_t m = 0;
  uint32_t ngroups = 1;
};

inline int64_t AlignUp(int64_t value, int64_t align) {
  return ((value + align - 1) / align) * align;
}

inline int64_t AlignDown(int64_t value, int64_t align) {
  return (value / align) * align;
}

inline int64_t LimitTailPadding(int64_t group_size,
                                int64_t chunk,
                                int64_t align) {
  chunk = std::min(chunk, group_size);
  chunk = AlignDown(chunk, align);
  chunk = std::max(chunk, align);
  for (int64_t candidate = chunk; candidate >= align; candidate -= align) {
    if (group_size % candidate == 0) {
      return candidate;
    }
  }
  for (int64_t candidate = chunk; candidate >= align; candidate -= align) {
    const int64_t tail = group_size % candidate;
    if (tail == 0 || candidate - tail <= 255) {
      return candidate;
    }
  }
  return align;
}

inline LaunchConfig MakeLaunchConfig(int64_t m,
                                     int64_t full_n,
                                     int64_t group_size,
                                     int64_t core_num,
                                     uint64_t ub_size,
                                     int64_t x_dtype_size,
                                     int64_t weight_dtype_size,
                                     bool has_bias,
                                     bool has_z) {
  LaunchConfig cfg;
  cfg.m = static_cast<uint32_t>(m);
  cfg.full_n = static_cast<uint32_t>(full_n);
  cfg.group_size = static_cast<uint32_t>(group_size);
  cfg.ngroups = static_cast<uint32_t>(full_n / group_size);

  const int64_t align_elems = std::max<int64_t>(1, 32 / x_dtype_size);
  const int64_t group_align = AlignUp(group_size, align_elems);
  cfg.group_align = static_cast<uint32_t>(group_align);

  const int64_t scalar_bytes = 32;
  const int64_t ub_reserve_bytes = 8 * 1024;
  const bool n128_fast_path = full_n == 128 && group_size == 128 && !has_bias;
  const int64_t n128_reduce_work_bytes = n128_fast_path ? 32 * 64 * 4 : 0;
  const int64_t reduce_tmp_bytes = std::max<int64_t>(group_align, 64) * 4;
  const int64_t param_bytes =
      group_align * (weight_dtype_size + 4 +
                     (has_bias ? (weight_dtype_size + 4) : 0));
  constexpr int64_t kBufferNum = 2;
  const int64_t queue_bytes =
      kBufferNum * group_align * (2 + (has_z ? 1 : 0)) * x_dtype_size;
  const int64_t compute_bytes = group_align * 2 * 4;
  const int64_t stat_bytes = 2 * 4;
  const int64_t per_row_bytes = queue_bytes + compute_bytes + stat_bytes;
  const int64_t full_row_fixed =
      scalar_bytes + reduce_tmp_bytes + param_bytes + n128_reduce_work_bytes;

  int64_t tile_rows = 0;
  if (static_cast<int64_t>(ub_size) > full_row_fixed + ub_reserve_bytes &&
      per_row_bytes > 0) {
    tile_rows = (static_cast<int64_t>(ub_size) - full_row_fixed -
                 ub_reserve_bytes) / per_row_bytes;
  }

  const int64_t logical_rows = m * cfg.ngroups;
  int64_t used_cores = std::max<int64_t>(1, std::min(core_num, logical_rows));
  // For short decode batches, two rows per AIV reduces task scheduling and
  // per-core weight reload overhead. Larger M still uses all available AIVs.
  if (full_n == 128 && group_size == 128 && has_z && !has_bias &&
      logical_rows > 1 && logical_rows <= 32) {
    used_cores = std::max<int64_t>(1, (logical_rows + 1) / 2);
  }
  cfg.block_dim = static_cast<uint32_t>(used_cores);

  if (tile_rows >= 1) {
    cfg.mode = KernelMode::kFullRow;
    if (m >= used_cores * 2) {
      const int64_t rows_for_occupancy =
          std::max<int64_t>(1, (m + used_cores - 1) / used_cores);
      tile_rows = std::min<int64_t>(tile_rows, rows_for_occupancy);
      tile_rows = std::min<int64_t>(tile_rows, 255);
    } else {
      tile_rows = 1;
    }
    cfg.tile_rows = static_cast<uint32_t>(std::max<int64_t>(1, tile_rows));
    cfg.chunk_size = cfg.group_align;
    return cfg;
  }

  const int64_t stream_fixed = scalar_bytes;
  const int64_t stream_queue_coeff =
      kBufferNum * (2 + (has_z ? 1 : 0)) * x_dtype_size;
  const int64_t stream_compute_coeff = 3 * 4;
  const int64_t stream_param_coeff =
      weight_dtype_size + 4 + (has_bias ? (weight_dtype_size + 4) : 0);
  const int64_t stream_coeff =
      stream_queue_coeff + stream_compute_coeff + stream_param_coeff;
  int64_t chunk = 0;
  if (static_cast<int64_t>(ub_size) > stream_fixed + ub_reserve_bytes &&
      stream_coeff > 0) {
    chunk = (static_cast<int64_t>(ub_size) - stream_fixed -
             ub_reserve_bytes) / stream_coeff;
  }
  chunk = std::max<int64_t>(align_elems, AlignDown(chunk, align_elems));
  chunk = std::min<int64_t>(chunk, group_size);
  chunk = std::min<int64_t>(chunk, 1024);
  chunk = LimitTailPadding(group_size, chunk, align_elems);

  cfg.mode = KernelMode::kStreamingTwoPass;
  cfg.tile_rows = 1;
  cfg.chunk_size = static_cast<uint32_t>(chunk);
  cfg.group_align = static_cast<uint32_t>(AlignUp(chunk, align_elems));
  return cfg;
}

}  // namespace xllm_layer_norm_fwd
