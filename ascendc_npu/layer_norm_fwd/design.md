# xllm_layer_norm_fwd AscendC design

## Source semantic

This operator is generated from the earliest tracked version of
`triton_npu/triton_src/test_layernorm_fwd.py` at commit
`38f12f9c088f452c20b52d1e3c9e7aaa804002c9`.

Input `x` is reshaped to `[M, fullN]`. `group_size` defaults to `fullN`.
`fullN % group_size == 0`, and `ngroups = fullN / group_size`.
For each `(row, group)`:

```text
xg = x[row, group * group_size : (group + 1) * group_size]
if z exists and norm_before_gate is false:
    xg = xg * silu(zg)

if is_rms_norm:
    rstd = rsqrt(mean(xg * xg) + eps)
    norm = xg * rstd
else:
    mean = mean(xg)
    rstd = rsqrt(mean((xg - mean) * (xg - mean)) + eps)
    norm = (xg - mean) * rstd

y = norm * weight_group
if bias exists:
    y = y + bias_group
if z exists and norm_before_gate is true:
    y = y * silu(zg)
```

The public wrapper returns `(y, mean, rstd)`. `mean` is an empty float tensor
for RMSNorm; `rstd` is laid out as `[ngroups, M]`, matching the Triton test.

## Prototype

```cpp
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> npu_layer_norm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    double eps,
    const c10::optional<torch::Tensor>& z,
    int64_t group_size,
    bool norm_before_gate,
    bool is_rms_norm);
```

The custom ACLNN entry expected by the wrapper is:

```text
aclnnXllmLayerNormFwd(
    Tensor x2d,
    Tensor weight,
    Tensor? bias,
    Tensor? z2d,
    float eps,
    int group_size,
    bool norm_before_gate,
    bool is_rms_norm,
    Tensor y,
    Tensor mean,
    Tensor rstd)
```

## Tiling strategy

The host side should choose one of two device paths.

### Mode 0: full row in UB

Used when one logical group fits in UB with FP32 compute buffers.

Per tile, one core handles `tile_rows` rows within the same group. This allows
one multi-row `DataCopyPad` for x, z, and y, while loading weight and bias once
per tile and reusing them across all rows.

Recommended host rule:

```text
group_align = align_up(group_size, 32 / sizeof(x_dtype))
row_bytes = group_align * (
    2 * sizeof(x_dtype)      // x and y queues
  + has_z * sizeof(x_dtype)  // z queue
  + 3 * sizeof(float)        // x_fp32, tmp_fp32, reduce backup
)
param_bytes = group_align * (
    sizeof(weight_dtype) + sizeof(float)
  + has_bias * (sizeof(weight_dtype) + sizeof(float))
)
tile_rows = floor((ub_size - param_bytes - reduce_tmp_bytes - scalar_bytes) / row_bytes)
tile_rows = clamp(tile_rows, 1, 255)
```

If `M < core_num`, set `tile_rows = 1` and distribute linear `(group,row)`
pairs across cores. Otherwise, distribute row blocks within each group.

### Mode 1: streaming two-pass

Used when a complete group does not fit in UB.

Each core handles one `(group,row)` at a time. Pass 1 streams chunks of
`chunk_elems` to accumulate `sum` and `sum_sq` in FP32. Pass 2 rereads the same
chunks, applies normalization, affine transform, optional post-gate, and writes
output. This intentionally avoids writing intermediate x/gate values to GM.

Recommended host rule:

```text
chunk_align = align_down(
    (ub_size - fixed_buffers) / buffer_coefficient,
    32 / sizeof(x_dtype))
chunk_elems = min(group_size, chunk_align)
```

## UB allocation

Mode 0 full-row path:

| Buffer | Count | Type | Bytes |
| --- | ---: | --- | ---: |
| x queue | 2 | input dtype | `2 * tile_rows * group_align * sizeof(T)` |
| y queue | 2 | input dtype | `2 * tile_rows * group_align * sizeof(T)` |
| z queue | 2 if z exists | input dtype | `2 * tile_rows * group_align * sizeof(T)` |
| x_fp32 | 1 | float | `tile_rows * group_align * 4` |
| tmp_fp32 | 1 | float | `tile_rows * group_align * 4` |
| reduce_tmp | 1 | float | `max(group_align, 64) * 4` |
| scalar | 1 | float | `8 * 4` |
| weight raw/fp32 | 1 each | weight dtype/float | `group_align * (sizeof(W) + 4)` |
| bias raw/fp32 | 1 each if bias exists | weight dtype/float | `group_align * (sizeof(W) + 4)` |

Mode 1 streaming path uses the same logical buffers with `tile_rows = 1` and
`group_align = chunk_align`.

## Performance notes learned from ops-nn

- Use `DataCopyPad` for all GM <-> UB movement.
- Keep x, z gate, weight, bias, normalized output, and temporaries in UB.
- Load weight/bias once per group tile and broadcast across rows.
- Use FP32 for all reductions and nonlinear gate computation.
- For small hidden sizes, batch multiple rows per core to raise single DMA
  burst size and amortize weight loads.
- For large hidden sizes, recompute from GM instead of spilling intermediate
  normalized/gated tensors to GM.
- Keep branches at tile level or row level; do not branch per element.

## Supported cases

- input dtype: fp16, bf16, fp32
- weight/bias dtype: same as input or fp32
- optional bias
- optional z gate
- both `norm_before_gate` orders
- normal LayerNorm and RMSNorm
- `group_size <= fullN`, `fullN % group_size == 0`

## Known integration requirement

This directory contains the generated AscendC kernel and the xLLM C++ wrapper.
To run it end to end, package the custom op so that `libcust_opapi.so` exports
`aclnnXllmLayerNormFwd` and `aclnnXllmLayerNormFwdGetWorkspaceSize`.
