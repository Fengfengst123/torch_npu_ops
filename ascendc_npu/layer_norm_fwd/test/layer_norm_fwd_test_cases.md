# xllm_layer_norm_fwd test cases

## SUPPORTED_DTYPES

- `torch.float16`
- `torch.bfloat16`
- `torch.float32`

## TEST_SHAPES

| x shape | group_size | bias | z | norm_before_gate | is_rms_norm |
| --- | ---: | --- | --- | --- | --- |
| `[2, 8, 128]` | 128 | false | true | true | false |
| `[2, 8, 128]` | 128 | true | true | false | false |
| `[4, 16, 256]` | 64 | true | false | true | false |
| `[1, 4, 64]` | 32 | false | true | false | true |
| `[1, 2048, 128]` | 128 | false | true | true | true |
| `[1, 512, 1024]` | 1024 | true | false | true | false |
| `[1, 128, 4096]` | 4096 | false | false | true | true |
| `[1, 32, 8192]` | 8192 | true | true | true | true |

## GENERAL_SHAPES

| x shape | group_size | reason |
| --- | ---: | --- |
| `[1, 1, 128]` | 128 | decode, low M |
| `[8, 1, 128]` | 128 | many batch rows |
| `[1, 4096, 128]` | 128 | prefill, small hidden |
| `[1, 128, 2048]` | 256 | grouped normalization |
| `[1, 16, 16384]` | 16384 | streaming two-pass path |

## BOUNDARY_VALUES

- all zeros
- all ones
- repeated constant rows
- small values around `1e-3`
- large values around `10.0`
- mixed positive and negative random values

## CPU reference

```python
def layer_norm_golden_cpu(x, weight, bias=None, eps=1e-6, z=None,
                          group_size=None, norm_before_gate=True,
                          is_rms_norm=False):
    x_shape = x.shape
    x = x.reshape(-1, x.shape[-1]).float().cpu().contiguous()
    m, full_n = x.shape
    if group_size is None or group_size < 0:
        group_size = full_n
    ngroups = full_n // group_size
    weight = weight.float().cpu().contiguous()
    bias = None if bias is None else bias.float().cpu().contiguous()
    z = None if z is None else z.reshape(m, full_n).float().cpu().contiguous()

    if z is not None and not norm_before_gate:
        x = x * (z * torch.sigmoid(z))

    xg = x.reshape(m, ngroups, group_size).reshape(-1, group_size)
    if is_rms_norm:
        rstd = torch.rsqrt(torch.mean(xg * xg, dim=-1, keepdim=True) + eps)
        yg = xg * rstd
    else:
        yg = torch.layer_norm(xg, (group_size,), eps=eps)

    y = yg.reshape(m, ngroups, group_size).reshape(m, full_n)
    y = y * weight
    if bias is not None:
        y = y + bias
    if z is not None and norm_before_gate:
        y = y * (z * torch.sigmoid(z))
    return y.reshape(x_shape).to(dtype=x.dtype)
```

## Precision thresholds

- fp32: `atol=1e-5`, `rtol=1e-5`
- fp16/bf16: `atol=2e-2`, `rtol=2e-2`

The broader low-precision threshold covers the fused gate path and reduction
order differences between CPU and NPU.
