#!/usr/bin/env python3
# Copyright 2025 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
npu_python_extension_lib = pytest.importorskip("npu_python_extension_lib")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_gemma_rms_norm(dtype):
    # Device selection (skip if no NPU available)
    try:
        torch_npu.npu.set_device(0)
    except Exception as e:
        pytest.skip(f"NPU device not available: {e}")

    torch.manual_seed(1234)

    batch = 4
    seq_len = 128
    hidden_size = 4096
    epsilon = 1e-6

    # Create input tensors
    x = torch.randn(batch, seq_len, hidden_size, dtype=dtype).npu()
    gamma = torch.randn(hidden_size, dtype=dtype).npu()

    # Call the NPU function
    rstd_out, y_out = npu_python_extension_lib.gemma_rms_norm(x, gamma, epsilon)

    # Basic checks
    assert y_out.shape == x.shape
    assert rstd_out.shape[0] == batch
    assert rstd_out.shape[1] == seq_len

    # Check that output is on NPU
    assert y_out.device.type == "npu"
    assert rstd_out.device.type == "npu"

    print(f"test_gemma_rms_norm passed with dtype={dtype}")


def _layer_norm_golden_cpu(
    x,
    weight,
    bias=None,
    eps=1e-6,
    z=None,
    group_size=-1,
    norm_before_gate=True,
    is_rms_norm=False,
):
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

    x_grouped = x.reshape(m, ngroups, group_size).reshape(-1, group_size)
    if is_rms_norm:
        rstd = torch.rsqrt(torch.mean(x_grouped * x_grouped, dim=-1, keepdim=True) + eps)
        y_grouped = x_grouped * rstd
    else:
        y_grouped = torch.layer_norm(x_grouped, (group_size,), eps=eps)

    y = y_grouped.reshape(m, ngroups, group_size).reshape(m, full_n)
    y = y * weight
    if bias is not None:
        y = y + bias
    if z is not None and norm_before_gate:
        y = y * (z * torch.sigmoid(z))
    return y.reshape(x_shape)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape,group_size,has_bias,has_z,norm_before_gate,is_rms_norm",
    [
        ((2, 8, 128), 128, False, True, True, False),
        ((2, 8, 128), 128, True, True, False, False),
        ((4, 16, 256), 64, True, False, True, False),
        ((1, 4, 64), 32, False, True, False, True),
        ((1, 128, 4096), 4096, False, False, True, True),
    ],
)
def test_layer_norm_fwd(dtype, shape, group_size, has_bias, has_z, norm_before_gate, is_rms_norm):
    try:
        torch_npu.npu.set_device(0)
    except Exception as e:
        pytest.skip(f"NPU device not available: {e}")

    torch.manual_seed(2026)
    x = torch.randn(*shape, dtype=dtype)
    weight = torch.randn(shape[-1], dtype=dtype)
    bias = torch.randn(shape[-1], dtype=dtype) if has_bias else None
    z = torch.randn(*shape, dtype=dtype) if has_z else None

    golden = _layer_norm_golden_cpu(
        x,
        weight,
        bias=bias,
        eps=1e-6,
        z=z,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
    )

    try:
        y_out, mean_out, rstd_out = npu_python_extension_lib.layer_norm_fwd(
            x.npu(),
            weight.npu(),
            None if bias is None else bias.npu(),
            1e-6,
            None if z is None else z.npu(),
            group_size,
            norm_before_gate,
            is_rms_norm,
        )
    except RuntimeError as e:
        if "aclnnXllmLayerNormFwd" in str(e):
            pytest.skip("custom aclnnXllmLayerNormFwd is not installed")
        raise

    atol, rtol = (1e-5, 1e-5) if dtype is torch.float32 else (2e-2, 2e-2)
    torch.testing.assert_close(y_out.cpu(), golden.to(dtype=dtype), atol=atol, rtol=rtol)
    assert rstd_out.shape == (shape[-1] // group_size * (x.numel() // shape[-1]),)
    if is_rms_norm:
        assert mean_out.numel() == 0
    else:
        assert mean_out.shape == rstd_out.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
