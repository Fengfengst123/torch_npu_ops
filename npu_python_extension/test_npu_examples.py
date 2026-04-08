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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
