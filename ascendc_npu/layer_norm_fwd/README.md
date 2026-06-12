# xllm_layer_norm_fwd

Generated AscendC replacement for the original Triton layernorm forward test.

Files:

- `design.md`: operator spec, tiling, and UB plan.
- `op_kernel/xllm_layer_norm_fwd.cpp`: AscendC kernel implementation.
- `op_host/xllm_layer_norm_fwd_tiling.h`: host-side launch config helper.
- `test/layer_norm_fwd_test_cases.md`: precision and performance case matrix.

The existing `npu_python_extension` wrapper exposes
`npu_python_extension_lib.layer_norm_fwd`. Runtime execution requires a custom
OPP package that exports `aclnnXllmLayerNormFwd`.
