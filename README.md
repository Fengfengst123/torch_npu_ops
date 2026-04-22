# torch_npu_ops

## triton_npu 算子编译流程

依赖：

```bash
pip install triton-ascend==3.2.0 pybind11
```

准备：调用 `triton_npu` 目录下的 `setup.py`。该步骤会在 xLLM 编译阶段自动执行，也可以手动运行来更新 Triton Ascend 的 AOT 产物。

```bash
python3 setup.py
```

生成产物默认在 `triton_npu/binary` 目录下。

## xLLM wheel 的 Triton NPU runtime 资产

`triton_npu/binary` 下的 `.npubin` 和 `.json` 文件属于运行时必需资产，不只是编译中间产物。

xLLM 根目录的 `python setup.py bdist_wheel` 会在 NPU wheel 构建阶段把这些 runtime 资产拷贝到 wheel 内的 `xllm/triton_npu/binary`。安装后的运行时查找顺序如下：

1. 显式传入的 `npubin_path`
2. 环境变量 `TRITON_BINARY_PATH`
3. 已安装模块或可执行文件旁边的 `triton_npu/binary`
4. 编译期的 `TRITON_BINARY_PATH` 兜底路径

如果安装后仍出现 `Failed to load binary file for kernel ...`，先检查 wheel 内是否包含 `xllm/triton_npu/binary/*.npubin` 与对应 `.json`，再检查是否有外部环境变量把查找路径覆盖到了错误目录。

