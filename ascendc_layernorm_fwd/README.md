# AscendC LayerNormFwd 算子说明

本文档说明 `ascendc_layernorm_fwd` 自定义算子的代码结构、执行逻辑和 xLLM 接入方式。该实现提供一个面向 NPU 的 LayerNorm/RMSNorm forward 算子，并支持可选 bias、可选 gated `z` 输入，以及按最后一维分组归一化。

## 目标

该算子用于替代 `triton_npu/triton_src/test_layernorm_fwd.py` 中已有的 Triton Ascend LayerNorm forward 路径。整体目标是：

- 在 AscendC 侧实现 `LayerNormFwd` 自定义算子。
- 通过 aclnn 形式暴露 `aclnnLayerNormFwd`。
- 在 `ascendc_npu` 中提供 torch C++ 包装函数。
- 在 `npu_python_extension` 中导出 Python 可调用接口。
- 在测试脚本中通过 `XLLM_LAYER_NORM_BACKEND=ascendc` 切换到 AscendC 后端。

## 文件结构

```text
ascendc_layernorm_fwd/
  CMakeLists.txt
  op_host/
    CMakeLists.txt
    layer_norm_fwd_def.cpp
    layer_norm_fwd_infershape.cpp
    layer_norm_fwd_tiling.cpp
    layer_norm_fwd_tiling.h
    op_api/
      layer_norm_fwd.h
      layer_norm_fwd.cpp
      aclnn_layer_norm_fwd.h
      aclnn_layer_norm_fwd.cpp
  op_kernel/
    CMakeLists.txt
    layer_norm_fwd.cpp
    layer_norm_fwd_common.h
    layer_norm_fwd_single_read.h
    layer_norm_fwd_single_read_z.h
    layer_norm_fwd_single_read_nomask.h
```

相关接入文件：

```text
ascendc_npu/
  ascendc_ops_api.h
  layer_norm_fwd.cpp
  CMakeLists.txt

npu_python_extension/
  op_register.cpp
  setup.py

triton_npu/triton_src/
  test_layernorm_fwd.py
```

## 算子接口

Host 侧算子名为 `LayerNormFwd`，aclnn 接口名为 `aclnnLayerNormFwd`。

输入：

- `x`: 必选，归一化输入。xLLM 包装层会把它 reshape 成二维 `[M, N]`。
- `gamma`: 可选，缩放权重，长度为 `N`。
- `beta`: 可选，偏置，长度为 `N`。
- `z`: 可选，门控输入，shape 与 `x` 相同。

输出：

- `y`: 必选，shape 与 `x` 相同，dtype 与 `x` 相同。
- `mean`: 可选，LayerNorm 时输出每个逻辑行的均值；RMSNorm 时不使用。
- `rstd`: 必选，输出每个逻辑行的 `1 / sqrt(var + eps)`。

属性：

- `eps`: 浮点数，默认 `1e-6`。
- `group_size`: 整数，默认 `0`。小于等于 0 时表示使用完整最后一维 `N`。
- `norm_before_gate`: 布尔值，默认 `true`。控制 `z * sigmoid(z)` 门控发生在归一化前还是归一化后。
- `is_rms_norm`: 布尔值，默认 `false`。为 `true` 时执行 RMSNorm，不计算 mean。

## 形状约定

xLLM C++ 包装层会把任意输入 shape 展平成二维：

```text
原始 x shape: [d0, d1, ..., N]
算子内部 x:  [M, N]
M = d0 * d1 * ...
```

如果 `group_size <= 0`，则：

```text
group_size = N
```

如果指定 `group_size`，必须满足：

```text
N % group_size == 0
```

归一化的逻辑行数是：

```text
group_count = N / group_size
logical_rows = M * group_count
```

每个逻辑行只处理连续的 `group_size` 个元素。`mean` 和 `rstd` 的长度都是 `logical_rows`，存储顺序与原 Triton 实现保持一致：

```text
param_offset = group * M + row
```

## 数学逻辑

### LayerNorm

当 `is_rms_norm == false` 时，对每个逻辑行 `x_row` 计算：

```text
mean = sum(x_row) / group_size
var = sum((x_row - mean)^2) / group_size
rstd = 1 / sqrt(var + eps)
x_norm = (x_row - mean) * rstd
```

然后应用 affine：

```text
y = x_norm * gamma + beta
```

如果没有传 `gamma`，则跳过乘法；如果没有传 `beta`，则跳过加法。

### RMSNorm

当 `is_rms_norm == true` 时，不计算均值：

```text
mean_square = sum(x_row^2) / group_size
rstd = 1 / sqrt(mean_square + eps)
x_norm = x_row * rstd
```

然后同样应用 `gamma` 和可选 `beta`。

### Gated z

如果传入 `z`，门控值为：

```text
gate = z * sigmoid(z)
```

当 `norm_before_gate == false` 时，先门控再归一化：

```text
x_input = x * gate
y = norm(x_input)
```

当 `norm_before_gate == true` 时，先归一化再门控：

```text
y = norm(x)
y = y * gate
```

这与原 Triton 测试脚本中的 golden reference 逻辑一致。

## Host 侧逻辑

### op 定义

`op_host/layer_norm_fwd_def.cpp` 注册 `LayerNormFwd`：

- 声明 `x/gamma/beta/z` 输入和 `y/mean/rstd` 输出。
- 支持 `float32` 和 `bfloat16` 输入。
- `gamma/beta` 支持 `float32` 或 `bfloat16`。
- 注册 `eps/group_size/norm_before_gate/is_rms_norm` 属性。
- 绑定 kernel 文件名 `layer_norm_fwd`。
- 添加 `ascend910b`、`ascend910_93`、`ascend950` 配置。

### infer shape

`op_host/layer_norm_fwd_infershape.cpp` 做输出 shape 和 dtype 推导：

- `y` shape 等于 `x` shape。
- `rstd` shape 为 `[M * group_count]`。
- LayerNorm 时 `mean` shape 为 `[M * group_count]`。
- RMSNorm 时 `mean` 不参与实际输出。
- `y` dtype 跟随 `x`，`mean/rstd` dtype 为 `float32`。

### tiling

`op_host/layer_norm_fwd_tiling.cpp` 将二维 `[M, N]` 转为逻辑行：

```text
logical_rows = M * group_count
row_size = group_size
row_align = align_up(row_size, 32 / sizeof(float))
```

当前实现每个 basic block 处理 1 个逻辑行：

```text
nRow = 1
tileLength = nRow * row_align
blockLength = nRow * row_size
```

block 数量按 AIV core 数和逻辑行数取较小值：

```text
numBlocks = min(logical_rows, core_num)
```

每个 block 处理一段连续 logical row，并通过 `loopCount/tailLoop/tailNRow` 描述普通循环和尾部。

tiling data 中保存：

- block 分布信息：`numBlocks/loopCount/tailLoop/tailNRow`
- 行尺寸信息：`rowSize/rowAlign/tileLength/blockLength`
- 功能开关：`nullptrGamma/nullptrBeta/hasZ/normBeforeGate/isRmsNorm/useNomask`
- 数学参数：`eps/coefficient`
- 分组信息：`inputRows/groupCount`

### tiling key

tiling key 用于 kernel 侧选择模板分支：

```text
100: 无 z，普通 single_read
200: 有 z，single_read_z
300: 无 z，nomask 分支
```

dtype 偏移：

```text
+0: x 为 float32
+20: x 为 bfloat16，gamma/beta 为 float32
+22: x 为 bfloat16，gamma/beta 为 bfloat16
```

因此实际 key 包括：

```text
100, 120, 122
200, 220, 222
300, 320, 322
```

`useNomask` 当前在 `!hasZ && rowSize == 128 && inputRows >= 1024` 时开启。当前 `nomask` 类继承普通 `single_read` 实现，保留独立 dispatch 点，便于后续替换成真正的无 mask 优化版本。

## Kernel 侧逻辑

入口文件是 `op_kernel/layer_norm_fwd.cpp`。

kernel 根据 tiling key 分发到三类模板：

- `LayerNormFwdSingleRead<Tfm, Tweight>`
- `LayerNormFwdSingleReadZ<Tfm, Tweight>`
- `LayerNormFwdSingleReadNomask<Tfm, Tweight>`

其中：

- `Tfm` 是激活输入/输出 dtype，支持 `float` 和 `bfloat16_t`。
- `Tweight` 是 `gamma/beta` dtype，支持 `float` 和 `bfloat16_t`。

### SingleRead 路径

`layer_norm_fwd_single_read.h` 实现无 `z` 的主路径。

每个 block 的执行流程：

1. `Init`
   - 读取 tiling data。
   - 根据 block id 计算当前 block 的起始 logical row。
   - 设置 `x/y/mean/rstd/gamma/beta` global tensor。
   - 初始化 local queue 和 buffer。

2. `Process`
   - 按 `loopCount/tailLoop` 遍历当前 block 负责的 logical row。
   - 每次调用 `ProcessBasicBlock`。

3. `LoadInput`
   - 从 GM 读取一行或多行到 local。
   - 如果输入是 bf16，先 cast 到 float32 做归一化计算。

4. 归一化计算
   - RMSNorm：计算 `sum(x^2) / row_size`，得到 `rstd`，再 `x * rstd`。
   - LayerNorm：先算 `mean`，再算 variance 和 `rstd`，再归一化。

5. `ApplyAffine`
   - 按 group offset 读取 `gamma/beta`。
   - bf16 权重会 cast 到 float32。
   - 依次执行乘 `gamma`、加 `beta`。

6. `CastAndStoreOutput`
   - 如果输出 dtype 是 bf16，将 float32 结果 cast 回 bf16。
   - 写回 GM 中对应 logical row。

### SingleReadZ 路径

`layer_norm_fwd_single_read_z.h` 在 SingleRead 的基础上增加 `z` 门控：

1. 读取 `x`。
2. 读取 `z` 并计算：

```text
gate = z * sigmoid(z)
```

3. 如果 `norm_before_gate == false`，先执行：

```text
x = x * gate
```

4. 执行 LayerNorm/RMSNorm 和 affine。
5. 如果 `norm_before_gate == true`，最后执行：

```text
y = y * gate
```

sigmoid 在 `layer_norm_fwd_common.h` 中封装为 `ComputeSigmoid`。默认使用：

```text
sigmoid(x) = reciprocal(exp(-x) + 1)
```

如果编译环境提供更合适的 AscendC sigmoid API，也可以通过宏切换。

## aclnn API 逻辑

`op_host/op_api/aclnn_layer_norm_fwd.cpp` 暴露两个 C 接口：

```cpp
aclnnLayerNormFwdGetWorkspaceSize(...)
aclnnLayerNormFwd(...)
```

`GetWorkspaceSize` 做以下事情：

1. 参数空指针检查。
2. dtype 检查。
3. shape 检查。
4. 对 `x/gamma/beta/z` 创建 contiguous 输入。
5. 调用 l0op `LayerNormFwd` 把算子加入 executor。
6. 返回 workspace size 和 executor。

`aclnnLayerNormFwd` 负责调用 `CommonOpExecutorRun` 真正执行 executor。

这里的 aclnn 接口会被 xLLM 侧 `EXEC_NPU_CMD(aclnnLayerNormFwd, ...)` 调用。

## xLLM C++ 包装逻辑

`ascendc_npu/layer_norm_fwd.cpp` 提供：

```cpp
torch::Tensor npu_ops::layer_norm_fwd(...)
```

主要逻辑：

1. 校验输入维度、`group_size`、`weight/bias/z` shape。
2. 将 `x` reshape 为 `[M, N]` 并 contiguous。
3. 将 `z` 同样 reshape 为 `[M, N]`。
4. 创建输出：

```text
y:    empty_like(x_2d)
mean: float32 [M * group_count]，仅 LayerNorm 创建
rstd: float32 [M * group_count]
```

5. 调用：

```cpp
EXEC_NPU_CMD(
    aclnnLayerNormFwd,
    x_2d,
    weight_contiguous,
    bias_contiguous,
    z_2d,
    y,
    mean,
    rstd,
    eps,
    group_size_val,
    norm_before_gate,
    is_rms_norm);
```

6. 将 `y` reshape 回原始输入 shape 后返回。

当前 Python 测试入口只需要返回 `y`，因此 C++ 包装层不返回 `mean/rstd`。

## Python 扩展接入

`npu_python_extension/setup.py` 将新增的 C++ 文件加入扩展编译：

```python
os.path.join(BASE_DIR, "..", "ascendc_npu", "layer_norm_fwd.cpp")
```

`npu_python_extension/op_register.cpp` 导出：

```python
npu_python_extension_lib.layer_norm_fwd(...)
```

这个接口直接调用 `npu_ops::layer_norm_fwd`。

## 测试脚本后端切换

`triton_npu/triton_src/test_layernorm_fwd.py` 中新增 AscendC 后端入口：

```python
XLLM_LAYER_NORM_BACKEND=ascendc
```

默认仍走原 Triton 后端：

```python
XLLM_LAYER_NORM_BACKEND=triton
```

当环境变量为 `ascendc` 时，`layer_norm_fwd` 会调用：

```python
_layer_norm_fwd_ascendc(...)
```

并最终调用：

```python
npu_python_extension_lib.layer_norm_fwd(...)
```

## 构建和运行步骤

在 Ascend/CANN 环境中需要先构建并安装自定义 OPP 包，然后构建 Python 扩展。

示意流程：

```bash
# 1. 构建并安装 ascendc_layernorm_fwd 自定义算子包
# 具体命令取决于当前 CANN 自定义算子工程脚手架。

# 2. 指向自定义 OPP 包
export ASCEND_CUSTOM_OPP_PATH=/path/to/custom_opp

# 3. 构建 Python 扩展
cd third_party/torch_npu_ops/npu_python_extension
python3 setup.py build bdist_wheel
pip3 install --force-reinstall dist/*.whl

# 4. 运行 LayerNorm 测试的 AscendC 后端
cd ../triton_npu/triton_src
XLLM_LAYER_NORM_BACKEND=ascendc pytest -v test_layernorm_fwd.py
```

## 当前实现边界

- 当前 kernel 采用 single-read、单 logical row 基本处理单元，逻辑清晰但不是最终性能上限。
- `nomask` 分支目前保留独立 tiling key 和 dispatch 类，但实现仍继承普通 `single_read`，后续可以替换成真正的无 mask 优化 kernel。
- 当前 Windows 开发环境无法完成 CANN 自定义算子编译和 NPU 运行验证，需要在 Ascend 环境中做最终构建和数值验证。
- Python 测试入口当前只返回 `y`，`mean/rstd` 用于内部计算和对齐算子接口，暂不暴露给 Python wrapper。

