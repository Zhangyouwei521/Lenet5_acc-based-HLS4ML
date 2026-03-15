# LeNet5 + hls4ml (MNIST, Zynq-7100, Vitis 2024.2)

该工程包含三条完整流程：
1. 使用 PyTorch 训练/测试 LeNet5 并导出 `.pth`。
2. 使用 hls4ml 将 PyTorch 模型转换为 HLS 工程，执行 C 仿真/综合/联合仿真并导出 IP。
3. 从 PyTorch 模型导出黄金数据，供 C 仿真与联合仿真测试使用。

## 1) 环境准备

- Python 3.10+（建议）
- 已安装并可用的 Xilinx 工具链（与你的软件版本一致）：
  - Vitis HLS 2024.2
  - Vivado 2024.2
- 板卡目标：Zynq-7100（默认器件：`xc7z100ffg900-2`，可在配置里改）

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

> 运行 hls4ml build 前，请先在终端 source 你的 Xilinx 环境脚本。

## 2) 训练 + 测试 + 导出 .pth

```bash
python train_lenet5.py --epochs 10 --output ./artifacts/lenet5_mnist.pth
```

训练脚本会在每个 epoch 输出 train/test 指标，并导出模型权重文件。

## 3) 导出黄金数据（给 C 仿真 / 联合仿真）

```bash
python export_golden_data.py \
  --model ./artifacts/lenet5_mnist.pth \
  --num-samples 200 \
  --out-dir ./golden_data \
  --export-layer-output
```

导出内容：
- `golden_data/inputs.npy`
- `golden_data/logits.npy`
- `golden_data/labels.npy`
- `golden_data/preds.npy`
- `golden_data/tb_input_features.dat`（每行一个样本，展平输入）
- `golden_data/tb_output_logits.dat`（每行一个样本，logits）
- `golden_data/tb_output_predictions.dat`（hls4ml testbench 直接读取的预测文件）
- `golden_data/tb_output_class.dat`（每行一个预测类别）
- `golden_data/layer_outputs/*.npy`（可选，每层输出）

## 4) PyTorch -> hls4ml 转换、仿真综合、导出 IP

`hls4ml` 配置已直接写在 `convert_hls4ml.py` 中，不再使用 `.yaml`。

你可直接在脚本顶部修改这 3 个字典：
- `HLS_MODEL_CONFIG`：项目、器件、时钟、默认精度等
- `HLS_BUILD_CONFIG`：`csim/synth/cosim/export/vsynth` 开关
- `LAYER_OPTIMIZATIONS`：逐层优化（`"*"` 全局默认 + 指定层覆盖）

执行：

```bash
python convert_hls4ml.py --model ./artifacts/lenet5_mnist.pth
```

如果你暂时还没 `source` Xilinx 环境，可先只做模型转换（不执行 build）：

```bash
python convert_hls4ml.py --model ./artifacts/lenet5_mnist.pth --no-build
```

脚本将：
- 自动从 PyTorch 模型生成完整 `hls_config`
- 合并脚本里定义的逐层优化设置
- 若存在 `golden_data`，自动复制 `tb_input_features.dat` 和 `tb_output_predictions.dat` 到 `hls_project/tb_data`
- 调用 `hls_model.build(...)` 执行 `csim/synth/cosim/export`

## 5) 逐层优化可配置说明（所有层可设置）

在 `convert_hls4ml.py` 的 `LAYER_OPTIMIZATIONS` 下：

- `"*"`：通配默认配置（会应用到所有层）
- `conv1 / conv2 / fc1 / fc2 / fc3 ...`：按层名覆盖

可配置项示例（按 hls4ml 层配置键）：
- `ReuseFactor`
- `Strategy`
- `Precision.result / Precision.weight / Precision.bias`
- 以及其它 hls4ml 支持的 layer config 字段

## 6) 与 Zynq-7100 / Vitis 2024.2 相关参数

默认已设置：
- `backend: Vitis`
- `part: xc7z100ffg900-2`
- `clock_period: 10`

你可按板卡工程实际需求修改 `convert_hls4ml.py` 中的 `HLS_MODEL_CONFIG`：
- 若你工程使用不同速度等级或封装，修改 `part`
- 若你追求更高频率，减小 `clock_period`
- 在 `HLS_BUILD_CONFIG` 中控制 `csim/synth/cosim/export` 开关

## 7) 建议执行顺序

```bash
python train_lenet5.py --epochs 10 --output ./artifacts/lenet5_mnist.pth
python export_golden_data.py --model ./artifacts/lenet5_mnist.pth --num-samples 200 --out-dir ./golden_data --export-layer-output
python convert_hls4ml.py --model ./artifacts/lenet5_mnist.pth
```
