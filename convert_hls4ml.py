import argparse
import copy
import shutil
from pathlib import Path
from typing import Any, Dict

import hls4ml
import torch

from model import LeNet5


# =========================
# 1) 在代码中直接配置 HLS 参数
# =========================
# 目标板卡：Zynq-7100（默认 part 为 xc7z100ffg900-2）
HLS_MODEL_CONFIG: Dict[str, Any] = {
    "backend": "Vitis",
    "project_name": "lenet5_hls",
    "output_dir": "./hls_project",
    "part": "xc7z100ffg900-2",
    "clock_period": 10,
    # io_stream：卷积层用滑窗流式处理，彻底消去 fill_buffer 膨胀，
    # 是综合速度慢和指令数爆炸的最主要解决手段；
    # io_parallel 会把所有权重缓冲展开，导致 10 万级指令数。
    "io_type": "io_stream",
    "default_precision": "ap_fixed<16,6,AP_RND,AP_SAT>",
    # 默认 ReuseFactor=1 表示完全展开，会生成巨量硬件资源。
    # 根据各层 fan_in 设置，综合工具压力大幅下降。
    "default_reuse_factor": 1,
    # Resource 策略配合 io_stream，让工具做共享乘法器复用，
    # 而不是 Latency 策略的全展开流水线。
    "strategy": "Resource",
}


# 控制 hls4ml build 的流程开关
HLS_BUILD_CONFIG: Dict[str, bool] = {
    "csim": False,
    "synth": False,
    "cosim": False,
    "export": False,
    "vsynth": True,
}


# 所有层都可配置：
# - "*" 表示给所有层的默认配置
# - 具体层名（如 conv1/fc1）用于覆盖默认配置
LAYER_OPTIMIZATIONS: Dict[str, Dict[str, Any]] = {
    # "*" 作用于所有层，单层配置会在此基础上覆盖
    "*": {
        "ReuseFactor": 1,
        "Strategy": "Resource",   # 全局用 Resource，与 io_stream 匹配
        "Precision": {
            "result": "ap_fixed<16,6,AP_RND,AP_SAT>",
            "weight": "ap_fixed<16,6,AP_RND,AP_SAT>",
            "bias": "ap_fixed<16,6,AP_RND,AP_SAT>",
            #"accum": "ap_fixed<32,12,AP_RND,AP_SAT>",
        },
    },
    # conv1: kernel=5x5, in_ch=1, out_ch=6
    #   fan_in = 5*5*1 = 25
    #   ReuseFactor 可取 1~25（25 表示最大复用，最省资源，延迟最高）
    "conv1": {
        "ReuseFactor": 25,
        "Strategy": "Resource",
    },
    # conv2: kernel=5x5, in_ch=6, out_ch=16
    #   fan_in = 5*5*6 = 150
    #   ReuseFactor 可取 1~150
    "conv2": {
        "ReuseFactor": 150,
        "Strategy": "Resource",
    },
    # fc1: in=256, out=120
    #   ReuseFactor 可取 1~256
    "fc1": {
        "ReuseFactor": 64,
        "Strategy": "Resource",
    },
    # fc2: in=120, out=84
    #   ReuseFactor 可取 1~120
    "fc2": {
        "ReuseFactor": 60,
        "Strategy": "Resource",
    },
    # fc3: in=84, out=10
    #   ReuseFactor 可取 1~84
    "fc3": {
        "ReuseFactor": 42,
        "Strategy": "Resource",
    },
}


def check_hls_toolchain(backend: str) -> None:
    backend_lower = backend.lower()
    if backend_lower == "vitis":
        required_cmd = "vitis_hls"
    elif backend_lower == "vivado":
        required_cmd = "vivado_hls"
    else:
        return

    if shutil.which(required_cmd) is None:
        raise RuntimeError(
            f"{required_cmd} not found in PATH. Please source your Xilinx 2024.2 environment first, "
            f"or run with --no-build to generate project files only."
        )


def deep_update(original: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(original)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_layer_optimizations(hls_config: Dict[str, Any], layer_cfg: Dict[str, Any]) -> Dict[str, Any]:
    # hls4ml 的按层配置都放在 LayerName 字段
    if "LayerName" not in hls_config:
        hls_config["LayerName"] = {}

    # 先应用全局默认（"*"）
    wildcard_cfg = layer_cfg.get("*", {})
    for layer_name in list(hls_config["LayerName"].keys()):
        hls_config["LayerName"][layer_name] = deep_update(hls_config["LayerName"][layer_name], wildcard_cfg)

    # 再应用每层覆盖配置
    for layer_name, config in layer_cfg.items():
        if layer_name == "*":
            continue
        if layer_name not in hls_config["LayerName"]:
            hls_config["LayerName"][layer_name] = {}
        hls_config["LayerName"][layer_name] = deep_update(hls_config["LayerName"][layer_name], config)

    return hls_config


def load_pytorch_checkpoint(model_path: Path, device: torch.device) -> LeNet5:
    checkpoint = torch.load(model_path, map_location=device)
    model = LeNet5().to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def copy_n_lines(src: Path, dst: Path, n: int) -> int:
    """将 src 的前 n 行写入 dst，返回实际写入的行数。"""
    with src.open("r") as fin, dst.open("w") as fout:
        for i, line in enumerate(fin):
            if i >= n:
                break
            fout.write(line)
    return min(n, sum(1 for _ in src.open()))


def copy_golden_data_to_tb_data(golden_dir: Path, output_dir: Path, cosim_samples: int) -> None:
    tb_data_dir = output_dir / "tb_data"
    tb_data_dir.mkdir(parents=True, exist_ok=True)

    input_file = golden_dir / "tb_input_features.dat"
    pred_file = golden_dir / "tb_output_predictions.dat"
    logits_file = golden_dir / "tb_output_logits.dat"

    if not input_file.exists():
        raise FileNotFoundError(f"Golden input file not found: {input_file}")

    if pred_file.exists():
        source_prediction_file = pred_file
    elif logits_file.exists():
        source_prediction_file = logits_file
    else:
        raise FileNotFoundError(
            "Golden prediction file not found. Expected tb_output_predictions.dat or tb_output_logits.dat"
        )

    # CoSim 只用验证功能正确性，截取前 N 个样本可大幅缩短 RTL 仿真时间。
    # 单次推理 Interval ≈ 25,090 cycles；200 样本会让 CoSim 跑数十分钟，5 个样本约 1 分钟。
    written = copy_n_lines(input_file,          tb_data_dir / "tb_input_features.dat",  cosim_samples)
    copy_n_lines(source_prediction_file,         tb_data_dir / "tb_output_predictions.dat", cosim_samples)
    print(f"CoSim testbench: {written} sample(s) written (--cosim-samples={cosim_samples}).")


def main():
    parser = argparse.ArgumentParser(description="Convert LeNet5 PyTorch model to hls4ml project")
    parser.add_argument("--model", type=str, default="./artifacts/lenet5_mnist.pth", help="Path to .pth")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs="+",
        default=[1, 28, 28],
        help="Input shape without batch dimension. Recommended: 1 28 28",
    )
    parser.add_argument(
        "--golden-dir",
        type=str,
        default="./golden_data",
        help="Directory containing exported golden data for csim/cosim",
    )
    parser.add_argument(
        "--cosim-samples",
        type=int,
        default=5,
        help="Number of test vectors copied into tb_data for CoSim (default: 5). "
             "RTL仿真每个样本约需 25,090 cycles，减少样本可大幅缩短 CoSim 时间。",
    )
    parser.add_argument("--no-build", action="store_true", help="Only convert and compile hls4ml model, do not run build")
    args = parser.parse_args()

    input_shape = list(args.input_shape)
    if len(input_shape) == 4:
        input_shape = input_shape[1:]

    if len(input_shape) != 3:
        raise ValueError(
            f"Invalid --input-shape {args.input_shape}. Use C H W (e.g. --input-shape 1 28 28)."
        )

    device = torch.device(args.device)
    model = load_pytorch_checkpoint(Path(args.model), device)

    # 2) 从 PyTorch 自动生成 hls4ml 基础配置
    default_precision = HLS_MODEL_CONFIG.get("default_precision", "ap_fixed<16,6>")
    default_reuse_factor = HLS_MODEL_CONFIG.get("default_reuse_factor", 1)
    strategy = HLS_MODEL_CONFIG.get("strategy", "Latency")

    hls_config = hls4ml.utils.config_from_pytorch_model(
        model,
        input_shape=tuple(input_shape),
        granularity="name",
        default_precision=default_precision,
        default_reuse_factor=default_reuse_factor,
    )

    hls_config.setdefault("Model", {})
    hls_config["Model"]["Strategy"] = strategy

    # 3) 合并逐层优化参数（所有层可配置）
    hls_config = apply_layer_optimizations(hls_config, LAYER_OPTIMIZATIONS)

    output_dir = Path(HLS_MODEL_CONFIG.get("output_dir", "./hls_project"))
    output_dir.mkdir(parents=True, exist_ok=True)

    golden_dir = Path(args.golden_dir)
    if golden_dir.exists():
        copy_golden_data_to_tb_data(golden_dir, output_dir, cosim_samples=args.cosim_samples)
        print(f"Copied golden data into testbench directory: {output_dir / 'tb_data'}")
    else:
        print(f"Golden data directory not found, csim/cosim may fall back to default input: {golden_dir}")

    # 4) 生成 hls4ml 工程对象
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        hls_config=hls_config,
        output_dir=str(output_dir),
        project_name=HLS_MODEL_CONFIG.get("project_name", "lenet5_hls"),
        backend=HLS_MODEL_CONFIG.get("backend", "Vivado"),
        part=HLS_MODEL_CONFIG.get("part", "xc7z100ffg900-2"),
        clock_period=HLS_MODEL_CONFIG.get("clock_period", 10),
        io_type=HLS_MODEL_CONFIG.get("io_type", "io_parallel"),
        input_shape=tuple(input_shape),
    )

    # 5) 编译并执行 CSim / Synthesis / CoSim / Export
    hls_model.compile()

    if args.no_build:
        print("--no-build enabled: skip hls_model.build().")
        print(f"HLS project generated at: {output_dir}")
        return

    check_hls_toolchain(HLS_MODEL_CONFIG.get("backend", "Vitis"))

    build_kwargs = {
        "csim": bool(HLS_BUILD_CONFIG.get("csim", True)),
        "synth": bool(HLS_BUILD_CONFIG.get("synth", True)),
        "cosim": bool(HLS_BUILD_CONFIG.get("cosim", True)),
        "export": bool(HLS_BUILD_CONFIG.get("export", True)),
        "vsynth": bool(HLS_BUILD_CONFIG.get("vsynth", False)),
    }

    print(f"Running hls_model.build with options: {build_kwargs}")
    hls_model.build(**build_kwargs)
    print(f"HLS project generated at: {output_dir}")


if __name__ == "__main__":
    main()
