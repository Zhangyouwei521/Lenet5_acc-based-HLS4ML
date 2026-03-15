import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import LeNet5


def load_model(model_path: Path, device: torch.device) -> LeNet5:
    checkpoint = torch.load(model_path, map_location=device)
    model = LeNet5().to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def save_txt_matrix(file_path: Path, array: np.ndarray) -> None:
    with file_path.open("w", encoding="utf-8") as f:
        for row in array:
            flat = row.reshape(-1)
            f.write(" ".join(f"{x:.8f}" for x in flat) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Export golden input/output data from PyTorch LeNet5")
    parser.add_argument("--model", type=str, default="./artifacts/lenet5_mnist.pth")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="./golden_data")
    parser.add_argument("--export-layer-output", action="store_true", help="Export each named layer output")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device(args.device)
    model = load_model(Path(args.model), device)

    all_inputs: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    layer_outputs: Dict[str, List[np.ndarray]] = {}

    hooks = []
    if args.export_layer_output:
        for name, module in model.named_modules():
            if name == "":
                continue

            def save_output(module_name):
                def _hook(_module, _input, output):
                    layer_outputs.setdefault(module_name, []).append(output.detach().cpu().numpy())

                return _hook

            hooks.append(module.register_forward_hook(save_output(name)))

    collected = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)

            batch_size = images.size(0)
            remaining = args.num_samples - collected
            take = min(batch_size, remaining)
            if take <= 0:
                break

            all_inputs.append(images[:take].cpu().numpy())
            all_logits.append(logits[:take].cpu().numpy())
            all_labels.append(labels[:take].cpu().numpy())

            collected += take
            if collected >= args.num_samples:
                break

    for h in hooks:
        h.remove()

    inputs = np.concatenate(all_inputs, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = logits.argmax(axis=1)

    np.save(out_dir / "inputs.npy", inputs)
    np.save(out_dir / "logits.npy", logits)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "preds.npy", preds)

    save_txt_matrix(out_dir / "tb_input_features.dat", inputs)
    save_txt_matrix(out_dir / "tb_output_logits.dat", logits)
    save_txt_matrix(out_dir / "tb_output_predictions.dat", logits)
    np.savetxt(out_dir / "tb_output_class.dat", preds.reshape(-1, 1), fmt="%d")

    if args.export_layer_output:
        layer_dir = out_dir / "layer_outputs"
        layer_dir.mkdir(parents=True, exist_ok=True)
        for name, chunks in layer_outputs.items():
            if not chunks:
                continue
            data = np.concatenate(chunks, axis=0)[: args.num_samples]
            np.save(layer_dir / f"{name}.npy", data)

    meta = {
        "num_samples": int(inputs.shape[0]),
        "input_shape": list(inputs.shape),
        "logits_shape": list(logits.shape),
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Golden data exported to: {out_dir}")


if __name__ == "__main__":
    main()
