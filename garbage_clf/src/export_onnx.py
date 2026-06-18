"""
export_onnx.py – Export PyTorch model sang ONNX
"""
import argparse, json
from pathlib import Path

import torch
import onnx


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True, help="Path to best_model.pt")
    p.add_argument("--config",      required=True, help="Path to config.json")
    p.add_argument("--output_onnx", default="outputs/models/best_model.onnx")
    p.add_argument("--img_size",    type=int, default=224)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    from src.models import build_model
    model_name = cfg.get("model_name", "cnn_baseline")
    train_mode = cfg.get("train_mode", "finetune")
    dropout    = cfg.get("dropout", 0.3)

    # Load class_to_idx để biết num_classes
    cfg_dir = Path(args.config).parent
    with open(cfg_dir / "class_to_idx.json") as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)

    model = build_model(model_name, num_classes, dropout, train_mode)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, args.img_size, args.img_size)
    Path(args.output_onnx).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, args.output_onnx,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )

    # Verify
    onnx_model = onnx.load(args.output_onnx)
    onnx.checker.check_model(onnx_model)

    size_mb = Path(args.output_onnx).stat().st_size / 1e6
    pt_size  = Path(args.checkpoint).stat().st_size / 1e6
    print(f"[export] ONNX saved: {args.output_onnx}")
    print(f"[export] PyTorch size: {pt_size:.1f} MB")
    print(f"[export] ONNX size:    {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
