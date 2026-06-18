"""
error_analysis.py – Phân tích 10+ mẫu dự đoán sai
"""
import argparse, json
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     required=True)
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--n_samples",  type=int, default=20)
    p.add_argument("--out_dir",    default="outputs/figures/error_analysis")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f: cfg = json.load(f)
    cfg_dir = Path(args.config).parent
    with open(cfg_dir / "class_to_idx.json") as f: class_to_idx = json.load(f)
    class_names = list(class_to_idx.keys())

    from src.models import build_model
    model = build_model(cfg["model_name"], len(class_names), cfg.get("dropout", 0.3), cfg.get("train_mode","finetune"))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    img_size = cfg.get("img_size", 224)
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    wrong_samples = []
    for cls_name, cls_idx in class_to_idx.items():
        cls_dir = Path(args.data_dir) / cls_name
        if not cls_dir.exists(): continue
        imgs = list(cls_dir.rglob("*.jpg")) + list(cls_dir.rglob("*.jpeg")) + list(cls_dir.rglob("*.png"))
        for img_path in imgs:
            img_t = tf(Image.open(img_path).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                logits = model(img_t)
                probs  = torch.softmax(logits, dim=1)[0]
                pred_idx = logits.argmax(1).item()
            if pred_idx != cls_idx:
                wrong_samples.append({
                    "path": img_path,
                    "true_cls": cls_name,
                    "true_idx": cls_idx,
                    "pred_cls": class_names[pred_idx],
                    "pred_idx": pred_idx,
                    "confidence": probs[pred_idx].item(),
                })
            if len(wrong_samples) >= args.n_samples * 3:
                break
        if len(wrong_samples) >= args.n_samples * 3:
            break

    # Lấy n_samples mẫu sai
    wrong_samples = wrong_samples[:args.n_samples]
    print(f"[error_analysis] Tìm thấy {len(wrong_samples)} mẫu sai")

    # Vẽ grid ảnh sai
    n = min(len(wrong_samples), 20)
    cols = 4; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    for i, sample in enumerate(wrong_samples[:n]):
        img = Image.open(sample["path"]).convert("RGB").resize((150, 150))
        axes[i].imshow(img)
        axes[i].set_title(f"True: {sample['true_cls']}\nPred: {sample['pred_cls']}\n({sample['confidence']:.0%})",
                          fontsize=8)
        axes[i].axis("off")

    for i in range(n, len(axes)): axes[i].axis("off")
    plt.suptitle("Wrong Predictions", fontsize=12); plt.tight_layout()
    fig.savefig(out_dir / "wrong_predictions.png", dpi=120); plt.close(fig)

    # Lưu CSV
    df = pd.DataFrame([{k: v for k, v in s.items() if k != "path"} for s in wrong_samples])
    df["image"] = [str(s["path"]) for s in wrong_samples]
    df.to_csv(out_dir / "error_analysis.csv", index=False)

    # Lưu markdown report
    with open("reports/error_analysis.md", "w", encoding="utf-8") as f:
        f.write("# Error Analysis\n\n")
        f.write(f"Tổng số mẫu sai phân tích: {len(wrong_samples)}\n\n")
        f.write("## Thống kê nhầm lẫn theo lớp\n\n")
        confusion_pairs = {}
        for s in wrong_samples:
            key = f"{s['true_cls']} → {s['pred_cls']}"
            confusion_pairs[key] = confusion_pairs.get(key, 0) + 1
        for pair, cnt in sorted(confusion_pairs.items(), key=lambda x: -x[1]):
            f.write(f"- **{pair}**: {cnt} lần\n")
        f.write("\n## Chi tiết 10 mẫu sai tiêu biểu\n\n")
        f.write("| # | True | Predicted | Confidence |\n")
        f.write("|---|------|-----------|------------|\n")
        for i, s in enumerate(wrong_samples[:10]):
            f.write(f"| {i+1} | {s['true_cls']} | {s['pred_cls']} | {s['confidence']:.2%} |\n")

    print(f"[error_analysis] Xong! Kết quả tại {out_dir}")
    print(f"[error_analysis] Report: reports/error_analysis.md")


if __name__ == "__main__":
    main()
