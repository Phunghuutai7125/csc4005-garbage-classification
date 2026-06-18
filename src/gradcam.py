"""
gradcam.py – Grad-CAM visualization cho BTL Garbage Classification
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


def get_gradcam(model, img_tensor, target_layer, class_idx=None):
    """Tính Grad-CAM cho một ảnh."""
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    fwd = target_layer.register_forward_hook(forward_hook)
    bwd = target_layer.register_backward_hook(backward_hook)

    model.eval()
    output = model(img_tensor.unsqueeze(0))

    if class_idx is None:
        class_idx = output.argmax(1).item()

    model.zero_grad()
    output[0, class_idx].backward()

    fwd.remove(); bwd.remove()

    grads = gradients[0].squeeze(0)      # (C, H, W)
    acts  = activations[0].squeeze(0)    # (C, H, W)

    weights = grads.mean(dim=(1, 2))     # (C,)
    cam = (weights[:, None, None] * acts).sum(0)  # (H, W)
    cam = torch.relu(cam)
    cam = cam - cam.min()
    if cam.max() > 0: cam = cam / cam.max()
    return cam.numpy(), class_idx


def visualize_gradcam(
    model, img_path, target_layer, class_names, class_to_idx,
    img_size=224, save_path=None, device="cpu"
):
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    img_orig = Image.open(img_path).convert("RGB").resize((img_size, img_size))
    img_tensor = tf(img_orig).to(device)

    cam, pred_idx = get_gradcam(model, img_tensor, target_layer)

    # Overlay
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((img_size, img_size)))
    heatmap = plt.cm.jet(cam_resized / 255)[:, :, :3]
    img_np  = np.array(img_orig) / 255.0
    overlay = 0.4 * heatmap + 0.6 * img_np
    overlay = np.clip(overlay, 0, 1)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_orig); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(cam_resized, cmap="jet"); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].set_title(f"Overlay\nPred: {class_names[pred_idx]}"); axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return pred_idx


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--config",       required=True)
    p.add_argument("--data_dir",     required=True)
    p.add_argument("--n_correct",    type=int, default=10)
    p.add_argument("--n_wrong",      type=int, default=10)
    p.add_argument("--img_size",     type=int, default=224)
    p.add_argument("--out_dir",      default="outputs/figures/gradcam")
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

    # Lấy target layer
    if cfg["model_name"].startswith("resnet"):
        target_layer = model.layer4[-1]
    elif cfg["model_name"].startswith("mobilenet"):
        target_layer = model.features[-1]
    elif cfg["model_name"].startswith("efficientnet"):
        target_layer = model.features[-1]
    else:
        target_layer = model.features[-1]  # CNN baseline

    # Lấy ảnh test
    from torchvision import transforms
    import random
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    all_imgs = []
    for cls_name, cls_idx in class_to_idx.items():
        cls_dir = Path(args.data_dir) / cls_name
        if cls_dir.exists():
            imgs = list(cls_dir.rglob("*.jpg")) + list(cls_dir.rglob("*.jpeg")) + list(cls_dir.rglob("*.png"))
            for img in imgs[:5]:
                all_imgs.append((img, cls_idx, cls_name))

    correct_count, wrong_count = 0, 0
    for img_path, true_idx, true_cls in all_imgs:
        img_t = tf(Image.open(img_path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            pred_idx = model(img_t).argmax(1).item()
        is_correct = (pred_idx == true_idx)

        if is_correct and correct_count < args.n_correct:
            save_path = out_dir / f"correct_{correct_count:02d}_{true_cls}.png"
            visualize_gradcam(model, img_path, target_layer, class_names, class_to_idx,
                              args.img_size, save_path)
            correct_count += 1
        elif not is_correct and wrong_count < args.n_wrong:
            save_path = out_dir / f"wrong_{wrong_count:02d}_{true_cls}_as_{class_names[pred_idx]}.png"
            visualize_gradcam(model, img_path, target_layer, class_names, class_to_idx,
                              args.img_size, save_path)
            wrong_count += 1

        if correct_count >= args.n_correct and wrong_count >= args.n_wrong:
            break

    print(f"[gradcam] Lưu {correct_count} correct + {wrong_count} wrong tại {out_dir}")


if __name__ == "__main__":
    main()
