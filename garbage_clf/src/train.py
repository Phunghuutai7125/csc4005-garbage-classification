"""
train.py – Training script cho BTL Garbage Classification
"""
import argparse, json, os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.dataset import build_dataloaders
from src.models import build_model, count_params
from src.utils import (Timer, EarlyStopping, save_history, plot_curves,
                       plot_confusion_matrix, compute_metrics, save_metrics)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--model_name",  default="cnn_baseline",
                   help="cnn_baseline | resnet18 | resnet50 | mobilenet_v2 | efficientnet_b0")
    p.add_argument("--train_mode",  default="finetune", help="finetune | head_only")
    p.add_argument("--img_size",    type=int,   default=224)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--patience",    type=int,   default=5)
    p.add_argument("--augment",     action="store_true")
    p.add_argument("--num_workers", type=int,   default=2)
    p.add_argument("--max_per_class", type=int, default=0)
    p.add_argument("--scheduler",   default="cosine", help="cosine | plateau | none")
    p.add_argument("--run_name",    default=None)
    p.add_argument("--project",     default="csc4005-khmt16-01-garbage-classification")
    p.add_argument("--use_wandb",   action="store_true")
    p.add_argument("--out_base",    default="outputs")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():        return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            preds = logits.argmax(1)
            total_loss += loss.item() * imgs.size(0)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / total, correct / total, macro_f1


@torch.no_grad()
def evaluate_test(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for imgs, labels in loader:
        preds = model(imgs.to(device)).argmax(1).cpu().tolist()
        y_pred.extend(preds); y_true.extend(labels.tolist())
    return y_true, y_pred


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.run_name is None:
        args.run_name = f"{args.model_name}_{args.train_mode}"

    out_dir = Path(args.out_base) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"[train] Device: {device}")

    loaders, class_names, class_to_idx = build_dataloaders(
        args.data_dir, img_size=args.img_size, batch_size=args.batch_size,
        augment=args.augment, num_workers=args.num_workers,
        max_per_class=args.max_per_class, seed=args.seed,
    )

    model = build_model(args.model_name, len(class_names), args.dropout, args.train_mode).to(device)
    total_p, train_p = count_params(model)
    print(f"[train] {args.model_name} | total={total_p:,} trainable={train_p:,}")

    # Lưu class_to_idx
    with open(out_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    es = EarlyStopping(patience=args.patience)

    wb_run = None
    if args.use_wandb:
        try:
            import wandb
            wb_run = wandb.init(project=args.project, name=args.run_name, config=vars(args))
        except ImportError:
            print("[W&B] wandb chưa cài.")

    history = {k: [] for k in ["epoch","train_loss","val_loss","train_acc","val_acc",
                                 "train_f1","val_f1","lr","epoch_time_sec"]}
    best_val_f1 = 0.0
    best_path = out_dir / "best_model.pt"
    timer = Timer()

    for epoch in range(1, args.epochs + 1):
        timer.start()
        tr_loss, tr_acc, tr_f1 = run_epoch(model, loaders["train"], criterion, optimizer, device, True)
        vl_loss, vl_acc, vl_f1 = run_epoch(model, loaders["val"],   criterion, optimizer, device, False)
        epoch_time = timer.elapsed()
        current_lr = optimizer.param_groups[0]["lr"]

        if scheduler:
            if args.scheduler == "plateau": scheduler.step(vl_loss)
            else: scheduler.step()

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            torch.save(model.state_dict(), best_path)

        for k, v in zip(["epoch","train_loss","val_loss","train_acc","val_acc",
                          "train_f1","val_f1","lr","epoch_time_sec"],
                         [epoch, tr_loss, vl_loss, tr_acc, vl_acc, tr_f1, vl_f1,
                          current_lr, epoch_time]):
            history[k].append(v)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"loss={tr_loss:.4f}/{vl_loss:.4f} | "
              f"acc={tr_acc:.4f}/{vl_acc:.4f} | "
              f"f1={tr_f1:.4f}/{vl_f1:.4f} | "
              f"lr={current_lr:.2e} | {epoch_time:.1f}s")

        if wb_run:
            import wandb
            wandb.log({"train_loss": tr_loss, "val_loss": vl_loss,
                       "train_acc": tr_acc, "val_acc": vl_acc,
                       "train_macro_f1": tr_f1, "val_macro_f1": vl_f1,
                       "lr": current_lr, "epoch_time_sec": epoch_time, "epoch": epoch})

        if es.step(vl_loss):
            print(f"[EarlyStopping] Dừng epoch {epoch}"); break

    save_history(history, out_dir)
    curves_path = plot_curves(history, out_dir)

    print("[train] Đánh giá test set...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    y_true, y_pred = evaluate_test(model, loaders["test"], device)
    test_acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    test_f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"[train] test_acc={test_acc:.4f} test_macro_f1={test_f1:.4f}")

    cm_path = plot_confusion_matrix(y_true, y_pred, class_names, out_dir)
    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics.update({
        "best_val_f1": best_val_f1, "test_acc": test_acc, "test_macro_f1": test_f1,
        "total_params": total_p, "trainable_params": train_p,
        "trainable_ratio": train_p / total_p,
        "avg_epoch_time_sec": sum(history["epoch_time_sec"]) / len(history["epoch_time_sec"]),
    })
    save_metrics(metrics, out_dir)

    if wb_run:
        import wandb
        wandb.log({"test_acc": test_acc, "test_macro_f1": test_f1,
                   "best_val_f1": best_val_f1, "total_params": total_p,
                   "trainable_params": train_p,
                   "trainable_ratio": train_p / total_p,
                   "confusion_matrix": wandb.Image(cm_path),
                   "learning_curves":  wandb.Image(curves_path)})
        wb_run.finish()

    print(f"\n[train] Xong! Outputs: {out_dir}")
    print(f"  test_acc      = {test_acc:.4f}")
    print(f"  test_macro_f1 = {test_f1:.4f}")


if __name__ == "__main__":
    main()
