"""utils.py – Tiện ích cho BTL Garbage Classification"""
import json, time
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def save_history(history, out_dir):
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)


def plot_curves(history, out_dir) -> str:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train"); axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, history["train_acc"], label="Train"); axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True)

    axes[2].plot(epochs, history["train_f1"], label="Train"); axes[2].plot(epochs, history["val_f1"], label="Val")
    axes[2].set_title("Macro F1"); axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    path = out_dir / "curves.png"
    fig.savefig(path, dpi=120); plt.close(fig)
    return str(path)


def plot_confusion_matrix(y_true, y_pred, class_names, out_dir, title="Confusion Matrix") -> str:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=class_names,
                yticklabels=class_names, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=120); plt.close(fig)
    return str(path)


def compute_metrics(y_true, y_pred, class_names) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    return {"accuracy": acc, "classification_report": report}


def save_metrics(metrics, out_dir):
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


class Timer:
    def __init__(self): self._start = None
    def start(self): self._start = time.time()
    def elapsed(self): return time.time() - self._start


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience; self.min_delta = min_delta
        self.best = None; self.counter = 0; self.should_stop = False

    def step(self, val_loss):
        if self.best is None or val_loss < self.best - self.min_delta:
            self.best = val_loss; self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.should_stop = True
        return self.should_stop
