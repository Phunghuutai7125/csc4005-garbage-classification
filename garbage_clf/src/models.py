"""
models.py – CNN Baseline + Transfer Learning cho BTL Garbage Classification
"""
import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple


# ---------------------------------------------------------------------------
# CNN Baseline (from scratch)
# ---------------------------------------------------------------------------

class CNNBaseline(nn.Module):
    """Simple CNN baseline – 3 conv blocks."""
    def __init__(self, num_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


# ---------------------------------------------------------------------------
# Transfer Learning models
# ---------------------------------------------------------------------------

TRANSFER_MODELS = {
    "resnet18":        (models.resnet18,        models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50":        (models.resnet50,        models.ResNet50_Weights.IMAGENET1K_V2),
    "mobilenet_v2":    (models.mobilenet_v2,    models.MobileNet_V2_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
}


def build_transfer_model(
    model_name: str,
    num_classes: int = 6,
    dropout: float = 0.3,
    train_mode: str = "finetune",  # "head_only" hoặc "finetune"
) -> nn.Module:
    if model_name not in TRANSFER_MODELS:
        raise ValueError(f"model_name phải là: {list(TRANSFER_MODELS.keys())}")

    fn, weights = TRANSFER_MODELS[model_name]
    model = fn(weights=weights)

    # Thay head
    if model_name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif model_name.startswith("mobilenet"):
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif model_name.startswith("efficientnet"):
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))

    # Freeze backbone nếu head_only
    if train_mode == "head_only":
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze head
        if model_name.startswith("resnet"):
            for param in model.fc.parameters():
                param.requires_grad = True
        else:
            for param in model.classifier.parameters():
                param.requires_grad = True

    return model


def build_model(
    model_name: str,
    num_classes: int = 6,
    dropout: float = 0.3,
    train_mode: str = "finetune",
) -> nn.Module:
    if model_name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes, dropout=dropout)
    return build_transfer_model(model_name, num_classes, dropout, train_mode)


def count_params(model: nn.Module) -> Tuple[int, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
