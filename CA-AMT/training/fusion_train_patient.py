from pathlib import Path
import csv
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import torchvision.models as models

from scr.datasets.patient_datasets import FusionDataset


# ========== 你只需要改这里 ==========
CSV_PATH = Path(r"D:\下载\Patient-level image data\Data\dataset_standard\patient_master.csv")
SAVE_DIR = Path(r"D:\下载\Patient-level image data\outputs\weight\weights_fusion_patient_main")
RESIZE = 128
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
SEED = 42

PRETRAINED = True
FEATURE_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 2
DROPOUT = 0.1
# ====================================


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_resnet18(pretrained=True):
    """
    兼容老/新 torchvision 的 resnet18 加载方式
    """
    try:
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
                return models.resnet18(weights=weights)
            except AttributeError:
                return models.resnet18(pretrained=True)
        else:
            try:
                return models.resnet18(weights=None)
            except TypeError:
                return models.resnet18(pretrained=False)
    except Exception as e:
        raise RuntimeError(f"构建 ResNet18 失败：{e}")


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid_channels = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.gap(x).view(b, c)
        y = self.attention(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ImageEncoder(nn.Module):
    """
    ResNet18 + SE + 特征投影
    """
    def __init__(self, input_channels=3, pretrained=True, feature_dim=256):
        super().__init__()

        resnet = build_resnet18(pretrained=pretrained)

        if input_channels != 3:
            old_weight = resnet.conv1.weight.data.clone()
            resnet.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            nn.init.kaiming_normal_(resnet.conv1.weight, mode="fan_out", nonlinearity="relu")

            if input_channels == 1:
                with torch.no_grad():
                    resnet.conv1.weight[:] = old_weight.mean(dim=1, keepdim=True)

        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.se_block = SEBlock(512)

        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.feature_dim = feature_dim

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.se_block(x)
        x = self.projection(x)
        return x


class AdaptiveFusionTransformer(nn.Module):
    """
    你的主融合模块（3-token版）：
    左手 token + 右手 token + 舌头 token
    - 可学习模态权重
    - Transformer 编码
    - 注意力池化
    """
    def __init__(
        self,
        input_dim=256,
        feature_dim=256,
        num_heads=8,
        num_layers=2,
        num_classes=2,
        dropout=0.1,
    ):
        super().__init__()

        self.feature_dim = feature_dim

        self.left_projection = nn.Linear(input_dim, feature_dim)
        self.right_projection = nn.Linear(input_dim, feature_dim)
        self.tongue_projection = nn.Linear(input_dim, feature_dim)

        # 3个模态：左手、右手、舌头
        self.modal_weights = nn.Parameter(torch.ones(3))

        # batch_first=True，用可学习位置编码更稳
        self.pos_embed = nn.Parameter(torch.zeros(1, 3, feature_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attention_pool = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, left_feat, right_feat, tongue_feat):
        left_proj = self.left_projection(left_feat)
        right_proj = self.right_projection(right_feat)
        tongue_proj = self.tongue_projection(tongue_feat)

        weights = F.softmax(self.modal_weights, dim=0)

        left_weighted = left_proj * weights[0]
        right_weighted = right_proj * weights[1]
        tongue_weighted = tongue_proj * weights[2]

        sequence = torch.stack(
            [left_weighted, right_weighted, tongue_weighted], dim=1
        )  # [B, 3, D]
        sequence = sequence + self.pos_embed

        transformer_output = self.transformer_encoder(sequence)  # [B, 3, D]

        attn_weights = F.softmax(self.attention_pool(transformer_output), dim=1)  # [B, 3, 1]
        context_vector = torch.sum(transformer_output * attn_weights, dim=1)  # [B, D]

        logits = self.classifier(context_vector)

        attn_dict = {
            "modal_weights": weights.detach().cpu(),
            "token_attention": attn_weights.squeeze(-1).detach().cpu(),
        }
        return logits, attn_dict


class HandTongueAdaptiveTransformerModel(nn.Module):
    """
    主模型（患者级三模态版）
    - 左右手共享 hand encoder
    - 舌头独立 tongue encoder
    - 自适应 Transformer 融合
    """
    def __init__(
        self,
        pretrained=True,
        feature_dim=256,
        num_heads=8,
        num_layers=2,
        num_classes=2,
        dropout=0.1,
    ):
        super().__init__()

        self.hand_encoder = ImageEncoder(
            input_channels=3,
            pretrained=pretrained,
            feature_dim=feature_dim,
        )
        self.tongue_encoder = ImageEncoder(
            input_channels=3,
            pretrained=pretrained,
            feature_dim=feature_dim,
        )

        self.fusion_transformer = AdaptiveFusionTransformer(
            input_dim=feature_dim,
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.modal_drop_prob = 0.1

    def forward(self, left_x, right_x, tongue_x):
        left_feat = self.hand_encoder(left_x)
        right_feat = self.hand_encoder(right_x)
        tongue_feat = self.tongue_encoder(tongue_x)

        if self.training:
            if torch.rand(1).item() < self.modal_drop_prob:
                left_feat = torch.zeros_like(left_feat)
            if torch.rand(1).item() < self.modal_drop_prob:
                right_feat = torch.zeros_like(right_feat)
            if torch.rand(1).item() < self.modal_drop_prob:
                tongue_feat = torch.zeros_like(tongue_feat)

        logits, attn = self.fusion_transformer(left_feat, right_feat, tongue_feat)
        return logits, attn


def compute_class_weights(dataset):
    labels = [item["label"] for item in dataset.samples]
    counts = np.bincount(labels, minlength=2)

    if np.any(counts == 0):
        return torch.tensor([1.0, 1.0], dtype=torch.float32)

    total = counts.sum()
    weights = total / (2.0 * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for left_x, right_x, tongue_x, y, _ in loader:
        left_x = left_x.to(device)
        right_x = right_x.to(device)
        tongue_x = tongue_x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, _ = model(left_x, right_x, tongue_x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    all_probs = []
    all_preds = []
    all_labels = []
    all_patient_ids = []
    all_modal_weights = []
    all_token_attn = []

    with torch.no_grad():
        for left_x, right_x, tongue_x, y, patient_ids in loader:
            left_x = left_x.to(device)
            right_x = right_x.to(device)
            tongue_x = tongue_x.to(device)
            y = y.to(device)

            logits, attn = model(left_x, right_x, tongue_x)
            loss = criterion(logits, y)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * y.size(0)

            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())
            all_patient_ids.extend(list(patient_ids))
            all_modal_weights.append(attn["modal_weights"].numpy())
            all_token_attn.append(attn["token_attention"].numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    if len(set(all_labels)) == 2:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = float("nan")

    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (recall + specificity) / 2.0

    mean_modal_weights = np.mean(np.stack(all_modal_weights, axis=0), axis=0)
    mean_token_attn = np.mean(np.concatenate(all_token_attn, axis=0), axis=0)

    metrics = {
        "loss": avg_loss,
        "acc": acc,
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "sensitivity": recall,
        "specificity": specificity,
        "balanced_acc": balanced_acc,
        "labels": all_labels,
        "preds": all_preds,
        "probs": all_probs,
        "patient_ids": all_patient_ids,
        "mean_modal_weights": mean_modal_weights,
        "mean_token_attn": mean_token_attn,
    }
    return metrics


def save_prediction_csv(metrics, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_uid", "label", "pred_label", "prob_positive"])

        for pid, label, pred, prob in zip(
            metrics["patient_ids"],
            metrics["labels"],
            metrics["preds"],
            metrics["probs"],
        ):
            writer.writerow([pid, label, pred, f"{prob:.6f}"])


def format_metric(x):
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{x:.4f}"


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"使用设备: {device}")

    train_ds = FusionDataset(CSV_PATH, mode="train", resize=RESIZE)
    val_ds = FusionDataset(CSV_PATH, mode="val", resize=RESIZE)
    test_ds = FusionDataset(CSV_PATH, mode="test", resize=RESIZE)

    print(f"train patients: {len(train_ds)}")
    print(f"val patients:   {len(val_ds)}")
    print(f"test patients:  {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = HandTongueAdaptiveTransformerModel(
        pretrained=PRETRAINED,
        feature_dim=FEATURE_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_classes=2,
        dropout=DROPOUT,
    ).to(device)

    class_weights = compute_class_weights(train_ds).to(device)
    print(f"class_weights: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6
    )

    best_score = -1.0
    best_epoch = -1
    best_path = SAVE_DIR / "best_fusion_patient_main_model.pth"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        current_score = (
            val_metrics["auc"]
            if not math.isnan(val_metrics["auc"])
            else val_metrics["balanced_acc"]
        )
        scheduler.step(current_score)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] | "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"val_loss: {val_metrics['loss']:.4f} | val_acc: {val_metrics['acc']:.4f} | "
            f"val_auc: {format_metric(val_metrics['auc'])} | "
            f"val_sens: {val_metrics['sensitivity']:.4f} | "
            f"val_spec: {val_metrics['specificity']:.4f}"
        )

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            print(f"✅ 保存最佳模型到: {best_path}")

    print("\n===== 训练结束 =====")
    print(f"best_epoch: {best_epoch}")
    print(f"best_score: {best_score:.4f}")

    try:
        state_dict = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(best_path, map_location=device)
    model.load_state_dict(state_dict)

    train_metrics = evaluate(model, train_loader, criterion, device)
    val_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)

    print("\n===== 最终评估（最佳模型） =====")
    print(
        f"Train | loss: {train_metrics['loss']:.4f} | acc: {train_metrics['acc']:.4f} | "
        f"auc: {format_metric(train_metrics['auc'])} | sens: {train_metrics['sensitivity']:.4f} | "
        f"spec: {train_metrics['specificity']:.4f} | f1: {train_metrics['f1']:.4f}"
    )
    print(
        f"Val   | loss: {val_metrics['loss']:.4f} | acc: {val_metrics['acc']:.4f} | "
        f"auc: {format_metric(val_metrics['auc'])} | sens: {val_metrics['sensitivity']:.4f} | "
        f"spec: {val_metrics['specificity']:.4f} | f1: {val_metrics['f1']:.4f}"
    )
    print(
        f"Test  | loss: {test_metrics['loss']:.4f} | acc: {test_metrics['acc']:.4f} | "
        f"auc: {format_metric(test_metrics['auc'])} | sens: {test_metrics['sensitivity']:.4f} | "
        f"spec: {test_metrics['specificity']:.4f} | f1: {test_metrics['f1']:.4f}"
    )

    print("\n===== 模态权重 / token注意力（平均） =====")
    print("Train modal weights [left, right, tongue]:", np.round(train_metrics["mean_modal_weights"], 4))
    print("Val   modal weights [left, right, tongue]:", np.round(val_metrics["mean_modal_weights"], 4))
    print("Test  modal weights [left, right, tongue]:", np.round(test_metrics["mean_modal_weights"], 4))

    print("Train token attention [left, right, tongue]:", np.round(train_metrics["mean_token_attn"], 4))
    print("Val   token attention [left, right, tongue]:", np.round(val_metrics["mean_token_attn"], 4))
    print("Test  token attention [left, right, tongue]:", np.round(test_metrics["mean_token_attn"], 4))

    pred_csv = SAVE_DIR / "fusion_test_predictions.csv"
    save_prediction_csv(test_metrics, pred_csv)
    print(f"\n测试集预测已保存: {pred_csv}")


if __name__ == "__main__":
    main()