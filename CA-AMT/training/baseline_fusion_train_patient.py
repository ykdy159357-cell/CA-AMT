from pathlib import Path
import csv
import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# 你的患者级数据集
from scr.datasets.patient_datasets import FusionDataset

# 统一模型库中的 ResNet 版 baseline
from scr.models.patient_networks import FusionPatientNet


# ========== 你只需要改这里 ==========
CSV_PATH = Path(r"D:\下载\Patient-level image data\Data\dataset_standard\patient_master.csv")
SAVE_DIR = Path(r"D:\下载\Patient-level image data\outputs\weight\weights_fusion_patient_resnet")
RESIZE = 128
BATCH_SIZE = 4
EPOCHS = 40
LR = 1e-4
SEED = 42
PRETRAINED = True
# ====================================


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    # drop_last=True 时，用真实参与训练的样本数更严谨
    epoch_loss = total_loss / len(all_labels)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    all_probs = []
    all_preds = []
    all_labels = []
    all_patient_ids = []

    with torch.no_grad():
        for left_x, right_x, tongue_x, y, patient_ids in loader:
            left_x = left_x.to(device)
            right_x = right_x.to(device)
            tongue_x = tongue_x.to(device)
            y = y.to(device)

            logits, _ = model(left_x, right_x, tongue_x)
            loss = criterion(logits, y)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * y.size(0)

            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())
            all_patient_ids.extend(list(patient_ids))

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
    }
    return metrics


def save_prediction_csv(metrics, save_path: Path):
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

    # train 用 drop_last=True，避免 BatchNorm 在最后一个 batch=1 时出错
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # 关键：这里直接调用统一模型库中的 ResNet 版 baseline
    model = FusionPatientNet(pretrained=PRETRAINED).to(device)

    class_weights = compute_class_weights(train_ds).to(device)
    print(f"class_weights: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_score = -1.0
    best_epoch = -1
    best_path = SAVE_DIR / "best_baseline_fusion_resnet_patient_model.pth"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        if not math.isnan(val_metrics["auc"]):
            current_score = val_metrics["auc"]
        else:
            current_score = val_metrics["balanced_acc"]

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

    model.load_state_dict(torch.load(best_path, map_location=device))

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

    pred_csv = SAVE_DIR / "baseline_fusion_resnet_test_predictions.csv"
    save_prediction_csv(test_metrics, pred_csv)
    print(f"\n测试集预测已保存: {pred_csv}")


if __name__ == "__main__":
    main()