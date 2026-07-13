# -*- coding: utf-8 -*-
"""
严格版 patient-level 5-fold CV + pooled OOF 评估脚本。

核心原则：
1. 每一折只把该折验证集患者的预测概率保存为 out-of-fold prediction。
2. 5 折结束后，合并全部 OOF 预测，统一计算 pooled OOF AUC。
3. 同时输出 Youden 阈值结果和 0.5 阈值结果。
4. AUC、Accuracy、Sensitivity、Specificity、F1 均提供 bootstrap 95%CI。

运行示例：
python cv_train_patient_oof_rigorous_with_transformer_fusion.py --model hand_only --checkpoint_selection best_auc
python cv_train_patient_oof_rigorous_with_transformer_fusion.py --model tongue_only --checkpoint_selection best_auc
python cv_train_patient_oof_rigorous_with_transformer_fusion.py --model baseline --checkpoint_selection best_auc
python cv_train_patient_oof_rigorous_with_transformer_fusion.py --model transformer_fusion --checkpoint_selection best_auc
python cv_train_patient_oof_rigorous_with_transformer_fusion.py --model main --checkpoint_selection best_auc

也可以一次性跑完五个模型：
python cv_train_patient_oof_rigorous_with_transformer_fusion.py --model all --checkpoint_selection best_auc
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from scr.models.patient_networks import (
    HandOnlyNet,
    TongueOnlyNet,
    FusionPatientNet,
    TransformerFusionPatientNet,
    HandTongueAdaptiveTransformerModel,
)

# ========== 基础配置：如路径变化，只改这里 ==========
CSV_PATH = Path(r"Data/dataset_standard/patient_master.csv")
SAVE_DIR = Path(r"D:\tcm_AI\Patient-level image data\outputs\cv_results_good")
RESIZE = 224  # 第一版修改：由 128 提高到 224，保留更多舌象/手掌细节
BATCH_SIZE = 4
EPOCHS = 40
PATIENCE = 10  # 🎯 新增：早停容忍度，连续 10 个 epoch 验证集没提升就停止
LR = 1e-4
CLIP_GRAD_NORM = 1.0  # 第一版修改：梯度裁剪，降低 Transformer 融合训练震荡
N_SPLITS = 5
SEED = 42
BOOTSTRAP_N = 2000

MODEL_TYPES = ["hand_only", "tongue_only", "baseline", "transformer_fusion", "main"]


# ========== 随机种子 ==========
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========== 数据集 ==========
def build_transforms(resize: int = 224):
    """
    第一版安全修改：
    1. 训练集使用轻量数据增强，降低小样本静态图片的过拟合风险。
    2. 验证集只做 Resize + ToTensor + Normalize，严禁增强。
    3. 不使用 Hue / Saturation / 强颜色扰动；不默认使用水平翻转，避免改变左右手和舌象方向语义。
    """
    train_transforms = T.Compose([
        T.Resize((resize, resize)),
        T.RandomAffine(
            degrees=8,
            translate=(0.03, 0.03),
            scale=(0.95, 1.05),
            fill=255,
        ),
        T.ColorJitter(brightness=0.08, contrast=0.08),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_transforms = T.Compose([
        T.Resize((resize, resize)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    return train_transforms, val_transforms


class DynamicFusionDataset(Dataset):
    def __init__(self, samples_list, transform=None):
        self.samples = samples_list
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def _load_img(self, image_path):
        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is None:
            raise ValueError("DynamicFusionDataset 必须传入 transform，以区分训练集增强和验证集标准化。")

        return self.transform(image)

    def __getitem__(self, idx):
        item = self.samples[idx]
        left_img = self._load_img(item["hand_left_path"])
        right_img = self._load_img(item["hand_right_path"])
        tongue_img = self._load_img(item["tongue_path"])
        label = torch.tensor(int(item["label"]), dtype=torch.long)
        return left_img, right_img, tongue_img, label, item["patient_uid"]


# ========== 模型 ==========
def build_model(model_type: str, device):
    if model_type == "hand_only":
        return HandOnlyNet(pretrained=True).to(device)
    if model_type == "tongue_only":
        return TongueOnlyNet(pretrained=True).to(device)
    if model_type == "baseline":
        return FusionPatientNet(pretrained=True).to(device)
    if model_type == "transformer_fusion":
        return TransformerFusionPatientNet(pretrained=True).to(device)
    if model_type == "main":
        return HandTongueAdaptiveTransformerModel(pretrained=True).to(device)
    raise ValueError(f"未知模型类型: {model_type}")


# ========== 训练与验证 ==========
def train_one_epoch(model, loader, optimizer, criterion, device, max_grad_norm: float = 1.0):
    model.train()
    total_loss = 0.0

    for left_x, right_x, tongue_x, y, _ in loader:
        left_x = left_x.to(device)
        right_x = right_x.to(device)
        tongue_x = tongue_x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, _ = model(left_x, right_x, tongue_x)
        loss = criterion(logits, y)
        loss.backward()

        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item() * y.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_patient_ids, all_labels, all_probs, all_preds_0p5 = [], [], [], []

    with torch.no_grad():
        for left_x, right_x, tongue_x, y, patient_ids in loader:
            left_x = left_x.to(device)
            right_x = right_x.to(device)
            tongue_x = tongue_x.to(device)

            logits, _ = model(left_x, right_x, tongue_x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds_0p5 = (probs >= 0.5).long()

            all_patient_ids.extend(list(patient_ids))
            all_labels.extend(y.numpy().astype(int).tolist())
            all_probs.extend(probs.cpu().numpy().astype(float).tolist())
            all_preds_0p5.extend(preds_0p5.cpu().numpy().astype(int).tolist())

    if len(set(all_labels)) == 2:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = float("nan")

    metrics_0p5 = compute_metrics_at_threshold(all_labels, all_probs, threshold=0.5)
    return auc, metrics_0p5, all_patient_ids, all_labels, all_probs, all_preds_0p5


# ========== 指标函数 ==========
def compute_youden_threshold(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid = np.isfinite(thresholds)
    if valid.sum() == 0:
        return 0.5
    youden = tpr[valid] - fpr[valid]
    threshold = thresholds[valid][np.argmax(youden)]
    return float(threshold)


def compute_metrics_at_threshold(y_true, y_prob, threshold: float):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_ci(y_true, y_prob, metric: str, threshold: float = None,
                 n_bootstrap: int = 2000, seed: int = 42):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]

        if len(np.unique(yt)) < 2:
            continue

        if metric == "auc":
            value = roc_auc_score(yt, yp)
        else:
            if threshold is None:
                raise ValueError("非 AUC 指标必须提供 threshold")
            m = compute_metrics_at_threshold(yt, yp, threshold)
            value = m[metric]

        if not np.isnan(value):
            values.append(value)

    if len(values) == 0:
        return np.nan, np.nan

    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def format_metric(value, lo, hi, digits: int = 3):
    return f"{value:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"


def build_summary_row(model_type: str, oof_df: pd.DataFrame, fold_aucs):
    y_true = oof_df["label"].astype(int).values
    y_prob = oof_df["prob_positive"].astype(float).values

    auc_value = roc_auc_score(y_true, y_prob)
    auc_lo, auc_hi = bootstrap_ci(y_true, y_prob, "auc", n_bootstrap=BOOTSTRAP_N, seed=SEED)

    youden_threshold = compute_youden_threshold(y_true, y_prob)
    metrics_youden = compute_metrics_at_threshold(y_true, y_prob, youden_threshold)
    metrics_0p5 = compute_metrics_at_threshold(y_true, y_prob, 0.5)

    row = {
        "model": model_type,
        "n": int(len(oof_df)),
        "positive_n": int((y_true == 1).sum()),
        "negative_n": int((y_true == 0).sum()),
        "pooled_oof_auc": auc_value,
        "pooled_oof_auc_95ci_low": auc_lo,
        "pooled_oof_auc_95ci_high": auc_hi,
        "pooled_oof_auc_formatted": format_metric(auc_value, auc_lo, auc_hi),
        "mean_best_fold_auc_for_log_only": float(np.mean(fold_aucs)),
        "sd_best_fold_auc_for_log_only": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else np.nan,
        "youden_threshold": youden_threshold,
    }

    for prefix, metric_dict, threshold in [
        ("youden", metrics_youden, youden_threshold),
        ("threshold_0p5", metrics_0p5, 0.5),
    ]:
        row[f"{prefix}_threshold"] = threshold
        for metric in ["accuracy", "sensitivity", "specificity", "f1"]:
            lo, hi = bootstrap_ci(
                y_true, y_prob, metric, threshold=threshold,
                n_bootstrap=BOOTSTRAP_N, seed=SEED
            )
            row[f"{prefix}_{metric}"] = metric_dict[metric]
            row[f"{prefix}_{metric}_95ci_low"] = lo
            row[f"{prefix}_{metric}_95ci_high"] = hi
            row[f"{prefix}_{metric}_formatted"] = format_metric(metric_dict[metric], lo, hi)
        for key in ["tn", "fp", "fn", "tp"]:
            row[f"{prefix}_{key}"] = metric_dict[key]

    return row


# ========== 保存与质量控制 ==========
def safe_save_dataframe(df: pd.DataFrame, canonical_path: Path):
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = canonical_path.with_name(canonical_path.stem + f"_{timestamp}" + canonical_path.suffix)

    try:
        df.to_csv(canonical_path, index=False, encoding="utf-8-sig")
        print(f"✅ 已保存: {canonical_path}")
        return canonical_path
    except PermissionError:
        df.to_csv(backup_path, index=False, encoding="utf-8-sig")
        print(f"⚠️ 标准文件被占用，已另存为: {backup_path}")
        return backup_path


def validate_oof_df(oof_df: pd.DataFrame, expected_n: int):
    required_cols = {"patient_uid", "fold", "label", "prob_positive"}
    missing = required_cols - set(oof_df.columns)
    if missing:
        raise ValueError(f"OOF 文件缺少必要列: {missing}")

    if len(oof_df) != expected_n:
        raise ValueError(f"OOF 行数错误: 当前 {len(oof_df)} 行，期望 {expected_n} 行")

    duplicated = oof_df["patient_uid"].duplicated().sum()
    if duplicated > 0:
        dup_ids = oof_df.loc[oof_df["patient_uid"].duplicated(), "patient_uid"].head(10).tolist()
        raise ValueError(f"patient_uid 存在重复，重复数量={duplicated}，示例={dup_ids}")

    if not oof_df["prob_positive"].between(0, 1).all():
        raise ValueError("prob_positive 存在不在 0-1 范围内的值")

    print("\n📌 OOF 质量检查通过")
    print(f"   N = {len(oof_df)}")
    print(f"   Positive = {(oof_df['label'] == 1).sum()} | Negative = {(oof_df['label'] == 0).sum()}")
    print(f"   Fold 分布: {oof_df['fold'].value_counts().sort_index().to_dict()}")



def save_run_config(run_save_dir: Path, model_type: str, checkpoint_selection: str):
    config = {
        "model_type": model_type,
        "checkpoint_selection": checkpoint_selection,
        "csv_path": str(CSV_PATH),
        "save_dir": str(SAVE_DIR),
        "resize": RESIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "lr": LR,
        "clip_grad_norm": CLIP_GRAD_NORM,
        "n_splits": N_SPLITS,
        "seed": SEED,
        "bootstrap_n": BOOTSTRAP_N,
        "train_augmentation": {
            "resize": f"{RESIZE}x{RESIZE}",
            "random_affine": {
                "degrees": 8,
                "translate": [0.03, 0.03],
                "scale": [0.95, 1.05],
                "fill": 255,
            },
            "color_jitter": {
                "brightness": 0.08,
                "contrast": 0.08,
                "saturation": 0,
                "hue": 0,
            },
            "horizontal_flip": False,
            "normalization": "mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]",
        },
        "val_transform": {
            "resize": f"{RESIZE}x{RESIZE}",
            "augmentation": False,
            "normalization": "mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]",
        },
    }

    config_path = run_save_dir / "run_config_first_version.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"✅ 已保存运行配置: {config_path}")


# ========== 读取 patient master ==========
def load_all_samples(csv_path: Path):
    all_samples = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_samples.append({
                "patient_uid": row["patient_uid"],
                "label": int(row["label"]),
                "hand_left_path": row["hand_left_path"],
                "hand_right_path": row["hand_right_path"],
                "tongue_path": row["tongue_path"],
            })

    df = pd.DataFrame(all_samples)
    if df.empty:
        raise ValueError(f"未从 CSV 读取到样本: {csv_path}")
    if df["patient_uid"].duplicated().any():
        print("⚠️ patient_master 中 patient_uid 有重复。若每个患者多张图，需要先做患者级聚合。")
    return df


# ========== 单模型主流程 ==========
def run_one_model(model_type: str, checkpoint_selection: str = "best_auc"):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_save_dir = SAVE_DIR / model_type
    run_save_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n🚀 开始执行 {N_SPLITS}-Fold CV | 模型: [{model_type.upper()}] | 设备: {device} | checkpoint_selection={checkpoint_selection}")

    train_transform, val_transform = build_transforms(RESIZE)
    save_run_config(run_save_dir, model_type, checkpoint_selection)

    df = load_all_samples(CSV_PATH)
    X = np.arange(len(df))
    y = df["label"].values.astype(int)
    groups = df["patient_uid"].values

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_records = []
    fold_summary_records = []
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), 1):
        print("\n" + "=" * 20 + f" Fold {fold}/{N_SPLITS} " + "=" * 20)

        train_samples = df.iloc[train_idx].to_dict("records")
        val_samples = df.iloc[val_idx].to_dict("records")

        # 第一版修改：保存每折患者级划分，保证后续主模型、baseline 和消融实验可以复现同一 fold。
        fold_split_df = pd.concat([
            pd.DataFrame(train_samples)[["patient_uid", "label"]].assign(fold=fold, split="train"),
            pd.DataFrame(val_samples)[["patient_uid", "label"]].assign(fold=fold, split="val"),
        ], ignore_index=True)
        safe_save_dataframe(fold_split_df, run_save_dir / f"fold_{fold}_split.csv")

        train_ds = DynamicFusionDataset(train_samples, transform=train_transform)
        val_ds = DynamicFusionDataset(val_samples, transform=val_transform)

        train_labels = np.array([s["label"] for s in train_samples]).astype(int)
        val_labels = np.array([s["label"] for s in val_samples]).astype(int)

        counts = np.bincount(train_labels, minlength=2)
        val_counts = np.bincount(val_labels, minlength=2)
        if (counts == 0).any():
            raise ValueError(f"Fold {fold} 训练集中存在空类别: counts={counts}")
        if (val_counts == 0).any():
            raise ValueError(f"Fold {fold} 验证集中存在空类别: val_counts={val_counts}")

        print(
            f"📊 Fold {fold} 类别分布 | "
            f"Train: negative={counts[0]}, positive={counts[1]} | "
            f"Val: negative={val_counts[0]}, positive={val_counts[1]}"
        )

        class_weights = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32).to(device)

        model = build_model(model_type, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        best_val_auc = -np.inf
        best_epoch = None
        best_oof_data = None
        best_model_path = run_save_dir / f"best_model_fold_{fold}.pth"

        epochs_no_improve = 0  # 🎯 新增：用于记录连续未提升的 epoch 数
        epoch_log_records = []  # 第一版修改：保存每个 epoch 的训练日志，便于排查过拟合和训练震荡。

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                max_grad_norm=CLIP_GRAD_NORM
            )
            val_auc, val_metrics_0p5, pids, labels, probs, preds_0p5 = evaluate(model, val_loader, device)
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch:02d}/{EPOCHS}] "
                f"Loss: {train_loss:.4f} | "
                f"Val AUC: {val_auc:.4f} | "
                f"Sens@0.5: {val_metrics_0p5['sensitivity']:.4f} | "
                f"Spec@0.5: {val_metrics_0p5['specificity']:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            epoch_log_records.append({
                "model": model_type,
                "fold": fold,
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_auc": float(val_auc),
                "sens_0p5": float(val_metrics_0p5["sensitivity"]),
                "spec_0p5": float(val_metrics_0p5["specificity"]),
                "acc_0p5": float(val_metrics_0p5["accuracy"]),
                "f1_0p5": float(val_metrics_0p5["f1"]),
                "lr": float(current_lr),
                "clip_grad_norm": CLIP_GRAD_NORM,
                "resize": RESIZE,
            })

            if checkpoint_selection == "best_auc":
                should_select = val_auc > best_val_auc
            elif checkpoint_selection == "last_epoch":
                should_select = True
            else:
                raise ValueError(f"未知 checkpoint_selection: {checkpoint_selection}")

            if should_select:
                best_val_auc = val_auc
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
                best_oof_data = list(zip(pids, labels, probs, preds_0p5))

                epochs_no_improve = 0  # 🎯 新增：如果有提升，重置计数器
            else:
                epochs_no_improve += 1  # 🎯 新增：如果没提升，计数器加 1

            # 🎯 新增：核心早停逻辑
            if epochs_no_improve >= PATIENCE:
                print(
                    f"🛑 早停触发！连续 {PATIENCE} 个 epoch 验证集 AUC 未提升，当前 Fold {fold} 提前结束于 epoch {epoch}。")
                break

        if best_oof_data is None:
            raise RuntimeError(f"Fold {fold} 未获得有效 OOF 预测")

        print(f"🏆 Fold {fold} 最佳 Val AUC: {best_val_auc:.4f} | best_epoch={best_epoch}")
        fold_aucs.append(float(best_val_auc))

        for pid, label, prob, pred_0p5 in best_oof_data:
            oof_records.append({
                "patient_uid": pid,
                "fold": fold,
                "label": int(label),
                "prob_positive": float(prob),
                "pred_label_0p5": int(pred_0p5),
                "best_epoch": int(best_epoch),
                "fold_val_auc": float(best_val_auc),
                "model": model_type,
                "seed": SEED,
            })

        fold_summary_records.append({
            "model": model_type,
            "fold": fold,
            "train_n": len(train_samples),
            "val_n": len(val_samples),
            "train_positive_n": int((train_labels == 1).sum()),
            "train_negative_n": int((train_labels == 0).sum()),
            "val_positive_n": int(sum(s["label"] == 1 for s in val_samples)),
            "val_negative_n": int(sum(s["label"] == 0 for s in val_samples)),
            "best_epoch": best_epoch,
            "best_val_auc": best_val_auc,
            "best_model_path": str(best_model_path),
        })

        epoch_log_df = pd.DataFrame(epoch_log_records)
        safe_save_dataframe(epoch_log_df, run_save_dir / f"train_log_fold_{fold}.csv")

        del model, optimizer, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_df = pd.DataFrame(oof_records)
    validate_oof_df(oof_df, expected_n=len(df))

    y_true = oof_df["label"].astype(int).values
    y_prob = oof_df["prob_positive"].astype(float).values
    youden_threshold = compute_youden_threshold(y_true, y_prob)
    oof_df["pred_label_youden"] = (y_prob >= youden_threshold).astype(int)

    canonical_oof = run_save_dir / f"oof_predictions_{model_type}.csv"
    latest_oof = run_save_dir / f"oof_predictions_{model_type}_latest.csv"
    saved_oof = safe_save_dataframe(oof_df, canonical_oof)
    safe_save_dataframe(oof_df, latest_oof)

    fold_summary_df = pd.DataFrame(fold_summary_records)
    safe_save_dataframe(fold_summary_df, run_save_dir / f"fold_summary_{model_type}.csv")

    summary_row = build_summary_row(model_type, oof_df, fold_aucs)
    summary_df = pd.DataFrame([summary_row])
    safe_save_dataframe(summary_df, run_save_dir / f"oof_metrics_{model_type}.csv")

    print("\n🎉 交叉验证完成。正式论文结果请优先使用 pooled OOF，而不是 best-fold AUC 均值。")
    print(f"📌 Pooled OOF AUC = {summary_row['pooled_oof_auc_formatted']}")
    print(f"📌 Youden threshold = {summary_row['youden_threshold']:.6f}")
    print(
        "📌 Youden 指标: "
        f"Accuracy={summary_row['youden_accuracy_formatted']}, "
        f"Sensitivity={summary_row['youden_sensitivity_formatted']}, "
        f"Specificity={summary_row['youden_specificity_formatted']}, "
        f"F1={summary_row['youden_f1_formatted']}"
    )
    print(
        "📌 0.5 阈值指标: "
        f"Accuracy={summary_row['threshold_0p5_accuracy_formatted']}, "
        f"Sensitivity={summary_row['threshold_0p5_sensitivity_formatted']}, "
        f"Specificity={summary_row['threshold_0p5_specificity_formatted']}, "
        f"F1={summary_row['threshold_0p5_f1_formatted']}"
    )
    print(f"📁 OOF 文件: {saved_oof}")


# ========== 总入口 ==========
def main():
    parser = argparse.ArgumentParser(description="严谨版 pooled OOF AUC 训练与评估")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODEL_TYPES + ["all"],
        help="选择模型: hand_only / tongue_only / baseline / transformer_fusion / main / all",
    )
    parser.add_argument(
        "--checkpoint_selection",
        type=str,
        default="best_auc",
        choices=["best_auc", "last_epoch"],
        help=(
            "best_auc: 沿用你当前流程，按外层验证折 AUC 选择最佳 epoch；"
            "last_epoch: 每折固定取最后一个 epoch，可减少验证折调参带来的乐观偏倚。"
        ),
    )
    args = parser.parse_args()

    models_to_run = MODEL_TYPES if args.model == "all" else [args.model]
    for model_type in models_to_run:
        run_one_model(model_type, checkpoint_selection=args.checkpoint_selection)


if __name__ == "__main__":
    main()