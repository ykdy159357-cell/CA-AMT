# -*- coding: utf-8 -*-
"""
External validation for patient-level multimodal CAD model.

原则：
1. 不训练。
2. 不调参。
3. 不交叉验证。
4. 使用开发集 5-fold CV 得到的 best models 做 ensemble。
5. 外部验证集只用于最终独立评估。
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
import matplotlib.pyplot as plt

from scr.models.patient_networks import HandTongueAdaptiveTransformerModel


# =========================
# 1. 路径配置
# =========================
PROJECT_ROOT = Path(r"D:\tcm_AI\Patient-level image data")

EXTERNAL_CSV = PROJECT_ROOT / r"Data\dataset_standard\external_patient_master.csv"

MODEL_DIR = PROJECT_ROOT / r"outputs\cv_results_good\main"

SAVE_DIR = PROJECT_ROOT / r"outputs\external_validation\main"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

RESIZE = 224
SEED = 42
BOOTSTRAP_N = 2000

# 优先从开发集 OOF 指标文件读取 Youden 阈值
OOF_METRICS_PATH = MODEL_DIR / "oof_metrics_main.csv"


# =========================
# 2. 验证集 transform：注意，外部验证绝对不能使用数据增强
# =========================
external_transform = T.Compose([
    T.Resize((RESIZE, RESIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# =========================
# 3. 工具函数
# =========================
def load_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
    return external_transform(img)


def load_external_samples(csv_path):
    samples = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                "patient_uid": row["patient_uid"],
                "label": int(row["label"]),
                "hand_left_path": row["hand_left_path"],
                "hand_right_path": row["hand_right_path"],
                "tongue_path": row["tongue_path"],
            })

    df = pd.DataFrame(samples)

    if df.empty:
        raise ValueError(f"外部验证 CSV 为空: {csv_path}")

    required_cols = {
        "patient_uid",
        "label",
        "hand_left_path",
        "hand_right_path",
        "tongue_path",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"外部验证 CSV 缺少必要列: {missing}")

    if df["patient_uid"].duplicated().any():
        duplicated = df.loc[df["patient_uid"].duplicated(), "patient_uid"].tolist()
        raise ValueError(f"外部验证集中 patient_uid 重复: {duplicated[:10]}")

    print("📌 外部验证集读取完成")
    print(f"   N = {len(df)}")
    print(f"   Positive = {(df['label'] == 1).sum()} | Negative = {(df['label'] == 0).sum()}")

    return df


def compute_metrics_at_threshold(y_true, y_prob, threshold):
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


def bootstrap_ci(y_true, y_prob, metric, threshold=None, n_bootstrap=2000, seed=42):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []

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


def format_metric(value, lo, hi, digits=3):
    return f"{value:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"


def load_development_youden_threshold():
    if not OOF_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"未找到开发集 OOF 指标文件: {OOF_METRICS_PATH}\n"
            f"请确认 main 模型已经完成 5-fold CV。"
        )

    df = pd.read_csv(OOF_METRICS_PATH)
    if "youden_threshold" not in df.columns:
        raise ValueError(f"{OOF_METRICS_PATH} 中没有 youden_threshold 列")

    threshold = float(df.loc[0, "youden_threshold"])
    print(f"📌 使用开发集 OOF Youden threshold = {threshold:.6f}")
    return threshold


def predict_with_one_model(model_path, external_df, device):
    model = HandTongueAdaptiveTransformerModel(pretrained=False).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    probs = []

    with torch.no_grad():
        for _, row in external_df.iterrows():
            left_x = load_image(row["hand_left_path"]).unsqueeze(0).to(device)
            right_x = load_image(row["hand_right_path"]).unsqueeze(0).to(device)
            tongue_x = load_image(row["tongue_path"]).unsqueeze(0).to(device)

            logits, _ = model(left_x, right_x, tongue_x)
            prob = torch.softmax(logits, dim=1)[:, 1].item()
            probs.append(float(prob))

    return np.asarray(probs, dtype=float)


def plot_external_roc(y_true, y_prob, auc_value, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"External validation AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("External validation ROC curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ 已保存外部验证 ROC: {save_path}")


# =========================
# 4. 主流程
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 External validation | device = {device}")

    external_df = load_external_samples(EXTERNAL_CSV)

    y_true = external_df["label"].astype(int).values

    fold_probs = []

    for fold in range(1, 6):
        model_path = MODEL_DIR / f"best_model_fold_{fold}.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"未找到模型文件: {model_path}")

        print(f"\n📌 正在使用 Fold {fold} 模型进行外部验证推理:")
        print(f"   {model_path}")

        prob = predict_with_one_model(model_path, external_df, device)
        fold_probs.append(prob)
        external_df[f"prob_fold_{fold}"] = prob

    fold_probs = np.vstack(fold_probs)  # [5, N]
    prob_mean = fold_probs.mean(axis=0)

    external_df["prob_positive_mean"] = prob_mean

    # 使用开发集阈值
    dev_youden_threshold = load_development_youden_threshold()

    external_df["pred_label_development_youden"] = (
        external_df["prob_positive_mean"].values >= dev_youden_threshold
    ).astype(int)

    external_df["pred_label_0p5"] = (
        external_df["prob_positive_mean"].values >= 0.5
    ).astype(int)

    # 保存外部预测结果
    pred_path = SAVE_DIR / "external_predictions_main_ensemble.csv"
    external_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 已保存外部验证预测结果: {pred_path}")

    # 计算 AUC
    auc_value = roc_auc_score(y_true, prob_mean)
    auc_lo, auc_hi = bootstrap_ci(
        y_true,
        prob_mean,
        metric="auc",
        n_bootstrap=BOOTSTRAP_N,
        seed=SEED,
    )

    # 开发集 Youden 阈值下的外部结果
    metrics_dev_youden = compute_metrics_at_threshold(
        y_true,
        prob_mean,
        threshold=dev_youden_threshold,
    )

    # 0.5 阈值下的外部结果
    metrics_0p5 = compute_metrics_at_threshold(
        y_true,
        prob_mean,
        threshold=0.5,
    )

    summary = {
        "model": "main_adaptive_transformer_ensemble",
        "cohort": "external_validation",
        "n": int(len(y_true)),
        "positive_n": int((y_true == 1).sum()),
        "negative_n": int((y_true == 0).sum()),
        "auc": auc_value,
        "auc_95ci_low": auc_lo,
        "auc_95ci_high": auc_hi,
        "auc_formatted": format_metric(auc_value, auc_lo, auc_hi),
        "development_youden_threshold": dev_youden_threshold,
    }

    for prefix, metric_dict, threshold in [
        ("development_youden", metrics_dev_youden, dev_youden_threshold),
        ("threshold_0p5", metrics_0p5, 0.5),
    ]:
        summary[f"{prefix}_threshold"] = threshold

        for metric in ["accuracy", "sensitivity", "specificity", "f1"]:
            lo, hi = bootstrap_ci(
                y_true,
                prob_mean,
                metric=metric,
                threshold=threshold,
                n_bootstrap=BOOTSTRAP_N,
                seed=SEED,
            )
            summary[f"{prefix}_{metric}"] = metric_dict[metric]
            summary[f"{prefix}_{metric}_95ci_low"] = lo
            summary[f"{prefix}_{metric}_95ci_high"] = hi
            summary[f"{prefix}_{metric}_formatted"] = format_metric(metric_dict[metric], lo, hi)

        for key in ["tn", "fp", "fn", "tp"]:
            summary[f"{prefix}_{key}"] = metric_dict[key]

    summary_df = pd.DataFrame([summary])
    summary_path = SAVE_DIR / "external_metrics_main_ensemble.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存外部验证指标: {summary_path}")

    # 保存 run config
    config = {
        "external_csv": str(EXTERNAL_CSV),
        "model_dir": str(MODEL_DIR),
        "save_dir": str(SAVE_DIR),
        "resize": RESIZE,
        "ensemble": "mean probability from 5 fold-specific best models",
        "no_training": True,
        "no_cross_validation": True,
        "no_hyperparameter_tuning": True,
        "threshold_source": "development cohort OOF Youden threshold",
        "development_youden_threshold": dev_youden_threshold,
    }

    config_path = SAVE_DIR / "external_validation_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"✅ 已保存外部验证配置: {config_path}")

    # 画 ROC
    roc_path = SAVE_DIR / "external_roc_main_ensemble.png"
    plot_external_roc(y_true, prob_mean, auc_value, roc_path)

    print("\n🎉 外部验证完成")
    print(f"📌 External AUC = {format_metric(auc_value, auc_lo, auc_hi)}")
    print(
        "📌 使用开发集 Youden 阈值的外部结果: "
        f"Accuracy={summary['development_youden_accuracy_formatted']}, "
        f"Sensitivity={summary['development_youden_sensitivity_formatted']}, "
        f"Specificity={summary['development_youden_specificity_formatted']}, "
        f"F1={summary['development_youden_f1_formatted']}"
    )
    print(
        "📌 0.5 阈值的外部结果: "
        f"Accuracy={summary['threshold_0p5_accuracy_formatted']}, "
        f"Sensitivity={summary['threshold_0p5_sensitivity_formatted']}, "
        f"Specificity={summary['threshold_0p5_specificity_formatted']}, "
        f"F1={summary['threshold_0p5_f1_formatted']}"
    )


if __name__ == "__main__":
    main()