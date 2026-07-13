# -*- coding: utf-8 -*-
"""
External validation for CA-AMT v1.

Principles:
  1. No training.
  2. No tuning.
  3. Uses five fold-specific models trained on development cohort.
  4. Uses each fold-specific clinical preprocessor saved during training.
  5. External validation set is evaluated once as an independent cohort.

Place this file at:
D:\tcm_AI\Patient-level image data\external_validate_camt_mlp_v1.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt

from scr.models.patient_networks_camt_v1 import ClinicalAwareAdaptiveMultimodalTransformerModel


CSV_PATH = Path(r"D:\tcm_AI\Patient-level image data\Data\dataset_standard\patient_master_clinical_external_added.csv")
MODEL_DIR = Path(r"D:\tcm_AI\Patient-level image data\outputs\cv_results_camt_mlp_v1\camt_mlp_v1")
SAVE_DIR = Path(r"D:\tcm_AI\Patient-level image data\outputs\external_validation\camt_mlp_v1")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

RESIZE = 224
SEED = 42
BOOTSTRAP_N = 2000
FEATURE_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 2
DROPOUT = 0.1
CLINICAL_HIDDEN_DIM = 128
CLINICAL_DROPOUT = 0.2
OOF_METRICS_PATH = MODEL_DIR / "oof_metrics_camt_mlp_v1.csv"

external_transform = T.Compose([
    T.Resize((RESIZE, RESIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Cannot decode CSV: {path}")


def load_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
    return external_transform(img)


def load_external_df(csv_path: Path) -> pd.DataFrame:
    df = read_csv_robust(csv_path)
    df = df[df["cohort"].astype(str).str.lower().eq("external")].copy()
    if df.empty:
        raise ValueError("No external cohort rows found.")
    if df["patient_uid"].duplicated().any():
        raise ValueError("patient_uid duplicated in external cohort.")

    required = ["patient_uid", "label", "hand_left_path", "hand_right_path", "tongue_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"External data missing columns: {missing}")

    print("📌 External cohort loaded")
    print(f"   N={len(df)}, Positive={(df['label'] == 1).sum()}, Negative={(df['label'] == 0).sum()}")
    return df


def compute_metrics_at_threshold(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ppv = precision_score(y_true, y_pred, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1": float(f1),
        "ppv": float(ppv),
        "npv": float(npv),
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
        elif metric == "auprc":
            value = average_precision_score(yt, yp)
        elif metric == "brier":
            value = brier_score_loss(yt, yp)
        else:
            if threshold is None:
                raise ValueError("threshold is required for this metric.")
            value = compute_metrics_at_threshold(yt, yp, threshold)[metric]
        if not np.isnan(value):
            values.append(value)
    if len(values) == 0:
        return np.nan, np.nan
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def format_metric(value, lo, hi, digits=3):
    return f"{value:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"


def load_development_youden_threshold():
    if not OOF_METRICS_PATH.exists():
        raise FileNotFoundError(f"OOF metrics not found: {OOF_METRICS_PATH}")
    df = pd.read_csv(OOF_METRICS_PATH, encoding="utf-8-sig")
    if "youden_threshold" not in df.columns:
        raise ValueError("youden_threshold column not found in OOF metrics.")
    threshold = float(df.loc[0, "youden_threshold"])
    print(f"📌 Development OOF Youden threshold = {threshold:.6f}")
    return threshold


def predict_with_one_model(model_path: Path, preprocessor_path: Path, external_df: pd.DataFrame, device):
    prep_pack = joblib.load(preprocessor_path)
    preprocessor = prep_pack["preprocessor"]
    clinical_cols = prep_pack["clinical_cols"]

    for c in clinical_cols:
        external_df[c] = pd.to_numeric(external_df[c], errors="coerce")
    x_clinical = preprocessor.transform(external_df[clinical_cols])

    model = ClinicalAwareAdaptiveMultimodalTransformerModel(
        num_clinical_features=len(clinical_cols),
        pretrained=False,
        feature_dim=FEATURE_DIM,
        clinical_hidden_dim=CLINICAL_HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_classes=2,
        dropout=DROPOUT,
        clinical_dropout=CLINICAL_DROPOUT,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    probs = []
    modal_records = []
    with torch.no_grad():
        for i, (_, row) in enumerate(external_df.iterrows()):
            left_x = load_image(row["hand_left_path"]).unsqueeze(0).to(device)
            right_x = load_image(row["hand_right_path"]).unsqueeze(0).to(device)
            tongue_x = load_image(row["tongue_path"]).unsqueeze(0).to(device)
            clinical_x = torch.tensor(x_clinical[i], dtype=torch.float32).unsqueeze(0).to(device)

            logits, attn = model(left_x, right_x, tongue_x, clinical_x)
            prob = torch.softmax(logits, dim=1)[:, 1].item()
            probs.append(float(prob))

            if attn is not None and "modal_weights" in attn:
                weights = attn["modal_weights"].numpy()[0]
                token_att = attn.get("token_attention", None)
                token_att_np = token_att.numpy()[0] if token_att is not None else np.full(4, np.nan)
                modal_records.append({
                    "patient_uid": row["patient_uid"],
                    "label": int(row["label"]),
                    "left_weight": float(weights[0]),
                    "right_weight": float(weights[1]),
                    "tongue_weight": float(weights[2]),
                    "clinical_weight": float(weights[3]),
                    "left_token_attention": float(token_att_np[0]),
                    "right_token_attention": float(token_att_np[1]),
                    "tongue_token_attention": float(token_att_np[2]),
                    "clinical_token_attention": float(token_att_np[3]),
                })

    return np.asarray(probs, dtype=float), modal_records, clinical_cols


def plot_external_roc(y_true, y_prob, auc_value, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"CA-AMT v1 external AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("External validation ROC curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved ROC: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 CA-AMT v1 external validation | device={device}")
    external_df = load_external_df(CSV_PATH)
    y_true = external_df["label"].astype(int).values

    fold_probs = []
    all_modal_records = []
    clinical_cols_final = None

    for fold in range(1, 6):
        model_path = MODEL_DIR / f"best_model_fold_{fold}.pth"
        prep_path = MODEL_DIR / f"clinical_preprocessor_fold_{fold}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")
        if not prep_path.exists():
            raise FileNotFoundError(f"Missing preprocessor: {prep_path}")

        print(f"\n📌 Predicting with fold {fold} model")
        prob, modal_records, clinical_cols = predict_with_one_model(model_path, prep_path, external_df.copy(), device)
        clinical_cols_final = clinical_cols
        fold_probs.append(prob)
        external_df[f"prob_fold_{fold}"] = prob
        for r in modal_records:
            r["fold"] = fold
            all_modal_records.append(r)

    fold_probs = np.vstack(fold_probs)
    prob_mean = fold_probs.mean(axis=0)
    external_df["prob_positive_mean"] = prob_mean

    dev_youden_threshold = load_development_youden_threshold()
    external_df["pred_label_development_youden"] = (prob_mean >= dev_youden_threshold).astype(int)
    external_df["pred_label_0p5"] = (prob_mean >= 0.5).astype(int)

    pred_path = SAVE_DIR / "external_predictions_camt_mlp_v1_ensemble.csv"
    external_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved external predictions: {pred_path}")

    auc_value = roc_auc_score(y_true, prob_mean)
    auprc_value = average_precision_score(y_true, prob_mean)
    brier_value = brier_score_loss(y_true, prob_mean)
    auc_lo, auc_hi = bootstrap_ci(y_true, prob_mean, "auc", n_bootstrap=BOOTSTRAP_N, seed=SEED)
    auprc_lo, auprc_hi = bootstrap_ci(y_true, prob_mean, "auprc", n_bootstrap=BOOTSTRAP_N, seed=SEED)
    brier_lo, brier_hi = bootstrap_ci(y_true, prob_mean, "brier", n_bootstrap=BOOTSTRAP_N, seed=SEED)

    metrics_dev_youden = compute_metrics_at_threshold(y_true, prob_mean, threshold=dev_youden_threshold)
    metrics_0p5 = compute_metrics_at_threshold(y_true, prob_mean, threshold=0.5)

    summary = {
        "model": "camt_mlp_v1_ensemble",
        "cohort": "external_validation",
        "n": int(len(y_true)),
        "positive_n": int((y_true == 1).sum()),
        "negative_n": int((y_true == 0).sum()),
        "auc": auc_value,
        "auc_formatted": format_metric(auc_value, auc_lo, auc_hi),
        "auprc": auprc_value,
        "auprc_formatted": format_metric(auprc_value, auprc_lo, auprc_hi),
        "brier": brier_value,
        "brier_formatted": format_metric(brier_value, brier_lo, brier_hi),
        "development_youden_threshold": dev_youden_threshold,
        "clinical_cols": ";".join(clinical_cols_final or []),
    }

    for prefix, metric_dict, threshold in [
        ("development_youden", metrics_dev_youden, dev_youden_threshold),
        ("threshold_0p5", metrics_0p5, 0.5),
    ]:
        summary[f"{prefix}_threshold"] = threshold
        for metric in ["accuracy", "sensitivity", "specificity", "f1", "ppv", "npv"]:
            lo, hi = bootstrap_ci(y_true, prob_mean, metric=metric, threshold=threshold, n_bootstrap=BOOTSTRAP_N, seed=SEED)
            summary[f"{prefix}_{metric}"] = metric_dict[metric]
            summary[f"{prefix}_{metric}_formatted"] = format_metric(metric_dict[metric], lo, hi)
        for key in ["tn", "fp", "fn", "tp"]:
            summary[f"{prefix}_{key}"] = metric_dict[key]

    summary_path = SAVE_DIR / "external_metrics_camt_mlp_v1_ensemble.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved external metrics: {summary_path}")

    if all_modal_records:
        modal_df = pd.DataFrame(all_modal_records)
        # Average modal weights across five fold models for each external patient.
        avg_cols = [
            "left_weight", "right_weight", "tongue_weight", "clinical_weight",
            "left_token_attention", "right_token_attention", "tongue_token_attention", "clinical_token_attention",
        ]
        modal_avg = modal_df.groupby(["patient_uid", "label"], as_index=False)[avg_cols].mean()
        modal_df.to_csv(SAVE_DIR / "external_modality_weights_by_fold_camt_mlp_v1.csv", index=False, encoding="utf-8-sig")
        modal_avg.to_csv(SAVE_DIR / "external_modality_weights_mean_camt_mlp_v1.csv", index=False, encoding="utf-8-sig")
        print("✅ Saved external modality weights.")

    config = {
        "csv_path": str(CSV_PATH),
        "model_dir": str(MODEL_DIR),
        "save_dir": str(SAVE_DIR),
        "resize": RESIZE,
        "ensemble": "mean probability from five fold-specific CA-AMT v1 models",
        "no_training": True,
        "no_hyperparameter_tuning": True,
        "threshold_source": "development cohort OOF Youden threshold",
        "development_youden_threshold": dev_youden_threshold,
    }
    with open(SAVE_DIR / "external_validation_config_camt_mlp_v1.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    plot_external_roc(y_true, prob_mean, auc_value, SAVE_DIR / "external_roc_camt_mlp_v1_ensemble.png")

    print("\n🎉 CA-AMT v1 external validation completed.")
    print(f"📌 External AUC = {format_metric(auc_value, auc_lo, auc_hi)}")
    print(f"📁 Output dir: {SAVE_DIR}")


if __name__ == "__main__":
    main()
