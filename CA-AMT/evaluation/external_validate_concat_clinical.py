# -*- coding: utf-8 -*-
"""
External validation for Image + Clinical concat fusion control model.

Purpose:
    This script evaluates the 5 fold-specific concat models on the external cohort.

Important methodological notes:
    1. No training.
    2. No hyperparameter tuning.
    3. No threshold selection using external validation.
    4. No imputation/scaling parameters fitted on external data.
    5. Each fold uses its own clinical_preprocessor_fold_*.joblib fitted in that fold's training data.
    6. Final external prediction is the mean probability from the 5 fold-specific models.

Place this file at:
    D:\tcm_AI\Patient-level image data\external_validate_concat_clinical.py
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
)

from scr.models.patient_networks_concat_clinical import ImageClinicalConcatNet


# =========================
# 1. PyCharm-friendly configuration
# =========================
CSV_PATH = Path(r"D:\tcm_AI\Patient-level image data\Data\dataset_standard\patient_master_clinical_external_added.csv")
MODEL_DIR = Path(r"D:\tcm_AI\Patient-level image data\outputs\cv_results_concat_clinical\concat_clinical")
SAVE_DIR = Path(r"D:\tcm_AI\Patient-level image data\outputs\external_validation\concat_clinical")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "concat_clinical"
RESIZE = 224
BATCH_SIZE = 4
SEED = 42
BOOTSTRAP_N = 2000

FEATURE_DIM = 256
CLINICAL_HIDDEN_DIM = 128
DROPOUT = 0.4
CLINICAL_DROPOUT = 0.2
NUM_CLASSES = 2

OOF_METRICS_PATH = MODEL_DIR / "oof_metrics_concat_clinical.csv"


# =========================
# 2. Utilities
# =========================
def read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Cannot decode CSV: {path}")


def safe_save_dataframe(df: pd.DataFrame, canonical_path: Path):
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(canonical_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved: {canonical_path}")
    return canonical_path


external_transform = T.Compose([
    T.Resize((RESIZE, RESIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class ExternalConcatDataset(torch.utils.data.Dataset):
    def __init__(self, samples_list):
        self.samples = samples_list

    def __len__(self):
        return len(self.samples)

    def _load_img(self, path):
        with Image.open(path) as image:
            image = image.convert("RGB")
        return external_transform(image)

    def __getitem__(self, idx):
        item = self.samples[idx]
        left_img = self._load_img(item["hand_left_path"])
        right_img = self._load_img(item["hand_right_path"])
        tongue_img = self._load_img(item["tongue_path"])
        clinical = torch.tensor(item["clinical_features"], dtype=torch.float32)
        label = torch.tensor(int(item["label"]), dtype=torch.long)
        return left_img, right_img, tongue_img, clinical, label, item["patient_uid"]


def load_external_df(csv_path: Path) -> pd.DataFrame:
    df = read_csv_robust(csv_path)
    required = ["patient_uid", "cohort", "label", "hand_left_path", "hand_right_path", "tongue_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df[df["cohort"].astype(str).str.lower().eq("external")].copy()
    if df.empty:
        raise ValueError("No external cohort rows found.")
    if df["patient_uid"].duplicated().any():
        raise ValueError("patient_uid duplicated in external cohort.")
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    print("📌 External cohort loaded")
    print(f"   N={len(df)}, Positive={(df['label'] == 1).sum()}, Negative={(df['label'] == 0).sum()}")
    return df


def make_samples(df_part: pd.DataFrame, clinical_array: np.ndarray):
    samples = []
    for i, (_, row) in enumerate(df_part.iterrows()):
        samples.append({
            "patient_uid": str(row["patient_uid"]),
            "label": int(row["label"]),
            "hand_left_path": str(row["hand_left_path"]),
            "hand_right_path": str(row["hand_right_path"]),
            "tongue_path": str(row["tongue_path"]),
            "clinical_features": clinical_array[i].astype(np.float32),
        })
    return samples


def build_model(num_clinical_features: int, device):
    model = ImageClinicalConcatNet(
        num_clinical_features=num_clinical_features,
        pretrained=False,
        feature_dim=FEATURE_DIM,
        clinical_hidden_dim=CLINICAL_HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        clinical_dropout=CLINICAL_DROPOUT,
    )
    return model.to(device)


# =========================
# 3. Metrics and plots
# =========================
def compute_metrics_at_threshold(y_true, y_prob, threshold: float):
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


def bootstrap_ci(y_true, y_prob, metric: str, threshold: float = None, n_bootstrap: int = 2000, seed: int = 42):
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
        elif metric == "auprc":
            value = average_precision_score(yt, yp)
        elif metric == "brier":
            value = brier_score_loss(yt, yp)
        else:
            if threshold is None:
                raise ValueError("threshold is required for threshold-dependent metrics.")
            value = compute_metrics_at_threshold(yt, yp, threshold)[metric]
        if not np.isnan(value):
            values.append(value)

    if len(values) == 0:
        return np.nan, np.nan
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def format_metric(value, lo, hi, digits: int = 3):
    return f"{value:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"


def load_development_youden_threshold():
    if not OOF_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find development OOF metrics: {OOF_METRICS_PATH}\n"
            f"Run cv_train_patient_oof_concat_clinical.py first."
        )
    df = pd.read_csv(OOF_METRICS_PATH)
    if "youden_threshold" not in df.columns:
        raise ValueError(f"{OOF_METRICS_PATH} has no youden_threshold column.")
    threshold = float(df.loc[0, "youden_threshold"])
    print(f"📌 Using development OOF Youden threshold = {threshold:.6f}")
    return threshold


def plot_roc_curve(y_true, y_prob, title: str, save_base: Path):
    auc_value = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2.2, label=f"AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Reference")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(str(save_base) + ".png", dpi=300)
    plt.savefig(str(save_base) + ".pdf")
    plt.close()


def plot_pr_curve(y_true, y_prob, title: str, save_base: Path):
    ap = average_precision_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    prevalence = float(np.mean(y_true))

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, linewidth=2.2, label=f"AP = {ap:.3f}")
    plt.axhline(prevalence, linestyle="--", linewidth=1.5, label=f"Prevalence = {prevalence:.3f}")
    plt.xlabel("Recall / Sensitivity")
    plt.ylabel("Precision / PPV")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(str(save_base) + ".png", dpi=300)
    plt.savefig(str(save_base) + ".pdf")
    plt.close()


def plot_calibration_curve(y_true, y_prob, title: str, save_base: Path, n_bins: int = 5):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    brier = brier_score_loss(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2.2, label=f"Brier = {brier:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Ideal")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed probability")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(str(save_base) + ".png", dpi=300)
    plt.savefig(str(save_base) + ".pdf")
    plt.close()


# =========================
# 4. Prediction
# =========================
def predict_with_one_fold(model_path: Path, prep_path: Path, external_df: pd.DataFrame, device):
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not prep_path.exists():
        raise FileNotFoundError(f"Missing clinical preprocessor file: {prep_path}")

    prep_pack = joblib.load(prep_path)
    preprocessor = prep_pack["preprocessor"]
    clinical_cols = prep_pack["clinical_cols"]
    feature_set = prep_pack.get("clinical_feature_set", "unknown")

    missing = [c for c in clinical_cols if c not in external_df.columns]
    if missing:
        raise ValueError(f"External CSV missing clinical columns required by fold preprocessor: {missing}")

    df = external_df.copy()
    for c in clinical_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Critical: transform only. Do NOT fit on external data.
    x_external_clinical = preprocessor.transform(df[clinical_cols])
    samples = make_samples(df, x_external_clinical)
    dataset = ExternalConcatDataset(samples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(num_clinical_features=len(clinical_cols), device=device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    probs = []
    patient_ids = []
    labels = []

    with torch.no_grad():
        for left_x, right_x, tongue_x, clinical_x, y, pids in loader:
            left_x = left_x.to(device)
            right_x = right_x.to(device)
            tongue_x = tongue_x.to(device)
            clinical_x = clinical_x.to(device)

            logits, _ = model(left_x, right_x, tongue_x, clinical_x)
            prob = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(prob.cpu().numpy().astype(float).tolist())
            patient_ids.extend(list(pids))
            labels.extend(y.numpy().astype(int).tolist())

    return np.asarray(probs, dtype=float), patient_ids, labels, clinical_cols, feature_set


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 External validation for Image + Clinical concat | device={device}")

    external_df = load_external_df(CSV_PATH)
    y_true = external_df["label"].astype(int).values

    fold_probs = []
    clinical_cols_by_fold = {}

    for fold in range(1, 6):
        model_path = MODEL_DIR / f"best_model_fold_{fold}.pth"
        prep_path = MODEL_DIR / f"clinical_preprocessor_fold_{fold}.joblib"
        print(f"\n📌 Fold {fold} external prediction")
        print(f"   model: {model_path}")
        print(f"   preprocessor: {prep_path}")

        prob, pids, labels, clinical_cols, feature_set = predict_with_one_fold(
            model_path=model_path,
            prep_path=prep_path,
            external_df=external_df,
            device=device,
        )
        fold_probs.append(prob)
        external_df[f"prob_fold_{fold}"] = prob
        clinical_cols_by_fold[f"fold_{fold}"] = {
            "clinical_cols": clinical_cols,
            "clinical_feature_set": feature_set,
        }

    fold_probs = np.vstack(fold_probs)  # [5, N]
    prob_mean = fold_probs.mean(axis=0)
    external_df["prob_positive_mean"] = prob_mean

    dev_youden_threshold = load_development_youden_threshold()
    external_df["pred_label_development_youden"] = (prob_mean >= dev_youden_threshold).astype(int)
    external_df["pred_label_0p5"] = (prob_mean >= 0.5).astype(int)

    # Required output name.
    pred_path = SAVE_DIR / "external_predictions_concat_clinical_ensemble.csv"
    safe_save_dataframe(external_df, pred_path)

    auc_value = roc_auc_score(y_true, prob_mean)
    auprc_value = average_precision_score(y_true, prob_mean)
    brier_value = brier_score_loss(y_true, prob_mean)
    auc_lo, auc_hi = bootstrap_ci(y_true, prob_mean, "auc", n_bootstrap=BOOTSTRAP_N, seed=SEED)
    auprc_lo, auprc_hi = bootstrap_ci(y_true, prob_mean, "auprc", n_bootstrap=BOOTSTRAP_N, seed=SEED)
    brier_lo, brier_hi = bootstrap_ci(y_true, prob_mean, "brier", n_bootstrap=BOOTSTRAP_N, seed=SEED)

    metrics_dev_youden = compute_metrics_at_threshold(y_true, prob_mean, dev_youden_threshold)
    metrics_0p5 = compute_metrics_at_threshold(y_true, prob_mean, 0.5)

    summary = {
        "model": MODEL_NAME,
        "model_role": "Image + Clinical concat fusion control model; no Transformer; no adaptive gating",
        "cohort": "external_validation",
        "n": int(len(y_true)),
        "positive_n": int((y_true == 1).sum()),
        "negative_n": int((y_true == 0).sum()),
        "auc": auc_value,
        "auc_95ci_low": auc_lo,
        "auc_95ci_high": auc_hi,
        "auc_formatted": format_metric(auc_value, auc_lo, auc_hi),
        "auprc": auprc_value,
        "auprc_formatted": format_metric(auprc_value, auprc_lo, auprc_hi),
        "brier": brier_value,
        "brier_formatted": format_metric(brier_value, brier_lo, brier_hi),
        "development_youden_threshold": dev_youden_threshold,
    }

    for prefix, metric_dict, threshold in [
        ("development_youden", metrics_dev_youden, dev_youden_threshold),
        ("threshold_0p5", metrics_0p5, 0.5),
    ]:
        summary[f"{prefix}_threshold"] = threshold
        for metric in ["accuracy", "sensitivity", "specificity", "f1", "ppv", "npv"]:
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
    summary_path = SAVE_DIR / "external_metrics_concat_clinical_ensemble.csv"
    safe_save_dataframe(summary_df, summary_path)

    plot_roc_curve(
        y_true,
        prob_mean,
        "Image + Clinical concat ROC: external validation",
        SAVE_DIR / "external_roc_concat_clinical_ensemble",
    )
    plot_pr_curve(
        y_true,
        prob_mean,
        "Image + Clinical concat PR: external validation",
        SAVE_DIR / "external_pr_concat_clinical_ensemble",
    )
    plot_calibration_curve(
        y_true,
        prob_mean,
        "Image + Clinical concat calibration: external validation",
        SAVE_DIR / "external_calibration_concat_clinical_ensemble",
    )

    config = {
        "model": MODEL_NAME,
        "model_dir": str(MODEL_DIR),
        "save_dir": str(SAVE_DIR),
        "csv_path": str(CSV_PATH),
        "ensemble": "mean probability from 5 fold-specific best models",
        "no_training": True,
        "no_hyperparameter_tuning": True,
        "no_external_threshold_selection": True,
        "threshold_source": "development pooled OOF Youden threshold",
        "development_youden_threshold": dev_youden_threshold,
        "clinical_cols_by_fold": clinical_cols_by_fold,
    }
    config_path = SAVE_DIR / "external_validation_config_concat_clinical.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved external validation config: {config_path}")

    readme = f"""
External validation outputs for Image + Clinical concat control model.

Main manuscript table:
- external_metrics_concat_clinical_ensemble.csv

External prediction audit:
- external_predictions_concat_clinical_ensemble.csv

Figures:
- external_roc_concat_clinical_ensemble.png/pdf
- external_pr_concat_clinical_ensemble.png/pdf
- external_calibration_concat_clinical_ensemble.png/pdf

Important:
- This model is not CA-AMT.
- It does not use Transformer.
- It does not use adaptive gating.
- It uses the development OOF Youden threshold.
- External validation is not used for training, imputation fitting, scaling fitting, threshold selection, or tuning.
""".strip()
    with open(SAVE_DIR / "README_external_outputs_and_usage_notes.txt", "w", encoding="utf-8") as f:
        f.write(readme)

    print("\n🎉 External validation completed.")
    print(f"📌 External AUC = {summary['auc_formatted']}")
    print(f"📌 External AUPRC = {summary['auprc_formatted']}")
    print(f"📌 External Brier = {summary['brier_formatted']}")
    print(
        "📌 0.5 threshold external result: "
        f"Accuracy={summary['threshold_0p5_accuracy_formatted']}, "
        f"Sensitivity={summary['threshold_0p5_sensitivity_formatted']}, "
        f"Specificity={summary['threshold_0p5_specificity_formatted']}, "
        f"F1={summary['threshold_0p5_f1_formatted']}"
    )
    print(f"📁 Output dir: {SAVE_DIR}")


if __name__ == "__main__":
    main()
