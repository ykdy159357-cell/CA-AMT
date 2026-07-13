# -*- coding: utf-8 -*-
"""
CA-AMT v1 patient-level 5-fold OOF training script.

Model:
  Left hand CNN + Right hand CNN + Tongue CNN + Clinical MLP
  -> Cross-modal Transformer + patient-specific adaptive gating
  -> CAD probability

Place this file at:
D:\tcm_AI\Patient-level image data\cv_train_patient_oof_camt_mlp_v1.py

Place patient_networks_camt_v1.py at:
D:\tcm_AI\Patient-level image data\scr\models\patient_networks_camt_v1.py

PyCharm usage:
  1. Check CSV_PATH and SAVE_DIR below.
  2. Keep QUICK_TEST=True for the first run.
  3. After it runs successfully, set QUICK_TEST=False and run again for final results.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from scr.models.patient_networks_camt_v1 import ClinicalAwareAdaptiveMultimodalTransformerModel


# =========================
# 1. PyCharm-friendly configuration
# =========================
CSV_PATH = Path(r"D:\tcm_AI\Patient-level image data\Data\dataset_standard\patient_master_clinical_external_added.csv")
SAVE_DIR = Path(r"D:\tcm_AI\Patient-level image data\outputs\cv_results_camt_mlp_v1")
MODEL_NAME = "camt_mlp_v1"

# Choose one: "base_symptoms", "base_no_symptoms", "full"
# For the first formal attempt, base_symptoms is more defensible than full.
CLINICAL_FEATURE_SET = "base_symptoms"

QUICK_TEST = False # First run should be True. Set False for final training.

RESIZE = 224
BATCH_SIZE = 4
EPOCHS = 6 if QUICK_TEST else 40
PATIENCE = 3 if QUICK_TEST else 10
LR = 1e-4
CLIP_GRAD_NORM = 1.0
N_SPLITS = 5
SEED = 42
BOOTSTRAP_N = 200 if QUICK_TEST else 2000

FEATURE_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 2
DROPOUT = 0.1
CLINICAL_HIDDEN_DIM = 128
CLINICAL_DROPOUT = 0.2

# Conservative variables: avoid downstream disease-consequence variables as the first version.
BASE_CLINICAL_COLS = [
    "age_years",
    "sex_female",
    "bmi_kg_m2",
    "fasting_glucose_mmol_L",
    "smoking",
    "alcohol",
    "hypertension",
    "diabetes",
    "dyslipidemia",
    "tc_mmol_L",
    "tg_mmol_L",
    "ldl_c_mmol_L",
    "hdl_c_mmol_L",
]
SYMPTOM_COLS = [
    "chest_tightness",
    "chest_pain",
]
FULL_EXTRA_COLS = [
    "family_history_cad",
    "prior_myocardial_infarction",
    "cerebral_infarction",
    "heart_failure",
    "renal_dysfunction",
    "peripheral_vascular_disease",
    "creatinine_umol_L",
    "uric_acid_umol_L",
    "lvef_percent",
]


# =========================
# 2. Utilities
# =========================
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Cannot decode CSV: {path}")


def get_clinical_cols(feature_set: str):
    if feature_set == "base_no_symptoms":
        return BASE_CLINICAL_COLS.copy()
    if feature_set == "base_symptoms":
        return BASE_CLINICAL_COLS + SYMPTOM_COLS
    if feature_set == "full":
        return BASE_CLINICAL_COLS + SYMPTOM_COLS + FULL_EXTRA_COLS
    raise ValueError(f"Unknown CLINICAL_FEATURE_SET: {feature_set}")


def safe_save_dataframe(df: pd.DataFrame, canonical_path: Path):
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = canonical_path.with_name(canonical_path.stem + f"_{timestamp}" + canonical_path.suffix)

    try:
        df.to_csv(canonical_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved: {canonical_path}")
        return canonical_path
    except PermissionError:
        df.to_csv(backup_path, index=False, encoding="utf-8-sig")
        print(f"⚠️ File occupied. Saved as: {backup_path}")
        return backup_path


def build_transforms(resize: int = 224):
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


class ClinicalMultimodalDataset(Dataset):
    def __init__(self, samples_list, transform=None):
        self.samples = samples_list
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def _load_img(self, image_path):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is None:
            raise ValueError("transform cannot be None.")
        return self.transform(image)

    def __getitem__(self, idx):
        item = self.samples[idx]
        left_img = self._load_img(item["hand_left_path"])
        right_img = self._load_img(item["hand_right_path"])
        tongue_img = self._load_img(item["tongue_path"])
        clinical = torch.tensor(item["clinical_features"], dtype=torch.float32)
        label = torch.tensor(int(item["label"]), dtype=torch.long)
        return left_img, right_img, tongue_img, clinical, label, item["patient_uid"]


def load_development_df(csv_path: Path, clinical_cols):
    df = read_csv_robust(csv_path)
    required = [
        "patient_uid", "cohort", "label", "hand_left_path", "hand_right_path", "tongue_path",
    ] + clinical_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df[df["cohort"].astype(str).str.lower().eq("development")].copy()
    if df.empty:
        raise ValueError("No development cohort rows found.")
    if df["patient_uid"].duplicated().any():
        raise ValueError("patient_uid duplicated in development cohort.")

    for c in clinical_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print("📌 Development cohort loaded")
    print(f"   N={len(df)}, Positive={(df['label'] == 1).sum()}, Negative={(df['label'] == 0).sum()}")
    print(f"   Clinical features ({len(clinical_cols)}): {clinical_cols}")
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


def build_clinical_preprocessor():
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def build_model(num_clinical_features: int, device):
    model = ClinicalAwareAdaptiveMultimodalTransformerModel(
        num_clinical_features=num_clinical_features,
        pretrained=True,
        feature_dim=FEATURE_DIM,
        clinical_hidden_dim=CLINICAL_HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_classes=2,
        dropout=DROPOUT,
        clinical_dropout=CLINICAL_DROPOUT,
    )
    return model.to(device)


# =========================
# 3. Metrics
# =========================
def compute_youden_threshold(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid = np.isfinite(thresholds)
    if valid.sum() == 0:
        return 0.5
    youden = tpr[valid] - fpr[valid]
    return float(thresholds[valid][np.argmax(youden)])


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


def build_summary_row(oof_df: pd.DataFrame, fold_aucs):
    y_true = oof_df["label"].astype(int).values
    y_prob = oof_df["prob_positive"].astype(float).values
    auc_value = roc_auc_score(y_true, y_prob)
    auprc_value = average_precision_score(y_true, y_prob)
    brier_value = brier_score_loss(y_true, y_prob)

    auc_lo, auc_hi = bootstrap_ci(y_true, y_prob, "auc", n_bootstrap=BOOTSTRAP_N, seed=SEED)
    auprc_lo, auprc_hi = bootstrap_ci(y_true, y_prob, "auprc", n_bootstrap=BOOTSTRAP_N, seed=SEED)
    brier_lo, brier_hi = bootstrap_ci(y_true, y_prob, "brier", n_bootstrap=BOOTSTRAP_N, seed=SEED)

    youden_threshold = compute_youden_threshold(y_true, y_prob)
    metrics_youden = compute_metrics_at_threshold(y_true, y_prob, youden_threshold)
    metrics_0p5 = compute_metrics_at_threshold(y_true, y_prob, 0.5)

    row = {
        "model": MODEL_NAME,
        "clinical_feature_set": CLINICAL_FEATURE_SET,
        "n": int(len(oof_df)),
        "positive_n": int((y_true == 1).sum()),
        "negative_n": int((y_true == 0).sum()),
        "pooled_oof_auc": auc_value,
        "pooled_oof_auc_95ci_low": auc_lo,
        "pooled_oof_auc_95ci_high": auc_hi,
        "pooled_oof_auc_formatted": format_metric(auc_value, auc_lo, auc_hi),
        "pooled_oof_auprc": auprc_value,
        "pooled_oof_auprc_formatted": format_metric(auprc_value, auprc_lo, auprc_hi),
        "pooled_oof_brier": brier_value,
        "pooled_oof_brier_formatted": format_metric(brier_value, brier_lo, brier_hi),
        "mean_best_fold_auc_for_log_only": float(np.mean(fold_aucs)),
        "sd_best_fold_auc_for_log_only": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else np.nan,
        "youden_threshold": youden_threshold,
    }

    for prefix, metric_dict, threshold in [
        ("youden", metrics_youden, youden_threshold),
        ("threshold_0p5", metrics_0p5, 0.5),
    ]:
        row[f"{prefix}_threshold"] = threshold
        for metric in ["accuracy", "sensitivity", "specificity", "f1", "ppv", "npv"]:
            lo, hi = bootstrap_ci(y_true, y_prob, metric, threshold=threshold, n_bootstrap=BOOTSTRAP_N, seed=SEED)
            row[f"{prefix}_{metric}"] = metric_dict[metric]
            row[f"{prefix}_{metric}_formatted"] = format_metric(metric_dict[metric], lo, hi)
        for key in ["tn", "fp", "fn", "tp"]:
            row[f"{prefix}_{key}"] = metric_dict[key]
    return row


# =========================
# 4. Training / evaluation
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device, max_grad_norm: float = 1.0):
    model.train()
    total_loss = 0.0

    for left_x, right_x, tongue_x, clinical_x, y, _ in loader:
        left_x = left_x.to(device)
        right_x = right_x.to(device)
        tongue_x = tongue_x.to(device)
        clinical_x = clinical_x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, _ = model(left_x, right_x, tongue_x, clinical_x)
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
    modal_weight_records = []

    with torch.no_grad():
        for left_x, right_x, tongue_x, clinical_x, y, patient_ids in loader:
            left_x = left_x.to(device)
            right_x = right_x.to(device)
            tongue_x = tongue_x.to(device)
            clinical_x = clinical_x.to(device)

            logits, attn = model(left_x, right_x, tongue_x, clinical_x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds_0p5 = (probs >= 0.5).long()

            all_patient_ids.extend(list(patient_ids))
            all_labels.extend(y.numpy().astype(int).tolist())
            all_probs.extend(probs.cpu().numpy().astype(float).tolist())
            all_preds_0p5.extend(preds_0p5.cpu().numpy().astype(int).tolist())

            if attn is not None and "modal_weights" in attn:
                weights = attn["modal_weights"].numpy()
                token_att = attn.get("token_attention", None)
                token_att_np = token_att.numpy() if token_att is not None else np.full_like(weights, np.nan)
                for i, pid in enumerate(patient_ids):
                    modal_weight_records.append({
                        "patient_uid": pid,
                        "left_weight": float(weights[i, 0]),
                        "right_weight": float(weights[i, 1]),
                        "tongue_weight": float(weights[i, 2]),
                        "clinical_weight": float(weights[i, 3]),
                        "left_token_attention": float(token_att_np[i, 0]),
                        "right_token_attention": float(token_att_np[i, 1]),
                        "tongue_token_attention": float(token_att_np[i, 2]),
                        "clinical_token_attention": float(token_att_np[i, 3]),
                    })

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) == 2 else float("nan")
    metrics_0p5 = compute_metrics_at_threshold(all_labels, all_probs, threshold=0.5)
    return auc, metrics_0p5, all_patient_ids, all_labels, all_probs, all_preds_0p5, modal_weight_records


def validate_oof_df(oof_df: pd.DataFrame, expected_n: int):
    required_cols = {"patient_uid", "fold", "label", "prob_positive"}
    missing = required_cols - set(oof_df.columns)
    if missing:
        raise ValueError(f"OOF missing columns: {missing}")
    if len(oof_df) != expected_n:
        raise ValueError(f"OOF row count error: {len(oof_df)} vs expected {expected_n}")
    if oof_df["patient_uid"].duplicated().sum() > 0:
        raise ValueError("patient_uid duplicated in OOF results.")
    if not oof_df["prob_positive"].between(0, 1).all():
        raise ValueError("prob_positive outside [0, 1].")
    print("📌 OOF QC passed")


def save_run_config(run_save_dir: Path, clinical_cols):
    config = {
        "model_name": MODEL_NAME,
        "csv_path": str(CSV_PATH),
        "save_dir": str(SAVE_DIR),
        "clinical_feature_set": CLINICAL_FEATURE_SET,
        "clinical_cols": clinical_cols,
        "quick_test": QUICK_TEST,
        "resize": RESIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "lr": LR,
        "clip_grad_norm": CLIP_GRAD_NORM,
        "n_splits": N_SPLITS,
        "seed": SEED,
        "bootstrap_n": BOOTSTRAP_N,
        "model": {
            "image_encoder": "existing ImageEncoder: ResNet18 pretrained + SE + projection",
            "clinical_encoder": "MLP",
            "fusion": "four-token cross-modal Transformer + patient-specific adaptive gating",
            "feature_dim": FEATURE_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
        },
        "preprocessing": "clinical imputer and scaler are fitted within each training fold only",
    }
    path = run_save_dir / "run_config_camt_mlp_v1.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved config: {path}")


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_save_dir = SAVE_DIR / MODEL_NAME
    run_save_dir.mkdir(parents=True, exist_ok=True)

    clinical_cols = get_clinical_cols(CLINICAL_FEATURE_SET)
    save_run_config(run_save_dir, clinical_cols)
    df = load_development_df(CSV_PATH, clinical_cols)

    feature_manifest = pd.DataFrame({
        "feature_name": clinical_cols,
        "feature_set": CLINICAL_FEATURE_SET,
        "used_in_model": 1,
    })
    safe_save_dataframe(feature_manifest, run_save_dir / "clinical_feature_manifest.csv")

    train_transform, val_transform = build_transforms(RESIZE)
    X_index = np.arange(len(df))
    y = df["label"].values.astype(int)
    groups = df["patient_uid"].values
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_records = []
    fold_summary_records = []
    fold_aucs = []
    all_modal_weight_records = []

    print(f"\n🚀 CA-AMT v1 training | device={device} | feature_set={CLINICAL_FEATURE_SET} | quick_test={QUICK_TEST}")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_index, y, groups), 1):
        print("\n" + "=" * 20 + f" Fold {fold}/{N_SPLITS} " + "=" * 20)
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        train_y = train_df["label"].astype(int).values
        val_y = val_df["label"].astype(int).values
        counts = np.bincount(train_y, minlength=2)
        val_counts = np.bincount(val_y, minlength=2)
        print(f"📊 Train: negative={counts[0]}, positive={counts[1]} | Val: negative={val_counts[0]}, positive={val_counts[1]}")

        # Fold-specific clinical preprocessing: prevents data leakage.
        preprocessor = build_clinical_preprocessor()
        x_train_clinical = preprocessor.fit_transform(train_df[clinical_cols])
        x_val_clinical = preprocessor.transform(val_df[clinical_cols])
        prep_path = run_save_dir / f"clinical_preprocessor_fold_{fold}.joblib"
        joblib.dump({
            "preprocessor": preprocessor,
            "clinical_cols": clinical_cols,
            "clinical_feature_set": CLINICAL_FEATURE_SET,
        }, prep_path)
        print(f"✅ Saved fold clinical preprocessor: {prep_path}")

        train_samples = make_samples(train_df, x_train_clinical)
        val_samples = make_samples(val_df, x_val_clinical)

        fold_split_df = pd.concat([
            train_df[["patient_uid", "label"]].assign(fold=fold, split="train"),
            val_df[["patient_uid", "label"]].assign(fold=fold, split="val"),
        ], ignore_index=True)
        safe_save_dataframe(fold_split_df, run_save_dir / f"fold_{fold}_split.csv")

        train_ds = ClinicalMultimodalDataset(train_samples, transform=train_transform)
        val_ds = ClinicalMultimodalDataset(val_samples, transform=val_transform)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        class_weights = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32).to(device)
        model = build_model(num_clinical_features=len(clinical_cols), device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

        best_val_auc = -np.inf
        best_epoch = None
        best_oof_data = None
        best_modal_weights = None
        best_model_path = run_save_dir / f"best_model_fold_{fold}.pth"
        epochs_no_improve = 0
        epoch_log_records = []

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, CLIP_GRAD_NORM)
            val_auc, val_metrics_0p5, pids, labels, probs, preds_0p5, modal_weight_records = evaluate(model, val_loader, device)

            print(
                f"Epoch [{epoch:02d}/{EPOCHS}] "
                f"Loss={train_loss:.4f} | Val AUC={val_auc:.4f} | "
                f"Sens@0.5={val_metrics_0p5['sensitivity']:.4f} | Spec@0.5={val_metrics_0p5['specificity']:.4f}"
            )

            epoch_log_records.append({
                "model": MODEL_NAME,
                "fold": fold,
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_auc": float(val_auc),
                "sens_0p5": float(val_metrics_0p5["sensitivity"]),
                "spec_0p5": float(val_metrics_0p5["specificity"]),
                "acc_0p5": float(val_metrics_0p5["accuracy"]),
                "f1_0p5": float(val_metrics_0p5["f1"]),
            })

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
                best_oof_data = list(zip(pids, labels, probs, preds_0p5))
                best_modal_weights = modal_weight_records
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= PATIENCE:
                print(f"🛑 Early stopping at epoch {epoch}")
                break

        if best_oof_data is None:
            raise RuntimeError(f"Fold {fold} produced no valid OOF predictions.")

        print(f"🏆 Fold {fold} best Val AUC={best_val_auc:.4f}, best_epoch={best_epoch}")
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
                "model": MODEL_NAME,
                "clinical_feature_set": CLINICAL_FEATURE_SET,
                "seed": SEED,
            })

        if best_modal_weights is not None:
            for r in best_modal_weights:
                r["fold"] = fold
                r["label"] = int(df.loc[df["patient_uid"].eq(r["patient_uid"]), "label"].iloc[0])
                r["model"] = MODEL_NAME
                all_modal_weight_records.append(r)

        fold_summary_records.append({
            "model": MODEL_NAME,
            "fold": fold,
            "train_n": len(train_df),
            "val_n": len(val_df),
            "train_positive_n": int((train_y == 1).sum()),
            "train_negative_n": int((train_y == 0).sum()),
            "val_positive_n": int((val_y == 1).sum()),
            "val_negative_n": int((val_y == 0).sum()),
            "best_epoch": int(best_epoch),
            "best_val_auc": float(best_val_auc),
            "best_model_path": str(best_model_path),
            "clinical_preprocessor_path": str(prep_path),
        })

        safe_save_dataframe(pd.DataFrame(epoch_log_records), run_save_dir / f"train_log_fold_{fold}.csv")

        del model, optimizer, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_df = pd.DataFrame(oof_records)
    validate_oof_df(oof_df, expected_n=len(df))
    y_true = oof_df["label"].astype(int).values
    y_prob = oof_df["prob_positive"].astype(float).values
    youden_threshold = compute_youden_threshold(y_true, y_prob)
    oof_df["pred_label_youden"] = (y_prob >= youden_threshold).astype(int)

    safe_save_dataframe(oof_df, run_save_dir / f"oof_predictions_{MODEL_NAME}.csv")
    safe_save_dataframe(oof_df, run_save_dir / f"oof_predictions_{MODEL_NAME}_latest.csv")
    safe_save_dataframe(pd.DataFrame(fold_summary_records), run_save_dir / f"fold_summary_{MODEL_NAME}.csv")

    summary_row = build_summary_row(oof_df, fold_aucs)
    summary_df = pd.DataFrame([summary_row])
    safe_save_dataframe(summary_df, run_save_dir / f"oof_metrics_{MODEL_NAME}.csv")

    if all_modal_weight_records:
        modal_df = pd.DataFrame(all_modal_weight_records)
        safe_save_dataframe(modal_df, run_save_dir / f"modality_weights_oof_{MODEL_NAME}.csv")

    print("\n🎉 CA-AMT v1 training completed.")
    print(f"📌 Pooled OOF AUC = {summary_row['pooled_oof_auc_formatted']}")
    print(f"📌 Youden threshold = {summary_row['youden_threshold']:.6f}")
    print(f"📁 Output dir: {run_save_dir}")


if __name__ == "__main__":
    main()
