# -*- coding: utf-8 -*-
r"""
Clinical-only modeling for CAD prediction - PyCharm version
==========================================================

Purpose
-------
Train and evaluate clinical-only models on patient-level structured data:
1) LASSO Logistic Regression: primary interpretable clinical baseline.
2) XGBoost: nonlinear comparator and SHAP interpretation model.

Designed for:
- development cohort: 5-fold patient-level out-of-fold evaluation
- external cohort: independent external validation

Default input file expected columns are compatible with:
patient_master_clinical_external_added.csv

PyCharm usage
-------------
1) Edit DATA_CSV and OUT_DIR in the settings block below.
2) Right-click this file in PyCharm and choose Run.
3) First run with QUICK_TEST=True. If no error occurs, set QUICK_TEST=False for final analysis.

Notes
-----
There is no generally accepted public pretrained model for this small tabular clinical-only CAD task.
This script trains the models on your development cohort, saves the fitted final models, and can reuse saved
final models for external prediction when --reuse_final_models is set.

Author: generated for patient-level CAD multimodal project
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from scipy.stats import loguniform, randint, uniform
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


CONTINUOUS_COLS = [
    "age_years",
    "bmi_kg_m2",
    "fasting_glucose_mmol_L",
    "tc_mmol_L",
    "tg_mmol_L",
    "ldl_c_mmol_L",
    "hdl_c_mmol_L",
    "creatinine_umol_L",
    "uric_acid_umol_L",
    "lvef_percent",
]

BINARY_COLS_WITHOUT_SYMPTOMS = [
    "sex_female",
    "smoking",
    "alcohol",
    "hypertension",
    "diabetes",
    "dyslipidemia",
    "family_history_cad",
    "prior_myocardial_infarction",
    "cerebral_infarction",
    "heart_failure",
    "renal_dysfunction",
    "peripheral_vascular_disease",
]

SYMPTOM_COLS = ["chest_pain", "chest_tightness"]

ID_COLS = [
    "patient_uid",
    "clinical_source_id",
    "cohort",
    "split",
    "label",
    "class_name",
]


@dataclass
class ModelResult:
    model_label: str
    feature_set: str
    oof_pred: pd.DataFrame
    external_pred: Optional[pd.DataFrame]
    metrics_rows: List[Dict]
    final_model_path: Path



# ======================================================================
# PyCharm 直接运行设置区
# 你只需要改下面这些路径和参数，然后在 PyCharm 里右键运行本文件即可。
# 不需要在 Terminal 里输入 python xxx.py --data_csv ...
# ======================================================================
DATA_CSV = r"D:\tcm_AI\Patient-level image data\Data\dataset_standard\patient_master_clinical_external_added.csv"
OUT_DIR = r"D:\tcm_AI\Patient-level image data\outputs\clinical_only_results_final"

# 基本设置
LABEL_COL = "label"
COHORT_COL = "cohort"
DEVELOPMENT_NAME = "development"
EXTERNAL_NAME = "external"
N_SPLITS = 5
SEED = 42
N_JOBS = -1

# 是否快速测试。第一次运行建议 True，确认没报错后改成 False 跑正式结果。
QUICK_TEST = False

# 正式运行时建议：N_BOOT_FULL = 2000, XGB_N_ITER_FULL = 15
# 快速测试时建议：N_BOOT_TEST = 200, XGB_N_ITER_TEST = 5
N_BOOT_FULL = 2000
XGB_N_ITER_FULL = 15
N_BOOT_TEST = 200
XGB_N_ITER_TEST = 5
XGB_INNER_CV = 3

# 模型开关
RUN_XGBOOST = True       # True：跑 LASSO + XGBoost；False：只跑 LASSO
RUN_SHAP = True          # True：生成 XGBoost 的 SHAP 图；False：跳过 SHAP，速度更快
REUSE_FINAL_MODELS = False  # True：复用已经保存的 final_*.joblib 模型


def parse_args() -> argparse.Namespace:
    """PyCharm版本：不读取命令行参数，直接使用上方设置区。"""
    n_boot = N_BOOT_TEST if QUICK_TEST else N_BOOT_FULL
    xgb_n_iter = XGB_N_ITER_TEST if QUICK_TEST else XGB_N_ITER_FULL
    return argparse.Namespace(
        data_csv=DATA_CSV,
        out_dir=OUT_DIR,
        label_col=LABEL_COL,
        cohort_col=COHORT_COL,
        development_name=DEVELOPMENT_NAME,
        external_name=EXTERNAL_NAME,
        n_splits=N_SPLITS,
        seed=SEED,
        n_boot=n_boot,
        xgb_n_iter=xgb_n_iter,
        xgb_inner_cv=XGB_INNER_CV,
        n_jobs=N_JOBS,
        reuse_final_models=REUSE_FINAL_MODELS,
        skip_xgboost=not RUN_XGBOOST,
        skip_shap=not RUN_SHAP,
    )

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_data(data_csv: str, label_col: str, cohort_col: str) -> pd.DataFrame:
    path = Path(data_csv)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find data_csv: {path}")
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Available columns: {list(df.columns)}")
    if cohort_col not in df.columns:
        raise ValueError(f"Cohort column '{cohort_col}' not found. Available columns: {list(df.columns)}")
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    if df[label_col].isna().any():
        bad = df[df[label_col].isna()].head(5)
        raise ValueError(f"Some labels are missing or non-numeric. Examples:\n{bad}")
    df[label_col] = df[label_col].astype(int)
    return df


def choose_existing_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def coerce_features_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def get_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    continuous = choose_existing_columns(df, CONTINUOUS_COLS)
    base_binary = choose_existing_columns(df, BINARY_COLS_WITHOUT_SYMPTOMS)
    symptoms = choose_existing_columns(df, SYMPTOM_COLS)
    without_symptoms = continuous + base_binary
    with_symptoms = continuous + base_binary + symptoms
    if not without_symptoms:
        raise ValueError("No clinical features found. Check column names in your CSV.")
    return {
        "without_symptoms": without_symptoms,
        "with_symptoms": with_symptoms,
    }


def split_numeric_types(feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    continuous = [c for c in feature_cols if c in CONTINUOUS_COLS]
    binary = [c for c in feature_cols if c not in CONTINUOUS_COLS]
    return continuous, binary


def build_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    continuous, binary = split_numeric_types(feature_cols)
    transformers = []
    if continuous:
        transformers.append(
            (
                "continuous",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                continuous,
            )
        )
    if binary:
        transformers.append(
            (
                "binary",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                    ]
                ),
                binary,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> List[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names = []
        for name, _, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            names.extend(list(cols))
        return names


def build_lasso_pipeline(feature_cols: List[str], seed: int, n_jobs: int) -> Pipeline:
    preprocessor = build_preprocessor(feature_cols)
    clf = LogisticRegressionCV(
        Cs=np.logspace(-4, 2, 15),
        cv=3,
        penalty="l1",
        solver="liblinear",
        scoring="roc_auc",
        max_iter=5000,
        class_weight="balanced",
        n_jobs=n_jobs,
        random_state=seed,
        refit=True,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


def build_xgb_pipeline(feature_cols: List[str], seed: int) -> Pipeline:
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost is not installed. Install with: pip install xgboost")
    preprocessor = build_preprocessor(feature_cols)
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=seed,
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        reg_alpha=0.0,
        n_jobs=1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", xgb)])


def tune_xgb_pipeline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_cols: List[str],
    seed: int,
    n_iter: int,
    inner_cv: int,
    n_jobs: int,
) -> Pipeline:
    pipe = build_xgb_pipeline(feature_cols, seed)
    if SCIPY_AVAILABLE:
        param_dist = {
            "model__n_estimators": randint(100, 600),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__max_depth": randint(2, 6),
            "model__min_child_weight": randint(1, 8),
            "model__subsample": uniform(0.65, 0.35),
            "model__colsample_bytree": uniform(0.65, 0.35),
            "model__reg_lambda": loguniform(0.1, 10.0),
            "model__reg_alpha": loguniform(1e-4, 1.0),
        }
    else:
        param_dist = {
            "model__n_estimators": [100, 200, 300, 500],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4, 5],
            "model__min_child_weight": [1, 3, 5, 7],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "model__reg_lambda": [0.1, 1.0, 3.0, 10.0],
            "model__reg_alpha": [0.0, 0.001, 0.01, 0.1],
        }
    cv = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=seed,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )
    search.fit(X_train[feature_cols], y_train)
    return search.best_estimator_


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx])


def compute_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        "auc": safe_auc(y_true, y_prob),
        "average_precision": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "brier_score": float(brier_score_loss(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision_ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    metric_name: str,
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    if n_boot <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y_true[idx]
        pp = y_prob[idx]
        if metric_name in {"auc", "average_precision"} and len(np.unique(yy)) < 2:
            continue
        try:
            m = compute_metrics_at_threshold(yy, pp, threshold)[metric_name]
            if not np.isnan(m):
                vals.append(m)
        except Exception:
            continue
    if len(vals) < 20:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def add_metric_rows(
    rows: List[Dict],
    model_label: str,
    feature_set: str,
    cohort_name: str,
    threshold_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_boot: int,
    seed: int,
) -> None:
    metrics = compute_metrics_at_threshold(y_true, y_prob, threshold)
    for metric_name in [
        "auc",
        "average_precision",
        "brier_score",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision_ppv",
        "npv",
        "f1",
    ]:
        lo, hi = bootstrap_ci(y_true, y_prob, threshold, metric_name, n_boot, seed)
        rows.append(
            {
                "model": model_label,
                "feature_set": feature_set,
                "cohort": cohort_name,
                "threshold_name": threshold_name,
                "metric": metric_name,
                "estimate": metrics[metric_name],
                "ci_lower": lo,
                "ci_upper": hi,
                "threshold": metrics["threshold"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
                "n": int(len(y_true)),
            }
        )


def train_predict_oof(
    df_dev: pd.DataFrame,
    feature_cols: List[str],
    y_col: str,
    model_label: str,
    seed: int,
    n_splits: int,
    xgb_n_iter: int,
    xgb_inner_cv: int,
    n_jobs: int,
    out_model_dir: Path,
) -> pd.DataFrame:
    y = df_dev[y_col].to_numpy(dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_prob = np.zeros(len(df_dev), dtype=float)
    fold_col = np.zeros(len(df_dev), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_dev[feature_cols], y), start=1):
        X_train = df_dev.iloc[train_idx]
        y_train = y[train_idx]
        X_val = df_dev.iloc[val_idx]

        if model_label == "lasso_logistic":
            model = build_lasso_pipeline(feature_cols, seed + fold, n_jobs)
            model.fit(X_train[feature_cols], y_train)
        elif model_label == "xgboost":
            model = tune_xgb_pipeline(
                X_train=X_train,
                y_train=y_train,
                feature_cols=feature_cols,
                seed=seed + fold,
                n_iter=xgb_n_iter,
                inner_cv=xgb_inner_cv,
                n_jobs=n_jobs,
            )
        else:
            raise ValueError(f"Unknown model_label: {model_label}")

        oof_prob[val_idx] = model.predict_proba(X_val[feature_cols])[:, 1]
        fold_col[val_idx] = fold
        fold_model_path = out_model_dir / f"fold{fold}_{model_label}.joblib"
        joblib.dump(model, fold_model_path)

    keep_cols = [c for c in ID_COLS if c in df_dev.columns]
    out = df_dev[keep_cols].copy()
    out["fold"] = fold_col
    out[f"prob_{model_label}"] = oof_prob
    out[f"pred_{model_label}_0p5"] = (oof_prob >= 0.5).astype(int)
    return out


def train_final_model(
    df_dev: pd.DataFrame,
    feature_cols: List[str],
    y_col: str,
    model_label: str,
    seed: int,
    xgb_n_iter: int,
    xgb_inner_cv: int,
    n_jobs: int,
) -> Pipeline:
    y = df_dev[y_col].to_numpy(dtype=int)
    if model_label == "lasso_logistic":
        model = build_lasso_pipeline(feature_cols, seed, n_jobs)
        model.fit(df_dev[feature_cols], y)
        return model
    if model_label == "xgboost":
        return tune_xgb_pipeline(
            X_train=df_dev,
            y_train=y,
            feature_cols=feature_cols,
            seed=seed,
            n_iter=xgb_n_iter,
            inner_cv=xgb_inner_cv,
            n_jobs=n_jobs,
        )
    raise ValueError(f"Unknown model_label: {model_label}")


def predict_external(
    df_ext: pd.DataFrame,
    final_model: Pipeline,
    feature_cols: List[str],
    model_label: str,
) -> pd.DataFrame:
    keep_cols = [c for c in ID_COLS if c in df_ext.columns]
    out = df_ext[keep_cols].copy()
    prob = final_model.predict_proba(df_ext[feature_cols])[:, 1]
    out[f"prob_{model_label}"] = prob
    out[f"pred_{model_label}_0p5"] = (prob >= 0.5).astype(int)
    return out


def save_lasso_coefficients(model: Pipeline, out_csv: Path) -> None:
    pre = model.named_steps["preprocess"]
    clf = model.named_steps["model"]
    names = get_feature_names_from_preprocessor(pre)
    coef = clf.coef_.ravel()
    odds_ratio = np.exp(coef)
    selected = np.abs(coef) > 1e-8
    df = pd.DataFrame(
        {
            "feature": names,
            "coefficient": coef,
            "odds_ratio_exp_coef": odds_ratio,
            "selected_by_lasso": selected,
        }
    ).sort_values("coefficient", key=lambda s: np.abs(s), ascending=False)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def save_xgb_importance(model: Pipeline, out_csv: Path) -> None:
    pre = model.named_steps["preprocess"]
    clf = model.named_steps["model"]
    names = get_feature_names_from_preprocessor(pre)
    try:
        importances = clf.feature_importances_
    except Exception:
        importances = np.full(len(names), np.nan)
    df = pd.DataFrame({"feature": names, "xgb_feature_importance": importances})
    df = df.sort_values("xgb_feature_importance", ascending=False)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def plot_roc_curves(prediction_sets: Dict[str, Tuple[np.ndarray, np.ndarray]], out_path: Path, title: str) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    plt.figure(figsize=(6.2, 5.2), dpi=300)
    for name, (y_true, y_prob) in prediction_sets.items():
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, linewidth=2.0, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="Reference")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def plot_pr_curves(prediction_sets: Dict[str, Tuple[np.ndarray, np.ndarray]], out_path: Path, title: str) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    plt.figure(figsize=(6.2, 5.2), dpi=300)
    for name, (y_true, y_prob) in prediction_sets.items():
        if len(np.unique(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, linewidth=2.0, label=f"{name} (AP={ap:.3f})")
    prevalence = np.mean(next(iter(prediction_sets.values()))[0]) if prediction_sets else 0
    plt.axhline(prevalence, linestyle="--", linewidth=1.0, label=f"Prevalence={prevalence:.3f}")
    plt.xlabel("Recall / Sensitivity")
    plt.ylabel("Precision / PPV")
    plt.title(title)
    plt.legend(loc="lower left", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def make_shap_plots(model: Pipeline, df_for_shap: pd.DataFrame, feature_cols: List[str], out_dir: Path, prefix: str) -> None:
    if not SHAP_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        return
    try:
        pre = model.named_steps["preprocess"]
        clf = model.named_steps["model"]
        X_trans = pre.transform(df_for_shap[feature_cols])
        feature_names = get_feature_names_from_preprocessor(pre)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_trans)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        # Beeswarm summary
        plt.figure(figsize=(7.2, 5.5), dpi=300)
        shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_shap_beeswarm.png", dpi=300, bbox_inches="tight")
        plt.savefig(out_dir / f"{prefix}_shap_beeswarm.pdf", bbox_inches="tight")
        plt.close()
        # Bar plot
        plt.figure(figsize=(7.2, 5.5), dpi=300)
        shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False, plot_type="bar", max_display=20)
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_shap_bar.png", dpi=300, bbox_inches="tight")
        plt.savefig(out_dir / f"{prefix}_shap_bar.pdf", bbox_inches="tight")
        plt.close()
        # Raw SHAP values
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        id_cols = [c for c in ID_COLS if c in df_for_shap.columns]
        shap_df = pd.concat([df_for_shap[id_cols].reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)
        shap_df.to_csv(out_dir / f"{prefix}_shap_values.csv", index=False, encoding="utf-8-sig")
    except Exception as e:
        with open(out_dir / f"{prefix}_shap_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


def run_one_model(
    df_dev: pd.DataFrame,
    df_ext: pd.DataFrame,
    feature_cols: List[str],
    feature_set_name: str,
    model_label: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> ModelResult:
    model_dir = out_dir / "models" / f"{model_label}_{feature_set_name}"
    ensure_dir(model_dir)

    df_dev = coerce_features_numeric(df_dev, feature_cols)
    df_ext = coerce_features_numeric(df_ext, feature_cols) if len(df_ext) else df_ext

    oof = train_predict_oof(
        df_dev=df_dev,
        feature_cols=feature_cols,
        y_col=args.label_col,
        model_label=model_label,
        seed=args.seed,
        n_splits=args.n_splits,
        xgb_n_iter=args.xgb_n_iter,
        xgb_inner_cv=args.xgb_inner_cv,
        n_jobs=args.n_jobs,
        out_model_dir=model_dir,
    )
    oof_prob = oof[f"prob_{model_label}"].to_numpy()
    y_dev = df_dev[args.label_col].to_numpy(dtype=int)
    thr_youden = youden_threshold(y_dev, oof_prob)
    oof[f"pred_{model_label}_youden"] = (oof_prob >= thr_youden).astype(int)

    final_model_path = model_dir / f"final_{model_label}_{feature_set_name}.joblib"
    if args.reuse_final_models and final_model_path.exists():
        final_model = joblib.load(final_model_path)
    else:
        final_model = train_final_model(
            df_dev=df_dev,
            feature_cols=feature_cols,
            y_col=args.label_col,
            model_label=model_label,
            seed=args.seed,
            xgb_n_iter=args.xgb_n_iter,
            xgb_inner_cv=args.xgb_inner_cv,
            n_jobs=args.n_jobs,
        )
        joblib.dump(final_model, final_model_path)

    external_pred = None
    if len(df_ext):
        external_pred = predict_external(df_ext, final_model, feature_cols, model_label)
        external_prob = external_pred[f"prob_{model_label}"].to_numpy()
        external_pred[f"pred_{model_label}_youden"] = (external_prob >= thr_youden).astype(int)

    # Save model-specific interpretability outputs
    if model_label == "lasso_logistic":
        save_lasso_coefficients(final_model, out_dir / f"coefficients_{model_label}_{feature_set_name}.csv")
    elif model_label == "xgboost":
        save_xgb_importance(final_model, out_dir / f"feature_importance_{model_label}_{feature_set_name}.csv")
        if not args.skip_shap:
            shap_source = df_ext if len(df_ext) else df_dev
            make_shap_plots(final_model, shap_source, feature_cols, out_dir, f"{model_label}_{feature_set_name}")

    # Metrics rows
    rows: List[Dict] = []
    for th_name, threshold in [("fixed_0.5", 0.5), ("development_oof_youden", thr_youden)]:
        add_metric_rows(
            rows,
            model_label=model_label,
            feature_set=feature_set_name,
            cohort_name="development_oof",
            threshold_name=th_name,
            y_true=y_dev,
            y_prob=oof_prob,
            threshold=threshold,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        if external_pred is not None:
            y_ext = df_ext[args.label_col].to_numpy(dtype=int)
            add_metric_rows(
                rows,
                model_label=model_label,
                feature_set=feature_set_name,
                cohort_name="external_validation",
                threshold_name=th_name,
                y_true=y_ext,
                y_prob=external_pred[f"prob_{model_label}"].to_numpy(),
                threshold=threshold,
                n_boot=args.n_boot,
                seed=args.seed + 1000,
            )

    oof.to_csv(out_dir / f"oof_predictions_{model_label}_{feature_set_name}.csv", index=False, encoding="utf-8-sig")
    if external_pred is not None:
        external_pred.to_csv(out_dir / f"external_predictions_{model_label}_{feature_set_name}.csv", index=False, encoding="utf-8-sig")

    return ModelResult(
        model_label=model_label,
        feature_set=feature_set_name,
        oof_pred=oof,
        external_pred=external_pred,
        metrics_rows=rows,
        final_model_path=final_model_path,
    )


def write_feature_manifest(out_dir: Path, feature_sets: Dict[str, List[str]], df: pd.DataFrame) -> None:
    rows = []
    for fs, cols in feature_sets.items():
        for c in cols:
            missing_rate = float(df[c].isna().mean()) if c in df.columns else np.nan
            rows.append({"feature_set": fs, "feature": c, "missing_rate_all": missing_rate})
    pd.DataFrame(rows).to_csv(out_dir / "clinical_feature_manifest.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "models")

    df = load_data(args.data_csv, args.label_col, args.cohort_col)
    feature_sets = get_feature_sets(df)
    all_features = sorted(set(sum(feature_sets.values(), [])))
    df = coerce_features_numeric(df, all_features)

    # Basic cohort checking
    df_dev = df[df[args.cohort_col].astype(str).str.lower() == args.development_name.lower()].copy()
    df_ext = df[df[args.cohort_col].astype(str).str.lower() == args.external_name.lower()].copy()
    if df_dev.empty:
        raise ValueError(f"No development cohort rows found using {args.cohort_col} == '{args.development_name}'")

    cohort_summary = (
        df.groupby([args.cohort_col, args.label_col])
        .size()
        .reset_index(name="n")
        .sort_values([args.cohort_col, args.label_col])
    )
    cohort_summary.to_csv(out_dir / "cohort_label_summary.csv", index=False, encoding="utf-8-sig")
    write_feature_manifest(out_dir, feature_sets, df)

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    results: List[ModelResult] = []
    for feature_set_name, feature_cols in feature_sets.items():
        # Always run LASSO primary model
        results.append(
            run_one_model(
                df_dev=df_dev,
                df_ext=df_ext,
                feature_cols=feature_cols,
                feature_set_name=feature_set_name,
                model_label="lasso_logistic",
                args=args,
                out_dir=out_dir,
            )
        )
        if not args.skip_xgboost:
            results.append(
                run_one_model(
                    df_dev=df_dev,
                    df_ext=df_ext,
                    feature_cols=feature_cols,
                    feature_set_name=feature_set_name,
                    model_label="xgboost",
                    args=args,
                    out_dir=out_dir,
                )
            )

    metrics_df = pd.DataFrame([r for res in results for r in res.metrics_rows])
    metrics_df.to_csv(out_dir / "clinical_only_metrics_long.csv", index=False, encoding="utf-8-sig")

    # Compact SCI-style performance table at Youden threshold
    compact = metrics_df[metrics_df["threshold_name"] == "development_oof_youden"].copy()
    compact["estimate_95ci"] = compact.apply(
        lambda r: f"{r['estimate']:.3f} ({r['ci_lower']:.3f}-{r['ci_upper']:.3f})"
        if pd.notna(r["ci_lower"])
        else f"{r['estimate']:.3f}",
        axis=1,
    )
    compact_pivot = compact.pivot_table(
        index=["model", "feature_set", "cohort"],
        columns="metric",
        values="estimate_95ci",
        aggfunc="first",
    ).reset_index()
    compact_pivot.to_csv(out_dir / "clinical_only_metrics_table_youden.csv", index=False, encoding="utf-8-sig")

    # Plot ROC and PR curves for development OOF and external validation
    for feature_set_name in feature_sets.keys():
        dev_sets = {}
        ext_sets = {}
        for res in results:
            if res.feature_set != feature_set_name:
                continue
            y_dev = df_dev[args.label_col].to_numpy(dtype=int)
            dev_sets[res.model_label] = (y_dev, res.oof_pred[f"prob_{res.model_label}"].to_numpy())
            if res.external_pred is not None and len(df_ext):
                y_ext = df_ext[args.label_col].to_numpy(dtype=int)
                ext_sets[res.model_label] = (y_ext, res.external_pred[f"prob_{res.model_label}"].to_numpy())
        plot_roc_curves(dev_sets, out_dir / f"ROC_development_oof_{feature_set_name}", f"Clinical-only ROC: development OOF ({feature_set_name})")
        plot_pr_curves(dev_sets, out_dir / f"PR_development_oof_{feature_set_name}", f"Clinical-only PR: development OOF ({feature_set_name})")
        if ext_sets:
            plot_roc_curves(ext_sets, out_dir / f"ROC_external_validation_{feature_set_name}", f"Clinical-only ROC: external validation ({feature_set_name})")
            plot_pr_curves(ext_sets, out_dir / f"PR_external_validation_{feature_set_name}", f"Clinical-only PR: external validation ({feature_set_name})")

    print("Clinical-only modeling completed.")
    print(f"Input CSV: {args.data_csv}")
    print(f"Development cohort: n={len(df_dev)}, positives={int(df_dev[args.label_col].sum())}, negatives={int((1-df_dev[args.label_col]).sum())}")
    if len(df_ext):
        print(f"External cohort: n={len(df_ext)}, positives={int(df_ext[args.label_col].sum())}, negatives={int((1-df_ext[args.label_col]).sum())}")
    print(f"Outputs saved to: {out_dir}")
    print("Key files:")
    print("- clinical_only_metrics_table_youden.csv")
    print("- clinical_only_metrics_long.csv")
    print("- oof_predictions_*.csv")
    print("- external_predictions_*.csv")
    print("- coefficients_lasso_logistic_*.csv")
    print("- feature_importance_xgboost_*.csv")
    print("- xgboost_*_shap_beeswarm.png/pdf, if SHAP is available")


if __name__ == "__main__":
    main()
