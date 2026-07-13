# -*- coding: utf-8 -*-
"""
Patient-level explanation cards for CA-AMT v1.

Default usage:
    python "D:/tcm_AI/Patient-level image data/make_camt_patient_explanation_cards.py"

Default run:
    Generates external TP/TN/FP/FN candidates, scores their clinical
    neutralization attribution, automatically selects one visually informative
    patient from each case type, and generates publication-ready cards.

Manual run:
    Pass --selection-mode manual, fill one patient_uid for each TP/TN/FP/FN row
    in manual_selected_patient_cases.csv, then rerun to validate the choices.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont, ImageOps

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpecFromSubplotSpec  # noqa: E402


PROJECT_ROOT = Path(r"D:\tcm_AI\Patient-level image data")
CSV_PATH = PROJECT_ROOT / "Data" / "dataset_standard" / "patient_master_clinical_external_added_298.csv"
MODEL_DIR = PROJECT_ROOT / "outputs" / "cv_results_camt_mlp_v1" / "camt_mlp_v1"
EXTERNAL_PRED_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "external_validation"
    / "camt_mlp_v1"
    / "external_predictions_camt_mlp_v1_ensemble.csv"
)
EXTERNAL_MODALITY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "external_validation"
    / "camt_mlp_v1"
    / "external_modality_weights_mean_camt_mlp_v1.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "camt_patient_explanation_cards"

MODEL_NAME = "camt_mlp_v1"
FIGURE_DPI = 600
DEFAULT_RESIZE = 224
DEFAULT_N_FOLDS = 5
DEFAULT_SEED = 42

DEFAULT_FEATURE_DIM = 256
DEFAULT_NUM_HEADS = 8
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.1
DEFAULT_CLINICAL_HIDDEN_DIM = 128
DEFAULT_CLINICAL_DROPOUT = 0.2

CASE_ORDER = ["TP", "TN", "FP", "FN"]
CASE_TITLES = {
    "TP": "True-positive CAD case",
    "TN": "True-negative Non-CAD case",
    "FP": "False-positive case",
    "FN": "False-negative case",
}
MODALITY_COLS = ["left_weight", "right_weight", "tongue_weight", "clinical_weight"]
MODALITY_LABELS = ["Left hand", "Right hand", "Tongue", "Clinical"]
MODALITY_COLORS = ["#547AA5", "#86BCB6", "#DDA448", "#8B6BB1"]
CAD_COLOR = "#B44441"
NONCAD_COLOR = "#4E6E8E"
NEUTRAL_COLOR = "#D9D9D9"
VISIBLE_ATTRIBUTION_THRESHOLD = 1e-3
MIN_ATTRIBUTION_THRESHOLD = 5e-4
NEAR_ZERO_ATTRIBUTION_THRESHOLD = 1e-4

CLINICAL_DISPLAY_NAMES = {
    "age_years": "Age",
    "sex_female": "Female sex",
    "bmi_kg_m2": "BMI",
    "fasting_glucose_mmol_L": "FPG",
    "smoking": "Smoking",
    "alcohol": "Alcohol",
    "hypertension": "Hypertension",
    "diabetes": "Diabetes",
    "dyslipidemia": "Dyslipidemia",
    "tc_mmol_L": "TC",
    "tg_mmol_L": "TG",
    "ldl_c_mmol_L": "LDL-C",
    "hdl_c_mmol_L": "HDL-C",
    "chest_tightness": "Chest tightness",
    "chest_pain": "Chest pain",
}


@dataclass
class PatientExplanation:
    patient_uid: str
    case_type: str
    true_label: int
    predicted_label: int
    probability: float
    overlay_images: Dict[str, np.ndarray]
    modality_weights: Dict[str, float]
    top_attributions: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CA-AMT patient-level explanation cards.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--external-pred-path", type=Path, default=EXTERNAL_PRED_PATH)
    parser.add_argument("--external-modality-path", type=Path, default=EXTERNAL_MODALITY_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--manual-selected-path", type=Path, default=None)
    parser.add_argument(
        "--selection-mode",
        choices=["auto", "manual"],
        default="auto",
        help="Use automatic attribution-ranked case selection or a manual selection CSV.",
    )
    parser.add_argument("--resize", type=int, default=DEFAULT_RESIZE)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--visible-attribution-threshold", type=float, default=VISIBLE_ATTRIBUTION_THRESHOLD)
    parser.add_argument("--min-attribution-threshold", type=float, default=MIN_ATTRIBUTION_THRESHOLD)
    parser.add_argument("--near-zero-attribution-threshold", type=float, default=NEAR_ZERO_ATTRIBUTION_THRESHOLD)
    parser.add_argument(
        "--force-candidate-attribution",
        action="store_true",
        help="Recompute candidate-level clinical attribution scores even when a cache exists.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate cards and refresh cached candidate attribution scores.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv_robust(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Cannot decode CSV: {path}")


def label_text(label: int) -> str:
    return "CAD" if int(label) == 1 else "Non-CAD"


def case_type_from_label_pred(label: int, pred: int) -> str:
    if int(label) == 1 and int(pred) == 1:
        return "TP"
    if int(label) == 0 and int(pred) == 0:
        return "TN"
    if int(label) == 0 and int(pred) == 1:
        return "FP"
    return "FN"


def load_run_config(model_dir: Path) -> Dict:
    path = model_dir / "run_config_camt_mlp_v1.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_model_params(config: Mapping) -> Dict[str, float]:
    model_cfg = config.get("model", {}) if isinstance(config, Mapping) else {}
    return {
        "feature_dim": int(model_cfg.get("feature_dim", DEFAULT_FEATURE_DIM)),
        "num_heads": int(model_cfg.get("num_heads", DEFAULT_NUM_HEADS)),
        "num_layers": int(model_cfg.get("num_layers", DEFAULT_NUM_LAYERS)),
        "dropout": float(model_cfg.get("dropout", DEFAULT_DROPOUT)),
        "clinical_hidden_dim": int(model_cfg.get("clinical_hidden_dim", DEFAULT_CLINICAL_HIDDEN_DIM)),
        "clinical_dropout": float(model_cfg.get("clinical_dropout", DEFAULT_CLINICAL_DROPOUT)),
    }


def build_transform(resize: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((resize, resize)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def load_model_class(project_root: Path):
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    from scr.models.patient_networks_camt_v1 import (  # noqa: WPS433
        ClinicalAwareAdaptiveMultimodalTransformerModel,
    )

    return ClinicalAwareAdaptiveMultimodalTransformerModel


def torch_load_state_dict(path: Path, device: torch.device) -> Mapping[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(
    project_root: Path,
    model_path: Path,
    num_clinical_features: int,
    model_params: Mapping[str, float],
    device: torch.device,
):
    model_cls = load_model_class(project_root)
    model = model_cls(
        num_clinical_features=num_clinical_features,
        pretrained=False,
        feature_dim=int(model_params["feature_dim"]),
        clinical_hidden_dim=int(model_params["clinical_hidden_dim"]),
        num_heads=int(model_params["num_heads"]),
        num_layers=int(model_params["num_layers"]),
        num_classes=2,
        dropout=float(model_params["dropout"]),
        clinical_dropout=float(model_params["clinical_dropout"]),
    ).to(device)
    model.load_state_dict(torch_load_state_dict(model_path, device))
    model.eval()
    return model


def load_preprocessor(model_dir: Path, fold: int):
    path = model_dir / f"clinical_preprocessor_fold_{fold}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing clinical preprocessor: {path}")
    pack = joblib.load(path)
    return pack["preprocessor"], list(pack["clinical_cols"]), path


def resolve_path(path_value: object, project_root: Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return project_root / path


def image_quality_metrics(path: Path) -> Dict[str, float]:
    try:
        with Image.open(path) as image:
            arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    except Exception:
        return {
            "image_readable": 0,
            "image_brightness": np.nan,
            "image_contrast": np.nan,
            "image_nonwhite_fraction": np.nan,
            "image_quality_score": np.nan,
        }
    gray = arr.mean(axis=2)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    nonwhite = float((arr.min(axis=2) < 0.93).mean())
    score = float(0.45 * min(contrast / 0.25, 1.0) + 0.35 * nonwhite + 0.20 * (1.0 - abs(brightness - 0.55)))
    return {
        "image_readable": 1,
        "image_brightness": brightness,
        "image_contrast": contrast,
        "image_nonwhite_fraction": nonwhite,
        "image_quality_score": score,
    }


def make_candidate_table(
    csv_path: Path,
    external_pred_path: Path,
    external_modality_path: Path,
    model_dir: Path,
    output_dir: Path,
    project_root: Path,
) -> pd.DataFrame:
    pred_df = read_csv_robust(external_pred_path).copy()
    modality_df = read_csv_robust(external_modality_path).copy()
    master_df = read_csv_robust(csv_path).copy()
    _, clinical_cols, _ = load_preprocessor(model_dir, 1)

    required = [
        "patient_uid",
        "label",
        "pred_label_0p5",
        "prob_positive_mean",
    ]
    missing = [col for col in required if col not in pred_df.columns]
    if missing:
        raise ValueError(f"External prediction CSV missing columns: {missing}")

    pred_df["patient_uid"] = pred_df["patient_uid"].astype(str)
    master_df["patient_uid"] = master_df["patient_uid"].astype(str)
    if "cohort" in master_df.columns:
        master_df = master_df[master_df["cohort"].astype(str).str.lower().eq("external")].copy()
    if master_df["patient_uid"].duplicated().any():
        duplicated = sorted(master_df.loc[master_df["patient_uid"].duplicated(), "patient_uid"].unique())
        raise ValueError(f"Master clinical CSV has duplicated external patient_uid values: {duplicated[:10]}")

    master_source_cols = [
        "clinical_source_id",
        "source_sequence",
        "cohort",
        "split",
        "class_name",
        "class_name_original",
        "hand_left_path",
        "hand_right_path",
        "tongue_path",
        "image_merge_status",
        "merge_basis",
        *clinical_cols,
        "family_history_cad",
        "prior_myocardial_infarction",
        "cerebral_infarction",
        "heart_failure",
        "renal_dysfunction",
        "peripheral_vascular_disease",
        "creatinine_umol_L",
        "uric_acid_umol_L",
        "lvef_percent",
        "data_source_note",
        "qc_note",
        "order_within_label_cohort",
        "order_within_label",
    ]
    master_source_cols = [col for col in dict.fromkeys(master_source_cols) if col in master_df.columns]
    pred_df = pred_df.drop(columns=[col for col in master_source_cols if col in pred_df.columns], errors="ignore")
    pred_df = pred_df.merge(master_df[["patient_uid"] + master_source_cols], on="patient_uid", how="left", validate="one_to_one")
    missing_master = pred_df[pred_df["hand_left_path"].isna() | pred_df["hand_right_path"].isna() | pred_df["tongue_path"].isna()]
    if not missing_master.empty:
        raise ValueError(
            "Some external predictions could not be matched to the master clinical CSV: "
            f"{missing_master['patient_uid'].astype(str).tolist()[:10]}"
        )

    pred_df["case_type"] = [
        case_type_from_label_pred(y, p)
        for y, p in zip(pred_df["label"].astype(int), pred_df["pred_label_0p5"].astype(int))
    ]
    pred_df["true_label_name"] = pred_df["label"].astype(int).map(label_text)
    pred_df["predicted_label_name"] = pred_df["pred_label_0p5"].astype(int).map(label_text)

    modality_df["patient_uid"] = modality_df["patient_uid"].astype(str)
    merge_cols = ["patient_uid"] + [c for c in MODALITY_COLS if c in modality_df.columns]
    candidate_df = pred_df.merge(modality_df[merge_cols], on="patient_uid", how="left")

    for col in clinical_cols:
        if col in candidate_df.columns:
            candidate_df[col] = pd.to_numeric(candidate_df[col], errors="coerce")
    candidate_df["clinical_missing_count"] = candidate_df[clinical_cols].isna().sum(axis=1)

    quality_records = []
    for _, row in candidate_df.iterrows():
        metrics = {}
        for prefix, col in [
            ("left", "hand_left_path"),
            ("right", "hand_right_path"),
            ("tongue", "tongue_path"),
        ]:
            q = image_quality_metrics(resolve_path(row[col], project_root))
            metrics.update({f"{prefix}_{k}": v for k, v in q.items()})
        metrics["mean_image_quality_score"] = float(
            np.nanmean(
                [
                    metrics["left_image_quality_score"],
                    metrics["right_image_quality_score"],
                    metrics["tongue_image_quality_score"],
                ]
            )
        )
        quality_records.append(metrics)
    candidate_df = pd.concat([candidate_df, pd.DataFrame(quality_records)], axis=1)

    sort_key = candidate_df["prob_positive_mean"].astype(float)
    candidate_df["manual_review_priority"] = np.where(
        candidate_df["case_type"].isin(["TP", "FP"]),
        -sort_key,
        sort_key,
    )
    candidate_df = candidate_df.sort_values(["case_type", "manual_review_priority", "patient_uid"]).reset_index(drop=True)

    summary = candidate_df["case_type"].value_counts().reindex(CASE_ORDER).fillna(0).astype(int)
    print("External candidate counts:")
    print(summary.to_string())

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_external_cases.csv"
    candidate_df.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    print(f"Saved candidate table: {candidate_path}")

    xlsx_path = output_dir / "candidate_external_cases_by_type.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path) as writer:
            for case_type in CASE_ORDER:
                candidate_df[candidate_df["case_type"].eq(case_type)].to_excel(
                    writer,
                    sheet_name=case_type,
                    index=False,
                )
            candidate_df.to_excel(writer, sheet_name="ALL", index=False)
        print(f"Saved candidate workbook: {xlsx_path}")
    except Exception as exc:
        print(f"Could not write xlsx candidate workbook ({exc}). CSV was saved.")

    return candidate_df


def draw_text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, fill=(20, 20, 20), font=None) -> None:
    draw.text(xy, text, fill=fill, font=font)


def load_contact_font(size: int = 16):
    for font_name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def thumbnail_with_label(path: Path, label: str, size=(110, 110)) -> Image.Image:
    font = load_contact_font(13)
    canvas = Image.new("RGB", (size[0], size[1] + 22), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        with Image.open(path) as image:
            thumb = ImageOps.contain(image.convert("RGB"), size, method=Image.Resampling.BILINEAR)
        x = (size[0] - thumb.width) // 2
        y = (size[1] - thumb.height) // 2
        canvas.paste(thumb, (x, y))
    except Exception:
        draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=(180, 0, 0), width=2)
        draw_text(draw, (8, 42), "missing", fill=(180, 0, 0), font=font)
    draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=(190, 190, 190), width=1)
    draw_text(draw, (4, size[1] + 3), label, font=font)
    return canvas


def create_contact_sheets(candidate_df: pd.DataFrame, output_dir: Path, project_root: Path) -> None:
    font = load_contact_font(16)
    small_font = load_contact_font(13)
    card_w, card_h = 390, 200
    cols = 3
    margin = 24

    for case_type in CASE_ORDER:
        sub = candidate_df[candidate_df["case_type"].eq(case_type)].copy()
        if sub.empty:
            continue
        rows = int(np.ceil(len(sub) / cols))
        sheet = Image.new("RGB", (margin * 2 + cols * card_w, margin * 2 + rows * card_h), "white")
        draw = ImageDraw.Draw(sheet)
        for idx, (_, row) in enumerate(sub.iterrows()):
            col_idx = idx % cols
            row_idx = idx // cols
            x0 = margin + col_idx * card_w
            y0 = margin + row_idx * card_h
            draw.rounded_rectangle(
                [x0, y0, x0 + card_w - 10, y0 + card_h - 10],
                radius=8,
                outline=(215, 215, 215),
                width=2,
                fill=(252, 252, 252),
            )
            title = f"{case_type}  {row['patient_uid']}  p={float(row['prob_positive_mean']):.3f}"
            draw_text(draw, (x0 + 10, y0 + 8), title, font=font)
            subtitle = f"True {row['true_label_name']} | Pred {row['predicted_label_name']}"
            if "max_abs_clinical_attribution" in row.index and pd.notna(row["max_abs_clinical_attribution"]):
                subtitle += f" | max clinical Delta p={float(row['max_abs_clinical_attribution']):.1e}"
            draw_text(draw, (x0 + 10, y0 + 31), subtitle, fill=(80, 80, 80), font=small_font)
            img_y = y0 + 55
            for j, (label, path_col) in enumerate(
                [("Left", "hand_left_path"), ("Right", "hand_right_path"), ("Tongue", "tongue_path")]
            ):
                thumb = thumbnail_with_label(resolve_path(row[path_col], project_root), label, size=(105, 105))
                sheet.paste(thumb, (x0 + 10 + j * 122, img_y))
        path = output_dir / f"candidate_contact_sheet_{case_type}.png"
        sheet.save(path, dpi=(FIGURE_DPI, FIGURE_DPI))
        print(f"Saved contact sheet: {path}")


def ensure_manual_template(output_dir: Path) -> Path:
    path = output_dir / "manual_selected_patient_cases.csv"
    if path.exists():
        print(f"Manual selection file already exists: {path}")
        return path
    template = pd.DataFrame(
        {
            "case_type": CASE_ORDER,
            "patient_uid": ["", "", "", ""],
            "include_in_main": [1, 1, 1, 1],
            "note": ["", "", "", ""],
        }
    )
    template.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved manual selection template: {path}")
    return path


def load_and_validate_manual_selection(manual_path: Path, candidate_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    manual_df = read_csv_robust(manual_path)
    required = {"case_type", "patient_uid", "include_in_main"}
    missing = required - set(manual_df.columns)
    if missing:
        raise ValueError(f"Manual selection CSV missing columns: {sorted(missing)}")

    manual_df["case_type"] = manual_df["case_type"].astype(str).str.strip().str.upper()
    manual_df["patient_uid"] = manual_df["patient_uid"].fillna("").astype(str).str.strip()
    manual_df["include_in_main"] = pd.to_numeric(manual_df["include_in_main"], errors="coerce").fillna(0).astype(int)
    selected = manual_df[manual_df["include_in_main"].eq(1) & manual_df["patient_uid"].ne("")].copy()

    if selected.empty:
        print("manual_selected_patient_cases.csv has no selected patient_uid yet. Candidate materials are ready.")
        return None

    problems = []
    if set(selected["case_type"]) != set(CASE_ORDER) or len(selected) != 4:
        problems.append("Manual selection must contain exactly one selected TP, TN, FP, and FN row.")
    if selected["patient_uid"].duplicated().any():
        problems.append("Manual selection contains duplicated patient_uid values.")

    candidate_key = candidate_df.set_index("patient_uid", drop=False)
    rows = []
    for _, row in selected.iterrows():
        pid = row["patient_uid"]
        case_type = row["case_type"]
        if pid not in candidate_key.index:
            problems.append(f"{pid} was not found in external candidates.")
            continue
        candidate_row = candidate_key.loc[pid]
        if isinstance(candidate_row, pd.DataFrame):
            candidate_row = candidate_row.iloc[0]
        actual_type = str(candidate_row["case_type"])
        if actual_type != case_type:
            problems.append(f"{pid} is {actual_type}, not requested {case_type}.")
        rows.append(candidate_row)

    if problems:
        raise ValueError("\n".join(problems))

    selected_df = pd.DataFrame(rows)
    selected_df["case_type"] = pd.Categorical(selected_df["case_type"], categories=CASE_ORDER, ordered=True)
    selected_df = selected_df.sort_values("case_type").reset_index(drop=True)
    return selected_df


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def load_tensor(path: Path, transform: T.Compose, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
    return tensor


def forward_encoder_with_map(encoder, x: torch.Tensor):
    fmap = encoder.feature_extractor(x)
    fmap = encoder.se_block(fmap)
    fmap.retain_grad()
    feat = encoder.projection(fmap)
    return feat, fmap


def build_cam(feature_map: torch.Tensor) -> np.ndarray:
    if feature_map.grad is None:
        raise RuntimeError("Grad-CAM gradient was not captured.")
    grads = feature_map.grad
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feature_map.detach()).sum(dim=1))[0]
    cam_np = cam.detach().cpu().numpy().astype(np.float32)
    if float(cam_np.max()) > 1e-12:
        cam_np = cam_np / float(cam_np.max())
    return cam_np


def resize_cam(cam: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    cam_uint8 = np.uint8(np.clip(cam, 0, 1) * 255)
    cam_img = Image.fromarray(cam_uint8, mode="L").resize(target_size, resample=Image.Resampling.BILINEAR)
    cam_resized = np.asarray(cam_img, dtype=np.float32) / 255.0
    if float(cam_resized.max()) > 1e-12:
        cam_resized = cam_resized / float(cam_resized.max())
    return cam_resized


def enhance_cam_for_display(cam: np.ndarray, original_rgb: np.ndarray) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32)
    whiteness = original_rgb.min(axis=2)
    foreground = (whiteness < 0.95).astype(np.float32)
    if foreground.mean() < 0.05:
        foreground = np.ones_like(cam, dtype=np.float32)
    valid = cam[foreground > 0]
    vmax = float(np.percentile(valid, 99.0)) if valid.size else 1.0
    cam = np.clip(cam / max(vmax, 1e-8), 0, 1)
    cam = np.power(cam, 1.15)
    cam = np.clip((cam - 0.16) / 0.84, 0, 1)
    return np.clip(cam * foreground, 0, 1)


def overlay_cam(original_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    cam_display = enhance_cam_for_display(cam, original_rgb)
    cmap = plt.get_cmap("turbo")
    heat = cmap(cam_display)[..., :3].astype(np.float32)
    alpha_map = cam_display[..., None] * alpha
    overlay = original_rgb * (1.0 - alpha_map) + heat * alpha_map
    return np.clip(overlay, 0, 1)


def compute_fold_gradcam_and_attribution(
    model,
    row: pd.Series,
    clinical_cols: Sequence[str],
    preprocessor,
    transform: T.Compose,
    project_root: Path,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, float]:
    left_path = resolve_path(row["hand_left_path"], project_root)
    right_path = resolve_path(row["hand_right_path"], project_root)
    tongue_path = resolve_path(row["tongue_path"], project_root)

    left_x = load_tensor(left_path, transform, device)
    right_x = load_tensor(right_path, transform, device)
    tongue_x = load_tensor(tongue_path, transform, device)

    raw_clinical = row[list(clinical_cols)].copy()
    raw_clinical = pd.DataFrame(
        [pd.to_numeric(raw_clinical, errors="coerce").to_numpy(dtype=float)],
        columns=clinical_cols,
        index=[0],
    )
    x_clinical = preprocessor.transform(raw_clinical)
    clinical_x = torch.tensor(x_clinical, dtype=torch.float32).to(device)

    target_class = int(row["pred_label_0p5"])
    model.zero_grad(set_to_none=True)
    left_feat, left_map = forward_encoder_with_map(model.hand_encoder, left_x)
    right_feat, right_map = forward_encoder_with_map(model.hand_encoder, right_x)
    tongue_feat, tongue_map = forward_encoder_with_map(model.tongue_encoder, tongue_x)
    clinical_feat = model.clinical_encoder(clinical_x)
    logits, _ = model.fusion_transformer(left_feat, right_feat, tongue_feat, clinical_feat)
    score = logits[:, target_class].sum()
    score.backward()

    original_left = load_rgb(left_path)
    original_right = load_rgb(right_path)
    original_tongue = load_rgb(tongue_path)
    cams = {
        "left": resize_cam(build_cam(left_map), (original_left.shape[1], original_left.shape[0])),
        "right": resize_cam(build_cam(right_map), (original_right.shape[1], original_right.shape[0])),
        "tongue": resize_cam(build_cam(tongue_map), (original_tongue.shape[1], original_tongue.shape[0])),
    }

    with torch.no_grad():
        logits, _ = model(left_x, right_x, tongue_x, clinical_x)
        fold_baseline_prob = torch.softmax(logits, dim=1)[:, 1].item()

        neutral_rows = []
        imputer = preprocessor.named_steps.get("imputer", None)
        imputer_stats = getattr(imputer, "statistics_", np.zeros(len(clinical_cols)))
        for feature_idx, feature_name in enumerate(clinical_cols):
            raw_mod = raw_clinical.copy()
            raw_mod.iloc[0, feature_idx] = imputer_stats[feature_idx]
            neutral_rows.append(raw_mod.iloc[0])
        neutral_df = pd.DataFrame(neutral_rows, columns=clinical_cols)
        neutral_x = torch.tensor(preprocessor.transform(neutral_df), dtype=torch.float32).to(device)
        n = neutral_x.size(0)
        logits_mod, _ = model(
            left_x.repeat(n, 1, 1, 1),
            right_x.repeat(n, 1, 1, 1),
            tongue_x.repeat(n, 1, 1, 1),
            neutral_x,
        )
        neutral_probs = torch.softmax(logits_mod, dim=1)[:, 1].detach().cpu().numpy()
    attributions = float(fold_baseline_prob) - neutral_probs
    return cams, attributions.astype(float), float(fold_baseline_prob)


def compute_fold_clinical_attribution_only(
    model,
    row: pd.Series,
    clinical_cols: Sequence[str],
    preprocessor,
    transform: T.Compose,
    project_root: Path,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    left_x = load_tensor(resolve_path(row["hand_left_path"], project_root), transform, device)
    right_x = load_tensor(resolve_path(row["hand_right_path"], project_root), transform, device)
    tongue_x = load_tensor(resolve_path(row["tongue_path"], project_root), transform, device)

    raw_clinical = row[list(clinical_cols)].copy()
    raw_clinical = pd.DataFrame(
        [pd.to_numeric(raw_clinical, errors="coerce").to_numpy(dtype=float)],
        columns=clinical_cols,
        index=[0],
    )
    clinical_x = torch.tensor(preprocessor.transform(raw_clinical), dtype=torch.float32).to(device)

    imputer = preprocessor.named_steps.get("imputer", None)
    imputer_stats = getattr(imputer, "statistics_", np.zeros(len(clinical_cols)))
    neutral_rows = []
    for feature_idx, feature_name in enumerate(clinical_cols):
        raw_mod = raw_clinical.copy()
        raw_mod.iloc[0, feature_idx] = imputer_stats[feature_idx]
        neutral_rows.append(raw_mod.iloc[0])
    neutral_df = pd.DataFrame(neutral_rows, columns=clinical_cols)
    neutral_x = torch.tensor(preprocessor.transform(neutral_df), dtype=torch.float32).to(device)

    with torch.no_grad():
        logits, _ = model(left_x, right_x, tongue_x, clinical_x)
        fold_baseline_prob = torch.softmax(logits, dim=1)[:, 1].item()
        n = neutral_x.size(0)
        logits_mod, _ = model(
            left_x.repeat(n, 1, 1, 1),
            right_x.repeat(n, 1, 1, 1),
            tongue_x.repeat(n, 1, 1, 1),
            neutral_x,
        )
        neutral_probs = torch.softmax(logits_mod, dim=1)[:, 1].detach().cpu().numpy()
    return (float(fold_baseline_prob) - neutral_probs).astype(float), float(fold_baseline_prob)


def clinical_attribution_status(max_abs: float, visible_threshold: float, min_threshold: float) -> str:
    if max_abs >= visible_threshold:
        return "visible"
    if max_abs >= min_threshold:
        return "borderline_visible"
    if max_abs >= NEAR_ZERO_ATTRIBUTION_THRESHOLD:
        return "low"
    return "near_zero"


def top_attribution_summary(clinical_cols: Sequence[str], mean_attr: np.ndarray, top_n: int = 5) -> str:
    order = np.argsort(-np.abs(mean_attr))[:top_n]
    parts = []
    for idx in order:
        name = CLINICAL_DISPLAY_NAMES.get(clinical_cols[idx], clinical_cols[idx])
        parts.append(f"{name}:{float(mean_attr[idx]):+.6g}")
    return "; ".join(parts)


def merge_candidate_scores(candidate_df: pd.DataFrame, score_df: pd.DataFrame) -> pd.DataFrame:
    score_cols = [
        "patient_uid",
        "max_abs_clinical_attribution",
        "top_clinical_factor",
        "top_clinical_attribution",
        "top_clinical_attribution_summary",
        "clinical_attribution_status",
        "recomputed_prob_mean_for_attribution",
    ]
    existing = [col for col in score_cols if col != "patient_uid" and col in candidate_df.columns]
    base = candidate_df.drop(columns=existing)
    return base.merge(score_df[[col for col in score_cols if col in score_df.columns]], on="patient_uid", how="left")


def score_candidate_clinical_attributions(
    candidate_df: pd.DataFrame,
    model_dir: Path,
    project_root: Path,
    output_dir: Path,
    resize: int,
    n_folds: int,
    device: torch.device,
    visible_threshold: float,
    min_threshold: float,
    force: bool = False,
) -> pd.DataFrame:
    score_path = output_dir / "candidate_clinical_attribution_scores.csv"
    long_path = output_dir / "candidate_clinical_attributions_long.csv"
    if score_path.exists() and not force:
        cached = read_csv_robust(score_path)
        cached_ids = set(cached["patient_uid"].astype(str))
        current_ids = set(candidate_df["patient_uid"].astype(str))
        if cached_ids == current_ids and "max_abs_clinical_attribution" in cached.columns:
            print(f"Using cached candidate clinical attribution scores: {score_path}")
            return merge_candidate_scores(candidate_df, cached)
        print("Cached candidate clinical attribution scores do not match current candidates; recomputing.")

    config = load_run_config(model_dir)
    model_params = get_model_params(config)
    transform = build_transform(resize)
    candidate_df = candidate_df.copy()
    candidate_df["patient_uid"] = candidate_df["patient_uid"].astype(str)

    attr_accum: Dict[str, List[np.ndarray]] = {pid: [] for pid in candidate_df["patient_uid"]}
    prob_accum: Dict[str, List[float]] = {pid: [] for pid in candidate_df["patient_uid"]}
    clinical_cols_final: List[str] = []

    for fold in range(1, n_folds + 1):
        print(f"Scoring candidate clinical attribution: fold {fold}/{n_folds}")
        preprocessor, clinical_cols, _ = load_preprocessor(model_dir, fold)
        clinical_cols_final = list(clinical_cols)
        model_path = model_dir / f"best_model_fold_{fold}.pth"
        model = build_model(project_root, model_path, len(clinical_cols), model_params, device)
        for idx, row in candidate_df.iterrows():
            pid = str(row["patient_uid"])
            attrs, fold_prob = compute_fold_clinical_attribution_only(
                model=model,
                row=row,
                clinical_cols=clinical_cols,
                preprocessor=preprocessor,
                transform=transform,
                project_root=project_root,
                device=device,
            )
            attr_accum[pid].append(attrs)
            prob_accum[pid].append(fold_prob)
            if (idx + 1) % 10 == 0 or (idx + 1) == len(candidate_df):
                print(f"  fold {fold}: scored {idx + 1}/{len(candidate_df)} candidates")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    score_records = []
    long_records = []
    for _, row in candidate_df.iterrows():
        pid = str(row["patient_uid"])
        mean_attr = np.mean(np.stack(attr_accum[pid], axis=0), axis=0)
        abs_attr = np.abs(mean_attr)
        top_idx = int(np.argmax(abs_attr)) if len(abs_attr) else -1
        max_abs = float(abs_attr[top_idx]) if top_idx >= 0 else 0.0
        top_feature_name = clinical_cols_final[top_idx] if top_idx >= 0 else ""
        top_feature_label = CLINICAL_DISPLAY_NAMES.get(top_feature_name, top_feature_name)
        score_records.append(
            {
                "patient_uid": pid,
                "case_type": row["case_type"],
                "prob_positive_mean": float(row["prob_positive_mean"]),
                "mean_image_quality_score": float(row.get("mean_image_quality_score", np.nan)),
                "clinical_missing_count": int(row.get("clinical_missing_count", 0)),
                "max_abs_clinical_attribution": max_abs,
                "top_clinical_factor": top_feature_label,
                "top_clinical_attribution": float(mean_attr[top_idx]) if top_idx >= 0 else 0.0,
                "top_clinical_attribution_summary": top_attribution_summary(clinical_cols_final, mean_attr),
                "clinical_attribution_status": clinical_attribution_status(max_abs, visible_threshold, min_threshold),
                "recomputed_prob_mean_for_attribution": float(np.mean(prob_accum[pid])),
            }
        )
        for feature_name, value in zip(clinical_cols_final, mean_attr):
            long_records.append(
                {
                    "patient_uid": pid,
                    "case_type": row["case_type"],
                    "feature_name": feature_name,
                    "feature_label": CLINICAL_DISPLAY_NAMES.get(feature_name, feature_name),
                    "camt_clinical_attribution": float(value),
                    "abs_attribution": float(abs(value)),
                    "direction": "towards_CAD" if value >= 0 else "towards_NonCAD",
                }
            )

    score_df = pd.DataFrame(score_records).sort_values(
        ["case_type", "max_abs_clinical_attribution", "mean_image_quality_score"],
        ascending=[True, False, False],
    )
    long_df = pd.DataFrame(long_records).sort_values(["case_type", "patient_uid", "abs_attribution"], ascending=[True, True, False])
    score_df.to_csv(score_path, index=False, encoding="utf-8-sig")
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
    print(f"Saved candidate clinical attribution scores: {score_path}")
    print(f"Saved candidate clinical attribution long table: {long_path}")
    return merge_candidate_scores(candidate_df, score_df)


def prediction_clarity(row: pd.Series) -> float:
    prob = float(row["prob_positive_mean"])
    pred = int(row["pred_label_0p5"])
    return prob if pred == 1 else 1.0 - prob


def auto_select_cases(
    candidate_df: pd.DataFrame,
    manual_path: Path,
    output_dir: Path,
    min_threshold: float,
) -> pd.DataFrame:
    required = {"max_abs_clinical_attribution", "mean_image_quality_score", "clinical_missing_count"}
    missing = required - set(candidate_df.columns)
    if missing:
        raise ValueError(f"Candidate attribution scoring missing columns: {sorted(missing)}")

    selected_rows = []
    manual_records = []
    for case_type in CASE_ORDER:
        sub = candidate_df[candidate_df["case_type"].eq(case_type)].copy()
        if sub.empty:
            raise ValueError(f"No external candidates available for {case_type}.")
        readable_cols = [
            "left_image_readable",
            "right_image_readable",
            "tongue_image_readable",
        ]
        readable_mask = np.ones(len(sub), dtype=bool)
        for col in readable_cols:
            if col in sub.columns:
                readable_mask &= pd.to_numeric(sub[col], errors="coerce").fillna(0).astype(int).eq(1).values
        eligible = sub[pd.to_numeric(sub["clinical_missing_count"], errors="coerce").fillna(999).eq(0) & readable_mask].copy()
        if eligible.empty:
            eligible = sub.copy()
        eligible["prediction_clarity"] = eligible.apply(prediction_clarity, axis=1)
        eligible["meets_min_clinical_attribution"] = (
            eligible["max_abs_clinical_attribution"].astype(float) >= float(min_threshold)
        ).astype(int)
        eligible = eligible.sort_values(
            [
                "meets_min_clinical_attribution",
                "max_abs_clinical_attribution",
                "mean_image_quality_score",
                "prediction_clarity",
                "patient_uid",
            ],
            ascending=[False, False, False, False, True],
        )
        chosen = eligible.iloc[0].copy()
        selected_rows.append(chosen)
        note = (
            f"auto-selected by max clinical attribution; "
            f"max_abs={float(chosen['max_abs_clinical_attribution']):.6g}; "
            f"status={chosen.get('clinical_attribution_status', '')}; "
            f"top={chosen.get('top_clinical_attribution_summary', '')}"
        )
        manual_records.append(
            {
                "case_type": case_type,
                "patient_uid": str(chosen["patient_uid"]),
                "include_in_main": 1,
                "note": note,
            }
        )

    selected_df = pd.DataFrame(selected_rows)
    selected_df["case_type"] = pd.Categorical(selected_df["case_type"], categories=CASE_ORDER, ordered=True)
    selected_df = selected_df.sort_values("case_type").reset_index(drop=True)
    manual_df = pd.DataFrame(manual_records)
    auto_path = output_dir / "auto_selected_patient_cases.csv"
    selected_df.to_csv(auto_path, index=False, encoding="utf-8-sig")
    print(f"Saved auto-selected cases: {auto_path}")
    try:
        manual_df.to_csv(manual_path, index=False, encoding="utf-8-sig")
        print(f"Updated manual selection file with auto-selected cases: {manual_path}")
    except PermissionError as exc:
        fallback_path = output_dir / "manual_selected_patient_cases_auto.csv"
        manual_df.to_csv(fallback_path, index=False, encoding="utf-8-sig")
        print(f"Could not update locked manual selection file ({exc}). Saved fallback: {fallback_path}")
    return selected_df


def save_overlay_image(arr: np.ndarray, path_prefix: Path) -> None:
    image = Image.fromarray(np.uint8(np.clip(arr, 0, 1) * 255))
    image.save(path_prefix.with_suffix(".png"), dpi=(FIGURE_DPI, FIGURE_DPI))
    image.save(path_prefix.with_suffix(".tiff"), dpi=(FIGURE_DPI, FIGURE_DPI), compression="tiff_lzw")


def compute_selected_explanations(
    selected_df: pd.DataFrame,
    model_dir: Path,
    project_root: Path,
    output_dir: Path,
    resize: int,
    n_folds: int,
    device: torch.device,
) -> Tuple[List[PatientExplanation], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = load_run_config(model_dir)
    model_params = get_model_params(config)
    transform = build_transform(resize)
    overlay_dir = output_dir / "gradcam_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    fold_preprocessors = {}
    clinical_cols_final = None
    for fold in range(1, n_folds + 1):
        preprocessor, clinical_cols, _ = load_preprocessor(model_dir, fold)
        fold_preprocessors[fold] = (preprocessor, clinical_cols)
        clinical_cols_final = clinical_cols
    clinical_cols_final = clinical_cols_final or []

    explanations: List[PatientExplanation] = []
    attribution_records = []
    modality_records = []
    fold_prob_records = []

    for _, row in selected_df.iterrows():
        pid = str(row["patient_uid"])
        case_type = str(row["case_type"])
        print(f"Explaining {case_type} {pid}")
        modality_weights = {label: float(row[col]) for label, col in zip(MODALITY_LABELS, MODALITY_COLS)}
        modality_records.append(
            {
                "patient_uid": pid,
                "case_type": case_type,
                **{col: float(row[col]) for col in MODALITY_COLS},
                "weight_sum": float(sum(float(row[col]) for col in MODALITY_COLS)),
            }
        )

        cam_accum = {"left": [], "right": [], "tongue": []}
        attr_accum = []
        fold_probs = []

        for fold in range(1, n_folds + 1):
            model_path = model_dir / f"best_model_fold_{fold}.pth"
            preprocessor, clinical_cols = fold_preprocessors[fold]
            model = build_model(project_root, model_path, len(clinical_cols), model_params, device)
            cams, attrs, fold_prob = compute_fold_gradcam_and_attribution(
                model=model,
                row=row,
                clinical_cols=clinical_cols,
                preprocessor=preprocessor,
                transform=transform,
                project_root=project_root,
                device=device,
            )
            for modality in cam_accum:
                cam_accum[modality].append(cams[modality])
            attr_accum.append(attrs)
            fold_probs.append(fold_prob)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        fold_prob_records.append(
            {
                "patient_uid": pid,
                "case_type": case_type,
                **{f"recomputed_prob_fold_{i+1}": float(v) for i, v in enumerate(fold_probs)},
                "recomputed_prob_mean": float(np.mean(fold_probs)),
                "csv_prob_positive_mean": float(row["prob_positive_mean"]),
            }
        )

        overlay_images = {}
        for modality, path_col in [
            ("left", "hand_left_path"),
            ("right", "hand_right_path"),
            ("tongue", "tongue_path"),
        ]:
            original = load_rgb(resolve_path(row[path_col], project_root))
            mean_cam = np.mean(np.stack(cam_accum[modality], axis=0), axis=0)
            overlay = overlay_cam(original, mean_cam)
            overlay_images[modality] = overlay
            save_overlay_image(overlay, overlay_dir / f"{case_type}_{pid}_{modality}_camt_gradcam")

        mean_attr = np.mean(np.stack(attr_accum, axis=0), axis=0)
        attr_df = pd.DataFrame(
            {
                "patient_uid": pid,
                "case_type": case_type,
                "feature_name": clinical_cols_final,
                "feature_label": [CLINICAL_DISPLAY_NAMES.get(c, c) for c in clinical_cols_final],
                "camt_clinical_attribution": mean_attr,
                "abs_attribution": np.abs(mean_attr),
                "direction": np.where(mean_attr >= 0, "towards_CAD", "towards_NonCAD"),
            }
        ).sort_values("abs_attribution", ascending=False)
        attribution_records.append(attr_df)
        top_attr = attr_df.head(5).iloc[::-1].reset_index(drop=True)

        explanations.append(
            PatientExplanation(
                patient_uid=pid,
                case_type=case_type,
                true_label=int(row["label"]),
                predicted_label=int(row["pred_label_0p5"]),
                probability=float(row["prob_positive_mean"]),
                overlay_images=overlay_images,
                modality_weights=modality_weights,
                top_attributions=top_attr,
            )
        )

    attribution_df = pd.concat(attribution_records, ignore_index=True)
    modality_df = pd.DataFrame(modality_records)
    fold_prob_df = pd.DataFrame(fold_prob_records)
    return explanations, attribution_df, modality_df, fold_prob_df


def save_figure(fig: plt.Figure, output_prefix: Path) -> None:
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
    fig.savefig(output_prefix.with_suffix(".png"), dpi=FIGURE_DPI, **kwargs)
    fig.savefig(output_prefix.with_suffix(".pdf"), **kwargs)
    fig.savefig(
        output_prefix.with_suffix(".tiff"),
        dpi=FIGURE_DPI,
        format="tiff",
        pil_kwargs={"compression": "tiff_lzw"},
        **kwargs,
    )
    plt.close(fig)


def format_attribution_label(value: float) -> str:
    value = float(value)
    abs_value = abs(value)
    if abs_value < NEAR_ZERO_ATTRIBUTION_THRESHOLD:
        return "<1e-4"
    if abs_value < 1e-3:
        return f"{value:+.1e}"
    return f"{value:+.3f}"


def draw_probability_axis(ax, explanation: PatientExplanation) -> None:
    prob = explanation.probability
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.65, 0.65)
    ax.barh([0], [1.0], height=0.22, color="#ECECEC", edgecolor="#BFBFBF", linewidth=0.5)
    ax.barh([0], [prob], height=0.22, color=CAD_COLOR if prob >= 0.5 else NONCAD_COLOR, alpha=0.82)
    ax.axvline(0.5, color="black", linewidth=0.8, linestyle="--")
    ax.scatter([prob], [0], s=20, color="black", zorder=3)
    text_box = {"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.92}
    ax.text(
        0.0,
        0.39,
        f"True: {label_text(explanation.true_label)}",
        ha="left",
        va="center",
        bbox=text_box,
        zorder=4,
    )
    ax.text(
        0.50,
        0.39,
        f"Pred: {label_text(explanation.predicted_label)}",
        ha="center",
        va="center",
        bbox=text_box,
        zorder=4,
    )
    ax.text(
        1.0,
        0.39,
        f"p(CAD)={prob:.3f}",
        ha="right",
        va="center",
        bbox=text_box,
        zorder=4,
    )
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([])
    ax.tick_params(axis="x", pad=1)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def draw_modality_axis(ax, explanation: PatientExplanation) -> None:
    labels = list(explanation.modality_weights.keys())[::-1]
    values = [explanation.modality_weights[label] for label in labels]
    colors = MODALITY_COLORS[::-1]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.56)
    for yi, value in zip(y, values):
        ax.text(min(value + 0.015, 0.98), yi, f"{value:.2f}", va="center", ha="left", fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Modality weight")
    ax.set_title("Fusion weights", pad=2)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_attribution_axis(ax, explanation: PatientExplanation) -> None:
    df = explanation.top_attributions.copy()
    values = df["camt_clinical_attribution"].astype(float).values
    labels = df["feature_label"].astype(str).values
    raw_max_abs = float(np.max(np.abs(values))) if len(values) else 0.0
    colors = [CAD_COLOR if v >= 0 else NONCAD_COLOR for v in values]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.56)
    max_abs = max(raw_max_abs, 0.01)
    label_pad = max_abs * 0.11
    x_limit = max(max_abs * 2.15, 0.12)
    ax.set_xlim(-x_limit, x_limit)
    for yi, value in zip(y, values):
        if value >= 0:
            x_pos = value + label_pad
            ha = "left"
        else:
            x_pos = value - label_pad
            ha = "right"
        ax.text(
            x_pos,
            yi,
            format_attribution_label(value),
            va="center",
            ha=ha,
            fontsize=6.4,
            clip_on=False,
        )
    ax.axvline(0, color="black", linewidth=0.8)
    if raw_max_abs < NEAR_ZERO_ATTRIBUTION_THRESHOLD:
        ax.text(
            0.98,
            0.04,
            "all |Delta p| < 1e-4",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.8,
            color="#555555",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Clinical attribution (Delta p)")
    ax.set_title("Top clinical factors", pad=2)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_case_card(fig: plt.Figure, outer_spec, explanation: PatientExplanation, panel_label: str) -> None:
    sub = GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=outer_spec,
        height_ratios=[1.05, 0.28, 0.70],
        width_ratios=[0.95, 1.05],
        hspace=0.58,
        wspace=0.52,
    )
    image_grid = GridSpecFromSubplotSpec(1, 3, subplot_spec=sub[0, :], wspace=0.07)
    title = f"{panel_label}. {CASE_TITLES[explanation.case_type]} ({explanation.patient_uid})"
    for idx, (modality, label) in enumerate([("left", "Left palm"), ("right", "Right palm"), ("tongue", "Tongue")]):
        ax = fig.add_subplot(image_grid[0, idx])
        ax.imshow(explanation.overlay_images[modality])
        ax.set_title(label, pad=2, fontsize=7.2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_color("#777777")
    title_ax = fig.add_subplot(sub[0, :], frameon=False)
    title_ax.set_title(title, loc="left", pad=14, fontsize=8.5, fontweight="bold")
    title_ax.set_xticks([])
    title_ax.set_yticks([])
    title_ax.patch.set_alpha(0.0)

    prob_ax = fig.add_subplot(sub[1, :])
    draw_probability_axis(prob_ax, explanation)

    mod_ax = fig.add_subplot(sub[2, 0])
    draw_modality_axis(mod_ax, explanation)

    attr_ax = fig.add_subplot(sub[2, 1])
    draw_attribution_axis(attr_ax, explanation)


def plot_main_2x2(explanations: Sequence[PatientExplanation], output_dir: Path) -> None:
    fig = plt.figure(figsize=(7.4, 8.4))
    outer = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.26)
    labels = ["A", "B", "C", "D"]
    for idx, explanation in enumerate(explanations):
        draw_case_card(fig, outer[idx // 2, idx % 2], explanation, labels[idx])
    save_figure(fig, output_dir / "Figure_patient_explanation_cards_2x2")


def plot_supplement_4row(explanations: Sequence[PatientExplanation], output_dir: Path) -> None:
    fig = plt.figure(figsize=(7.4, 14.0))
    outer = fig.add_gridspec(4, 1, hspace=0.48)
    labels = ["A", "B", "C", "D"]
    for idx, explanation in enumerate(explanations):
        draw_case_card(fig, outer[idx, 0], explanation, labels[idx])
    save_figure(fig, output_dir / "Supplementary_patient_explanation_cards_4row")


def validate_outputs(output_dir: Path) -> pd.DataFrame:
    records = []
    for path in sorted(output_dir.glob("*.png")) + sorted(output_dir.glob("*.tiff")):
        with Image.open(path) as image:
            records.append(
                {
                    "file": path.name,
                    "width_px": image.size[0],
                    "height_px": image.size[1],
                    "dpi": image.info.get("dpi"),
                    "bytes": path.stat().st_size,
                }
            )
    return pd.DataFrame(records)


def write_figure_caption(output_dir: Path) -> Path:
    caption = (
        "Patient-level CA-AMT explanations. For each case, Grad-CAM overlays show image regions "
        "supporting the predicted class. Fusion weights summarize the adaptive contribution of "
        "left palm, right palm, tongue, and clinical tokens. Clinical factor bars show CA-AMT-specific "
        "single-feature neutralization attribution, defined as the change in p(CAD) after replacing "
        "one clinical feature with its fold-specific training reference value; positive values increase "
        "predicted CAD probability and negative values decrease it. Panels with all |Delta p| < 1e-4 "
        "indicate locally near-zero single-feature clinical attribution rather than missing clinical data."
    )
    path = output_dir / "Figure_patient_explanation_cards_caption.txt"
    path.write_text(caption, encoding="utf-8")
    print(f"Saved figure caption draft: {path}")
    return path


def main() -> None:
    global NEAR_ZERO_ATTRIBUTION_THRESHOLD
    args = parse_args()
    NEAR_ZERO_ATTRIBUTION_THRESHOLD = float(args.near_zero_attribution_threshold)
    set_seed(args.seed)
    set_publication_style()

    project_root = args.project_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manual_path = args.manual_selected_path or (output_dir / "manual_selected_patient_cases.csv")

    candidate_df = make_candidate_table(
        csv_path=args.csv_path,
        external_pred_path=args.external_pred_path,
        external_modality_path=args.external_modality_path,
        model_dir=args.model_dir,
        output_dir=output_dir,
        project_root=project_root,
    )
    device = torch.device(args.device)
    candidate_df = score_candidate_clinical_attributions(
        candidate_df=candidate_df,
        model_dir=args.model_dir,
        project_root=project_root,
        output_dir=output_dir,
        resize=args.resize,
        n_folds=args.n_folds,
        device=device,
        visible_threshold=args.visible_attribution_threshold,
        min_threshold=args.min_attribution_threshold,
        force=bool(args.force or args.force_candidate_attribution),
    )
    candidate_df.to_csv(output_dir / "candidate_external_cases.csv", index=False, encoding="utf-8-sig")
    create_contact_sheets(candidate_df, output_dir, project_root)

    if args.selection_mode == "auto":
        selected_df = auto_select_cases(
            candidate_df=candidate_df,
            manual_path=manual_path,
            output_dir=output_dir,
            min_threshold=args.min_attribution_threshold,
        )
    else:
        ensure_manual_template(output_dir)
        selected_df = load_and_validate_manual_selection(manual_path, candidate_df)
        if selected_df is None:
            manifest = {
                "status": "candidate_materials_only",
                "message": "Fill manual_selected_patient_cases.csv with one TP/TN/FP/FN patient_uid, then rerun.",
                "candidate_counts": candidate_df["case_type"].value_counts().reindex(CASE_ORDER).fillna(0).astype(int).to_dict(),
                "manual_selected_path": str(manual_path),
            }
            with open(output_dir / "patient_level_card_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            print("\nStopped after candidate generation. Fill the manual selection CSV and rerun.")
            return

    selected_path = output_dir / "selected_patient_explanation_cases.csv"
    selected_df.to_csv(selected_path, index=False, encoding="utf-8-sig")
    print(f"Saved validated selected cases: {selected_path}")

    explanations, attribution_df, modality_df, fold_prob_df = compute_selected_explanations(
        selected_df=selected_df,
        model_dir=args.model_dir,
        project_root=project_root,
        output_dir=output_dir,
        resize=args.resize,
        n_folds=args.n_folds,
        device=device,
    )

    attribution_df.to_csv(output_dir / "patient_level_clinical_attributions.csv", index=False, encoding="utf-8-sig")
    modality_df.to_csv(output_dir / "patient_level_modality_weights.csv", index=False, encoding="utf-8-sig")
    fold_prob_df.to_csv(output_dir / "patient_level_recomputed_fold_probabilities.csv", index=False, encoding="utf-8-sig")

    plot_main_2x2(explanations, output_dir)
    plot_supplement_4row(explanations, output_dir)
    caption_path = write_figure_caption(output_dir)

    image_qc = validate_outputs(output_dir)
    image_qc.to_csv(output_dir / "patient_level_card_image_qc.csv", index=False, encoding="utf-8-sig")

    selected_case_cols = [
        "case_type",
        "patient_uid",
        "label",
        "pred_label_0p5",
        "prob_positive_mean",
        "max_abs_clinical_attribution",
        "clinical_attribution_status",
        "top_clinical_attribution_summary",
    ]
    selected_case_cols = [col for col in selected_case_cols if col in selected_df.columns]
    manifest = {
        "status": "completed",
        "model": MODEL_NAME,
        "selection_mode": args.selection_mode,
        "case_definition": "external validation pred_label_0p5",
        "clinical_explanation": "CA-AMT-specific neutralization attribution; positive values push predicted CAD probability upward.",
        "clinical_explanation_caption": str(caption_path),
        "visible_attribution_threshold": args.visible_attribution_threshold,
        "minimum_attribution_threshold": args.min_attribution_threshold,
        "near_zero_attribution_threshold": args.near_zero_attribution_threshold,
        "gradcam_target": "predicted class from pred_label_0p5",
        "figure_dpi": FIGURE_DPI,
        "selected_cases": selected_df[selected_case_cols].to_dict("records"),
    }
    with open(output_dir / "patient_level_card_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nCompleted patient-level explanation cards.")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
