# -*- coding: utf-8 -*-
"""
CA-AMT v1 modality ablation and cross-modal token attention analysis.

Default usage:
    python "D:/tcm_AI/Patient-level image data/make_camt_modality_ablation_and_token_attention.py"

The script reuses the saved five fold CA-AMT models, fold-specific clinical
preprocessors, and fold split CSV files. It does not train or tune the model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(r"D:\tcm_AI\Patient-level image data")
CSV_PATH = PROJECT_ROOT / "Data" / "dataset_standard" / "patient_master_clinical_external_added.csv"
MODEL_DIR = PROJECT_ROOT / "outputs" / "cv_results_camt_mlp_v1" / "camt_mlp_v1"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "camt_modality_ablation_token_attention"

MODEL_NAME = "camt_mlp_v1"
FIGURE_DPI = 600
TOKEN_NAMES = ["left_hand", "right_hand", "tongue", "clinical"]
TOKEN_LABELS = ["Left hand", "Right hand", "Tongue", "Clinical"]
POS_LABEL_NAME = "CAD"
NEG_LABEL_NAME = "Non-CAD"

DEFAULT_RESIZE = 224
DEFAULT_BATCH_SIZE = 8
DEFAULT_N_FOLDS = 5
DEFAULT_SEED = 42
DEFAULT_FEATURE_DIM = 256
DEFAULT_NUM_HEADS = 8
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.1
DEFAULT_CLINICAL_HIDDEN_DIM = 128
DEFAULT_CLINICAL_DROPOUT = 0.2


@dataclass(frozen=True)
class AblationSpec:
    name: str
    label: str
    ablate_indices: Tuple[int, ...]


ABLATIONS: Tuple[AblationSpec, ...] = (
    AblationSpec("baseline", "Baseline", ()),
    AblationSpec("mask_left_hand", "Mask left hand", (0,)),
    AblationSpec("mask_right_hand", "Mask right hand", (1,)),
    AblationSpec("mask_tongue", "Mask tongue", (2,)),
    AblationSpec("mask_clinical", "Mask clinical", (3,)),
    AblationSpec("mask_all_images_keep_clinical", "Keep clinical only", (0, 1, 2)),
    AblationSpec("mask_clinical_keep_images", "Keep images only", (3,)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CA-AMT modality ablation and 4x4 token attention analysis."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--resize", type=int, default=DEFAULT_RESIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Use cuda, cpu, or a specific torch device string.",
    )
    parser.add_argument(
        "--skip-development",
        action="store_true",
        help="Only run external validation analysis.",
    )
    parser.add_argument(
        "--skip-external",
        action="store_true",
        help="Only run development OOF analysis.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_publication_style() -> None:
    """Use a restrained, journal-friendly plotting style."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
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


def safe_auc(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true_arr)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true_arr, y_prob_arr))


def load_run_config(model_dir: Path) -> Dict:
    config_path = model_dir / "run_config_camt_mlp_v1.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
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


def resolve_image_path(path_value: object, project_root: Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return project_root / path


class CAMTDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        clinical_array: np.ndarray,
        transform: T.Compose,
        project_root: Path,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.clinical_array = clinical_array.astype(np.float32)
        self.transform = transform
        self.project_root = project_root

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path_value: object) -> torch.Tensor:
        path = resolve_image_path(path_value, self.project_root)
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        left_x = self._load_image(row["hand_left_path"])
        right_x = self._load_image(row["hand_right_path"])
        tongue_x = self._load_image(row["tongue_path"])
        clinical_x = torch.tensor(self.clinical_array[idx], dtype=torch.float32)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        patient_uid = str(row["patient_uid"])
        return left_x, right_x, tongue_x, clinical_x, label, patient_uid


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
    state_dict = torch_load_state_dict(model_path, device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def disable_mha_fastpath() -> None:
    mha_backend = getattr(torch.backends, "mha", None)
    setter = getattr(mha_backend, "set_fastpath_enabled", None)
    if setter is not None:
        setter(False)


def patch_transformer_attention_capture(model) -> List[torch.nn.Module]:
    """
    Make TransformerEncoderLayer keep per-head self-attention weights.

    PyTorch's TransformerEncoderLayer normally calls MultiheadAttention with
    need_weights=False. This patch is inference-only and restores the 4x4
    query-token by key-token matrix needed for cross-modal heatmaps.
    """

    disable_mha_fastpath()
    layers = list(model.fusion_transformer.transformer_encoder.layers)

    for layer in layers:
        layer._camt_last_attn_weights = None
        if getattr(layer, "_camt_attention_patch", False):
            continue

        def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
            kwargs = {
                "attn_mask": attn_mask,
                "key_padding_mask": key_padding_mask,
                "need_weights": True,
            }
            try:
                attn_output, attn_weights = self.self_attn(
                    x,
                    x,
                    x,
                    average_attn_weights=False,
                    is_causal=is_causal,
                    **kwargs,
                )
            except TypeError:
                try:
                    attn_output, attn_weights = self.self_attn(
                        x,
                        x,
                        x,
                        average_attn_weights=False,
                        **kwargs,
                    )
                except TypeError:
                    attn_output, attn_weights = self.self_attn(x, x, x, **kwargs)
                    if attn_weights is not None and attn_weights.dim() == 3:
                        attn_weights = attn_weights.unsqueeze(1)

            self._camt_last_attn_weights = (
                attn_weights.detach().cpu() if attn_weights is not None else None
            )
            return self.dropout1(attn_output)

        layer._sa_block = types.MethodType(_sa_block, layer)
        layer._camt_attention_patch = True

    return layers


def clear_captured_attention(layers: Iterable[torch.nn.Module]) -> None:
    for layer in layers:
        layer._camt_last_attn_weights = None


def collect_captured_attention(layers: Sequence[torch.nn.Module]) -> Optional[np.ndarray]:
    weights = []
    for layer in layers:
        attn = getattr(layer, "_camt_last_attn_weights", None)
        if attn is None:
            return None
        if attn.dim() == 3:
            attn = attn.unsqueeze(1)
        weights.append(attn.float())
    if not weights:
        return None
    stacked = torch.stack(weights, dim=0)  # [layers, batch, heads, query, key]
    return stacked.mean(dim=0).mean(dim=1).numpy()  # [batch, query, key]


def encode_modalities(model, left_x, right_x, tongue_x, clinical_x):
    left_feat = model.hand_encoder(left_x)
    right_feat = model.hand_encoder(right_x)
    tongue_feat = model.tongue_encoder(tongue_x)
    clinical_feat = model.clinical_encoder(clinical_x)
    return left_feat, right_feat, tongue_feat, clinical_feat


def fusion_forward_from_features(model, features, ablate_indices: Sequence[int]) -> torch.Tensor:
    fusion = model.fusion_transformer
    left_feat, right_feat, tongue_feat, clinical_feat = features

    sequence = torch.stack(
        [
            fusion.left_projection(left_feat),
            fusion.right_projection(right_feat),
            fusion.tongue_projection(tongue_feat),
            fusion.clinical_projection(clinical_feat),
        ],
        dim=1,
    )

    token_sequence = sequence
    if ablate_indices:
        token_sequence = sequence.clone()
        token_sequence[:, list(ablate_indices), :] = 0.0

    gate_logits = fusion.gate_network(token_sequence.reshape(token_sequence.size(0), -1))
    modal_weights = torch.softmax(gate_logits, dim=1)
    gated_sequence = token_sequence * modal_weights.unsqueeze(-1)

    transformer_input = gated_sequence + fusion.modality_embed
    if ablate_indices:
        transformer_input = transformer_input.clone()
        transformer_input[:, list(ablate_indices), :] = 0.0

    transformer_output = fusion.transformer_encoder(transformer_input)
    token_attention = torch.softmax(fusion.attention_pool(transformer_output), dim=1)
    context_vector = torch.sum(transformer_output * token_attention, dim=1)
    return fusion.classifier(context_vector)


def predict_all_conditions(
    model,
    loader: DataLoader,
    device: torch.device,
    fold: int,
    cohort: str,
    capture_attention: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    layers = patch_transformer_attention_capture(model) if capture_attention else []
    prediction_records: Dict[str, List[Dict]] = {spec.name: [] for spec in ABLATIONS}
    attention_records: List[Dict] = []

    with torch.no_grad():
        for left_x, right_x, tongue_x, clinical_x, y, patient_ids in loader:
            left_x = left_x.to(device)
            right_x = right_x.to(device)
            tongue_x = tongue_x.to(device)
            clinical_x = clinical_x.to(device)
            labels = y.numpy().astype(int).tolist()
            patient_ids = [str(pid) for pid in patient_ids]

            features = encode_modalities(model, left_x, right_x, tongue_x, clinical_x)

            for spec in ABLATIONS:
                if capture_attention and spec.name == "baseline":
                    clear_captured_attention(layers)

                logits = fusion_forward_from_features(model, features, spec.ablate_indices)
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

                for pid, label, prob in zip(patient_ids, labels, probs):
                    prediction_records[spec.name].append(
                        {
                            "cohort": cohort,
                            "fold": int(fold),
                            "patient_uid": pid,
                            "label": int(label),
                            "condition": spec.name,
                            "condition_label": spec.label,
                            "prob_positive": float(prob),
                        }
                    )

                if capture_attention and spec.name == "baseline":
                    batch_attn = collect_captured_attention(layers)
                    if batch_attn is not None:
                        for row_idx, (pid, label) in enumerate(zip(patient_ids, labels)):
                            matrix = batch_attn[row_idx]
                            for q_idx, source in enumerate(TOKEN_NAMES):
                                for k_idx, target in enumerate(TOKEN_NAMES):
                                    attention_records.append(
                                        {
                                            "cohort": cohort,
                                            "fold": int(fold),
                                            "patient_uid": pid,
                                            "label": int(label),
                                            "source_token": source,
                                            "target_token": target,
                                            "attention": float(matrix[q_idx, k_idx]),
                                        }
                                    )

    pred_dfs = {name: pd.DataFrame(records) for name, records in prediction_records.items()}
    return pred_dfs, pd.DataFrame(attention_records)


def load_preprocessor(model_dir: Path, fold: int):
    path = model_dir / f"clinical_preprocessor_fold_{fold}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing clinical preprocessor: {path}")
    pack = joblib.load(path)
    return pack["preprocessor"], list(pack["clinical_cols"]), path


def build_loader_for_df(
    df: pd.DataFrame,
    clinical_array: np.ndarray,
    transform: T.Compose,
    project_root: Path,
    batch_size: int,
) -> DataLoader:
    ds = CAMTDataset(df, clinical_array, transform=transform, project_root=project_root)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)


def select_development_fold_df(full_df: pd.DataFrame, split_path: Path) -> pd.DataFrame:
    split_df = read_csv_robust(split_path)
    split_df["patient_uid"] = split_df["patient_uid"].astype(str)
    val_ids = split_df.loc[split_df["split"].astype(str).str.lower().eq("val"), "patient_uid"].tolist()
    if not val_ids:
        raise ValueError(f"No validation rows in {split_path}")

    dev_df = full_df[full_df["cohort"].astype(str).str.lower().eq("development")].copy()
    dev_df["patient_uid"] = dev_df["patient_uid"].astype(str)
    dev_df = dev_df.set_index("patient_uid", drop=False)
    missing = [pid for pid in val_ids if pid not in dev_df.index]
    if missing:
        raise ValueError(f"{len(missing)} validation IDs missing from master CSV: {missing[:5]}")
    return dev_df.loc[val_ids].reset_index(drop=True)


def merge_prediction_dicts(dicts: Sequence[Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    merged = {}
    for spec in ABLATIONS:
        parts = [d[spec.name] for d in dicts if spec.name in d and not d[spec.name].empty]
        merged[spec.name] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return merged


def average_external_predictions(pred_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    averaged = {}
    for spec in ABLATIONS:
        df = pred_dfs[spec.name]
        grouped = (
            df.groupby(["cohort", "patient_uid", "label", "condition", "condition_label"], as_index=False)
            .agg(prob_positive=("prob_positive", "mean"))
            .sort_values("patient_uid")
        )
        averaged[spec.name] = grouped
    return averaged


def make_probability_shift_df(pred_dfs: Dict[str, pd.DataFrame], cohort: str) -> pd.DataFrame:
    baseline = pred_dfs["baseline"][
        ["patient_uid", "label", "prob_positive"]
    ].rename(columns={"prob_positive": "baseline_prob"})
    rows = []
    for spec in ABLATIONS:
        if spec.name == "baseline":
            continue
        current = pred_dfs[spec.name][["patient_uid", "label", "prob_positive"]].rename(
            columns={"prob_positive": "ablated_prob"}
        )
        merged = baseline.merge(current, on=["patient_uid", "label"], how="inner")
        merged["cohort"] = cohort
        merged["condition"] = spec.name
        merged["condition_label"] = spec.label
        merged["delta_prob"] = merged["ablated_prob"] - merged["baseline_prob"]
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compute_ablation_table(pred_dfs: Dict[str, pd.DataFrame], shift_df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    baseline_auc = safe_auc(
        pred_dfs["baseline"]["label"].values,
        pred_dfs["baseline"]["prob_positive"].values,
    )
    rows = []
    for spec in ABLATIONS:
        df = pred_dfs[spec.name].copy()
        auc = safe_auc(df["label"].values, df["prob_positive"].values)
        row = {
            "cohort": cohort,
            "condition": spec.name,
            "condition_label": spec.label,
            "ablated_tokens": ";".join(TOKEN_NAMES[i] for i in spec.ablate_indices) or "none",
            "n": int(len(df)),
            "positive_n": int((df["label"].astype(int) == 1).sum()),
            "negative_n": int((df["label"].astype(int) == 0).sum()),
            "auc": auc,
            "delta_auc_vs_baseline": auc - baseline_auc if np.isfinite(auc) and np.isfinite(baseline_auc) else np.nan,
            "baseline_auc": baseline_auc,
        }
        if spec.name == "baseline" or shift_df.empty:
            row.update(
                {
                    "mean_delta_prob": 0.0,
                    "mean_abs_delta_prob": 0.0,
                    "mean_delta_prob_cad": 0.0,
                    "mean_delta_prob_non_cad": 0.0,
                }
            )
        else:
            sub = shift_df[shift_df["condition"].eq(spec.name)]
            row.update(
                {
                    "mean_delta_prob": float(sub["delta_prob"].mean()),
                    "mean_abs_delta_prob": float(sub["delta_prob"].abs().mean()),
                    "mean_delta_prob_cad": float(sub.loc[sub["label"].eq(1), "delta_prob"].mean()),
                    "mean_delta_prob_non_cad": float(sub.loc[sub["label"].eq(0), "delta_prob"].mean()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def patient_mean_attention(attn_df: pd.DataFrame) -> pd.DataFrame:
    if attn_df.empty:
        return attn_df
    return (
        attn_df.groupby(
            ["cohort", "patient_uid", "label", "source_token", "target_token"],
            as_index=False,
        )
        .agg(attention=("attention", "mean"))
        .sort_values(["patient_uid", "source_token", "target_token"])
    )


def attention_group_matrices(attn_patient_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    matrices: Dict[str, np.ndarray] = {}
    groups = {
        "Overall": attn_patient_df,
        POS_LABEL_NAME: attn_patient_df[attn_patient_df["label"].astype(int).eq(1)],
        NEG_LABEL_NAME: attn_patient_df[attn_patient_df["label"].astype(int).eq(0)],
    }
    for group_name, group_df in groups.items():
        matrix = np.full((len(TOKEN_NAMES), len(TOKEN_NAMES)), np.nan, dtype=float)
        for i, source in enumerate(TOKEN_NAMES):
            for j, target in enumerate(TOKEN_NAMES):
                values = group_df.loc[
                    group_df["source_token"].eq(source) & group_df["target_token"].eq(target),
                    "attention",
                ]
                matrix[i, j] = float(values.mean()) if len(values) else np.nan
        matrices[group_name] = matrix
    matrices[f"{POS_LABEL_NAME} minus {NEG_LABEL_NAME}"] = matrices[POS_LABEL_NAME] - matrices[NEG_LABEL_NAME]
    return matrices


def save_attention_matrix_table(matrices: Mapping[str, np.ndarray], cohort: str, output_dir: Path) -> pd.DataFrame:
    rows = []
    for group_name, matrix in matrices.items():
        for i, source in enumerate(TOKEN_NAMES):
            for j, target in enumerate(TOKEN_NAMES):
                rows.append(
                    {
                        "cohort": cohort,
                        "group": group_name,
                        "source_token": source,
                        "target_token": target,
                        "attention": float(matrix[i, j]),
                    }
                )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(
        output_dir / f"Table_token_attention_{cohort}_matrices.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return out_df


def save_figure(fig: plt.Figure, output_prefix: Path, tight_layout: bool = True) -> None:
    if tight_layout and not fig.get_constrained_layout():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This figure includes Axes that are not compatible with tight_layout.*",
            )
            fig.tight_layout()
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
    fig.savefig(output_prefix.with_suffix(".png"), dpi=FIGURE_DPI, **save_kwargs)
    fig.savefig(output_prefix.with_suffix(".pdf"), **save_kwargs)
    try:
        fig.savefig(
            output_prefix.with_suffix(".tiff"),
            dpi=FIGURE_DPI,
            format="tiff",
            pil_kwargs={"compression": "tiff_lzw"},
            **save_kwargs,
        )
    except TypeError:
        fig.savefig(output_prefix.with_suffix(".tiff"), dpi=FIGURE_DPI, format="tiff", **save_kwargs)
    plt.close(fig)


def plot_delta_auc(dev_table: Optional[pd.DataFrame], ext_table: Optional[pd.DataFrame], output_dir: Path) -> None:
    tables = []
    if dev_table is not None and not dev_table.empty:
        tables.append(dev_table)
    if ext_table is not None and not ext_table.empty:
        tables.append(ext_table)
    if not tables:
        return
    df = pd.concat(tables, ignore_index=True)
    df = df[~df["condition"].eq("baseline")].copy()

    labels = [spec.label for spec in ABLATIONS if spec.name != "baseline"]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    cohorts = list(df["cohort"].drop_duplicates())
    colors = {"development": "#4C78A8", "external": "#F58518"}
    cohort_label_map = {
        "development": "development",
        "external": "validation",
    }

    for idx, cohort in enumerate(cohorts):
        values = []
        cohort_df = df[df["cohort"].eq(cohort)].set_index("condition_label")
        for label in labels:
            values.append(float(cohort_df.loc[label, "delta_auc_vs_baseline"]) if label in cohort_df.index else np.nan)
        offset = (idx - (len(cohorts) - 1) / 2.0) * width
        display_label = cohort_label_map.get(str(cohort).lower(), cohort)
        ax.bar(x + offset, values, width=width, label=display_label, color=colors.get(cohort, None), alpha=0.88)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Delta AUC vs baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax.legend(frameon=False, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, output_dir / "Figure_modality_ablation_delta_auc")


def plot_probability_shift(shift_df: pd.DataFrame, output_dir: Path) -> None:
    if shift_df.empty:
        return
    cohorts = list(shift_df["cohort"].drop_duplicates())
    labels = [spec.label for spec in ABLATIONS if spec.name != "baseline"]
    fig, axes = plt.subplots(1, len(cohorts), figsize=(max(7.2, 4.3 * len(cohorts)), 3.7), sharey=True)
    if len(cohorts) == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    for ax, cohort in zip(axes, cohorts):
        sub = shift_df[shift_df["cohort"].eq(cohort)]
        box_data = [
            sub.loc[sub["condition_label"].eq(label), "delta_prob"].dropna().values
            for label in labels
        ]
        ax.boxplot(
            box_data,
            positions=np.arange(1, len(labels) + 1),
            widths=0.55,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#E8EDF3", "edgecolor": "#4A5568"},
            medianprops={"color": "#1A202C", "linewidth": 1.4},
            whiskerprops={"color": "#4A5568"},
            capprops={"color": "#4A5568"},
        )
        for pos, label in enumerate(labels, start=1):
            points = sub[sub["condition_label"].eq(label)]
            for label_value, color in [(0, "#4C78A8"), (1, "#E45756")]:
                vals = points.loc[points["label"].eq(label_value), "delta_prob"].dropna().values
                if len(vals) == 0:
                    continue
                jitter = rng.normal(0, 0.035, size=len(vals))
                ax.scatter(
                    np.full(len(vals), pos) + jitter,
                    vals,
                    s=14,
                    color=color,
                    alpha=0.55,
                    linewidths=0,
                    label=NEG_LABEL_NAME if label_value == 0 and pos == 1 else POS_LABEL_NAME if label_value == 1 and pos == 1 else None,
                )
        ax.axhline(0, color="black", linewidth=1)
        cohort_title_map = {
            "development": "Development",
            "external": "Validation",
        }
        ax.set_title(cohort_title_map.get(str(cohort).lower(), str(cohort).capitalize()))
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Ablated probability - baseline probability")
    axes[0].legend(frameon=False, loc="upper left")
    save_figure(fig, output_dir / "Figure_patient_probability_shift")


def draw_heatmap(ax, matrix: np.ndarray, title: str, cmap: str, vmin: float, vmax: float) -> None:
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(TOKEN_LABELS)))
    ax.set_yticks(np.arange(len(TOKEN_LABELS)))
    ax.set_xticklabels(TOKEN_LABELS, rotation=35, ha="right")
    ax.set_yticklabels(TOKEN_LABELS)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=7.5)
    ax.tick_params(axis="both", which="both", length=3)
    return im


def plot_attention_figures(matrices: Mapping[str, np.ndarray], cohort: str, output_dir: Path) -> None:
    if not matrices:
        return
    base_mats = [matrices["Overall"], matrices[POS_LABEL_NAME], matrices[NEG_LABEL_NAME]]
    finite_values = np.concatenate([m[np.isfinite(m)].ravel() for m in base_mats if np.isfinite(m).any()])
    vmin = float(finite_values.min()) if len(finite_values) else 0.0
    vmax = float(finite_values.max()) if len(finite_values) else 1.0

    fig = plt.figure(figsize=(7.6, 2.7), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.055], wspace=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    last_im = None
    for ax, group_name in zip(axes, ["Overall", POS_LABEL_NAME, NEG_LABEL_NAME]):
        last_im = draw_heatmap(ax, matrices[group_name], group_name, "viridis", vmin, vmax)
        ax.set_xlabel("Key token")
    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    axes[0].set_ylabel("Query token")
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("Attention")
    cbar.ax.tick_params(labelsize=7)
    prefix = (
        output_dir / "Figure_token_attention_overall_CAD_NonCAD"
        if cohort == "external"
        else output_dir / "Figure_token_attention_development_overall_CAD_NonCAD"
    )
    save_figure(fig, prefix, tight_layout=False)

    diff_name = f"{POS_LABEL_NAME} minus {NEG_LABEL_NAME}"
    diff = matrices[diff_name]
    finite_diff = diff[np.isfinite(diff)]
    max_abs = float(np.max(np.abs(finite_diff))) if len(finite_diff) else 1.0
    max_abs = max(max_abs, 1e-6)
    fig = plt.figure(figsize=(3.7, 3.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.055], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    im = draw_heatmap(ax, diff, "CAD - Non-CAD", "coolwarm", -max_abs, max_abs)
    ax.set_xlabel("Key token")
    ax.set_ylabel("Query token")
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Attention difference")
    cbar.ax.tick_params(labelsize=7)
    prefix = (
        output_dir / "Figure_token_attention_difference_CAD_minus_NonCAD"
        if cohort == "external"
        else output_dir / "Figure_token_attention_development_difference_CAD_minus_NonCAD"
    )
    save_figure(fig, prefix, tight_layout=False)


def run_development(
    full_df: pd.DataFrame,
    model_dir: Path,
    project_root: Path,
    model_params: Mapping[str, float],
    transform: T.Compose,
    batch_size: int,
    n_folds: int,
    device: torch.device,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n=== Development OOF replay ===")
    fold_prediction_dicts: List[Dict[str, pd.DataFrame]] = []
    attention_parts = []

    for fold in range(1, n_folds + 1):
        print(f"Fold {fold}/{n_folds}")
        split_path = model_dir / f"fold_{fold}_split.csv"
        model_path = model_dir / f"best_model_fold_{fold}.pth"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing fold split: {split_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")

        preprocessor, clinical_cols, _ = load_preprocessor(model_dir, fold)
        fold_df = select_development_fold_df(full_df, split_path)
        for col in clinical_cols:
            fold_df[col] = pd.to_numeric(fold_df[col], errors="coerce")
        x_clinical = preprocessor.transform(fold_df[clinical_cols])
        loader = build_loader_for_df(fold_df, x_clinical, transform, project_root, batch_size)

        model = build_model(project_root, model_path, len(clinical_cols), model_params, device)
        pred_dfs, attn_df = predict_all_conditions(
            model, loader, device, fold=fold, cohort="development", capture_attention=True
        )
        fold_prediction_dicts.append(pred_dfs)
        if not attn_df.empty:
            attention_parts.append(attn_df)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pred_dfs = merge_prediction_dicts(fold_prediction_dicts)
    for spec in ABLATIONS:
        pred_dfs[spec.name].to_csv(
            output_dir / f"Predictions_development_{spec.name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    shift_df = make_probability_shift_df(pred_dfs, "development")
    table = compute_ablation_table(pred_dfs, shift_df, "development")
    attention_df = pd.concat(attention_parts, ignore_index=True) if attention_parts else pd.DataFrame()
    return table, shift_df, attention_df


def run_external(
    full_df: pd.DataFrame,
    model_dir: Path,
    project_root: Path,
    model_params: Mapping[str, float],
    transform: T.Compose,
    batch_size: int,
    n_folds: int,
    device: torch.device,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n=== External ensemble replay ===")
    external_df = full_df[full_df["cohort"].astype(str).str.lower().eq("external")].copy()
    if external_df.empty:
        raise ValueError("No external rows found in master CSV.")
    external_df["patient_uid"] = external_df["patient_uid"].astype(str)

    fold_prediction_dicts: List[Dict[str, pd.DataFrame]] = []
    attention_parts = []

    for fold in range(1, n_folds + 1):
        print(f"Fold {fold}/{n_folds}")
        model_path = model_dir / f"best_model_fold_{fold}.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")

        preprocessor, clinical_cols, _ = load_preprocessor(model_dir, fold)
        fold_df = external_df.copy()
        for col in clinical_cols:
            fold_df[col] = pd.to_numeric(fold_df[col], errors="coerce")
        x_clinical = preprocessor.transform(fold_df[clinical_cols])
        loader = build_loader_for_df(fold_df, x_clinical, transform, project_root, batch_size)

        model = build_model(project_root, model_path, len(clinical_cols), model_params, device)
        pred_dfs, attn_df = predict_all_conditions(
            model, loader, device, fold=fold, cohort="external", capture_attention=True
        )
        fold_prediction_dicts.append(pred_dfs)
        if not attn_df.empty:
            attention_parts.append(attn_df)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_pred_dfs = merge_prediction_dicts(fold_prediction_dicts)
    for spec in ABLATIONS:
        fold_pred_dfs[spec.name].to_csv(
            output_dir / f"Predictions_external_by_fold_{spec.name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    ensemble_pred_dfs = average_external_predictions(fold_pred_dfs)
    for spec in ABLATIONS:
        ensemble_pred_dfs[spec.name].to_csv(
            output_dir / f"Predictions_external_ensemble_{spec.name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    shift_df = make_probability_shift_df(ensemble_pred_dfs, "external")
    table = compute_ablation_table(ensemble_pred_dfs, shift_df, "external")
    attention_by_fold = pd.concat(attention_parts, ignore_index=True) if attention_parts else pd.DataFrame()
    attention_patient = patient_mean_attention(attention_by_fold)
    return table, shift_df, attention_by_fold, attention_patient


def write_outputs(
    output_dir: Path,
    dev_table: Optional[pd.DataFrame],
    dev_shift: Optional[pd.DataFrame],
    dev_attn: Optional[pd.DataFrame],
    ext_table: Optional[pd.DataFrame],
    ext_shift: Optional[pd.DataFrame],
    ext_attn_by_fold: Optional[pd.DataFrame],
    ext_attn_patient: Optional[pd.DataFrame],
) -> None:
    if dev_table is not None:
        dev_table.to_csv(output_dir / "Table_modality_ablation_development.csv", index=False, encoding="utf-8-sig")
    if ext_table is not None:
        ext_table.to_csv(output_dir / "Table_modality_ablation_external.csv", index=False, encoding="utf-8-sig")
    if dev_shift is not None:
        dev_shift.to_csv(output_dir / "Patient_probability_shift_development.csv", index=False, encoding="utf-8-sig")
    if ext_shift is not None:
        ext_shift.to_csv(output_dir / "Patient_probability_shift_external.csv", index=False, encoding="utf-8-sig")

    if dev_attn is not None and not dev_attn.empty:
        dev_attn.to_csv(output_dir / "Token_attention_development_patient_long.csv", index=False, encoding="utf-8-sig")
        dev_mats = attention_group_matrices(dev_attn)
        save_attention_matrix_table(dev_mats, "development", output_dir)
        plot_attention_figures(dev_mats, "development", output_dir)
    if ext_attn_by_fold is not None and not ext_attn_by_fold.empty:
        ext_attn_by_fold.to_csv(
            output_dir / "Token_attention_external_by_fold_long.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if ext_attn_patient is not None and not ext_attn_patient.empty:
        ext_attn_patient.to_csv(
            output_dir / "Token_attention_external_patient_mean_long.csv",
            index=False,
            encoding="utf-8-sig",
        )
        ext_mats = attention_group_matrices(ext_attn_patient)
        save_attention_matrix_table(ext_mats, "external", output_dir)
        plot_attention_figures(ext_mats, "external", output_dir)

    plot_delta_auc(dev_table, ext_table, output_dir)
    shift_parts = [df for df in [dev_shift, ext_shift] if df is not None and not df.empty]
    if shift_parts:
        plot_probability_shift(pd.concat(shift_parts, ignore_index=True), output_dir)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    set_publication_style()
    project_root = args.project_root
    model_dir = args.model_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_run_config(model_dir)
    model_params = get_model_params(config)
    device = torch.device(args.device)
    transform = build_transform(args.resize)
    full_df = read_csv_robust(args.csv_path)
    full_df["patient_uid"] = full_df["patient_uid"].astype(str)

    required_cols = ["patient_uid", "cohort", "label", "hand_left_path", "hand_right_path", "tongue_path"]
    missing = [col for col in required_cols if col not in full_df.columns]
    if missing:
        raise ValueError(f"Master CSV missing required columns: {missing}")

    run_manifest = {
        "project_root": str(project_root),
        "csv_path": str(args.csv_path),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "resize": int(args.resize),
        "batch_size": int(args.batch_size),
        "n_folds": int(args.n_folds),
        "ablation_method": (
            "token-level ablation: zero selected projected modality token before gate, "
            "then zero the corresponding transformer input token including modality embedding"
        ),
        "attention_method": "mean over transformer layers and heads; rows=query tokens, columns=key tokens",
        "model_params": dict(model_params),
        "source_run_config": config,
    }
    with open(output_dir / "analysis_config.json", "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, ensure_ascii=False, indent=2)

    cohort_summary = full_df.groupby("cohort")["label"].agg(["count", "sum"]).reset_index()
    cohort_summary.to_csv(output_dir / "Input_cohort_summary.csv", index=False, encoding="utf-8-sig")
    print(cohort_summary.to_string(index=False))
    print(f"Device: {device}")
    print(f"Output dir: {output_dir}")

    dev_table = dev_shift = dev_attn = None
    ext_table = ext_shift = ext_attn_by_fold = ext_attn_patient = None

    if not args.skip_development:
        dev_table, dev_shift, dev_attn = run_development(
            full_df=full_df,
            model_dir=model_dir,
            project_root=project_root,
            model_params=model_params,
            transform=transform,
            batch_size=args.batch_size,
            n_folds=args.n_folds,
            device=device,
            output_dir=output_dir,
        )

    if not args.skip_external:
        ext_table, ext_shift, ext_attn_by_fold, ext_attn_patient = run_external(
            full_df=full_df,
            model_dir=model_dir,
            project_root=project_root,
            model_params=model_params,
            transform=transform,
            batch_size=args.batch_size,
            n_folds=args.n_folds,
            device=device,
            output_dir=output_dir,
        )

    write_outputs(
        output_dir=output_dir,
        dev_table=dev_table,
        dev_shift=dev_shift,
        dev_attn=dev_attn,
        ext_table=ext_table,
        ext_shift=ext_shift,
        ext_attn_by_fold=ext_attn_by_fold,
        ext_attn_patient=ext_attn_patient,
    )

    print("\nCompleted CA-AMT modality ablation and token attention analysis.")
    if dev_table is not None:
        print("\nDevelopment ablation AUC:")
        print(dev_table[["condition_label", "auc", "delta_auc_vs_baseline"]].to_string(index=False))
    if ext_table is not None:
        print("\nExternal ablation AUC:")
        print(ext_table[["condition_label", "auc", "delta_auc_vs_baseline"]].to_string(index=False))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
