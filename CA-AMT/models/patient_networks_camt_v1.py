# -*- coding: utf-8 -*-
"""
CA-AMT v1: Clinical-aware Adaptive Multimodal Transformer, MLP clinical branch.

Place this file at:
D:\tcm_AI\Patient-level image data\scr\models\patient_networks_camt_v1.py

It reuses the existing ImageEncoder from scr.models.patient_networks to keep image
encoders consistent with the previous image-only model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scr.models.patient_networks import ImageEncoder


class ClinicalMLPEncoder(nn.Module):
    """
    Lightweight clinical encoder for small-sample tabular clinical variables.

    Input:  [B, num_clinical_features]
    Output: [B, feature_dim]
    """

    def __init__(
        self,
        num_clinical_features: int,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_clinical_features <= 0:
            raise ValueError("num_clinical_features must be positive.")

        self.num_clinical_features = int(num_clinical_features)
        self.feature_dim = int(feature_dim)

        self.encoder = nn.Sequential(
            nn.Linear(num_clinical_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, clinical_x: torch.Tensor) -> torch.Tensor:
        clinical_x = clinical_x.float()
        return self.encoder(clinical_x)


class ClinicalAwareAdaptiveFusionTransformer(nn.Module):
    """
    Four-modality fusion module:
    left hand token, right hand token, tongue token, clinical token.

    Main design:
    1. Each modality is projected to a common feature_dim.
    2. A cross-modal Transformer models interactions among the four tokens.
    3. A patient-specific gate generates [B, 4] modality weights for each patient.
    4. An attention-pooling layer aggregates the gated token sequence for CAD prediction.

    This is the key difference from the previous global modal_weights design.
    """

    def __init__(
        self,
        input_dim: int = 256,
        feature_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        gate_hidden_dim: int = 128,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_modalities = 4

        self.left_projection = nn.Linear(input_dim, feature_dim)
        self.right_projection = nn.Linear(input_dim, feature_dim)
        self.tongue_projection = nn.Linear(input_dim, feature_dim)
        self.clinical_projection = nn.Linear(input_dim, feature_dim)

        self.modality_embed = nn.Parameter(torch.zeros(1, self.num_modalities, feature_dim))
        nn.init.trunc_normal_(self.modality_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Patient-specific adaptive gating.
        # It generates one 4-dimensional gate vector for each patient.
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim * self.num_modalities, gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, self.num_modalities),
        )

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

    def forward(self, left_feat, right_feat, tongue_feat, clinical_feat):
        left_proj = self.left_projection(left_feat)
        right_proj = self.right_projection(right_feat)
        tongue_proj = self.tongue_projection(tongue_feat)
        clinical_proj = self.clinical_projection(clinical_feat)

        sequence = torch.stack(
            [left_proj, right_proj, tongue_proj, clinical_proj],
            dim=1,
        )  # [B, 4, D]

        # Gate is calculated from the patient-level multimodal token set.
        gate_logits = self.gate_network(sequence.reshape(sequence.size(0), -1))
        modal_weights = F.softmax(gate_logits, dim=1)  # [B, 4]

        gated_sequence = sequence * modal_weights.unsqueeze(-1)
        transformer_input = gated_sequence + self.modality_embed
        transformer_output = self.transformer_encoder(transformer_input)  # [B, 4, D]

        token_attention = F.softmax(self.attention_pool(transformer_output), dim=1)  # [B, 4, 1]
        context_vector = torch.sum(transformer_output * token_attention, dim=1)  # [B, D]
        logits = self.classifier(context_vector)

        attn_dict = {
            "modal_weights": modal_weights.detach().cpu(),
            "token_attention": token_attention.squeeze(-1).detach().cpu(),
            "modal_names": ["left_hand", "right_hand", "tongue", "clinical"],
        }
        return logits, attn_dict


class ClinicalAwareAdaptiveMultimodalTransformerModel(nn.Module):
    """
    CA-AMT v1 model.

    Image encoders:
      - shared hand ImageEncoder for left and right hands
      - independent tongue ImageEncoder
    Clinical encoder:
      - lightweight MLP encoder
    Fusion:
      - four-token cross-modal Transformer
      - patient-specific adaptive modality gating
    """

    def __init__(
        self,
        num_clinical_features: int,
        pretrained: bool = True,
        feature_dim: int = 256,
        clinical_hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        clinical_dropout: float = 0.2,
    ):
        super().__init__()
        self.num_clinical_features = int(num_clinical_features)
        self.feature_dim = int(feature_dim)

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
        self.clinical_encoder = ClinicalMLPEncoder(
            num_clinical_features=num_clinical_features,
            feature_dim=feature_dim,
            hidden_dim=clinical_hidden_dim,
            dropout=clinical_dropout,
        )
        self.fusion_transformer = ClinicalAwareAdaptiveFusionTransformer(
            input_dim=feature_dim,
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, left_x, right_x, tongue_x, clinical_x):
        left_feat = self.hand_encoder(left_x)
        right_feat = self.hand_encoder(right_x)
        tongue_feat = self.tongue_encoder(tongue_x)
        clinical_feat = self.clinical_encoder(clinical_x)

        logits, attn = self.fusion_transformer(
            left_feat,
            right_feat,
            tongue_feat,
            clinical_feat,
        )
        return logits, attn
