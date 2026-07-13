# -*- coding: utf-8 -*-
"""
Image + Clinical concat fusion control model.

Purpose:
    This file defines a clean ablation/control model for SCI comparison:
    Image + Clinical concat fusion.

Important:
    - This is NOT the CA-AMT final model.
    - It does NOT use Transformer.
    - It does NOT use adaptive gating.
    - It does NOT output patient-specific modality weights.
    - Its purpose is to test whether CA-AMT is better than simple feature concatenation.

Place this file at:
    D:\tcm_AI\Patient-level image data\scr\models\patient_networks_concat_clinical.py

It reuses the existing ImageEncoder from:
    scr.models.patient_networks
"""

import torch
import torch.nn as nn

from scr.models.patient_networks import ImageEncoder


class ClinicalMLPEncoder(nn.Module):
    """
    Lightweight clinical encoder for tabular clinical variables.

    Input:
        clinical_x: [B, num_clinical_features]

    Output:
        clinical_feat: [B, feature_dim]
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
        return self.encoder(clinical_x.float())


class ImageClinicalConcatNet(nn.Module):
    """
    Image + Clinical concat fusion control model.

    Architecture:
        left hand image  -> ImageEncoder -> left_feat
        right hand image -> ImageEncoder -> right_feat
        tongue image     -> ImageEncoder -> tongue_feat
        clinical vector  -> ClinicalMLPEncoder -> clinical_feat

        concat([left_feat, right_feat, tongue_feat, clinical_feat])
        -> MLP classifier
        -> CAD probability

    This model intentionally removes:
        - cross-modal Transformer
        - adaptive modality gating
        - token attention pooling
        - patient-specific modality weights

    Therefore, it is a clean control model for comparing against CA-AMT.
    """

    def __init__(
        self,
        num_clinical_features: int,
        pretrained: bool = True,
        feature_dim: int = 256,
        clinical_hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.4,
        clinical_dropout: float = 0.2,
    ):
        super().__init__()
        self.num_clinical_features = int(num_clinical_features)
        self.feature_dim = int(feature_dim)

        # Keep the same image encoders as the image-only and CA-AMT models
        # so the comparison is methodologically fair.
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

        concat_dim = feature_dim * 4
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(concat_dim),
            nn.Linear(concat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.75),
            nn.Linear(64, num_classes),
        )

    def forward(self, left_x, right_x, tongue_x, clinical_x):
        left_feat = self.hand_encoder(left_x)
        right_feat = self.hand_encoder(right_x)
        tongue_feat = self.tongue_encoder(tongue_x)
        clinical_feat = self.clinical_encoder(clinical_x)

        fused = torch.cat(
            [left_feat, right_feat, tongue_feat, clinical_feat],
            dim=1,
        )
        logits = self.classifier(fused)

        # Keep the same return style as other models.
        # No attention or modality weights are returned for this control model.
        return logits, None
