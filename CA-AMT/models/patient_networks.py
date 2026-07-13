import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ==========================================
# 0. 共享基础组件 (严格控制变量：所有模型共用同一套特征提取器)
# ==========================================
def build_resnet18(pretrained=True):
    try:
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
                return models.resnet18(weights=weights)
            except AttributeError:
                return models.resnet18(pretrained=True)
        else:
            try:
                return models.resnet18(weights=None)
            except TypeError:
                return models.resnet18(pretrained=False)
    except Exception as e:
        raise RuntimeError(f"构建 ResNet18 失败：{e}")


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid_channels = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.gap(x).view(b, c)
        y = self.attention(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ImageEncoder(nn.Module):
    """
    统一的图像编码器：ResNet18 + SE + 特征投影
    """

    def __init__(self, input_channels=3, pretrained=True, feature_dim=256):
        super().__init__()
        resnet = build_resnet18(pretrained=pretrained)

        if input_channels != 3:
            old_weight = resnet.conv1.weight.data.clone()
            resnet.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            nn.init.kaiming_normal_(resnet.conv1.weight, mode="fan_out", nonlinearity="relu")
            if input_channels == 1:
                with torch.no_grad():
                    resnet.conv1.weight[:] = old_weight.mean(dim=1, keepdim=True)

        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.se_block = SEBlock(512)

        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.feature_dim = feature_dim

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.se_block(x)
        x = self.projection(x)
        return x


# ==========================================
# 1. 消融实验一：仅手部模态模型 (Hand Only)
# ==========================================
class HandOnlyNet(nn.Module):
    def __init__(self, pretrained=True, feature_dim=256, num_classes=2):
        super().__init__()
        self.hand_encoder = ImageEncoder(pretrained=pretrained, feature_dim=feature_dim)
        # 左右手拼接：feature_dim * 2
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim * 2),
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, left_x, right_x, tongue_x):
        # 物理隔离：完全不处理舌部图像
        left_feat = self.hand_encoder(left_x)
        right_feat = self.hand_encoder(right_x)
        fused = torch.cat([left_feat, right_feat], dim=1)
        logits = self.classifier(fused)
        return logits, None  # 统一返回格式


# ==========================================
# 2. 消融实验二：仅舌部模态模型 (Tongue Only)
# ==========================================
class TongueOnlyNet(nn.Module):
    def __init__(self, pretrained=True, feature_dim=256, num_classes=2):
        super().__init__()
        self.tongue_encoder = ImageEncoder(pretrained=pretrained, feature_dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, left_x, right_x, tongue_x):
        # 物理隔离：完全不处理手部图像
        tongue_feat = self.tongue_encoder(tongue_x)
        logits = self.classifier(tongue_feat)
        return logits, None


# ==========================================
# 3. 对比实验：简单特征拼接融合基线 (Baseline Fusion)
# ==========================================
class FusionPatientNet(nn.Module):
    def __init__(self, pretrained=True, feature_dim=256, num_classes=2):
        super().__init__()
        self.hand_encoder = ImageEncoder(pretrained=pretrained, feature_dim=feature_dim)
        self.tongue_encoder = ImageEncoder(pretrained=pretrained, feature_dim=feature_dim)

        # 左右手 + 舌头：feature_dim * 3
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim * 3),
            nn.Linear(feature_dim * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, left_x, right_x, tongue_x):
        left_feat = self.hand_encoder(left_x)
        right_feat = self.hand_encoder(right_x)
        tongue_feat = self.tongue_encoder(tongue_x)

        fused = torch.cat([left_feat, right_feat, tongue_feat], dim=1)
        logits = self.classifier(fused)
        return logits, None


# ==========================================
# 4. 融合模块消融：仅保留 Transformer 跨模态交互
#    Transformer Fusion without adaptive modality weighting
# ==========================================
class TransformerFusionModule(nn.Module):
    """
    纯 Transformer 跨模态交互模块：
    - 输入：左手、右手、舌象三个 feature token
    - 保留：TransformerEncoder 跨模态交互 + 位置/模态嵌入
    - 去掉：modal_weights、自适应模态加权、attention_pool、modal_dropout
    - 池化：mean pooling，避免引入额外自适应权重
    """

    def __init__(
        self,
        input_dim=256,
        feature_dim=256,
        num_heads=8,
        num_layers=2,
        num_classes=2,
        dropout=0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        self.left_projection = nn.Linear(input_dim, feature_dim)
        self.right_projection = nn.Linear(input_dim, feature_dim)
        self.tongue_projection = nn.Linear(input_dim, feature_dim)

        # 3个 token 分别对应：左手、右手、舌象。
        # 这里仅用于区分模态位置，不代表自适应模态权重。
        self.pos_embed = nn.Parameter(torch.zeros(1, 3, feature_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

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

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, left_feat, right_feat, tongue_feat):
        left_proj = self.left_projection(left_feat)
        right_proj = self.right_projection(right_feat)
        tongue_proj = self.tongue_projection(tongue_feat)

        sequence = torch.stack(
            [left_proj, right_proj, tongue_proj],
            dim=1,
        )  # [B, 3, D]
        sequence = sequence + self.pos_embed

        transformer_output = self.transformer_encoder(sequence)  # [B, 3, D]

        # 关键：使用 mean pooling，而不是 attention pooling。
        # 这样该模型只验证 Transformer 跨模态交互本身，不引入额外自适应加权。
        context_vector = transformer_output.mean(dim=1)  # [B, D]
        logits = self.classifier(context_vector)

        return logits, None


class TransformerFusionPatientNet(nn.Module):
    """
    融合模块消融模型：Transformer fusion without adaptive weighting。

    与主模型保持一致：
    - 输入：左手、右手、舌象
    - 编码器：ImageEncoder，即 ResNet18 + SE + projection
    - Transformer 参数接口：feature_dim、num_heads、num_layers、dropout

    与主模型不同：
    - 不使用 learnable modal_weights
    - 不使用 attention pooling
    - 不使用 modal dropout
    - Transformer 输出后采用 mean pooling
    """

    def __init__(
        self,
        pretrained=True,
        feature_dim=256,
        num_heads=8,
        num_layers=2,
        num_classes=2,
        dropout=0.0,
    ):
        super().__init__()
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
        self.fusion_transformer = TransformerFusionModule(
            input_dim=feature_dim,
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, left_x, right_x, tongue_x):
        left_feat = self.hand_encoder(left_x)
        right_feat = self.hand_encoder(right_x)
        tongue_feat = self.tongue_encoder(tongue_x)

        logits, attn = self.fusion_transformer(left_feat, right_feat, tongue_feat)
        return logits, attn


# ==========================================
# 5. 核心创新：自适应 Transformer 融合主模型 (Main Model)
# ==========================================
class AdaptiveFusionTransformer(nn.Module):
    # (完全保留你的原版代码)
    def __init__(self, input_dim=256, feature_dim=256, num_heads=8, num_layers=2, num_classes=2, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.left_projection = nn.Linear(input_dim, feature_dim)
        self.right_projection = nn.Linear(input_dim, feature_dim)
        self.tongue_projection = nn.Linear(input_dim, feature_dim)
        self.modal_weights = nn.Parameter(torch.zeros(3))
        self.pos_embed = nn.Parameter(torch.zeros(1, 3, feature_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=num_heads, dim_feedforward=feature_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_pool = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.Tanh(), nn.Linear(64, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(128, num_classes),
        )

    def forward(self, left_feat, right_feat, tongue_feat):
        left_proj = self.left_projection(left_feat)
        right_proj = self.right_projection(right_feat)
        tongue_proj = self.tongue_projection(tongue_feat)

        gates = 1.0 + 0.5 * torch.tanh(self.modal_weights)

        left_weighted = left_proj * gates[0]
        right_weighted = right_proj * gates[1]
        tongue_weighted = tongue_proj * gates[2]

        weights = F.softmax(self.modal_weights, dim=0)

        sequence = torch.stack([left_weighted, right_weighted, tongue_weighted], dim=1)
        sequence = sequence + self.pos_embed
        transformer_output = self.transformer_encoder(sequence)

        attn_weights = F.softmax(self.attention_pool(transformer_output), dim=1)
        context_vector = torch.sum(transformer_output * attn_weights, dim=1)
        logits = self.classifier(context_vector)

        attn_dict = {
            "modal_weights": weights.detach().cpu(),
            "token_attention": attn_weights.squeeze(-1).detach().cpu(),
        }
        return logits, attn_dict


class HandTongueAdaptiveTransformerModel(nn.Module):
    # (完全保留你的原版架构，继承了统一的特征提取器)
    def __init__(self, pretrained=True, feature_dim=256, num_heads=8, num_layers=2, num_classes=2, dropout=0.1):
        super().__init__()
        self.hand_encoder = ImageEncoder(input_channels=3, pretrained=pretrained, feature_dim=feature_dim)
        self.tongue_encoder = ImageEncoder(input_channels=3, pretrained=pretrained, feature_dim=feature_dim)
        self.fusion_transformer = AdaptiveFusionTransformer(
            input_dim=feature_dim, feature_dim=feature_dim, num_heads=num_heads,
            num_layers=num_layers, num_classes=num_classes, dropout=dropout,
        )
        self.modal_drop_prob = 0.0

    def forward(self, left_x, right_x, tongue_x):
        left_feat = self.hand_encoder(left_x)
        right_feat = self.hand_encoder(right_x)
        tongue_feat = self.tongue_encoder(tongue_x)

        if self.training:
            if torch.rand(1).item() < self.modal_drop_prob:
                left_feat = torch.zeros_like(left_feat)
            if torch.rand(1).item() < self.modal_drop_prob:
                right_feat = torch.zeros_like(right_feat)
            if torch.rand(1).item() < self.modal_drop_prob:
                tongue_feat = torch.zeros_like(tongue_feat)

        logits, attn = self.fusion_transformer(left_feat, right_feat, tongue_feat)
        return logits, attn