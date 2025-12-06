import torch
import torch.nn as nn
from torchvision import models


__all__ = ["EfficientNetBackbone", "CBAM", "OrangeNetV1"]


class EfficientNetBackbone(nn.Module):
    """EfficientNet-B0 backbone returning convolutional features only."""

    def __init__(self, backbone_name: str = "efficientnet_b0", pretrained: bool = True):
        super().__init__()
        if backbone_name != "efficientnet_b0":
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None

        base_model = models.efficientnet_b0(weights=weights)
        self.features = base_model.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=False)
        max_pool = torch.amax(x, dim=(2, 3), keepdim=False)

        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        weight = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * weight


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        weight = self.sigmoid(self.conv(concat))
        return x * weight


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out


class OrangeNetV1(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        dropout_rate: float = 0.4,
        cbam_reduction_ratio: int = 16,
        backbone_name: str = "efficientnet_b0",
        use_mid_fc: bool = False,
    ):
        super().__init__()
        self.normalize = NormalizeLayer()
        self.backbone = EfficientNetBackbone(backbone_name=backbone_name, pretrained=True)
        sample_input = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            feature_dim = self.backbone(sample_input).shape[1]
        self.cbam = CBAM(feature_dim, reduction=cbam_reduction_ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.use_mid_fc = use_mid_fc

        if use_mid_fc:
            hidden_dim = 512
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        features = self.backbone(x)
        features = self.cbam(features)
        pooled = self.pool(features).flatten(1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class NormalizeLayer(nn.Module):
    """Applies ImageNet normalization inside the model for convenience."""

    def __init__(self):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std
