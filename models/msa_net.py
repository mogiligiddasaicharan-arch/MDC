"""
MSA-Net: Multi-Scale Attention Network for Manufacturing Defect Classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_channels = out_channels // 3
        remainder = out_channels - branch_channels * 3

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.branch_dilated = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels + remainder, kernel_size=3,
                       padding=2, dilation=2),
            nn.BatchNorm2d(branch_channels + remainder),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b1 = self.branch3x3(x)
        b2 = self.branch5x5(x)
        b3 = self.branch_dilated(x)
        out = torch.cat([b1, b2, b3], dim=1)
        return self.fuse(out)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)
        mask = self.sigmoid(self.conv(pooled))
        return x * mask


class MSABlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.multi_scale = MultiScaleConv(in_channels, out_channels)
        self.channel_attn = ChannelAttention(out_channels)
        self.spatial_attn = SpatialAttention()

        self.residual_proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

        self.downsample = (
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3,
                           stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ) if downsample else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual_proj(x)
        out = self.multi_scale(x)
        out = self.channel_attn(out)
        out = self.spatial_attn(out)
        out = out + residual
        out = self.downsample(out)
        return out


class MSANet(nn.Module):
    def __init__(self, num_classes, base_channels=32, in_channels=3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2,
                       padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        c1, c2, c3, c4 = (base_channels, base_channels * 2,
                           base_channels * 4, base_channels * 8)

        self.stage1 = MSABlock(base_channels, c1, downsample=True)
        self.stage2 = MSABlock(c1, c2, downsample=True)
        self.stage3 = MSABlock(c2, c3, downsample=True)
        self.stage4 = MSABlock(c3, c4, downsample=False)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(c4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        features = self.stage4(x)
        pooled = self.gap(features).flatten(1)
        logits = self.classifier(pooled)

        if return_features:
            return logits, features
        return logits


if __name__ == "__main__":
    model = MSANet(num_classes=6)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")