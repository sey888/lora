import math
import torch
import torch.nn as nn

class PolarAttention(nn.Module):
    """
    Simplified Polar Coordinate Attention module for ERP feature maps.
    Designed for Stage 2 training:
    1. Initialized with gamma=0 to ensure we start from the exact Stage 1 performance.
    2. Placed at decoder side to decouple from frozen encoder LoRA.
    """
    def __init__(self, channels: int):
        super(PolarAttention, self).__init__()

        # A simple spatial attention branch
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # Learnable scale for the residual (initialized to 0 for safe Stage 2 start)
        self.gamma = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_latitude_prior(self, H: int, device, dtype) -> torch.Tensor:
        row_idx = torch.arange(H, device=device, dtype=dtype)
        phi = math.pi * (row_idx + 0.5) / H
        # Inverse sine prior: focus on poles
        w = 1.0 - torch.sin(phi)
        # Normalize to mean 1
        w = w / (w.mean() + 1e-6)
        return w.view(1, 1, H, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        attn = self.spatial_attn(x)
        lat_prior = self._get_latitude_prior(H, x.device, x.dtype)
        attn = attn * lat_prior
        out = x * attn
        return x + self.gamma * out
