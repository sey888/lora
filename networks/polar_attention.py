import math
import torch
import torch.nn as nn

class PolarAttention(nn.Module):
    """
    Refined Polar Coordinate Attention for stable Joint Training.
    Key design: 
    1. Residual gating with gamma initialized to 0.0 to ensure 
       it starts from the exact LoRA-only baseline.
    2. Decoupled from the encoder to prevent gradient interference.
    """
    def __init__(self, channels: int):
        super(PolarAttention, self).__init__()

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # Gamma is the gating factor, initialized to 0.0.
        # This ensures that at the beginning of joint training, 
        # the model is mathematically identical to the LoRA-only baseline.
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
        # Latitude-aware weight: focus more on the polar regions where ERP distortion is high
        w = 1.0 - torch.sin(phi)
        w = w / (w.mean() + 1e-6)
        return w.view(1, 1, H, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Spatial attention map
        attn = self.spatial_attn(x)
        # Apply latitude geometric prior
        lat_prior = self._get_latitude_prior(H, x.device, x.dtype)
        attn = attn * lat_prior
        
        # Residual gating: x_out = x + gamma * (x * attention)
        # When gamma=0, x_out = x (the stable LoRA feature)
        return x + self.gamma * (x * attn)
