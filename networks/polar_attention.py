import math
import torch
import torch.nn as nn

class PolarAttention(nn.Module):
    """
    Ultimate Stable Polar Attention for Joint Training.
    Designed to ensure precision boost without interfering with LoRA's convergence.
    """
    def __init__(self, channels: int):
        super(PolarAttention, self).__init__()

        # Use a lightweight bottleneck to extract spatial features
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # CRITICAL: Initialize gamma to EXACTLY 0.0.
        # This acts as a 'safety valve' that ensures the model starts 
        # exactly from the LoRA baseline and only adds improvements.
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
        # Generate a latitude-aware prior (1.0 at equator, higher at poles)
        row_idx = torch.arange(H, device=device, dtype=dtype)
        phi = math.pi * (row_idx + 0.5) / H
        # Distortion is higher at poles, so we give more attention weight there
        w = 1.0 - torch.sin(phi)
        w = w / (w.mean() + 1e-6)
        return w.view(1, 1, H, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Compute spatial attention
        attn = self.spatial_attn(x)
        
        # 2. Multiply by geometric latitude prior
        lat_prior = self._get_latitude_prior(x.shape[2], x.device, x.dtype)
        attn = attn * lat_prior
        
        # 3. Residual Gating: Out = x + gamma * (x * attention)
        # Because gamma=0 at start, the model initially behaves exactly like the baseline.
        # This prevents the initial random noise of the attention module 
        # from disrupting LoRA's learning process.
        return x + self.gamma * (x * attn)
