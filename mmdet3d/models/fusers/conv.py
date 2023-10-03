from typing import Union, List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["ConvFuser"]


@FUSERS.register_module()
class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))


@FUSERS.register_module()
class MapInputConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(self.in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(self.out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)

@FUSERS.register_module()
class SEBlock(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16) -> None:
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        self.reduced_channels = max(1, in_channels // reduction_ratio)
        
        # Squeeze operation.
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation operation.
        self.fc1 = nn.Linear(in_channels, self.reduced_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.reduced_channels, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        out = self.global_avg_pool(x).view(b, c)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).view(b, c, 1, 1)
        return x * out
