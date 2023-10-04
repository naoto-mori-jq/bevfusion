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

@FUSERS.register_module()
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels: int) -> None:
        """
        Initialize 2D PositionalEncoding.
        
        Parameters:
        - channels (int): The number of channels in the input feature map.
        """
        super(PositionalEncoding2D, self).__init__()
        self.channels = channels
        self.dim = channels // 2
        self.encoding = None

    def create_positional_encoding(self, height: int, width: int) -> torch.Tensor:
        """
        Create 2D positional encoding.
        
        Parameters:
        - height (int): The height of the input feature map.
        - width (int): The width of the input feature map.
        
        Returns:
        - torch.Tensor: The positional encoding.
        """
        y_pos = torch.arange(height).unsqueeze(1).repeat(1, width).float()
        x_pos = torch.arange(width).unsqueeze(0).repeat(height, 1).float()
        
        div_term = torch.exp(torch.arange(0, self.dim, 2).float() * (-math.log(10000.0) / self.dim))
        
        pos_enc = torch.zeros(height, width, self.channels)
        pos_enc[:, :, 0:self.dim:2] = torch.sin(y_pos * div_term) + torch.sin(x_pos * div_term)
        pos_enc[:, :, 1:self.dim:2] = torch.cos(y_pos * div_term) + torch.cos(x_pos * div_term)

        return pos_enc.permute(2, 0, 1).unsqueeze(0)  # Shape: [1, C, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method for PositionalEncoding2D.
        
        Parameters:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Output tensor with positional encoding added.
        """
        b, _, h, w = x.size()
        if self.encoding is None or self.encoding.size(2) != h or self.encoding.size(3) != w:
            self.encoding = self.create_positional_encoding(h, w).to(x.device)
        
        return x + self.encoding
