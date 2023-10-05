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
class MapInputSeparatebleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)
    
@FUSERS.register_module()
class ChannelAttentionLayer(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.fc = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels//8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channels//8, self.in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 入力テンソル x: [batch_size, channels, height, width]
        # 平均プーリング
        avg_pool = torch.mean(x, dim=[2, 3], keepdim=True) # [batch_size, channels, 1, 1]
        avg_pool = avg_pool.view(avg_pool.size(0), -1)     # [batch_size, channels]
        
        # チャネルアテンション
        y = self.fc(avg_pool)   # [batch_size, channels]
        y = y.view(y.size(0), -1, 1, 1)  # [batch_size, channels, 1, 1]

        # アテンションを適用
        attention_applied = x * y  # [batch_size, channels, height, width]

        return attention_applied

@FUSERS.register_module()
class CustomAttentionModel(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int):
        super(CustomAttentionModel, self).__init__()
        self.attention_layers = nn.ModuleList([
            ChannelAttentionLayer(ch) for ch in in_channels
        ])
        self.conv_fuser = ConvFuser(in_channels, out_channels)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # 各入力に対してアテンションを適用
        attended_inputs = [
            att_layer(inp) for att_layer, inp in zip(self.attention_layers, inputs)
        ]
        # アテンションを適用したテンソルを連結
        # concatenated = torch.cat(attended_inputs, dim=1)
        # 畳み込みレイヤーを通して出力を計算
        output = self.conv_fuser(attended_inputs)
        return output
