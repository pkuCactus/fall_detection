"""
copyright (c) 2024 hjz. All rights reserved.
Use of this source is goverend by the Apache License that can be found in the LICENSE file
"""

from ultralytics.nn.modules.conv import autopad
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def __repr__(self):
        s = 'ConvBNReLU(in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={})'
        return s.format(self.layers[0].in_channels, self.layers[0].out_channels, self.layers[0].kernel_size,
                        self.layers[0].stride, self.layers[0].padding)

    def forward(self, x):
        return self.layers(x)


class ConvUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=3):
        super().__init__()
        self.layer = nn.Sequential(
             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=autopad(kernel_size), bias=False),
             nn.Upsample(scale_factor=scale_factor, mode='nearest'),
             nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)
