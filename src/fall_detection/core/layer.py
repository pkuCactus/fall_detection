"""
copyright (c) 2024 hjz. All rights reserved.
Use of this source is goverend by the Apache License that can be found in the LICENSE file
"""

import math
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import autopad, Conv


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


class AnchorDetect(nn.Module):
    """YOLO anchor-based detection head.

    Args:
        nc: number of classes
        anchors: list of anchor boxes per detection layer, e.g., [[10,13,16,30,33,23], [30,61,62,45,59,119], [116,90,156,198,373,326]]
        ch: input channels tuple for each detection layer
        inplace: inplace operation for inference speed
    """
    dynamic = False
    export = False
    shape = None

    def __init__(self, nc=80, anchors=None, *args, inplace=True, **kwargs):
        """Initialize AnchorDetect head.

        Args:
            nc: number of classes
            anchors: list of anchor definitions
            *args: additional args (reg_max, end2end from parse_model - ignored)
            inplace: whether to use inplace operations
            **kwargs: additional kwargs (ch from parse_model)
        """
        super().__init__()

        # Extract ch from kwargs (passed by parse_model as last element of args)
        ch = kwargs.get('ch', [])
        if not ch and len(args) >= 1:
            # ch might be passed as the last positional arg from parse_model
            last_arg = args[-1]
            if isinstance(last_arg, (list, tuple)) and all(isinstance(x, int) for x in last_arg):
                ch = last_arg

        # Determine number of layers from anchors or ch
        if anchors is not None:
            self.nl = len(anchors)
        elif ch:
            self.nl = len(ch)
        else:
            self.nl = 3  # default

        self.nc = nc  # number of classes
        self.na = len(anchors[0]) // 2 if anchors else 3  # number of anchors per layer
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

        # Build anchors
        if anchors is None:
            # Default YOLOv5 anchors
            anchors = [[10, 13, 16, 30, 33, 23],
                       [30, 61, 62, 45, 59, 119],
                       [116, 90, 156, 198, 373, 326]]

        # Register anchors as buffers
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid (not buffer, dynamic)

        self.m = nn.ModuleList(nn.Conv2d(x, self.na * (self.nc + 5), 1) for x in ch)  # output conv

        # Initialize
        self._initialize_biases()

    def _initialize_biases(self):
        """Initialize detection head biases."""
        for mi, s in zip(self.m, [8, 16, 32]):  # stride assumptions
            b = mi.bias.view(self.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + self.nc] += math.log(0.6 / (self.nc - 0.999999)) if self.nc == 1 else torch.log(
                torch.tensor([0.6 / (self.nc - 0.999999)] * self.nc))  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        """Forward pass through anchor-based detection head.

        Args:
            x: list of feature maps from backbone/neck

        Returns:
            During training: list of feature maps
            During inference: concatenated predictions or (predictions, features)
        """
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.nc + 5, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.nc + 5))

        return x if self.training else (torch.cat(z, 1), x) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        """Generate detection grid and anchor grid."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

    @property
    def stride(self):
        """Calculate stride from model if available, otherwise use default."""
        # Try to get stride from parent model
        if hasattr(self, '_stride'):
            return self._stride
        # Default strides for P3, P4, P5
        return [8, 16, 32]

    @stride.setter
    def stride(self, value):
        self._stride = value
