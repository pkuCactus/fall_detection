"""
copyright (c) 2024 pkuCactus. All rights reserved.
Use of this source is goverend by the Apache License that can be found in the LICENSE file
"""

import math
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import autopad


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

    Supports two usage modes:
      1) External output convolutions (e.g. custom_yolo.yaml) where input channels
         already equal na*(nc+5) per layer.
      2) Internal output convolutions (e.g. ori_detector_anchor.yaml) where this
         module creates 1x1 convs to project features to detection space.

    Args:
        nc: number of classes
        anchors: list of anchor boxes per detection layer
        ch: input channels tuple for each detection layer (passed by parse_model)
        inplace: inplace operation for inference speed
    """
    dynamic = False
    export = False
    shape = None

    def __init__(self, nc=80, anchors=None, version="v3", *args, inplace=True, **kwargs):
        super().__init__()

        # Extract ch from kwargs (passed by parse_model as last element of args)
        ch = kwargs.get('ch', [])
        if not ch and len(args) >= 1:
            last_arg = args[-1]
            if isinstance(last_arg, (list, tuple)) and all(isinstance(x, int) for x in last_arg):
                ch = last_arg

        if anchors is None:
            anchors = [[10, 13, 16, 30, 33, 23],
                       [30, 61, 62, 45, 59, 119],
                       [116, 90, 156, 198, 373, 326]]

        self.nl = len(anchors)
        self.nc = nc
        self.na = [len(a) // 2 for a in anchors]  # anchors per layer (may differ)
        self.out_chs = [na * (nc + 5) for na in self.na]
        self.inplace = inplace

        # Decide whether input already has detection channels (mode 1) or we need internal convs (mode 2)
        if ch and len(ch) == self.nl and all(ch[i] == self.out_chs[i] for i in range(self.nl)):
            self.m = None  # external convs provided
        else:
            ch = ch if ch and len(ch) == self.nl else self.out_chs
            self.m = nn.ModuleList(nn.Conv2d(ch[i], self.out_chs[i], 1) for i in range(self.nl))
            self._initialize_biases()

        # Register anchors as buffers
        anchors_t = [torch.tensor(a).float().view(1, -1, 2) for a in anchors]
        max_na = max(self.na)
        anchors_padded = torch.zeros(self.nl, max_na, 2)
        for i, a in enumerate(anchors_t):
            anchors_padded[i, :self.na[i]] = a[0]
        self.register_buffer('anchors', anchors_padded)  # shape(nl, max_na, 2)
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]
        self.grid = [torch.empty(0) for _ in range(self.nl)]
        self._post = {
            "yolo": self._yolo_post,
            "v3": self._yolov3_post,
            "v5": self._yolov5_post,
        }.get(version, self._yolo_post)

    def _initialize_biases(self):
        """Initialize detection head biases."""
        if self.m is None:
            return
        for mi, s in zip(self.m, [8, 16, 32]):
            b = mi.bias.view(-1)  # (na*(nc+5),)
            b = b.view(self.na[0] if len(set(self.na)) == 1 else mi.out_channels // (self.nc + 5), -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:5 + self.nc] += math.log(0.6 / (self.nc - 0.999999)) if self.nc == 1 else math.log(0.6 / (self.nc - 0.999999))
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
            if self.m is not None:
                x[i] = self.m[i](x[i])
            if self.training or self.export:
                continue
            # inference decoding
            bs, _, ny, nx = x[i].shape
            y = x[i].view(bs, self.na[i], self.nc + 5, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
            y = self._post(y, i)  # decode predictions in-place
            z.append(y)

        return x if (self.training or self.export) else (torch.cat(z, 1), x)

    def _yolo_post(self, x, i: int = 0):
        y = x.sigmoid()
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
        y[..., 2:4] = y[..., 2:4] ** 2 * 2 * self.anchor_grid[i] # wh
        return y.view(x.shape[0], -1, self.nc + 5)

    def _yolov3_post(self, x, i: int = 0, class_softmax: bool = False):
        x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid[i]) * self.stride[i]  # xy
        x[..., 2:4] = x[..., 2:4].exp() * self.anchor_grid[i]  # wh
        x[..., 4] = x[..., 4].sigmoid()  # conf
        if class_softmax:
            x[..., 5:] = torch.softmax(x[..., 5:], dim=-1)  # class scores
        else:
            x[..., 5:] = x[..., 5:].sigmoid()  # class scores
        return x.view(x.shape[0], -1, self.nc + 5)

    def _yolov5_post(self, x, i: int = 0, with_nmsnet: bool = False, nms_ouptut: torch.Tensor = None):
        out_dim = self.nc + 9
        y[..., 0:2] = (y[..., 0:2].sigmoid() * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
        y[..., 2:4] = y[..., 2:4].exp() * self.anchor_grid[i]  # wh
        y[..., 4:6] = (y[..., 4:6].sigmoid() * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
        y[..., 6:8] = y[..., 6:8].exp() * self.anchor_grid[i]  # wh
        y[..., 8:] = y[..., 8:].sigmoid()  # conf and class scores
        if with_nmsnet and nms_ouptut is not None:
            # Apply NMSNet post-processing to conf and class scores
            y = torch.cat((y, nms_ouptut.sigmoid().unsqueeze(-1)), dim=-1)
            out_dim += 1
        return y.view(x.shape[0], -1, out_dim)

    def _make_grid(self, nx=20, ny=20, i=0):
        """Generate detection grid and anchor grid."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na[i], ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((xv, yv), 2).expand(shape)
        anchor_grid = (self.anchors[i, :self.na[i]]).view((1, self.na[i], 1, 1, 2)).expand(shape)
        return grid, anchor_grid

    @property
    def stride(self):
        """Calculate stride from model if available, otherwise use default."""
        if hasattr(self, '_stride'):
            return self._stride
        return [8, 16, 32]

    @stride.setter
    def stride(self, value):
        self._stride = value
