"""
copyright (c) 2024 pkuCactus. All rights reserved.
Use of this source is goverend by the Apache License that can be found in the LICENSE file
"""

import ast
import contextlib
from copy import deepcopy

import torch
import torch.nn as nn
from ultralytics import YOLO, YOLOWorld
from ultralytics.nn.modules import (
    Conv, Classify, Concat, Detect, Segment, Pose,
    ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP,
    SPPF, C2fPSA, C2PSA, DWConv, Focus, BottleneckCSP, C1, C2,
    C2f, C3k2, RepNCSPELAN4, ELAN1, ADown, AConv, SPPELAN, C2fAttn,
    C3, C3TR, C3Ghost, DWConvTranspose2d, C3x, RepC3, PSA, SCDown, C2fCIB, A2C2f,
    AIFI, HGStem, HGBlock, ResNetLayer, ImagePoolingAttn, RTDETRDecoder, CBLinear, CBFuse, TorchVision, Index,
    WorldDetect, YOLOEDetect, Segment26, YOLOESegment, YOLOESegment26, Pose26, OBB, OBB26, v10Detect
)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.ops import make_divisible

from fall_detection.core.layer import ConvUpsample, AnchorDetect

DEFAULT_IMG_SIZE = 640


def custom_parse_model(d, ch, verbose=True):
    """Parse a YOLO model.yaml dictionary into a PyTorch model.

    Args:
        d (dict): Model dictionary.
        ch (int): Input channels.
        verbose (bool): Whether to print model details.

    Returns:
        (torch.nn.Sequential): PyTorch model.
        (list): Sorted list of layer indices whose outputs need to be saved.
    """
    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
    reg_max = d.get("reg_max", 16)
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")
    version = d.get("version", "v3")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = torch.nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvUpsample,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            torch.nn.Conv2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
        }
    )
    repeat_modules = frozenset(  # modules with 'repeat' arguments
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fPSA,
            C2fCIB,
            C2PSA,
            A2C2f,
        }
    )
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else d[a] if a in d else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc and m is not nn.Conv2d:  # if c2 != nc (e.g., Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:  # set 1) embed channels and 2) num heads
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":  # for L/X sizes
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in frozenset(
            {
                Detect,
                WorldDetect,
                YOLOEDetect,
                Segment,
                Segment26,
                YOLOESegment,
                YOLOESegment26,
                Pose,
                Pose26,
                OBB,
                OBB26,
            }
        ):
            args.extend([reg_max, end2end, [ch[x] for x in f]])
            if m is Segment or m is YOLOESegment or m is Segment26 or m is YOLOESegment26:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, AnchorDetect, YOLOEDetect, Segment, Segment26, YOLOESegment, YOLOESegment26, Pose, Pose26, OBB, OBB26}:
                m.legacy = legacy
        elif m is AnchorDetect:
            args.extend([version, [ch[x] for x in f]])
        elif m is v10Detect:
            args.append([ch[x] for x in f])
        elif m is ImagePoolingAttn:
            args.insert(1, [ch[x] for x in f])  # channels as second arg
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)


class CustomDetectionModel(DetectionModel):
    """Custom detection model with anchor-based detection head support.

    This class extends the BaseModel to support:
    - Custom layers: ConvBNReLU, ConvUpsample
    - Anchor-based detection: AnchorDetect
    """

    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):
        """Initialize CustomDetectionModel.

        Args:
            cfg: model configuration file or dict
            ch: number of input channels
            nc: number of classes (overrides cfg if provided)
            verbose: print model info
        """
        nn.Module.__init__(self)  # Initialize Module directly to bypass DetectionModel default build
        self.yaml = cfg if isinstance(cfg, dict) else self._load_yaml(cfg)

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)

        # Override nc if provided
        if nc and nc != self.yaml.get('nc'):
            LOGGER.info(f"Overriding model.yaml nc={self.yaml.get('nc')} with nc={nc}")
            self.yaml['nc'] = nc

        # Build model using custom parse_model
        self.model, self.save = custom_parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose and RANK == -1)

        # Set stride and other properties
        m = self.model[-1]  # last layer (Detect)
        if isinstance(m, AnchorDetect):
            # infernce to get stride
            self.train(True)  # ensure model is in training mode for correct output shape
            outputs = self.forward(torch.zeros(1, ch, DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE))
            m.stride = torch.tensor([DEFAULT_IMG_SIZE // x.shape[2] for x in outputs])  # P5, P4, P3 strides
            LOGGER.info(f"AnchorDetect strides: {m.stride}")
            self.stride = m.stride
            self._initialize_biases()  # Initialize biases for AnchorDetect

        # Build strides
        m = self.model[-1]
        if isinstance(m, (AnchorDetect,)):
            self.stride = m.stride

        # Initialize weights
        self._initialize_weights(verbose)

    def _load_yaml(self, path):
        """Load yaml file."""
        from ultralytics.utils.files import check_yaml
        from pathlib import Path
        import yaml as pyyaml

        path = check_yaml(path)
        with open(path, 'r') as f:
            d = pyyaml.safe_load(f)
        d['yaml_file'] = str(path)
        return d

    def _initialize_weights(self, verbose=True):
        """Initialize model weights."""
        for m in self.model.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True

        # Initialize detection head biases
        m = self.model[-1]
        if isinstance(m, AnchorDetect) and getattr(m, 'm', None) is not None:
            m._initialize_biases()

    def _initialize_biases(self):
        """Initialize detection head biases for AnchorDetect."""
        m = self.model[-1]
        if isinstance(m, AnchorDetect) and getattr(m, 'm', None) is not None:
            for mi, s in zip(m.m, [32, 16, 8]):
                b = mi.bias.view(-1)
                na = mi.out_channels // (m.nc + 5)
                b = b.view(na, -1)
                b.data[:, 4] += torch.log(torch.tensor(8 / (640 / s) ** 2))
                b.data[:, 5:5 + m.nc] += torch.log(torch.tensor(0.6 / (m.nc - 0.999999)))
                mi.bias = nn.Parameter(b.view(-1), requires_grad=True)

    def init_criterion(self):
        """Initialize loss criterion for training."""
        return AnchorDetectionLoss(self)

    def forward(self, x, *args, **kwargs):
        """Forward pass."""
        y = []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)  # run
            y.append(x if m.i in self.save else None)

        return x


class AnchorDetectionLoss(nn.Module):
    """Anchor-based detection loss for CustomDetectionModel."""

    def __init__(self, model):
        super().__init__()
        self.device = next(model.parameters()).device
        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=self.device))
        self.bce_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=self.device))
        from ultralytics.utils.metrics import bbox_iou
        self.bbox_iou = bbox_iou

        m = model.model[-1]
        self.na = m.na  # list of anchors per layer
        self.nc = m.nc
        self.nl = m.nl
        self.stride = m.stride if hasattr(m.stride, '__iter__') else [m.stride] * self.nl
        self.anchors = m.anchors  # (nl, max_na, 2)
        self.gr = 1.0
        self.box_gain = 0.05
        self.cls_gain = 0.5
        self.obj_gain = 1.0

    def __call__(self, preds, batch):
        """Compute detection loss for anchor-based head.

        Args:
            preds: list of (bs, na, ny, nx, nc+5) for each layer
            batch: dict with 'batch_idx', 'cls', 'bboxes'
        """
        device = preds[0].device
        l_cls = torch.zeros(1, device=device)
        l_box = torch.zeros(1, device=device)
        l_obj = torch.zeros(1, device=device)

        targets = self._build_targets(preds, batch)
        if targets is None:
            for i, pi in enumerate(preds):
                tobj = torch.zeros_like(pi[..., 4])
                l_obj += self.bce_obj(pi[..., 4], tobj).mean()
            return l_box, l_cls, l_box + l_cls + l_obj

        for i, pi in enumerate(preds):
            b, a, gj, gi = targets[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 4])
            n = b.shape[0]
            if n:
                ps = pi[b, a, gj, gi]  # (n, nc+5)
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * targets['anchors'][i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = self.bbox_iou(pbox, targets['tbox'][i], CIoU=True).squeeze()
                l_box += (1.0 - iou).mean()
                tobj[b, a, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)
                if self.nc > 1:
                    t = torch.zeros_like(ps[:, 5:])
                    t[range(n), targets['tcls'][i].long()] = 1
                    l_cls += self.bce_cls(ps[:, 5:], t).mean()
            l_obj += self.bce_obj(pi[..., 4], tobj).mean()

        bs = preds[0].shape[0]
        l_box *= self.box_gain * bs
        l_obj *= self.obj_gain * bs
        l_cls *= self.cls_gain * bs
        return l_box, l_cls, l_box + l_obj + l_cls

    def _build_targets(self, preds, batch):
        """Build targets for anchor-based loss."""
        batch_idx = batch.get('batch_idx')
        if batch_idx is None or len(batch_idx) == 0:
            return None

        targets = torch.cat((batch_idx.unsqueeze(1), batch['cls'].unsqueeze(1), batch['bboxes']), 1)
        nt = targets.shape[0]
        if nt == 0:
            return None

        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(6, device=self.device)

        for i in range(self.nl):
            na = self.na[i]
            anchors = self.anchors[i, :na] / self.stride[i]  # normalized anchors
            gain[2:6] = torch.tensor(preds[i].shape)[[3, 2, 3, 2]].float()  # nx, ny, nx, ny

            t = targets * gain
            if nt == 0:
                r = torch.empty((0, 4), device=self.device)
            else:
                r = t[:, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < 4.0  # anchor match threshold
            t = t.unsqueeze(0).repeat(na, 1, 1)[j]
            offsets = torch.zeros_like(t[:, 2:4])

            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = gxy.long()
            gi, gj = gij.T

            a = torch.arange(na, device=self.device)[:, None].repeat(1, nt)[j].long()

            tbox.append(torch.cat((gxy - gij.float(), gwh), 1))
            anch.append(anchors[a] * self.stride[i])
            indices.append((b, a, gj.clamp_(0, gain[3].long() - 1), gi.clamp_(0, gain[2].long() - 1)))
            tcls.append(c)

        return {'tcls': tcls, 'tbox': tbox, 'anchors': anch, **{i: indices[i] for i in range(self.nl)}}


class CustomYOLO(YOLO):
    """Custom YOLO model with anchor-based detection support.

    Usage:
        # With anchor-based detection head
        model = CustomYOLO("configs/model/custom_yolo.yaml")
        model.train(data="coco.yaml", epochs=100)

        # With standard YOLO model
        model = CustomYOLO("yolov8n.pt")
    """

    def __init__(self, model='yolov8n.pt', task=None, verbose=False):
        """Initialize CustomYOLO model.

        Args:
            model: model path or config dict
            task: task type (detect, segment, classify, pose)
            verbose: verbosity flag
        """
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        task_map = super().task_map
        task_map["detect"]["model"] = CustomDetectionModel
        return task_map
