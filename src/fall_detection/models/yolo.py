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
                AnchorDetect,
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
        self.model, self.save = custom_parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)

        # Set stride and other properties
        m = self.model[-1]  # last layer (Detect)
        if isinstance(m, AnchorDetect):
            m.stride = torch.tensor([32.0, 16.0, 8.0])  # P5, P4, P3 strides
            self.stride = m.stride

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

    def init_criterion(self):
        """Initialize loss criterion for training."""
        from ultralytics.utils.loss import BboxLoss

        # For AnchorDetect, use a custom loss or standard detection loss
        class AnchorDetectionLoss(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.bce = nn.BCEWithLogitsLoss(reduction='none')

            def __call__(self, preds, batch):
                """Compute detection loss for anchor-based head."""
                # This is a simplified loss - you may want to implement full YOLO loss
                device = preds[0].device
                loss = torch.tensor(0.0, device=device)

                for i, pi in enumerate(preds):
                    # pi shape: (bs, na, ny, nx, nc+5)
                    # batch should contain targets
                    if batch.get('batch_idx') is None:
                        continue

                    # Basic box regression + classification loss
                    # This is a placeholder - implement proper YOLO anchor-based loss
                    obj_loss = self.bce(pi[..., 4], torch.zeros_like(pi[..., 4]))
                    cls_loss = self.bce(pi[..., 5:], torch.zeros_like(pi[..., 5:]))
                    loss += obj_loss.mean() + cls_loss.mean()

                return loss * 0, loss * 0, loss  # box_loss, cls_loss, total_loss

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


class CustomYOLO(YOLO):
    """Custom YOLO model with anchor-based detection support.

    Usage:
        # With anchor-based detection head
        model = CustomYOLO("configs/model/custom_yolo.yaml")
        model.train(data="coco.yaml", epochs=100)

        # With standard YOLO model
        model = CustomYOLO("yolov8n.pt")
    """

    def __init__(self, model='yolov8n.pt', task="detect", verbose=False):
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
