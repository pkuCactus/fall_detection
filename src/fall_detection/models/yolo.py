"""
copyright (c) 2024 hjz. All rights reserved.
Use of this source is goverend by the Apache License that can be found in the LICENSE file
"""

from copy import deepcopy
from typing import Dict, Any

import torch
import torch.nn as nn
from ultralytics import YOLO, YOLOWorld
from ultralytics.nn import tasks
from ultralytics.nn.modules import conv
from ultralytics.nn.tasks import DetectionModel, BaseModel
from ultralytics.utils import LOGGER, RANK

from fall_detection.core.layer import ConvBNReLU, ConvUpsample, AnchorDetect


# Register custom layers to ultralytics parse_model
setattr(conv, "ConvBNReLU", ConvBNReLU)
setattr(conv, "ConvUpsample", ConvUpsample)
setattr(conv, "AnchorDetect", AnchorDetect)

tasks.parse_model.__globals__["ConvBNReLU"] = ConvBNReLU
tasks.parse_model.__globals__["ConvUpsample"] = ConvUpsample
tasks.parse_model.__globals__["AnchorDetect"] = AnchorDetect


def custom_parse_model(d, ch, verbose=True):
    """Custom parse_model that supports AnchorDetect and custom layers.

    Args:
        d: model dict from yaml
        ch: input channels
        verbose: print model info

    Returns:
        nn.Sequential model and save list
    """
    import ast

    # Get args from yaml
    anchors = d.get('anchors', None)
    nc, act, scale = (d.get(x) for x in ('nc', 'activation', 'scale'))
    reg_max = d.get('reg_max', 16)
    end2end = d.get('end2end', False)
    legacy = d.get('legacy', False)

    # Import necessary modules
    import torch.nn as nn
    from ultralytics.nn.modules import Conv, Concat

    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        # Resolve module
        if m == 'nn.Conv2d':
            m_ = nn.Conv2d
        elif m == 'ConvBNReLU':
            m_ = ConvBNReLU
        elif m == 'ConvUpsample':
            m_ = ConvUpsample
        elif m == 'AnchorDetect':
            m_ = AnchorDetect
        elif m == 'Concat':
            m_ = Concat
        elif m == 'Detect':
            from ultralytics.nn.modules import Detect
            m_ = Detect
        elif m == 'C2f':
            m_ = C2f
        elif m == 'SPPF':
            m_ = SPPF
        elif m == 'Conv':
            m_ = Conv
        elif m == 'Upsample':
            m_ = nn.Upsample
        else:
            # Try to get from ultralytics modules
            try:
                m_ = eval(m)
            except:
                raise ValueError(f"Unknown module: {m}")

        # Resolve args
        if args is None:
            args = []
        if isinstance(args, (int, float, str)):
            args = [args]
        elif isinstance(args, (list, tuple)) and len(args) == 1 and isinstance(args[0], (list, tuple)):
            # Handle case where args is [[a, b, c]] instead of [a, b, c]
            args = list(args[0])

        n_ = n
        n = n_ = max(round(n * 1), 1) if n > 1 else n  # depth gain

        # Get input channels
        if f == -1:
            c1 = ch[-1]
        elif isinstance(f, int):
            c1 = ch[f]
        elif isinstance(f, (list, tuple)):
            c1 = [ch[x] for x in f]
        else:
            c1 = ch[f]

        # Handle specific module types
        if m_ is ConvBNReLU:
            c2 = args[0]
            args = [c1, *args]
        elif m_ is ConvUpsample:
            c2 = args[0]
            args = [c1, *args]
        elif m_ is nn.Conv2d:
            c2 = args[0]
            args = [c1, *args]
        elif m_ is Conv:
            c2 = args[0]
            args = [c1, *args]
        elif m_ is Concat:
            c2 = sum(ch[x] for x in f)
        elif m_ is AnchorDetect:
            # Handle AnchorDetect specially
            # Parse nc and anchors from args, supporting string references
            arg_nc = args[0] if args else nc
            arg_anchors = args[1] if len(args) > 1 else anchors
            # Resolve string references
            if isinstance(arg_nc, str):
                arg_nc = nc if arg_nc == 'nc' else int(arg_nc)
            if isinstance(arg_anchors, str):
                arg_anchors = anchors if arg_anchors == 'anchors' else eval(arg_anchors)
            # Get input channels from f (multi-input)
            ch_list = [ch[x] for x in f] if isinstance(f, (list, tuple)) else [ch[f]]
            args = [arg_nc, arg_anchors, ch_list]
            # c2 is not set for Detect layers
        elif m_ in (Detect,):
            # Handle standard Detect
            from ultralytics.nn.modules import Detect
            args.extend([reg_max, end2end, [ch[x] for x in f]])
        elif m_ is nn.Upsample:
            c2 = c1
        else:
            c2 = c1

        # Create module
        m_ = nn.Sequential(*(m_(*args) for _ in range(n))) if n > 1 else m_(*args)
        t = str(m_)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t

        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")

        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)

        if i == 0:
            ch = []
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)


class CustomDetectionModel(BaseModel):
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
        super().__init__()  # Initialize Module first
        self.yaml = cfg if isinstance(cfg, dict) else self._load_yaml(cfg)

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)

        # Override nc if provided
        if nc and nc != self.yaml.get('nc'):
            LOGGER.info(f"Overriding model.yaml nc={self.yaml.get('nc')} with nc={nc}")
            self.yaml['nc'] = nc

        # Build model using custom parse_model
        self.model, self.save = custom_parse_model(deepcopy(self.yaml), ch=[ch], verbose=verbose and RANK == -1)

        # Set stride and other properties
        m = self.model[-1]  # last layer (Detect)
        if isinstance(m, AnchorDetect):
            m.stride = torch.tensor([32.0, 16.0, 8.0])  # P5, P4, P3 strides
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
        if isinstance(m, AnchorDetect):
            m._initialize_biases()

    def _initialize_biases(self):
        """Initialize detection head biases for AnchorDetect."""
        m = self.model[-1]
        if isinstance(m, AnchorDetect):
            for mi, s in zip(m.m, [32, 16, 8]):
                b = mi.bias.view(m.na, -1)
                b.data[:, 4] += torch.log(torch.tensor(8 / (640 / s) ** 2))
                b.data[:, 5:5 + m.nc] += torch.log(torch.tensor(0.6 / (m.nc - 0.999999)))
                mi.bias = nn.Parameter(b.view(-1), requires_grad=True)

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
        model = CustomYOLO("configs/model/ori_detector_anchor.yaml")
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
        # Check if model is a custom anchor-based yaml
        if isinstance(model, str) and model.endswith('.yaml'):
            import yaml
            with open(model, 'r') as f:
                cfg = yaml.safe_load(f)
            if cfg.get('anchors') is not None or any('AnchorDetect' in str(layer) for layer in cfg.get('head', [])):
                # Use custom model for anchor-based detection
                self._custom_anchor = True
            else:
                self._custom_anchor = False
        else:
            self._custom_anchor = False

        super().__init__(model=model, task=task, verbose=verbose)

    def _new(self, model=None, task=None, verbose=False):
        """Create new model."""
        from ultralytics.nn.tasks import DetectionModel as BaseDetectionModel, yaml_model_load

        # Load yaml config
        cfg = model if isinstance(model, dict) else yaml_model_load(model)
        self.model_cfg = cfg  # Store for later use

        if getattr(self, '_custom_anchor', False):
            # Use CustomDetectionModel for anchor-based models
            self.model = CustomDetectionModel(cfg, verbose=verbose)
            self.task = 'detect'
        else:
            # Use standard DetectionModel
            self.model = BaseDetectionModel(cfg, ch=3, verbose=verbose and RANK == -1)
            self.task = task or 'detect'


# Backward compatibility
__all__ = ["YOLO", "YOLOWorld", "CustomYOLO", "CustomDetectionModel", "ConvBNReLU", "ConvUpsample", "AnchorDetect", "custom_parse_model"]
