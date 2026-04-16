"""
copyright (c) 2024 hjz. All rights reserved.
Use of this source is goverend by the Apache License that can be found in the LICENSE file
"""

from ultralytics import YOLO, YOLOWorld
from ultralytics.nn import tasks
from ultralytics.nn.modules import conv

from fall_detection.core.layer import ConvBNReLU, ConvUpsample

setattr(conv, "ConvBNReLU", ConvBNReLU)
setattr(conv, "ConvUpsample", ConvUpsample)

tasks.parse_model.__globals__["ConvBNReLU"] = ConvBNReLU
tasks.parse_model.__globals__["ConvUpsample"] = ConvUpsample

__all__ = ["YOLO", "YOLOWorld"]
