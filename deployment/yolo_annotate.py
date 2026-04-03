"""YOLOv8-based image annotation tool.

Uses pre-trained YOLOv8 model to detect persons and generate Pascal VOC format annotations.
Much faster and more accurate than VLM-based annotation for person detection.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "src")


def load_yolo_model(model_path: str = "yolov8n.pt", device: str = "auto"):
    """Load YOLOv8 model.

    Args:
        model_path: Model path or name (yolov8n.pt, yolov8s.pt, etc.)
        device: Device to run on (cpu, cuda, auto)

    Returns:
        Loaded YOLO model
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics package required. Install: pip install ultralytics"
        )

    # Auto-download if model doesn't exist
    if not Path(model_path).exists() and not model_path.startswith("yolov8"):
        raise ValueError(f"Model not found: {model_path}")

    model = YOLO(model_path)

    # Set device
    if device == "auto":
        device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

    return model


class YOLODetector:
    """YOLOv8-based person detector."""

    # COCO class names that YOLOv8 uses
    COCO_CLASSES = {
        0: "person",
        1: "bicycle",
        2: "car",
        # ... full list not needed, we only care about person (0)
    }

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
    ):
        """
        Args:
            model_path: YOLO model path or name
            conf_threshold: Confidence threshold for detections
            iou_threshold: NMS IoU threshold
            device: Device to run on
        """
        self.model = load_yolo_model(model_path, device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, image_path: str) -> List[Dict]:
        """
        Detect persons in image.

        Returns:
            List of detections: [{"class": "person", "bbox": [x1, y1, x2, y2], "confidence": 0.95}]
        """
        # Run inference
        results = self.model(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[0],  # Only detect persons (class 0)
            verbose=False,
        )[0]

        detections = []

        # Extract boxes
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = box.astype(int)

                detections.append({
                    "class": "person",
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(conf),
                })

        return detections


class VOCAnnotationWriter:
    """Write annotations in Pascal VOC format."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self, image_path: str, detections: List[Dict], image_size: Tuple[int, int]
    ):
        """
        Write VOC XML annotation file.

        Args:
            image_path: Path to source image
            detections: List of detection dicts
            image_size: (width, height, channels)
        """
        img_path = Path(image_path)
        filename = img_path.name
        name = img_path.stem

        # Create XML structure
        annotation = ET.Element("annotation")

        # Folder and filename
        ET.SubElement(annotation, "folder").text = img_path.parent.name
        ET.SubElement(annotation, "filename").text = filename
        ET.SubElement(annotation, "path").text = str(image_path)

        # Source
        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "YOLO Auto Generated"

        # Size
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(image_size[0])
        ET.SubElement(size, "height").text = str(image_size[1])
        ET.SubElement(size, "depth").text = str(image_size[2])

        # Segmented
        ET.SubElement(annotation, "segmented").text = "0"

        # Objects
        for det in detections:
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = det["class"]
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"

            # Confidence as attribute
            if "confidence" in det:
                ET.SubElement(obj, "confidence").text = str(det["confidence"])

            # BBox
            bbox = ET.SubElement(obj, "bndbox")
            x1, y1, x2, y2 = map(int, det["bbox"])
            ET.SubElement(bbox, "xmin").text = str(x1)
            ET.SubElement(bbox, "ymin").text = str(y1)
            ET.SubElement(bbox, "xmax").text = str(x2)
            ET.SubElement(bbox, "ymax").text = str(y2)

        # Write to file
        xml_path = self.output_dir / f"{name}.xml"
        tree = ET.ElementTree(annotation)
        ET.indent(tree, space="  ")
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

        return xml_path


class Visualizer:
    """Visualize detection results on images."""

    COLORS = {
        "person": (0, 255, 0),  # Green for person
    }

    def __init__(self, font_size: int = 20):
        self.font_size = font_size

    def draw(self, image_path: str, detections: List[Dict], output_path: str):
        """
        Draw bounding boxes and labels on image.

        Args:
            image_path: Source image path
            detections: List of detection dicts
            output_path: Where to save visualization
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert to PIL for better text rendering
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Try to load font
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.font_size
            )
        except:
            font = ImageFont.load_default()

        # Draw each detection
        for det in detections:
            class_name = det["class"]
            bbox = det["bbox"]
            conf = det.get("confidence", 1.0)

            x1, y1, x2, y2 = map(int, bbox)
            color = self.COLORS.get(class_name, (128, 128, 128))

            # Convert color to RGB tuple
            if isinstance(color, tuple) and len(color) == 3:
                color_rgb = color
            else:
                color_rgb = (128, 128, 128)

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)

            # Prepare label
            label = f"{class_name}: {conf:.2f}"

            # Get text size
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]

            # Draw text background
            draw.rectangle(
                [x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=color_rgb
            )

            # Draw text (white)
            draw.text((x1 + 2, y1 - text_h - 2), label, fill=(255, 255, 255), font=font)

        # Convert back to OpenCV and save
        result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_img)

        return output_path


def process_image(
    image_path: str,
    detector: YOLODetector,
    voc_writer: VOCAnnotationWriter,
    visualizer: Visualizer,
    vis_output_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Process a single image: detect, save VOC annotation, visualize.

    Returns:
        Detections list
    """
    print(f"Processing: {image_path}")

    # Get image size first
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Warning: Failed to load image")
        return []

    h, w, c = img.shape
    print(f"  Image size: {w}x{h}")

    # Detect
    detections = detector.detect(image_path)
    print(f"  Found {len(detections)} persons")

    for det in detections:
        print(f"    - {det['class']}: {det['bbox']} (conf: {det.get('confidence', 1.0):.2f})")

    # Write VOC annotation
    xml_path = voc_writer.write(image_path, detections, (w, h, c))
    print(f"  Saved annotation: {xml_path}")

    # Visualize
    if vis_output_dir:
        vis_dir = Path(vis_output_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_path = vis_dir / Path(image_path).name
        visualizer.draw(image_path, detections, str(vis_path))
        print(f"  Saved visualization: {vis_path}")

    return detections


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8-based person detection and annotation tool"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input image or directory"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="outputs/yolo_annotations",
        help="VOC output directory",
    )
    parser.add_argument(
        "--vis-dir",
        "-v",
        default=None,
        help="Visualization output directory (default: no visualization)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="yolov8n.pt",
        help="YOLO model path or name (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)",
    )
    parser.add_argument(
        "--conf-threshold",
        "-c",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--device",
        "-d",
        default="auto",
        choices=["auto", "cpu", "cuda", "0", "1"],
        help="Device to run on",
    )
    parser.add_argument(
        "--extensions",
        "-e",
        default="jpg,jpeg,png,bmp",
        help="Image extensions to process",
    )
    args = parser.parse_args()

    # Initialize components
    print(f"Loading YOLO model: {args.model}")
    detector = YOLODetector(
        model_path=args.model,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device,
    )
    print("Model loaded successfully")

    voc_writer = VOCAnnotationWriter(args.output_dir)
    visualizer = Visualizer()

    # Process input
    input_path = Path(args.input)
    extensions = args.extensions.split(",")

    if input_path.is_file():
        # Single image
        process_image(str(input_path), detector, voc_writer, visualizer, args.vis_dir)
    elif input_path.is_dir():
        # Directory
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*.{ext}"))
            image_files.extend(input_path.glob(f"*.{ext.upper()}"))

        print(f"Found {len(image_files)} images in {input_path}")

        for img_path in sorted(image_files):
            try:
                process_image(
                    str(img_path), detector, voc_writer, visualizer, args.vis_dir
                )
                print()
            except Exception as e:
                print(f"  Error processing {img_path}: {e}\n")
    else:
        print(f"Error: Input path not found: {input_path}")
        sys.exit(1)

    print(f"\nDone! Annotations saved to: {args.output_dir}")
    if args.vis_dir:
        print(f"Visualizations saved to: {args.vis_dir}")


if __name__ == "__main__":
    main()
