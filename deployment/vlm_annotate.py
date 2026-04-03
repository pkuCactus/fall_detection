"""VLM-based image annotation tool.

Uses Vision-Language Model API to analyze images and generate Pascal VOC format annotations.
Supports person state detection (fall/stand/sit/etc.) and bounding box localization.
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "src")


class VLMDetector:
    """VLM-based person detector and state classifier."""

    def __init__(self, api_url: str, api_key: Optional[str] = None, model: str = "claude"):
        """
        Args:
            api_url: VLM API endpoint
            api_key: API authentication key
            model: Model name (claude, gpt4v, etc.)
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API."""
        import base64

        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def detect(self, image_path: str) -> List[Dict]:
        """
        Detect persons in image and classify their state.

        Returns:
            List of detections: [{"class": "fall", "bbox": [x1, y1, x2, y2], "confidence": 0.95}]
        """
        # Read image for dimension check
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        img_h, img_w = img.shape[:2]

        # Prepare prompt for VLM
        system_prompt = """You are a computer vision expert. Analyze the image and detect all persons.

For each person, provide:
1. State: one of [fall_down, kneel, half_up, crawl, stand, sit, squat, bend]
   - fall_down, kneel, half_up, crawl -> person in abnormal/dangerous state
   - stand, sit, squat, bend -> person in normal state
2. Bounding box coordinates [x1, y1, x2, y2] in pixel values
3. Confidence score (0-1)

Focus on detecting people in abnormal states (fall_down, kneel, half_up, crawl).

Respond in JSON format:
{
  "detections": [
    {"class": "fall_down", "bbox": [100, 200, 300, 400], "confidence": 0.95}
  ]
}

If no persons detected, return {"detections": []}."""

        # Call VLM API
        if self.model == "claude":
            return self._call_claude(image_path, system_prompt)
        elif self.model == "gpt4v":
            return self._call_gpt4v(image_path, system_prompt)
        elif self.model == "bailian":
            return self._call_bailian(image_path, system_prompt)
        else:
            return self._call_generic(image_path, system_prompt)

    def _call_claude(self, image_path: str, prompt: str) -> List[Dict]:
        """Call Anthropic Claude API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install: pip install anthropic")

        client = anthropic.Anthropic(api_key=self.api_key)

        # Encode image
        image_base64 = self._encode_image(image_path)

        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            system=prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Analyze this image and detect all persons with their states and bounding boxes.",
                        },
                    ],
                }
            ],
        )

        # Parse JSON response
        try:
            response_text = message.content[0].text
            # Extract JSON from response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(response_text[json_start:json_end])
                return result.get("detections", [])
            return []
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Failed to parse VLM response: {e}")
            print(f"Response: {response_text}")
            return []

    def _call_gpt4v(self, image_path: str, prompt: str) -> List[Dict]:
        """Call OpenAI GPT-4V API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install: pip install openai")

        client = OpenAI(api_key=self.api_key)

        # Encode image
        image_base64 = self._encode_image(image_path)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image and detect all persons with their states and bounding boxes.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                },
            ],
            max_tokens=4096,
        )

        # Parse JSON response
        try:
            response_text = response.choices[0].message.content
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(response_text[json_start:json_end])
                return result.get("detections", [])
            return []
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Failed to parse VLM response: {e}")
            return []

    def _call_bailian(self, image_path: str, prompt: str) -> List[Dict]:
        """Call Alibaba Bailian (百炼) API.

        百炼平台使用 OpenAI 兼容格式，支持 Qwen-VL 等模型。
        API Key 从 https://bailian.console.aliyun.com/ 获取
        """
        import requests

        # 百炼平台默认使用 OpenAI 兼容格式
        api_url = self.api_url or "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 编码图像
        image_base64 = self._encode_image(image_path)

        # 构建请求体 - 百炼/Qwen-VL 格式
        payload = {
            "model": self.api_url or "qwen-vl-max",  # 默认使用 qwen-vl-max
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": prompt}]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "请分析这张图片，检测图中的人物状态并提供边界框坐标。"
                        }
                    ]
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.7
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()

            # 解析响应
            response_text = result["choices"][0]["message"]["content"]

            # 提取 JSON
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response_text[json_start:json_end])
                return data.get("detections", [])

            # 如果没有找到 JSON，尝试直接返回解析的内容
            print(f"Warning: No JSON found in response: {response_text[:200]}")
            return []

        except requests.exceptions.RequestException as e:
            print(f"Bailian API request failed: {e}")
            return []
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse Bailian response: {e}")
            return []

    def _call_generic(self, image_path: str, prompt: str) -> List[Dict]:
        """Call generic VLM API endpoint."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        image_base64 = self._encode_image(image_path)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                    ],
                }
            ],
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        # Extract detections from generic response format
        try:
            response_text = result["choices"][0]["message"]["content"]
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response_text[json_start:json_end])
                return data.get("detections", [])
            return []
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Failed to parse response: {e}")
            return []


class VOCAnnotationWriter:
    """Write annotations in Pascal VOC format."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, image_path: str, detections: List[Dict], image_size: Tuple[int, int]):
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
        ET.SubElement(source, "database").text = "VLM Generated"

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

    # Color map for different classes
    # Red for fall classes (label=1), Green for normal classes (label=0)
    COLORS = {
        # Fall classes (abnormal/dangerous) - Red
        "fall_down": (0, 0, 255),   # Red
        "kneel": (0, 0, 255),       # Red
        "half_up": (0, 0, 255),     # Red
        "crawl": (0, 0, 255),       # Red
        # Normal classes - Green
        "stand": (0, 255, 0),       # Green
        "sit": (0, 255, 0),         # Green
        "squat": (0, 255, 0),       # Green
        "bend": (0, 255, 0),        # Green
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
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.font_size)
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
            draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=color_rgb)

            # Draw text (white)
            draw.text((x1 + 2, y1 - text_h - 2), label, fill=(255, 255, 255), font=font)

        # Convert back to OpenCV and save
        result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_img)

        return output_path


def process_image(
    image_path: str,
    detector: VLMDetector,
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

    # Detect
    detections = detector.detect(image_path)
    print(f"  Found {len(detections)} persons")

    for det in detections:
        print(f"    - {det['class']}: {det['bbox']} (conf: {det.get('confidence', 1.0):.2f})")

    # Get image size
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Warning: Failed to load image")
        return []

    h, w, c = img.shape

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
    parser = argparse.ArgumentParser(description="VLM-based image annotation tool")
    parser.add_argument("--input", "-i", required=True, help="Input image or directory")
    parser.add_argument("--output-dir", "-o", default="outputs/vlm_annotations", help="VOC output directory")
    parser.add_argument("--vis-dir", "-v", default="outputs/vlm_visualizations", help="Visualization output directory")
    parser.add_argument("--api-key", "-k", help="VLM API key (or set ANTHROPIC_API_KEY / DASHSCOPE_API_KEY env var)")
    parser.add_argument("--model", "-m", default="claude", choices=["claude", "gpt4v", "bailian"], help="VLM model")
    parser.add_argument("--extensions", "-e", default="jpg,jpeg,png,bmp", help="Image extensions to process")
    args = parser.parse_args()

    # Get API key
    # Get API key based on model type
    if args.model == "bailian":
        api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY")
    else:
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        if args.model == "bailian":
            print("Error: API key required. Provide --api-key or set DASHSCOPE_API_KEY env var")
        else:
            print("Error: API key required. Provide --api-key or set ANTHROPIC_API_KEY / OPENAI_API_KEY env var")
        sys.exit(1)

    # Initialize components
    detector = VLMDetector(api_url="", api_key=api_key, model=args.model)
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
                process_image(str(img_path), detector, voc_writer, visualizer, args.vis_dir)
                print()
            except Exception as e:
                print(f"  Error processing {img_path}: {e}\n")
    else:
        print(f"Error: Input path not found: {input_path}")
        sys.exit(1)

    print(f"\nDone! Annotations saved to: {args.output_dir}")
    print(f"Visualizations saved to: {args.vis_dir}")


if __name__ == "__main__":
    main()
