"""Unit tests for dataset loading modules."""

import json
import os
import xml.etree.ElementTree as ET
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from fall_detection.data.datasets import CocoFallDataset, VOCFallDataset


class TestCocoFallDataset:
    """Tests for CocoFallDataset class."""

    def test_init_with_mock_coco_json(self, tmp_path):
        """Test CocoFallDataset initialization with mock COCO JSON."""
        # Create mock COCO JSON
        coco_data = {
            "images": [
                {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "image2.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 80]},
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [200, 150, 60, 90]},
                {"id": 3, "image_id": 2, "category_id": 1, "bbox": [50, 50, 100, 150]},
            ],
            "categories": [
                {"id": 1, "name": "person"},
            ],
        }

        # Create temporary files
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        coco_json = tmp_path / "annotations.json"
        with open(coco_json, 'w') as f:
            json.dump(coco_data, f)

        # Create dummy images
        for img_info in coco_data["images"]:
            img = Image.new('RGB', (640, 480), color=(128, 128, 128))
            img.save(image_dir / img_info["file_name"])

        # Initialize dataset
        dataset = CocoFallDataset(
            image_dir=str(image_dir),
            coco_json=str(coco_json),
            target_size=96,
            use_letterbox=True,
        )

        # Verify initialization
        assert len(dataset) == 3
        assert dataset.target_size == 96
        assert dataset.use_letterbox is True
        assert 1 in dataset.images
        assert 2 in dataset.images

    def test_getitem_with_mock_image(self, tmp_path, monkeypatch):
        """Test CocoFallDataset __getitem__ with mock image and annotations."""
        # Create mock COCO JSON
        coco_data = {
            "images": [
                {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 80]},
            ],
            "categories": [
                {"id": 1, "name": "person"},
            ],
        }

        # Create temporary files
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        coco_json = tmp_path / "annotations.json"
        with open(coco_json, 'w') as f:
            json.dump(coco_data, f)

        # Create dummy image
        img = Image.new('RGB', (640, 480), color=(128, 128, 128))
        img.save(image_dir / "image1.jpg")

        # Mock cv2.imread to return a numpy array
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        monkeypatch.setattr('cv2.imread', lambda path: mock_img)

        # Initialize dataset
        dataset = CocoFallDataset(
            image_dir=str(image_dir),
            coco_json=str(coco_json),
            target_size=96,
            use_letterbox=True,
        )

        # Get item
        roi, label = dataset[0]

        # Verify output
        assert isinstance(roi, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert roi.shape == (3, 96, 96)  # (C, H, W)
        assert label.dtype == torch.long
        assert label.item() in [0, 1]

    def test_different_fall_category_id(self, tmp_path, monkeypatch):
        """Test CocoFallDataset with different fall_category_id."""
        # Create mock COCO JSON with fall category
        coco_data = {
            "images": [
                {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 5, "bbox": [100, 100, 50, 80]},
            ],
            "categories": [
                {"id": 5, "name": "fall"},
            ],
        }

        # Create temporary files
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        coco_json = tmp_path / "annotations.json"
        with open(coco_json, 'w') as f:
            json.dump(coco_data, f)

        # Create dummy image
        img = Image.new('RGB', (640, 480), color=(128, 128, 128))
        img.save(image_dir / "image1.jpg")

        # Mock cv2.imread
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        monkeypatch.setattr('cv2.imread', lambda path: mock_img)

        # Initialize dataset with fall_category_id=5
        dataset = CocoFallDataset(
            image_dir=str(image_dir),
            coco_json=str(coco_json),
            target_size=96,
            use_letterbox=True,
            fall_category_id=5,
        )

        # Get item and verify label is 1 (fall)
        roi, label = dataset[0]
        assert label.item() == 1

    def test_len_with_empty_annotations(self, tmp_path):
        """Test CocoFallDataset len with empty annotations."""
        # Create mock COCO JSON with no annotations
        coco_data = {
            "images": [
                {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
            ],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "person"},
            ],
        }

        # Create temporary files
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        coco_json = tmp_path / "annotations.json"
        with open(coco_json, 'w') as f:
            json.dump(coco_data, f)

        # Initialize dataset
        dataset = CocoFallDataset(
            image_dir=str(image_dir),
            coco_json=str(coco_json),
            target_size=96,
            use_letterbox=True,
        )

        # Verify length is 0
        assert len(dataset) == 0

    def test_label_from_attributes(self, tmp_path, monkeypatch):
        """Test CocoFallDataset label extraction from attributes."""
        # Create mock COCO JSON with fall in attributes
        coco_data = {
            "images": [
                {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 80], "attributes": {"fall": 1}},
            ],
            "categories": [
                {"id": 1, "name": "person"},
            ],
        }

        # Create temporary files
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        coco_json = tmp_path / "annotations.json"
        with open(coco_json, 'w') as f:
            json.dump(coco_data, f)

        # Create dummy image
        img = Image.new('RGB', (640, 480), color=(128, 128, 128))
        img.save(image_dir / "image1.jpg")

        # Mock cv2.imread
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        monkeypatch.setattr('cv2.imread', lambda path: mock_img)

        # Initialize dataset
        dataset = CocoFallDataset(
            image_dir=str(image_dir),
            coco_json=str(coco_json),
            target_size=96,
            use_letterbox=True,
            fall_category_id=2,  # Different from annotation category_id
        )

        # Get item and verify label is 1 (fall from attributes)
        roi, label = dataset[0]
        assert label.item() == 1

    def test_label_from_fall_field(self, tmp_path, monkeypatch):
        """Test CocoFallDataset label extraction from fall field."""
        # Create mock COCO JSON with fall field
        coco_data = {
            "images": [
                {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 80], "fall": 1},
            ],
            "categories": [
                {"id": 1, "name": "person"},
            ],
        }

        # Create temporary files
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        coco_json = tmp_path / "annotations.json"
        with open(coco_json, 'w') as f:
            json.dump(coco_data, f)

        # Create dummy image
        img = Image.new('RGB', (640, 480), color=(128, 128, 128))
        img.save(image_dir / "image1.jpg")

        # Mock cv2.imread
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        monkeypatch.setattr('cv2.imread', lambda path: mock_img)

        # Initialize dataset
        dataset = CocoFallDataset(
            image_dir=str(image_dir),
            coco_json=str(coco_json),
            target_size=96,
            use_letterbox=True,
            fall_category_id=2,  # Different from annotation category_id
        )

        # Get item and verify label is 1 (fall from fall field)
        roi, label = dataset[0]
        assert label.item() == 1

    def test_bbox_conversion(self, tmp_path, monkeypatch):
        """Test COCO bbox [x, y, w, h] to [x1, y1, x2, y2] conversion."""
        # Create mock COCO JSON
        coco_data = {
            "images": [
                {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 80]},  # x=100, y=100, w=50, h=80
            ],
            "categories": [
                {"id": 1, "name": "person"},
            ],
        }

        # Create temporary files
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        coco_json = tmp_path / "annotations.json"
        with open(coco_json, 'w') as f:
            json.dump(coco_data, f)

        # Create dummy image
        img = Image.new('RGB', (640, 480), color=(128, 128, 128))
        img.save(image_dir / "image1.jpg")

        # Initialize dataset
        dataset = CocoFallDataset(
            image_dir=str(image_dir),
            coco_json=str(coco_json),
            target_size=96,
            use_letterbox=True,
        )

        # Verify bbox conversion: [100, 100, 50, 80] -> [100, 100, 150, 180]
        assert dataset.samples[0][1] == [100, 100, 150, 180]

    def test_image_load_failure(self, tmp_path):
        """Test CocoFallDataset image load failure handling."""
        # Create mock COCO JSON
        coco_data = {
            "images": [
                {"id": 1, "file_name": "nonexistent.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 80]},
            ],
            "categories": [
                {"id": 1, "name": "person"},
            ],
        }

        # Create temporary files
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        coco_json = tmp_path / "annotations.json"
        with open(coco_json, 'w') as f:
            json.dump(coco_data, f)

        # Initialize dataset
        dataset = CocoFallDataset(
            image_dir=str(image_dir),
            coco_json=str(coco_json),
            target_size=96,
            use_letterbox=True,
        )

        # Should raise ValueError when trying to load non-existent image
        with pytest.raises(ValueError, match="Failed to load image"):
            _ = dataset[0]

    def test_no_letterbox_resize(self, tmp_path, monkeypatch):
        """Test CocoFallDataset without letterbox (direct resize)."""
        # Create mock COCO JSON
        coco_data = {
            "images": [
                {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 80]},
            ],
            "categories": [
                {"id": 1, "name": "person"},
            ],
        }

        # Create temporary files
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        coco_json = tmp_path / "annotations.json"
        with open(coco_json, 'w') as f:
            json.dump(coco_data, f)

        # Create dummy image
        img = Image.new('RGB', (640, 480), color=(128, 128, 128))
        img.save(image_dir / "image1.jpg")

        # Mock cv2.imread
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        monkeypatch.setattr('cv2.imread', lambda path: mock_img)

        # Initialize dataset without letterbox
        dataset = CocoFallDataset(
            image_dir=str(image_dir),
            coco_json=str(coco_json),
            target_size=96,
            use_letterbox=False,
        )

        # Get item
        roi, label = dataset[0]

        # Verify output shape
        assert roi.shape == (3, 96, 96)


class TestVOCFallDataset:
    """Tests for VOCFallDataset class."""

    def _create_voc_structure(self, base_path, image_ids, annotations):
        """Helper to create VOC directory structure."""
        # Create directories
        jpeg_dir = base_path / "JPEGImages"
        anno_dir = base_path / "Annotations"
        imageset_dir = base_path / "ImageSets" / "Main"

        jpeg_dir.mkdir(parents=True)
        anno_dir.mkdir(parents=True)
        imageset_dir.mkdir(parents=True)

        # Create images
        for img_id in image_ids:
            img = Image.new('RGB', (640, 480), color=(128, 128, 128))
            img.save(jpeg_dir / f"{img_id}.jpg")

        # Create annotations
        for img_id, objects in annotations.items():
            root = ET.Element("annotation")
            ET.SubElement(root, "filename").text = f"{img_id}.jpg"

            for obj in objects:
                obj_elem = ET.SubElement(root, "object")
                ET.SubElement(obj_elem, "name").text = obj["name"]
                bndbox = ET.SubElement(obj_elem, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(obj["xmin"])
                ET.SubElement(bndbox, "ymin").text = str(obj["ymin"])
                ET.SubElement(bndbox, "xmax").text = str(obj["xmax"])
                ET.SubElement(bndbox, "ymax").text = str(obj["ymax"])

            tree = ET.ElementTree(root)
            tree.write(anno_dir / f"{img_id}.xml")

        return jpeg_dir, anno_dir, imageset_dir

    def test_init_with_mock_voc_structure(self, tmp_path, monkeypatch):
        """Test VOCFallDataset initialization with mock VOC structure."""
        # Create VOC structure
        image_ids = ["img001", "img002"]
        annotations = {
            "img001": [{"name": "fall", "xmin": 100, "ymin": 100, "xmax": 200, "ymax": 300}],
            "img002": [{"name": "person", "xmin": 50, "ymin": 50, "xmax": 150, "ymax": 250}],
        }

        jpeg_dir, anno_dir, imageset_dir = self._create_voc_structure(
            tmp_path, image_ids, annotations
        )

        # Create train.txt
        with open(imageset_dir / "train.txt", 'w') as f:
            for img_id in image_ids:
                f.write(f"{img_id}\n")

        # Mock cv2.imread
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        monkeypatch.setattr('cv2.imread', lambda path: mock_img)

        # Initialize dataset
        dataset = VOCFallDataset(
            data_dirs=[str(tmp_path)],
            split="train",
            target_size=96,
            use_letterbox=True,
        )

        # Verify initialization
        assert len(dataset) == 2
        assert dataset.split == "train"
        assert dataset.target_size == 96

    def test_getitem(self, tmp_path, monkeypatch):
        """Test VOCFallDataset __getitem__."""
        # Create VOC structure
        image_ids = ["img001"]
        annotations = {
            "img001": [{"name": "fall", "xmin": 100, "ymin": 100, "xmax": 200, "ymax": 300}],
        }

        jpeg_dir, anno_dir, imageset_dir = self._create_voc_structure(
            tmp_path, image_ids, annotations
        )

        # Create train.txt
        with open(imageset_dir / "train.txt", 'w') as f:
            f.write("img001\n")

        # Mock cv2.imread
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        monkeypatch.setattr('cv2.imread', lambda path: mock_img)

        # Initialize dataset
        dataset = VOCFallDataset(
            data_dirs=[str(tmp_path)],
            split="train",
            target_size=96,
            use_letterbox=True,
        )

        # Get item
        roi, label = dataset[0]

        # Verify output
        assert isinstance(roi, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert roi.shape == (3, 96, 96)
        assert label.dtype == torch.long

    def test_fall_classes_filtering(self, tmp_path, monkeypatch):
        """Test VOCFallDataset with fall_classes filtering."""
        # Create VOC structure - only fall classes
        image_ids = ["img001", "img002"]
        annotations = {
            "img001": [
                {"name": "fall", "xmin": 100, "ymin": 100, "xmax": 200, "ymax": 300},
                {"name": "lying", "xmin": 50, "ymin": 50, "xmax": 150, "ymax": 200},
            ],
            "img002": [{"name": "fall", "xmin": 50, "ymin": 50, "xmax": 150, "ymax": 250}],
        }

        jpeg_dir, anno_dir, imageset_dir = self._create_voc_structure(
            tmp_path, image_ids, annotations
        )

        # Create train.txt
        with open(imageset_dir / "train.txt", 'w') as f:
            for img_id in image_ids:
                f.write(f"{img_id}\n")

        # Mock cv2.imread
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        monkeypatch.setattr('cv2.imread', lambda path: mock_img)

        # Initialize dataset with custom fall_classes
        dataset = VOCFallDataset(
            data_dirs=[str(tmp_path)],
            split="train",
            target_size=96,
            use_letterbox=True,
            fall_classes=["fall", "lying"],  # Both should be label=1
        )

        # Verify all samples are falls (label=1)
        assert len(dataset) == 3  # 2 from img001 + 1 from img002
        for i in range(len(dataset)):
            roi, label = dataset[i]
            assert label.item() == 1

    def test_normal_classes_filtering(self, tmp_path, monkeypatch):
        """Test VOCFallDataset with normal_classes filtering."""
        # Create VOC structure
        image_ids = ["img001", "img002"]
        annotations = {
            "img001": [
                {"name": "fall", "xmin": 100, "ymin": 100, "xmax": 200, "ymax": 300},
            ],
            "img002": [
                {"name": "person", "xmin": 50, "ymin": 50, "xmax": 150, "ymax": 250},
                {"name": "walking", "xmin": 200, "ymin": 200, "xmax": 300, "ymax": 350},
            ],
        }

        jpeg_dir, anno_dir, imageset_dir = self._create_voc_structure(
            tmp_path, image_ids, annotations
        )

        # Create train.txt
        with open(imageset_dir / "train.txt", 'w') as f:
            for img_id in image_ids:
                f.write(f"{img_id}\n")

        # Mock cv2.imread
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        monkeypatch.setattr('cv2.imread', lambda path: mock_img)

        # Initialize dataset with normal_classes
        dataset = VOCFallDataset(
            data_dirs=[str(tmp_path)],
            split="train",
            target_size=96,
            use_letterbox=True,
            fall_classes=["fall"],
            normal_classes=["person", "walking"],  # These should be label=0
        )

        # Verify samples have correct labels
        labels = [dataset[i][1].item() for i in range(len(dataset))]
        assert 0 in labels  # Should have normal samples
        assert 1 in labels  # Should have fall samples

    def test_unknown_classes_skipped(self, tmp_path, monkeypatch):
        """Test that unknown classes are skipped when normal_classes is specified."""
        # Create VOC structure
        image_ids = ["img001"]
        annotations = {
            "img001": [
                {"name": "fall", "xmin": 100, "ymin": 100, "xmax": 200, "ymax": 300},
                {"name": "unknown_class", "xmin": 50, "ymin": 50, "xmax": 150, "ymax": 200},
            ],
        }

        jpeg_dir, anno_dir, imageset_dir = self._create_voc_structure(
            tmp_path, image_ids, annotations
        )

        # Create train.txt
        with open(imageset_dir / "train.txt", 'w') as f:
            f.write("img001\n")

        # Mock cv2.imread
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        monkeypatch.setattr('cv2.imread', lambda path: mock_img)

        # Initialize dataset with specific normal_classes
        dataset = VOCFallDataset(
            data_dirs=[str(tmp_path)],
            split="train",
            target_size=96,
            use_letterbox=True,
            fall_classes=["fall"],
            normal_classes=["person"],  # unknown_class is not in fall or normal
        )

        # Only fall sample should be included (unknown_class skipped)
        assert len(dataset) == 1
        roi, label = dataset[0]
        assert label.item() == 1  # fall

    def test_multiple_data_directories(self, tmp_path, monkeypatch):
        """Test VOCFallDataset with multiple data directories."""
        # Create first VOC structure
        dir1 = tmp_path / "dataset1"
        image_ids1 = ["img001"]
        annotations1 = {
            "img001": [{"name": "fall", "xmin": 100, "ymin": 100, "xmax": 200, "ymax": 300}],
        }
        jpeg_dir1, anno_dir1, imageset_dir1 = self._create_voc_structure(
            dir1, image_ids1, annotations1
        )
        with open(imageset_dir1 / "train.txt", 'w') as f:
            f.write("img001\n")

        # Create second VOC structure
        dir2 = tmp_path / "dataset2"
        image_ids2 = ["img002"]
        annotations2 = {
            "img002": [{"name": "person", "xmin": 50, "ymin": 50, "xmax": 150, "ymax": 250}],
        }
        jpeg_dir2, anno_dir2, imageset_dir2 = self._create_voc_structure(
            dir2, image_ids2, annotations2
        )
        with open(imageset_dir2 / "train.txt", 'w') as f:
            f.write("img002\n")

        # Mock cv2.imread
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        monkeypatch.setattr('cv2.imread', lambda path: mock_img)

        # Initialize dataset with multiple directories
        dataset = VOCFallDataset(
            data_dirs=[str(dir1), str(dir2)],
            split="train",
            target_size=96,
            use_letterbox=True,
        )

        # Verify samples from both directories are included
        assert len(dataset) == 2

    def test_split_handling_train_val(self, tmp_path, monkeypatch):
        """Test VOCFallDataset split handling for train and val."""
        # Create VOC structure
        image_ids = ["img001", "img002", "img003"]
        annotations = {
            "img001": [{"name": "fall", "xmin": 100, "ymin": 100, "xmax": 200, "ymax": 300}],
            "img002": [{"name": "person", "xmin": 50, "ymin": 50, "xmax": 150, "ymax": 250}],
            "img003": [{"name": "fall", "xmin": 200, "ymin": 200, "xmax": 300, "ymax": 400}],
        }

        jpeg_dir, anno_dir, imageset_dir = self._create_voc_structure(
            tmp_path, image_ids, annotations
        )

        # Create train.txt and val.txt
        with open(imageset_dir / "train.txt", 'w') as f:
            f.write("img001\n")
            f.write("img002\n")

        with open(imageset_dir / "val.txt", 'w') as f:
            f.write("img003\n")

        # Mock cv2.imread
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        monkeypatch.setattr('cv2.imread', lambda path: mock_img)

        # Initialize train dataset
        train_dataset = VOCFallDataset(
            data_dirs=[str(tmp_path)],
            split="train",
            target_size=96,
            use_letterbox=True,
        )

        # Initialize val dataset
        val_dataset = VOCFallDataset(
            data_dirs=[str(tmp_path)],
            split="val",
            target_size=96,
            use_letterbox=True,
        )

        # Verify splits
        assert len(train_dataset) == 2
        assert len(val_dataset) == 1

    def test_no_imageset_file_scan_all_xml(self, tmp_path, monkeypatch):
        """Test VOCFallDataset scans all XML when ImageSet file doesn't exist."""
        # Create VOC structure
        image_ids = ["img001", "img002"]
        annotations = {
            "img001": [{"name": "fall", "xmin": 100, "ymin": 100, "xmax": 200, "ymax": 300}],
            "img002": [{"name": "person", "xmin": 50, "ymin": 50, "xmax": 150, "ymax": 250}],
        }

        jpeg_dir, anno_dir, imageset_dir = self._create_voc_structure(
            tmp_path, image_ids, annotations
        )

        # Don't create train.txt - should scan all XML files

        # Mock cv2.imread
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        monkeypatch.setattr('cv2.imread', lambda path: mock_img)

        # Initialize dataset
        dataset = VOCFallDataset(
            data_dirs=[str(tmp_path)],
            split="train",  # ImageSet file doesn't exist for this split
            target_size=96,
            use_letterbox=True,
        )

        # Should scan all XML files
        assert len(dataset) == 2

    def test_missing_annotation_file(self, tmp_path, monkeypatch):
        """Test VOCFallDataset handles missing annotation files gracefully."""
        # Create VOC structure
        image_ids = ["img001", "img002"]
        annotations = {
            "img001": [{"name": "fall", "xmin": 100, "ymin": 100, "xmax": 200, "ymax": 300}],
            # img002 annotation is missing
        }

        jpeg_dir, anno_dir, imageset_dir = self._create_voc_structure(
            tmp_path, image_ids, annotations
        )

        # Create train.txt with both images
        with open(imageset_dir / "train.txt", 'w') as f:
            f.write("img001\n")
            f.write("img002\n")

        # Mock cv2.imread
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        monkeypatch.setattr('cv2.imread', lambda path: mock_img)

        # Initialize dataset
        dataset = VOCFallDataset(
            data_dirs=[str(tmp_path)],
            split="train",
            target_size=96,
            use_letterbox=True,
        )

        # Only img001 should be included (img002 has no annotation)
        assert len(dataset) == 1

    def test_missing_image_file(self, tmp_path, monkeypatch):
        """Test VOCFallDataset handles missing image files gracefully."""
        # Create VOC structure
        image_ids = ["img001", "img002"]
        annotations = {
            "img001": [{"name": "fall", "xmin": 100, "ymin": 100, "xmax": 200, "ymax": 300}],
            "img002": [{"name": "person", "xmin": 50, "ymin": 50, "xmax": 150, "ymax": 250}],
        }

        jpeg_dir, anno_dir, imageset_dir = self._create_voc_structure(
            tmp_path, image_ids, annotations
        )

        # Remove img002.jpg
        (jpeg_dir / "img002.jpg").unlink()

        # Create train.txt
        with open(imageset_dir / "train.txt", 'w') as f:
            f.write("img001\n")
            f.write("img002\n")

        # Mock cv2.imread for existing image only
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        original_imread = lambda path: mock_img if "img001" in path else None
        monkeypatch.setattr('cv2.imread', original_imread)

        # Initialize dataset
        dataset = VOCFallDataset(
            data_dirs=[str(tmp_path)],
            split="train",
            target_size=96,
            use_letterbox=True,
        )

        # Only img001 should be included
        assert len(dataset) == 1

    def test_invalid_bbox_in_xml(self, tmp_path, monkeypatch):
        """Test VOCFallDataset skips invalid bboxes."""
        # Create VOC structure with invalid bbox
        image_ids = ["img001"]

        jpeg_dir, anno_dir, imageset_dir = self._create_voc_structure(
            tmp_path, image_ids, {}
        )

        # Create annotation with invalid bbox (xmax <= xmin)
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = "img001.jpg"

        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "fall"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = "200"
        ET.SubElement(bndbox, "ymin").text = "100"
        ET.SubElement(bndbox, "xmax").text = "100"  # Invalid: xmax < xmin
        ET.SubElement(bndbox, "ymax").text = "300"

        tree = ET.ElementTree(root)
        tree.write(anno_dir / "img001.xml")

        # Create train.txt
        with open(imageset_dir / "train.txt", 'w') as f:
            f.write("img001\n")

        # Mock cv2.imread
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        monkeypatch.setattr('cv2.imread', lambda path: mock_img)

        # Invalid bbox should be skipped, resulting in no valid samples
        with pytest.raises(ValueError, match="No valid samples found"):
            VOCFallDataset(
                data_dirs=[str(tmp_path)],
                split="train",
                target_size=96,
                use_letterbox=True,
            )

    def test_missing_bbox_fields(self, tmp_path, monkeypatch):
        """Test VOCFallDataset handles missing bbox fields."""
        # Create VOC structure
        image_ids = ["img001"]

        jpeg_dir, anno_dir, imageset_dir = self._create_voc_structure(
            tmp_path, image_ids, {}
        )

        # Create annotation with missing bbox fields
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = "img001.jpg"

        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "fall"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = "100"
        # Missing ymin, xmax, ymax

        tree = ET.ElementTree(root)
        tree.write(anno_dir / "img001.xml")

        # Create train.txt
        with open(imageset_dir / "train.txt", 'w') as f:
            f.write("img001\n")

        # Mock cv2.imread
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        monkeypatch.setattr('cv2.imread', lambda path: mock_img)

        # Invalid bbox should be skipped, resulting in no valid samples
        with pytest.raises(ValueError, match="No valid samples found"):
            VOCFallDataset(
                data_dirs=[str(tmp_path)],
                split="train",
                target_size=96,
                use_letterbox=True,
            )

    def test_missing_object_name(self, tmp_path, monkeypatch):
        """Test VOCFallDataset handles missing object name."""
        # Create VOC structure
        image_ids = ["img001"]

        jpeg_dir, anno_dir, imageset_dir = self._create_voc_structure(
            tmp_path, image_ids, {}
        )

        # Create annotation with missing name
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = "img001.jpg"

        obj = ET.SubElement(root, "object")
        # Missing name element
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = "100"
        ET.SubElement(bndbox, "ymin").text = "100"
        ET.SubElement(bndbox, "xmax").text = "200"
        ET.SubElement(bndbox, "ymax").text = "300"

        tree = ET.ElementTree(root)
        tree.write(anno_dir / "img001.xml")

        # Create train.txt
        with open(imageset_dir / "train.txt", 'w') as f:
            f.write("img001\n")

        # Mock cv2.imread
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        monkeypatch.setattr('cv2.imread', lambda path: mock_img)

        # Object without name should be skipped, resulting in no valid samples
        with pytest.raises(ValueError, match="No valid samples found"):
            VOCFallDataset(
                data_dirs=[str(tmp_path)],
                split="train",
                target_size=96,
                use_letterbox=True,
            )

    def test_xml_parse_error(self, tmp_path, monkeypatch):
        """Test VOCFallDataset handles XML parse errors gracefully."""
        # Create VOC structure
        image_ids = ["img001"]

        jpeg_dir, anno_dir, imageset_dir = self._create_voc_structure(
            tmp_path, image_ids, {}
        )

        # Create invalid XML
        with open(anno_dir / "img001.xml", 'w') as f:
            f.write("<invalid>xml content</unmatched>")

        # Create train.txt
        with open(imageset_dir / "train.txt", 'w') as f:
            f.write("img001\n")

        # Mock cv2.imread
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        monkeypatch.setattr('cv2.imread', lambda path: mock_img)

        # Invalid XML should be skipped, resulting in no valid samples
        with pytest.raises(ValueError, match="No valid samples found"):
            VOCFallDataset(
                data_dirs=[str(tmp_path)],
                split="train",
                target_size=96,
                use_letterbox=True,
            )

    def test_empty_dataset_raises_error(self, tmp_path):
        """Test VOCFallDataset raises error when no valid samples found."""
        # Create empty VOC structure
        jpeg_dir = tmp_path / "JPEGImages"
        anno_dir = tmp_path / "Annotations"
        imageset_dir = tmp_path / "ImageSets" / "Main"

        jpeg_dir.mkdir(parents=True)
        anno_dir.mkdir(parents=True)
        imageset_dir.mkdir(parents=True)

        # Create empty train.txt
        with open(imageset_dir / "train.txt", 'w') as f:
            pass

        # Should raise ValueError
        with pytest.raises(ValueError, match="No valid samples found"):
            VOCFallDataset(
                data_dirs=[str(tmp_path)],
                split="train",
                target_size=96,
                use_letterbox=True,
            )

    def test_missing_annotations_dir(self, tmp_path, monkeypatch):
        """Test VOCFallDataset handles missing annotations directory."""
        # Create VOC structure without Annotations directory
        jpeg_dir = tmp_path / "JPEGImages"
        imageset_dir = tmp_path / "ImageSets" / "Main"

        jpeg_dir.mkdir(parents=True)
        imageset_dir.mkdir(parents=True)

        # Create empty train.txt
        with open(imageset_dir / "train.txt", 'w') as f:
            pass

        # Mock cv2.imread
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        monkeypatch.setattr('cv2.imread', lambda path: mock_img)

        # Should raise ValueError (no valid samples)
        with pytest.raises(ValueError, match="No valid samples found"):
            VOCFallDataset(
                data_dirs=[str(tmp_path)],
                split="train",
                target_size=96,
                use_letterbox=True,
            )

    def test_single_string_data_dirs(self, tmp_path, monkeypatch):
        """Test VOCFallDataset accepts single string for data_dirs."""
        # Create VOC structure
        image_ids = ["img001"]
        annotations = {
            "img001": [{"name": "fall", "xmin": 100, "ymin": 100, "xmax": 200, "ymax": 300}],
        }

        jpeg_dir, anno_dir, imageset_dir = self._create_voc_structure(
            tmp_path, image_ids, annotations
        )

        with open(imageset_dir / "train.txt", 'w') as f:
            f.write("img001\n")

        # Mock cv2.imread
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        monkeypatch.setattr('cv2.imread', lambda path: mock_img)

        # Pass single string instead of list
        dataset = VOCFallDataset(
            data_dirs=str(tmp_path),  # Single string
            split="train",
            target_size=96,
            use_letterbox=True,
        )

        assert len(dataset) == 1

    def test_different_image_extensions(self, tmp_path, monkeypatch):
        """Test VOCFallDataset finds images with different extensions."""
        # Create VOC structure
        image_ids = ["img001"]

        jpeg_dir = tmp_path / "JPEGImages"
        anno_dir = tmp_path / "Annotations"
        imageset_dir = tmp_path / "ImageSets" / "Main"

        jpeg_dir.mkdir(parents=True)
        anno_dir.mkdir(parents=True)
        imageset_dir.mkdir(parents=True)

        # Create PNG image instead of JPG
        img = Image.new('RGB', (640, 480), color=(128, 128, 128))
        img.save(jpeg_dir / "img001.png")

        # Create annotation
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = "img001.png"
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "fall"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = "100"
        ET.SubElement(bndbox, "ymin").text = "100"
        ET.SubElement(bndbox, "xmax").text = "200"
        ET.SubElement(bndbox, "ymax").text = "300"
        tree = ET.ElementTree(root)
        tree.write(anno_dir / "img001.xml")

        with open(imageset_dir / "train.txt", 'w') as f:
            f.write("img001\n")

        # Mock cv2.imread
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        monkeypatch.setattr('cv2.imread', lambda path: mock_img)

        # Initialize dataset
        dataset = VOCFallDataset(
            data_dirs=[str(tmp_path)],
            split="train",
            target_size=96,
            use_letterbox=True,
        )

        assert len(dataset) == 1

    def test_no_letterbox_resize(self, tmp_path, monkeypatch):
        """Test VOCFallDataset without letterbox (direct resize)."""
        # Create VOC structure
        image_ids = ["img001"]
        annotations = {
            "img001": [{"name": "fall", "xmin": 100, "ymin": 100, "xmax": 200, "ymax": 300}],
        }

        jpeg_dir, anno_dir, imageset_dir = self._create_voc_structure(
            tmp_path, image_ids, annotations
        )

        with open(imageset_dir / "train.txt", 'w') as f:
            f.write("img001\n")

        # Mock cv2.imread
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        monkeypatch.setattr('cv2.imread', lambda path: mock_img)

        # Initialize dataset without letterbox
        dataset = VOCFallDataset(
            data_dirs=[str(tmp_path)],
            split="train",
            target_size=96,
            use_letterbox=False,
        )

        # Get item
        roi, label = dataset[0]

        # Verify output shape
        assert roi.shape == (3, 96, 96)

    def test_image_load_failure_voc(self, tmp_path, monkeypatch):
        """Test VOCFallDataset image load failure handling during __getitem__."""
        # Create VOC structure
        image_ids = ["img001"]
        annotations = {
            "img001": [{"name": "fall", "xmin": 100, "ymin": 100, "xmax": 200, "ymax": 300}],
        }

        jpeg_dir, anno_dir, imageset_dir = self._create_voc_structure(
            tmp_path, image_ids, annotations
        )

        with open(imageset_dir / "train.txt", 'w') as f:
            f.write("img001\n")

        # Mock cv2.imread to return None (simulating image load failure)
        monkeypatch.setattr('cv2.imread', lambda path: None)

        # Initialize dataset (this should succeed as it only parses annotations)
        dataset = VOCFallDataset(
            data_dirs=[str(tmp_path)],
            split="train",
            target_size=96,
            use_letterbox=True,
        )

        # Should raise ValueError when trying to load the image
        with pytest.raises(ValueError, match="Failed to load image"):
            _ = dataset[0]
