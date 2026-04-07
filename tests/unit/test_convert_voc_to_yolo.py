"""Tests for convert_voc_to_yolo.py script."""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

# Add scripts/tools to path for import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts' / 'tools'))

from convert_voc_to_yolo import (
    parse_voc_xml,
    load_config,
    get_class_mapping,
    get_output_config,
    read_imageset_split,
    get_image_path,
    get_xml_files_from_dir,
    convert_dataset_split,
    create_yolo_yaml,
)


@pytest.fixture
def sample_voc_xml(tmp_path):
    """Create a sample VOC XML annotation file."""
    xml_content = """<?xml version="1.0"?>
<annotation>
    <filename>test_image.jpg</filename>
    <size>
        <width>640</width>
        <height>480</height>
        <depth>3</depth>
    </size>
    <object>
        <name>person</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>100</ymin>
            <xmax>200</xmax>
            <ymax>300</ymax>
        </bndbox>
    </object>
    <object>
        <name>fall_down</name>
        <bndbox>
            <xmin>300</xmin>
            <ymin>350</ymin>
            <xmax>400</xmax>
            <ymax>450</ymax>
        </bndbox>
    </object>
</annotation>"""
    xml_path = tmp_path / 'test.xml'
    xml_path.write_text(xml_content)
    return xml_path


@pytest.fixture
def sample_voc_dataset(tmp_path):
    """Create a sample VOC dataset structure."""
    # Create directories
    ann_dir = tmp_path / 'Annotations'
    jpeg_dir = tmp_path / 'JPEGImages'
    imagesets_dir = tmp_path / 'ImageSets' / 'Main'
    ann_dir.mkdir(parents=True)
    jpeg_dir.mkdir(parents=True)
    imagesets_dir.mkdir(parents=True)

    # Create annotation files
    for i in range(5):
        xml_content = f"""<?xml version="1.0"?>
<annotation>
    <filename>image_{i:03d}.jpg</filename>
    <size>
        <width>640</width>
        <height>480</height>
        <depth>3</depth>
    </size>
    <object>
        <name>stand</name>
        <bndbox>
            <xmin>{100 + i * 10}</xmin>
            <ymin>100</ymin>
            <xmax>{200 + i * 10}</xmax>
            <ymax>300</ymax>
        </bndbox>
    </object>
</annotation>"""
        (ann_dir / f'image_{i:03d}.xml').write_text(xml_content)

        # Create dummy image files
        (jpeg_dir / f'image_{i:03d}.jpg').write_text('dummy')

    # Create ImageSets
    (imagesets_dir / 'train.txt').write_text('image_000\nimage_001\n')
    (imagesets_dir / 'val.txt').write_text('image_002\nimage_003\n')
    (imagesets_dir / 'test.txt').write_text('image_004\n')

    return tmp_path


@pytest.fixture
def sample_config(tmp_path, sample_voc_dataset):
    """Create a sample configuration dictionary."""
    return {
        'datasets': {
            'train_dirs': [str(sample_voc_dataset)],
            'val_dirs': [str(sample_voc_dataset)],
        },
        'class_mapping': {
            'stand': 'person',
            'sit': 'person',
            'fall_down': 'person',
        },
        'names': ['person'],
        'output': {
            'dir': str(tmp_path / 'output'),
            'images_dir': 'images',
            'labels_dir': 'labels',
            'yaml_path': str(tmp_path / 'output' / 'data.yaml'),
        },
        'use_imagesets': True,
        'copy_images': True,
    }


@pytest.fixture
def config_file(tmp_path, sample_voc_dataset):
    """Create a sample config file."""
    config = {
        'datasets': {
            'train_dirs': [str(sample_voc_dataset)],
            'val_dirs': [str(sample_voc_dataset)],
        },
        'class_mapping': {
            'stand': 'person',
            'fall_down': 'person',
        },
        'names': ['person'],
        'output': {
            'dir': str(tmp_path / 'output'),
            'images_dir': 'images',
            'labels_dir': 'labels',
            'yaml_path': str(tmp_path / 'output' / 'data.yaml'),
        },
        'use_imagesets': True,
        'copy_images': True,
    }
    config_path = tmp_path / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return config_path


class TestParseVocXml:
    """Tests for parse_voc_xml function."""

    def test_parse_valid_xml(self, sample_voc_xml):
        """Test parsing a valid VOC XML file."""
        boxes, width, height = parse_voc_xml(sample_voc_xml)

        assert width == 640
        assert height == 480
        assert len(boxes) == 2

        # Check first box (person)
        assert boxes[0]['class'] == 'person'
        assert 0 <= boxes[0]['x_center'] <= 1
        assert 0 <= boxes[0]['y_center'] <= 1
        assert 0 <= boxes[0]['width'] <= 1
        assert 0 <= boxes[0]['height'] <= 1

        # Check second box (fall_down)
        assert boxes[1]['class'] == 'fall_down'

    def test_parse_empty_annotation(self, tmp_path):
        """Test parsing XML with no objects."""
        xml_content = """<?xml version="1.0"?>
<annotation>
    <filename>empty.jpg</filename>
    <size>
        <width>100</width>
        <height>100</height>
    </size>
</annotation>"""
        xml_path = tmp_path / 'empty.xml'
        xml_path.write_text(xml_content)

        boxes, width, height = parse_voc_xml(xml_path)
        assert len(boxes) == 0
        assert width == 100
        assert height == 100

    def test_normalized_coordinates(self, sample_voc_xml):
        """Test that coordinates are properly normalized."""
        boxes, width, height = parse_voc_xml(sample_voc_xml)

        for box in boxes:
            assert 0 <= box['x_center'] <= 1
            assert 0 <= box['y_center'] <= 1
            assert 0 <= box['width'] <= 1
            assert 0 <= box['height'] <= 1


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self, config_file):
        """Test loading a valid config file."""
        config = load_config(config_file)

        assert 'datasets' in config
        assert 'class_mapping' in config
        assert config['class_mapping']['stand'] == 'person'

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading a non-existent config file."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / 'nonexistent.yaml')


class TestGetClassMapping:
    """Tests for get_class_mapping function."""

    def test_basic_mapping(self):
        """Test basic class mapping."""
        config = {
            'class_mapping': {
                'stand': 'person',
                'sit': 'person',
                'fall_down': 'fall',
            },
            'names': ['person', 'fall'],
        }
        voc_to_yolo, yolo_to_id = get_class_mapping(config)

        assert voc_to_yolo['stand'] == 'person'
        assert voc_to_yolo['sit'] == 'person'
        assert voc_to_yolo['fall_down'] == 'fall'

        assert yolo_to_id['person'] == 0
        assert yolo_to_id['fall'] == 1

    def test_no_names_list(self):
        """Test extracting names from class_mapping values."""
        config = {
            'class_mapping': {
                'stand': 'person',
                'sit': 'person',
                'fall_down': 'fall',
            },
        }
        voc_to_yolo, yolo_to_id = get_class_mapping(config)

        # Should extract unique values and sort them
        assert 'person' in yolo_to_id
        assert 'fall' in yolo_to_id

    def test_case_insensitive(self):
        """Test that VOC class names are case-insensitive."""
        config = {
            'class_mapping': {
                'Stand': 'person',
                'SIT': 'person',
            },
            'names': ['person'],
        }
        voc_to_yolo, _ = get_class_mapping(config)

        assert voc_to_yolo['stand'] == 'person'
        assert voc_to_yolo['sit'] == 'person'


class TestGetOutputConfig:
    """Tests for get_output_config function."""

    def test_default_output_config(self):
        """Test default output configuration."""
        config = {}
        output_cfg = get_output_config(config)

        assert output_cfg['output_dir'] == Path('data/yolo')
        assert output_cfg['images_dir'] == 'images'
        assert output_cfg['labels_dir'] == 'labels'

    def test_custom_output_config(self):
        """Test custom output configuration."""
        config = {
            'output': {
                'dir': 'custom/output',
                'images_dir': 'imgs',
                'labels_dir': 'lbls',
                'yaml_path': 'custom/data.yaml',
            }
        }
        output_cfg = get_output_config(config)

        assert output_cfg['output_dir'] == Path('custom/output')
        assert output_cfg['images_dir'] == 'imgs'
        assert output_cfg['labels_dir'] == 'lbls'
        assert output_cfg['yaml_path'] == Path('custom/data.yaml')


class TestReadImagesetSplit:
    """Tests for read_imageset_split function."""

    def test_read_train_split(self, sample_voc_dataset):
        """Test reading train split from ImageSets."""
        image_ids = read_imageset_split(sample_voc_dataset, 'train')

        assert image_ids is not None
        assert 'image_000' in image_ids
        assert 'image_001' in image_ids
        assert 'image_002' not in image_ids

    def test_read_val_split(self, sample_voc_dataset):
        """Test reading val split from ImageSets."""
        image_ids = read_imageset_split(sample_voc_dataset, 'val')

        assert image_ids is not None
        assert 'image_002' in image_ids
        assert 'image_003' in image_ids

    def test_read_missing_split(self, sample_voc_dataset):
        """Test reading a non-existent split file."""
        image_ids = read_imageset_split(sample_voc_dataset, 'nonexistent')
        assert image_ids is None

    def test_read_nonexistent_dataset(self, tmp_path):
        """Test reading from a directory without ImageSets."""
        image_ids = read_imageset_split(tmp_path, 'train')
        assert image_ids is None

    def test_handle_annotated_format(self, tmp_path):
        """Test handling ImageSets with +/- 1 annotations."""
        imagesets_dir = tmp_path / 'ImageSets' / 'Main'
        imagesets_dir.mkdir(parents=True)

        # Format: "image_id 1" for positive, "image_id -1" for negative
        (imagesets_dir / 'train.txt').write_text(
            'image_001 1\nimage_002 -1\nimage_003 1\n'
        )

        image_ids = read_imageset_split(tmp_path, 'train')
        assert 'image_001' in image_ids
        assert 'image_002' in image_ids  # Still included
        assert 'image_003' in image_ids


class TestGetImagePath:
    """Tests for get_image_path function."""

    def test_find_jpg(self, tmp_path):
        """Test finding .jpg image."""
        (tmp_path / 'test.jpg').write_text('dummy')
        result = get_image_path(tmp_path, 'test')
        assert result == tmp_path / 'test.jpg'

    def test_find_png(self, tmp_path):
        """Test finding .png image."""
        (tmp_path / 'test.png').write_text('dummy')
        result = get_image_path(tmp_path, 'test')
        assert result == tmp_path / 'test.png'

    def test_find_preference_order(self, tmp_path):
        """Test that .jpg is preferred over other formats."""
        (tmp_path / 'test.png').write_text('dummy')
        (tmp_path / 'test.jpg').write_text('dummy')
        result = get_image_path(tmp_path, 'test')
        assert result == tmp_path / 'test.jpg'

    def test_not_found(self, tmp_path):
        """Test when image is not found."""
        result = get_image_path(tmp_path, 'nonexistent')
        assert result is None


class TestGetXmlFilesFromDir:
    """Tests for get_xml_files_from_dir function."""

    def test_get_xml_files(self, sample_voc_dataset):
        """Test getting all XML files from directory."""
        xml_files = get_xml_files_from_dir(sample_voc_dataset)

        assert len(xml_files) == 5
        assert 'image_000' in xml_files
        assert 'image_001' in xml_files
        assert xml_files['image_000'].exists()

    def test_empty_directory(self, tmp_path):
        """Test empty Annotations directory."""
        ann_dir = tmp_path / 'Annotations'
        ann_dir.mkdir()

        xml_files = get_xml_files_from_dir(tmp_path)
        assert len(xml_files) == 0

    def test_no_annotations_dir(self, tmp_path):
        """Test when Annotations directory doesn't exist."""
        xml_files = get_xml_files_from_dir(tmp_path)
        assert len(xml_files) == 0


class TestConvertDatasetSplit:
    """Tests for convert_dataset_split function."""

    def test_convert_with_imagesets(self, sample_voc_dataset, tmp_path):
        """Test conversion using ImageSets for splits."""
        output_dir = tmp_path / 'output'
        voc_to_yolo = {'stand': 'person'}
        yolo_to_id = {'person': 0}

        converted, skipped = convert_dataset_split(
            data_dirs=[sample_voc_dataset],
            output_dir=output_dir,
            split_name='train',
            voc_to_yolo_name=voc_to_yolo,
            yolo_name_to_id=yolo_to_id,
            use_imagesets=True,
            copy_images=True,
        )

        assert converted == 2  # image_000 and image_001
        assert skipped == 0

        # Check output structure
        assert (output_dir / 'labels' / 'train').exists()
        assert (output_dir / 'images' / 'train').exists()
        assert (output_dir / 'labels' / 'train' / 'image_000.txt').exists()
        assert (output_dir / 'labels' / 'train' / 'image_001.txt').exists()

    def test_convert_without_imagesets(self, sample_voc_dataset, tmp_path):
        """Test conversion without ImageSets (all images)."""
        output_dir = tmp_path / 'output'
        voc_to_yolo = {'stand': 'person'}
        yolo_to_id = {'person': 0}

        converted, skipped = convert_dataset_split(
            data_dirs=[sample_voc_dataset],
            output_dir=output_dir,
            split_name='train',
            voc_to_yolo_name=voc_to_yolo,
            yolo_name_to_id=yolo_to_id,
            use_imagesets=False,
            copy_images=True,
        )

        assert converted == 5  # All 5 images

    def test_unknown_class_warning(self, sample_voc_dataset, tmp_path, capsys):
        """Test warning for unknown classes."""
        output_dir = tmp_path / 'output'
        # Only map 'unknown_class', not 'stand'
        voc_to_yolo = {'unknown_class': 'unknown'}
        yolo_to_id = {'unknown': 99}

        convert_dataset_split(
            data_dirs=[sample_voc_dataset],
            output_dir=output_dir,
            split_name='train',
            voc_to_yolo_name=voc_to_yolo,
            yolo_name_to_id=yolo_to_id,
            use_imagesets=True,
            copy_images=True,
        )

        captured = capsys.readouterr()
        assert "not in class_mapping" in captured.out or "not in class_mapping" in captured.err

    def test_convert_multiple_dirs(self, sample_voc_dataset, tmp_path):
        """Test conversion from multiple source directories."""
        # Create second dataset
        dataset2 = tmp_path / 'dataset2'
        ann_dir = dataset2 / 'Annotations'
        jpeg_dir = dataset2 / 'JPEGImages'
        imagesets_dir = dataset2 / 'ImageSets' / 'Main'
        ann_dir.mkdir(parents=True)
        jpeg_dir.mkdir(parents=True)
        imagesets_dir.mkdir(parents=True)

        # Create 2 annotations
        for i in range(2):
            xml_content = f"""<?xml version="1.0"?>
<annotation>
    <filename>img_{i}.jpg</filename>
    <size><width>100</width><height>100</height></size>
    <object><name>stand</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>20</xmax><ymax>20</ymax></bndbox></object>
</annotation>"""
            (ann_dir / f'img_{i}.xml').write_text(xml_content)
            (jpeg_dir / f'img_{i}.jpg').write_text('dummy')

        (imagesets_dir / 'train.txt').write_text('img_0\nimg_1\n')

        output_dir = tmp_path / 'output'
        voc_to_yolo = {'stand': 'person'}
        yolo_to_id = {'person': 0}

        converted, skipped = convert_dataset_split(
            data_dirs=[sample_voc_dataset, dataset2],
            output_dir=output_dir,
            split_name='train',
            voc_to_yolo_name=voc_to_yolo,
            yolo_name_to_id=yolo_to_id,
            use_imagesets=True,
            copy_images=True,
        )

        # sample_voc_dataset: 2 train images
        # dataset2: 2 train images
        assert converted == 4


class TestCreateYoloYaml:
    """Tests for create_yolo_yaml function."""

    def test_create_yaml(self, tmp_path):
        """Test creating data.yaml file."""
        config = {
            'names': ['person', 'fall'],
        }
        output_cfg = {
            'output_dir': tmp_path / 'output',
            'images_dir': 'images',
            'labels_dir': 'labels',
            'yaml_path': tmp_path / 'output' / 'data.yaml',
        }
        output_cfg['output_dir'].mkdir()

        yaml_path = create_yolo_yaml(config, output_cfg)

        assert yaml_path.exists()

        # Read and verify content
        with open(yaml_path, 'r') as f:
            content = f.read()

        assert 'person' in content
        assert 'fall' in content
        assert 'nc: 2' in content
        assert str(output_cfg['output_dir'].absolute()) in content

    def test_extract_names_from_class_mapping(self, tmp_path):
        """Test extracting names from class_mapping when names not provided."""
        config = {
            'class_mapping': {
                'stand': 'person',
                'fall_down': 'fall',
            },
            # No 'names' list
        }
        output_cfg = {
            'output_dir': tmp_path / 'output',
            'images_dir': 'images',
            'labels_dir': 'labels',
            'yaml_path': tmp_path / 'output' / 'data.yaml',
        }
        output_cfg['output_dir'].mkdir()

        yaml_path = create_yolo_yaml(config, output_cfg)

        with open(yaml_path, 'r') as f:
            content = f.read()

        assert 'person' in content
        assert 'fall' in content


class TestIntegration:
    """Integration tests for the full conversion pipeline."""

    def test_full_conversion_with_config(self, sample_voc_dataset, tmp_path):
        """Test full conversion using a config file."""
        # Create config
        config = {
            'datasets': {
                'train_dirs': [str(sample_voc_dataset)],
                'val_dirs': [str(sample_voc_dataset)],
            },
            'class_mapping': {
                'stand': 'person',
                'sit': 'person',
                'fall_down': 'person',
            },
            'names': ['person'],
            'output': {
                'dir': str(tmp_path / 'yolo_output'),
                'images_dir': 'images',
                'labels_dir': 'labels',
                'yaml_path': str(tmp_path / 'yolo_output' / 'data.yaml'),
            },
            'use_imagesets': True,
            'copy_images': True,
        }

        # Save config
        config_path = tmp_path / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Load and process
        loaded_config = load_config(config_path)
        voc_to_yolo, yolo_to_id = get_class_mapping(loaded_config)
        output_cfg = get_output_config(loaded_config)

        output_cfg['output_dir'].mkdir(parents=True, exist_ok=True)

        # Convert train split
        train_dirs = [Path(d) for d in loaded_config['datasets']['train_dirs']]
        train_conv, train_skip = convert_dataset_split(
            data_dirs=train_dirs,
            output_dir=output_cfg['output_dir'],
            split_name='train',
            voc_to_yolo_name=voc_to_yolo,
            yolo_name_to_id=yolo_to_id,
            use_imagesets=True,
            copy_images=True,
        )

        # Convert val split
        val_dirs = [Path(d) for d in loaded_config['datasets']['val_dirs']]
        val_conv, val_skip = convert_dataset_split(
            data_dirs=val_dirs,
            output_dir=output_cfg['output_dir'],
            split_name='val',
            voc_to_yolo_name=voc_to_yolo,
            yolo_name_to_id=yolo_to_id,
            use_imagesets=True,
            copy_images=True,
        )

        # Verify results
        assert train_conv == 2  # image_000, image_001
        assert val_conv == 2    # image_002, image_003

        # Verify label file content
        label_file = output_cfg['output_dir'] / 'labels' / 'train' / 'image_000.txt'
        assert label_file.exists()

        with open(label_file, 'r') as f:
            content = f.read().strip()
        parts = content.split()
        assert parts[0] == '0'  # person -> class 0

        # Create YAML and verify
        yaml_path = create_yolo_yaml(loaded_config, output_cfg)
        assert yaml_path.exists()

        with open(yaml_path, 'r') as f:
            yaml_content = f.read()
        assert 'person' in yaml_content
        assert 'nc: 1' in yaml_content

    def test_label_file_format(self, sample_voc_dataset, tmp_path):
        """Test that label files have correct YOLO format."""
        output_dir = tmp_path / 'output'
        voc_to_yolo = {'stand': 'custom_class'}
        yolo_to_id = {'custom_class': 3}

        convert_dataset_split(
            data_dirs=[sample_voc_dataset],
            output_dir=output_dir,
            split_name='train',
            voc_to_yolo_name=voc_to_yolo,
            yolo_name_to_id=yolo_to_id,
            use_imagesets=True,
            copy_images=True,
        )

        label_file = output_dir / 'labels' / 'train' / 'image_000.txt'
        with open(label_file, 'r') as f:
            line = f.read().strip()

        parts = line.split()
        assert len(parts) == 5  # class_id, x, y, w, h

        class_id = int(parts[0])
        coords = [float(p) for p in parts[1:]]

        assert class_id == 3
        assert all(0 <= c <= 1 for c in coords)

    def test_yolo_world_style_mapping(self, sample_voc_dataset, tmp_path):
        """Test YOLO-World style mapping where all classes map to one."""
        config = {
            'datasets': {
                'train_dirs': [str(sample_voc_dataset)],
            },
            'class_mapping': {
                'stand': 'person',
                'sit': 'person',
                'squat': 'person',
                'bend': 'person',
                'half_up': 'person',
                'kneel': 'person',
                'crawl': 'person',
                'fall_down': 'person',
            },
            'names': ['person'],
            'output': {
                'dir': str(tmp_path / 'output'),
                'images_dir': 'images',
                'labels_dir': 'labels',
                'yaml_path': str(tmp_path / 'output' / 'data.yaml'),
            },
            'use_imagesets': True,
            'copy_images': True,
        }

        voc_to_yolo, yolo_to_id = get_class_mapping(config)
        output_cfg = get_output_config(config)
        output_cfg['output_dir'].mkdir(parents=True, exist_ok=True)

        train_dirs = [Path(d) for d in config['datasets']['train_dirs']]
        converted, _ = convert_dataset_split(
            data_dirs=train_dirs,
            output_dir=output_cfg['output_dir'],
            split_name='train',
            voc_to_yolo_name=voc_to_yolo,
            yolo_name_to_id=yolo_to_id,
            use_imagesets=True,
            copy_images=True,
        )

        assert converted == 2

        # All boxes should be class 0 (person)
        label_file = output_cfg['output_dir'] / 'labels' / 'train' / 'image_000.txt'
        with open(label_file, 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                assert class_id == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
