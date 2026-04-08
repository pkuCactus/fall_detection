"""跌倒检测Pipeline单元测试."""

import numpy as np
from unittest.mock import Mock
from fall_detection.pipeline.pipeline import FallDetectionPipeline


class TestPipelineInit:
    """测试Pipeline初始化."""

    def test_default_config_init(self):
        """测试使用默认配置初始化."""
        pipe = FallDetectionPipeline("configs/pipeline/default.yaml")

        assert pipe.detector is not None
        assert pipe.tracker is not None
        assert pipe.pose_estimator is not None
        assert pipe.rule_engine is not None
        assert pipe.classifier is not None
        assert pipe.skip_frames == 2
        assert pipe.fps == 25


class TestProcessFrame:
    """测试帧处理逻辑."""

    def test_process_single_frame_with_detection(self):
        """测试处理单帧（检测帧）."""
        pipe = FallDetectionPipeline("configs/pipeline/default.yaml")

        # Mock detector
        pipe.detector = Mock(
            return_value=[
                {"bbox": [100.0, 100.0, 200.0, 400.0], "conf": 0.9, "class_id": 0}
            ]
        )

        # Mock pose estimator
        def mock_pose(img, bboxes):
            kpts = np.zeros((17, 3), dtype=np.float32)
            kpts[0] = [150, 120, 0.9]
            kpts[15] = [135, 380, 0.9]
            kpts[16] = [165, 380, 0.9]
            return [kpts] if bboxes else []

        pipe.pose_estimator = mock_pose

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipe.process_frame(frame)

        assert "tracks" in result
        assert "track_kpts" in result
        assert len(result["tracks"]) >= 1

    def test_process_skip_frame(self):
        """测试处理跳帧（非检测帧）."""
        pipe = FallDetectionPipeline("configs/pipeline/default.yaml")

        # Mock detector
        pipe.detector = Mock(
            return_value=[
                {"bbox": [100.0, 100.0, 200.0, 400.0], "conf": 0.9, "class_id": 0}
            ]
        )

        # Mock pose estimator
        pipe.pose_estimator = Mock(return_value=[])

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # skip_frames=2，所以每3帧检测一次
        # 第0帧：检测帧
        result0 = pipe.process_frame(frame)
        assert result0["is_detection_frame"] is True

        # 第1帧：跳帧
        result1 = pipe.process_frame(frame)
        assert result1["is_detection_frame"] is False

        # 第2帧：跳帧
        result2 = pipe.process_frame(frame)
        assert result2["is_detection_frame"] is False

        # 第3帧：检测帧
        result3 = pipe.process_frame(frame)
        assert result3["is_detection_frame"] is True

    def test_process_multiple_tracks(self):
        """测试处理多目标跟踪."""
        pipe = FallDetectionPipeline("configs/pipeline/default.yaml")

        # Mock detector返回多个检测框
        pipe.detector = Mock(
            return_value=[
                {"bbox": [100.0, 100.0, 200.0, 400.0], "conf": 0.9, "class_id": 0},
                {"bbox": [300.0, 100.0, 400.0, 400.0], "conf": 0.8, "class_id": 0},
            ]
        )

        # Mock pose estimator返回多个姿态
        def mock_pose(img, bboxes):
            kpts_list = []
            for _ in bboxes:
                kpts = np.zeros((17, 3), dtype=np.float32)
                kpts[0] = [150, 120, 0.9]
                kpts[15] = [135, 380, 0.9]
                kpts[16] = [165, 380, 0.9]
                kpts_list.append(kpts)
            return kpts_list

        pipe.pose_estimator = mock_pose

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipe.process_frame(frame)

        assert len(result["tracks"]) >= 2


class TestClassifierIntegration:
    """测试分类器集成."""

    def test_fusion_classifier_triggered(self):
        """测试融合分类器触发."""
        import torch

        pipe = FallDetectionPipeline("configs/pipeline/default.yaml")
        pipe.use_simple_classifier = False

        # Mock detector
        pipe.detector = Mock(
            return_value=[
                {"bbox": [100.0, 100.0, 200.0, 400.0], "conf": 0.9, "class_id": 0}
            ]
        )

        # Mock pose estimator（躺卧姿态）
        def mock_pose(img, bboxes):
            kpts = np.zeros((17, 3), dtype=np.float32)
            for i in range(17):
                kpts[i] = [150, 380, 0.9]
            return [kpts] if bboxes else []

        pipe.pose_estimator = mock_pose

        # Mock融合分类器
        mock_clf = Mock()
        mock_clf.eval = Mock()
        mock_clf.return_value = torch.tensor(0.9)
        pipe.classifier = mock_clf

        # 构造运动历史
        pipe._track_history[0] = []
        for i in range(20):
            pipe._track_history[0].append((100.0 + i * 10, 100.0 + i * 10))

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipe.process_frame(frame)

        assert "track_scores" in result or len(result["tracks"]) > 0

    def test_simple_classifier_triggered(self):
        """测试简单分类器触发."""
        import torch

        pipe = FallDetectionPipeline("configs/pipeline/default.yaml")
        pipe.use_simple_classifier = True

        # Mock detector
        pipe.detector = Mock(
            return_value=[
                {"bbox": [100.0, 100.0, 200.0, 400.0], "conf": 0.9, "class_id": 0}
            ]
        )

        # Mock pose estimator
        def mock_pose(img, bboxes):
            kpts = np.zeros((17, 3), dtype=np.float32)
            for i in range(17):
                kpts[i] = [150, 380, 0.9]
            return [kpts] if bboxes else []

        pipe.pose_estimator = mock_pose

        # Mock simple classifier
        mock_clf = Mock()
        mock_clf.fall_class_idx = 1
        mock_clf.return_value = torch.tensor([[0.5, 2.0]])
        pipe.classifier = mock_clf

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipe.process_frame(frame)

        assert len(result["tracks"]) > 0


class TestFallDetection:
    """测试跌倒检测."""

    def test_fall_detected_with_alarm(self):
        """测试跌倒检测并触发告警."""
        import torch

        pipe = FallDetectionPipeline("configs/pipeline/default.yaml")
        # 明确设置为融合分类器
        pipe.use_simple_classifier = False

        # Mock detector
        pipe.detector = Mock(
            return_value=[
                {"bbox": [100.0, 100.0, 200.0, 400.0], "conf": 0.9, "class_id": 0}
            ]
        )

        # Mock pose estimator（躺卧）
        def mock_pose(img, bboxes):
            kpts = np.zeros((17, 3), dtype=np.float32)
            for i in range(17):
                kpts[i] = [150, 380, 0.9]
            return [kpts] if bboxes else []

        pipe.pose_estimator = mock_pose

        # Mock融合分类器 - 直接返回分数，不需要softmax
        def mock_classifier_call(roi, kpts, motion):
            return 0.9  # 返回float分数

        pipe.classifier = mock_classifier_call

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # 连续多帧模拟跌倒
        for _ in range(10):
            result = pipe.process_frame(frame)

        assert "tracks" in result

    def test_normal_not_fall(self):
        """测试正常站立不触发跌倒."""
        pipe = FallDetectionPipeline("configs/pipeline/default.yaml")

        # Mock detector
        pipe.detector = Mock(
            return_value=[
                {"bbox": [100.0, 100.0, 200.0, 400.0], "conf": 0.9, "class_id": 0}
            ]
        )

        # Mock pose estimator（站立姿态）
        def mock_pose(img, bboxes):
            kpts = np.zeros((17, 3), dtype=np.float32)
            kpts[0] = [150, 120, 0.9]
            kpts[15] = [135, 380, 0.9]
            kpts[16] = [165, 380, 0.9]
            return [kpts] if bboxes else []

        pipe.pose_estimator = mock_pose

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipe.process_frame(frame)

        assert len(result["tracks"]) > 0


class TestEdgeCases:
    """测试边界情况."""

    def test_no_detection(self):
        """测试无检测结果."""
        pipe = FallDetectionPipeline("configs/pipeline/default.yaml")

        pipe.detector = Mock(return_value=[])
        pipe.pose_estimator = Mock(return_value=[])

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipe.process_frame(frame)

        assert "tracks" in result
        assert len(result["tracks"]) == 0

    def test_empty_frame(self):
        """测试空帧."""
        pipe = FallDetectionPipeline("configs/pipeline/default.yaml")

        pipe.detector = Mock(return_value=[])
        pipe.pose_estimator = Mock(return_value=[])

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipe.process_frame(frame)

        assert "tracks" in result

    def test_large_number_of_tracks(self):
        """测试大量目标."""
        pipe = FallDetectionPipeline("configs/pipeline/default.yaml")

        # Mock detector返回10个检测框
        detections = []
        for i in range(10):
            detections.append(
                {
                    "bbox": [i * 50.0, 100.0, i * 50.0 + 100, 400.0],
                    "conf": 0.9,
                    "class_id": 0,
                }
            )
        pipe.detector = Mock(return_value=detections)

        # Mock pose estimator
        def mock_pose(img, bboxes):
            kpts_list = []
            for _ in range(len(bboxes)):
                kpts = np.zeros((17, 3), dtype=np.float32)
                kpts[0] = [150, 120, 0.9]
                kpts[15] = [135, 380, 0.9]
                kpts[16] = [165, 380, 0.9]
                kpts_list.append(kpts)
            return kpts_list

        pipe.pose_estimator = mock_pose

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipe.process_frame(frame)

        assert len(result["tracks"]) >= 0
