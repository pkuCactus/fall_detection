#!/usr/bin/env python3
"""Pipeline详细调试脚本 - 输出每一步的中间结果到日志."""

import argparse
import sys
import logging
import os
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, "src")
from fall_detection.pipeline import FallDetectionPipeline
from fall_detection.utils import draw_results


def setup_logger(log_file):
    """设置详细日志."""
    logger = logging.getLogger('fall_detection')
    logger.setLevel(logging.DEBUG)

    # 文件处理器
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def main():
    parser = argparse.ArgumentParser(description="Fall Detection Pipeline Debug")
    parser.add_argument("--video", default="data/fall_test.mp4", help="Video path")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--log", default="logs/pipeline_debug.log", help="Log file path")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()

    # 创建日志目录
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    # 设置日志
    logger = setup_logger(args.log)
    logger.info("=" * 80)
    logger.info("Fall Detection Pipeline Debug Session")
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"Video: {args.video}")
    logger.info(f"Config: {args.config}")
    logger.info("=" * 80)

    # 初始化pipeline
    logger.info("Initializing pipeline...")
    pipeline = FallDetectionPipeline(args.config)
    logger.info(f"  - Trigger threshold: {pipeline.trigger_thresh}")
    logger.info(f"  - Skip frames: {pipeline.skip_frames}")
    logger.info(f"  - FPS: {pipeline.fps}")

    # 打开视频
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video info: {w}x{h} @ {fps:.1f}fps, {total_frames} frames")

    # 准备输出
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
        logger.info(f"Output video: {args.output}")

    frame_idx = 0
    fall_detected_count = 0

    try:
        while True:
            if args.max_frames and frame_idx >= args.max_frames:
                logger.info(f"Reached max frames limit: {args.max_frames}")
                break

            ret, frame = cap.read()
            if not ret:
                logger.info("End of video")
                break

            # 记录帧信息
            logger.info(f"\n{'='*60}")
            logger.info(f"FRAME {frame_idx}")
            logger.info(f"{'='*60}")

            # 是否运行检测
            run_detection = (pipeline._frame_counter % (pipeline.skip_frames + 1)) == 0
            logger.info(f"[1] Detection phase: {'YES' if run_detection else 'SKIP (use cached)'}")

            # 处理帧
            results = pipeline.process_frame(frame)

            # 记录检测结果
            tracks = results.get("tracks", [])
            logger.info(f"[2] Tracking: {len(tracks)} active tracks")

            for track in tracks:
                tid = track.track_id
                bbox = track.to_tlbr()
                state = getattr(track, 'state', 'unknown')
                hits = getattr(track, 'hits', 0)

                logger.info(f"    Track {tid}:")
                logger.info(f"      - BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                logger.info(f"      - State: {state}, Hits: {hits}")
                logger.info(f"      - Center: ({bbox[0]+(bbox[2]-bbox[0])/2:.1f}, {bbox[1]+(bbox[3]-bbox[1])/2:.1f})")

            # 记录关键点检测结果
            track_kpts = results.get("track_kpts", {})
            logger.info(f"[3] Pose estimation:")
            for tid, kpts in track_kpts.items():
                visible = sum(1 for k in kpts if k[2] > 0.1)
                logger.info(f"    Track {tid}: {visible}/17 keypoints visible")

                # 详细记录每个关键点
                names = ['nose', 'leye', 'reye', 'lear', 'rear', 'lsho', 'rsho',
                        'lelb', 'relb', 'lwri', 'rwri', 'lhip', 'rhip', 'lkne', 'rkne', 'lank', 'rank']
                for i, (x, y, c) in enumerate(kpts):
                    if c > 0.1:
                        logger.debug(f"      - {names[i]}: ({x:.1f}, {y:.1f}, conf={c:.3f})")

            # 记录规则判定情况
            track_scores = results.get("track_scores", {})
            logger.info(f"[4] Rule engine evaluation:")
            for tid, scores in track_scores.items():
                rule_score = scores.get('rule', 0)
                triggered = rule_score >= pipeline.trigger_thresh
                logger.info(f"    Track {tid}:")
                logger.info(f"      - Rule score: {rule_score:.3f} (threshold: {pipeline.trigger_thresh})")
                logger.info(f"      - Trigger classifier: {'YES' if triggered else 'NO'}")

            # 记录分类器结果
            logger.info(f"[5] Classifier evaluation:")
            for tid, scores in track_scores.items():
                cls_score = scores.get('cls', 0)
                logger.info(f"    Track {tid}: cls_score={cls_score:.3f}")

            # 记录融合决策
            track_falling = results.get("track_falling", {})
            logger.info(f"[6] Fusion decision:")
            for tid, scores in track_scores.items():
                final_score = scores.get('final', 0)
                is_falling = track_falling.get(tid, False)
                logger.info(f"    Track {tid}:")
                logger.info(f"      - Final score: {final_score:.3f}")
                logger.info(f"      - Is falling: {is_falling}")

                if is_falling:
                    fall_detected_count += 1

            # 绘制结果
            frame = draw_results(
                frame, tracks, track_kpts, track_scores, track_falling
            )
            cv2.putText(frame, f"Frame: {frame_idx}", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if writer:
                writer.write(frame)

            frame_idx += 1

            # 每30帧打印一次进度摘要
            if frame_idx % 30 == 0:
                logger.info(f"\n[PROGRESS] Processed {frame_idx} frames, detected {fall_detected_count} fall frames")

    finally:
        cap.release()
        if writer:
            writer.release()

        logger.info("=" * 80)
        logger.info(f"Session complete")
        logger.info(f"Total frames processed: {frame_idx}")
        logger.info(f"Fall detected in {fall_detected_count} frames")
        logger.info(f"Log saved to: {args.log}")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
