#!/usr/bin/env python3
"""Pipeline演示脚本 - 根据配置自动选择标准或YOLO-World跌倒检测Pipeline."""

import argparse
import sys
import logging
import os
from datetime import datetime

import cv2
import numpy as np
import yaml

sys.path.insert(0, "src")
from fall_detection.pipeline.pipeline import FallDetectionPipeline
from fall_detection.pipeline.yolo_world_pipeline import YOLOWorldFallPipeline
from fall_detection.utils.visualization import draw_results


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


def _format_track_log(track, run_detection: bool) -> str:
    """格式化单条跟踪轨迹日志."""
    tid = track.track_id
    bbox = track.to_tlbr()
    state = getattr(track, 'state', 'unknown')
    time_since = getattr(track, 'time_since_update', 0)
    bbox_str = f"[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]"
    if run_detection:
        hits = getattr(track, 'hits', 0)
        status = "matched" if time_since == 0 else f"predicted({time_since}f)"
        return f"    [Track {tid}] BBox={bbox_str}, State={state}, Hits={hits}, Status={status}"
    return f"    [Track {tid}] BBox={bbox_str}, State={state}, Predicted={time_since}f"


def log_frame_details(logger, pipeline, frame_idx, results, is_yolo_world: bool = False):
    """记录帧处理详细信息."""
    logger.info(f"\n{'='*60}")
    logger.info(f"FRAME {frame_idx}")
    logger.info(f"{'='*60}")

    # 是否运行检测（使用pipeline返回的结果，避免counter已递增的问题）
    run_detection = results.get("is_detection_frame", True)
    phase_label = "DETECTION" if run_detection else "TRACKING (skip frame)"
    logger.info(f"[1] Phase: {phase_label}")

    # 检测帧：分开打印检测框和跟踪框
    if run_detection:
        # 原始检测框（YOLO输出）
        detections = results.get("detections", [])
        logger.info(f"[2] Detections (YOLO): {len(detections)} found")
        for i, bbox in enumerate(detections):
            logger.info(f"    [Det {i+1}] BBox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")

        # 跟踪框（关联后的轨迹）
        tracks = results.get("tracks", [])
        logger.info(f"[3] Tracks (after matching): {len(tracks)} active")
        for track in tracks:
            logger.info(_format_track_log(track, run_detection=True))
    else:
        # 跳帧：只打印跟踪预测的框
        tracks = results.get("tracks", [])
        logger.info(f"[2] Tracks (predicted): {len(tracks)} active")
        for track in tracks:
            logger.info(_format_track_log(track, run_detection=False))

    # 关键点检测（仅标准pipeline有）
    track_kpts = results.get("track_kpts", {})
    if not is_yolo_world and track_kpts:
        pose_idx = "[4]" if run_detection else "[3]"
        pose_label = "Pose (detected)" if run_detection else "Pose (cached)"
        logger.info(f"{pose_idx} {pose_label}:")
        for tid, kpts in track_kpts.items():
            visible = sum(1 for k in kpts if k[2] > 0.1)
            tag = "[Detection]" if run_detection else "[Cached]"
            logger.info(f"    {tag} Track {tid}: {visible}/17 keypoints visible")

            names = ['nose', 'leye', 'reye', 'lear', 'rear', 'lsho', 'rsho',
                    'lelb', 'relb', 'lwri', 'rwri', 'lhip', 'rhip', 'lkne', 'rkne', 'lank', 'rank']
            for i, (x, y, c) in enumerate(kpts):
                if c > 0.1:
                    logger.debug(f"      - {names[i]}: ({x:.1f}, {y:.1f}, conf={c:.3f})")

    # 分类器结果 / 检测类别结果
    track_scores = results.get("track_scores", {})
    cls_idx = "[5]" if (run_detection and not is_yolo_world) else ("[4]" if (run_detection and is_yolo_world) else "[3]")
    logger.info(f"{cls_idx} Classifier (per-frame):")
    for tid, scores in track_scores.items():
        cls_score = scores.get('cls', 0)
        debug = scores.get('debug', {})
        if is_yolo_world:
            logger.info(f"    Track {tid}: cls_score={cls_score:.3f}, class={debug.get('class_name', 'unknown')}")
        else:
            logger.info(f"    Track {tid}: cls_score={cls_score:.3f}")

    # 规则判定（已引入分类器得分做姿态辅助修正）
    rule_idx = "[6]" if (run_detection and not is_yolo_world) else ("[5]" if (run_detection and is_yolo_world) else "[4]")
    logger.info(f"{rule_idx} Rule engine evaluation:")
    for tid, scores in track_scores.items():
        rule_score = scores.get('rule', 0)
        flags = scores.get('flags', {})
        debug = scores.get('debug', {})
        cls_score = scores.get('cls', 0)
        logger.info(f"    Track {tid}:")
        logger.info(f"      - Rule score: {rule_score:.3f}")
        if not is_yolo_world:
            logger.info(f"      - Rules: A={flags.get('A', False)}, B={flags.get('B', False)}, "
                        f"C={flags.get('C', False)}, D={flags.get('D', False)}, E={flags.get('E', False)}, F={flags.get('F', False)}")
        posture = debug.get('posture', 'unknown')
        logger.info(f"      - Posture: {posture} (cls_score={cls_score:.3f})")
        if is_yolo_world:
            logger.info(f"      - Debug: aspect={debug.get('aspect', 0):.2f}, "
                        f"motion_bonus={debug.get('motion_bonus', 0):.3f}, det_conf={debug.get('det_conf', 0):.3f}")
        else:
            logger.info(f"      - Debug: h_ratio={debug.get('h_ratio', 0):.3f}, kpt_aspect={debug.get('kpt_aspect', 0):.2f}, "
                        f"torso_angle={debug.get('torso_angle', 0):.1f}, n_ground={debug.get('n_ground', 0)}, "
                        f"vy={debug.get('vy_px_s', 0):.1f}, centers={debug.get('centers_len', 0)}")

    # 融合决策
    track_falling = results.get("track_falling", {})
    new_alarms = results.get("new_alarms", [])
    fusion_idx = "[7]" if (run_detection and not is_yolo_world) else ("[5]" if run_detection else "[4]")
    logger.info(f"{fusion_idx} Fusion decision:")
    for tid, scores in track_scores.items():
        final_score = scores.get('final', 0)
        is_falling = track_falling.get(tid, False)
        state = scores.get('state', 'unknown')
        is_new_alarm = tid in new_alarms
        cls_score = scores.get('cls', 0)
        logger.info(f"    Track {tid}:")
        logger.info(f"      - Inputs: rule={scores.get('rule', 0):.3f}, cls={cls_score:.3f}, posture={scores.get('debug', {}).get('posture', 'unknown')}")
        logger.info(f"      - Final score: {final_score:.3f}")
        logger.info(f"      - State: {state}")
        logger.info(f"      - Is falling: {is_falling}")
        if tid in pipeline.fusion:
            fusion_state = pipeline.fusion[tid].get_state()
            logger.info(f"      - Fusion detail: S_temporal={fusion_state['S_temporal']:.3f}, "
                        f"alarm_frames={fusion_state['alarm_frames']}, cooldown={fusion_state['cooldown_remaining']}")
        if is_new_alarm:
            logger.info(f"      - *** NEW ALARM TRIGGERED ***")


def load_pipeline(config_path: str, device: str = None):
    """根据配置文件内容自动选择并实例化对应 pipeline."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    is_yolo_world = cfg is not None and "yolo_world_fall" in cfg
    if is_yolo_world:
        return YOLOWorldFallPipeline(config_path, device=device), True
    return FallDetectionPipeline(config_path, device=device), False


def main():
    parser = argparse.ArgumentParser(description="Fall Detection Pipeline Demo")
    parser.add_argument("--video", default="data/sample.mp4",
                        help="Video path (or 0 for webcam)")
    parser.add_argument("--config", default="configs/pipeline/default.yaml",
                        help="Config file path")
    parser.add_argument("--output", default=None,
                        help="Output video path")
    parser.add_argument("--save-frames", default=None,
                        help="Directory to save frames")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI display (for server)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable detailed debug logging")
    parser.add_argument("--log", default=None,
                        help="Log file path (used with --debug). Defaults to logs/{video_name}.log")
    parser.add_argument("--device", default=None,
                        help="Device to use (e.g., 'cuda', 'cuda:0', 'cpu'). Auto-detect if not specified.")
    args = parser.parse_args()

    # 如果没有指定log路径，使用视频文件名作为默认日志名
    if args.log is None and args.debug:
        out_dir = os.path.dirname(args.output) if args.output else "."
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        args.log = f"{out_dir}/logs/{video_name}.log"

    # 设置调试日志
    logger = None
    if args.debug:
        os.makedirs(os.path.dirname(args.log), exist_ok=True)
        logger = setup_logger(args.log)
        logger.info("=" * 80)
        logger.info("Fall Detection Pipeline Debug Session")
        logger.info(f"Start time: {datetime.now()}")
        logger.info(f"Video: {args.video}")
        logger.info(f"Config: {args.config}")
        logger.info("=" * 80)

    print("Initializing pipeline...")
    pipeline, is_yolo_world = load_pipeline(args.config, device=args.device)
    print(f"  Using device: {args.device if args.device else 'auto'}")
    if is_yolo_world:
        print(f"  Pipeline type: YOLO-World Fall Detection")
    else:
        print(f"  Pipeline type: Standard Fall Detection")

    # 打印检测器输入分辨率
    detector_input_size = pipeline.detector.input_size
    print(f"  Detector input size: {detector_input_size}x{detector_input_size}")

    if logger:
        logger.info(f"  - Pipeline type: {'YOLO-World' if is_yolo_world else 'Standard'}")
        if not is_yolo_world:
            logger.info(f"  - Trigger threshold: {pipeline.trigger_thresh}")
        logger.info(f"  - Skip frames: {pipeline.skip_frames}")
        logger.info(f"  - FPS: {pipeline.fps}")
        logger.info(f"  - Detector input size: {detector_input_size}x{detector_input_size}")

    video_path = 0 if args.video == "0" else args.video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Warning: video not found, generating blank test frames.")
        if logger:
            logger.warning("Video not found, generating blank test frames")
        # 生成测试序列
        for i in range(100):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            results = pipeline.process_frame(frame)
            frame = draw_results(
                frame,
                results["tracks"],
                results["track_kpts"],
                results["track_scores"],
                results["track_falling"],
                results.get("fusion_histories", {}),
            )
            if not args.headless:
                window_name = "YOLO-World Fall Demo" if is_yolo_world else "Pipeline Demo"
                cv2.imshow(window_name, frame)
                if cv2.waitKey(100) == 27:
                    break
        if not args.headless:
            cv2.destroyAllWindows()
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay = int(1000 / fps)

    if logger:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video info: {w}x{h} @ {fps:.1f}fps, {total_frames} frames")

    writer = None
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
        print(f"Output video: {args.output} ({w}x{h} @ {fps:.1f}fps)")
        if logger:
            logger.info(f"Output video: {args.output}")

    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)

    if not args.headless:
        print("\nControls:")
        print("  ESC - quit")
        print("  p - pause/resume")
        print("  s - save current frame")
    else:
        print("\nRunning in headless mode (no GUI display)")

    paused = False
    frame_idx = 0
    fall_detected_count = 0
    new_alarm_count = 0
    active_alarms = set()

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video")
                    if logger:
                        logger.info("End of video")
                    break

                results = pipeline.process_frame(frame)
                frame = draw_results(
                    frame,
                    results["tracks"],
                    results["track_kpts"],
                    results["track_scores"],
                    results["track_falling"],
                    results.get("fusion_histories", {}),
                )

                # 添加帧信息（左上角，分数信息下面）
                # 计算信息栏高度（与utils.py中一致）
                track_scores = results.get("track_scores", {})
                info_bar_height = 25 * len(track_scores) + 40 if track_scores else 50
                cv2.putText(frame, f"Frame: {frame_idx}", (10, info_bar_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Alarms: {new_alarm_count}", (10, info_bar_height + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # 统计跌倒帧和新告警事件
                if any(results.get("track_falling", {}).values()):
                    fall_detected_count += 1

                # 检测新的告警事件
                for tid in results.get("new_alarms", []):
                    if tid not in active_alarms:
                        new_alarm_count += 1
                        active_alarms.add(tid)
                        print(f"*** ALARM: New fall detected for Track {tid} at frame {frame_idx} ***")
                        if logger:
                            logger.warning(f"NEW FALL ALARM: Track {tid} at frame {frame_idx}")

                # 清理已恢复的告警
                current_falling = set(tid for tid, falling in results.get("track_falling", {}).items() if falling)
                active_alarms &= current_falling

                # 调试日志
                if logger:
                    log_frame_details(logger, pipeline, frame_idx, results, is_yolo_world=is_yolo_world)

                frame_idx += 1
            else:
                cv2.putText(frame, "PAUSED", (w//2 - 60, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            # 保存输出（共通逻辑）
            if writer:
                writer.write(frame)
            if args.save_frames:
                fname = f"{args.save_frames}/frame_{frame_idx:06d}.jpg"
                cv2.imwrite(fname, frame)

            if not args.headless:
                window_name = "YOLO-World Fall Demo" if is_yolo_world else "Pipeline Demo"
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(delay) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord('s'):
                    fname = f"pipeline_frame_{frame_idx:04d}.png"
                    cv2.imwrite(fname, frame)
                    print(f"Saved: {fname}")
            else:
                # Headless mode progress print
                if frame_idx % 30 == 0:
                    active_tracks = len(results.get("tracks", []))
                    is_falling = any(results.get("track_falling", {}).values())
                    print(f"Frame {frame_idx}: {active_tracks} tracks, fall={is_falling}")
                    if logger:
                        logger.info(f"[PROGRESS] Processed {frame_idx} frames, detected {fall_detected_count} fall frames")
    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

        if logger:
            logger.info("=" * 80)
            logger.info(f"Session complete")
            logger.info(f"Total frames processed: {frame_idx}")
            logger.info(f"Fall detected in {fall_detected_count} frames")
            logger.info(f"New alarm events: {new_alarm_count}")
            logger.info(f"Log saved to: {args.log}")
            logger.info("=" * 80)

    print(f"\nDone. Processed {frame_idx} frames.")
    print(f"Fall detected in {fall_detected_count} frames.")
    print(f"New alarm events: {new_alarm_count} (each fall event only reported once)")


if __name__ == "__main__":
    main()
