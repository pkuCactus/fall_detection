from typing import Dict, List
import numpy as np
import cv2


# COCO 17 关键点骨架连线
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # 头部
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上肢
    (5, 11), (6, 12), (11, 12),            # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16),  # 下肢
]


def draw_results(
    frame: np.ndarray,
    tracks,
    track_kpts: Dict[int, np.ndarray],
    track_scores: Dict[int, dict],
    track_falling: Dict[int, bool],
    fusion_histories: Dict[int, list] = None,
) -> np.ndarray:
    """在帧上绘制 bbox、track_id、关键点骨架和得分信息."""
    h, w = frame.shape[:2]

    # 绘制全局状态栏
    any_fall = any(track_falling.get(track.track_id, False) for track in tracks)
    if any_fall:
        # 红色告警背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, "FALL DETECTED!", (w//2 - 150, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    for track in tracks:
        tid = track.track_id
        x1, y1, x2, y2 = map(int, track.to_tlbr())
        is_falling = track_falling.get(tid, False)

        # 跌倒用红色，正常用绿色
        if is_falling:
            color = (0, 0, 255)  # 红色
            thickness = 4  # 更粗的框
            # 添加闪烁效果的填充
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        else:
            color = (0, 255, 0)  # 绿色
            thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # 状态标签
        status = "FALL!" if is_falling else f"ID:{tid}"
        label_y = max(y1 - 10, 25)
        cv2.putText(frame, status, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3 if is_falling else 2)

        # 绘制关键点（跌倒时用更明显的颜色）
        kpts = track_kpts.get(tid)
        if kpts is not None and kpts.shape == (17, 3):
            kpt_color = (0, 100, 255) if is_falling else color
            for (p1, p2) in COCO_SKELETON:
                x_a, y_a, c_a = kpts[p1]
                x_b, y_b, c_b = kpts[p2]
                if c_a > 0.1 and c_b > 0.1:
                    pt_a = (int(x_a), int(y_a))
                    pt_b = (int(x_b), int(y_b))
                    cv2.line(frame, pt_a, pt_b, kpt_color, 3 if is_falling else 2)
            for i in range(17):
                x, y, c = kpts[i]
                if c > 0.1:
                    radius = 5 if is_falling else 3
                    cv2.circle(frame, (int(x), int(y)), radius, kpt_color, -1)
                    # 跌倒时在关键点周围加白色圆圈
                    if is_falling:
                        cv2.circle(frame, (int(x), int(y)), radius + 2, (255, 255, 255), 1)

    # 左上角信息栏 - 显示所有跟踪目标得分和时序信息
    bar_height = 25 * len(track_scores) + 25 if track_scores else 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (450, bar_height + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y_offset = 25
    cv2.putText(frame, "Track Scores (Rule/Cls/Final | State):", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += 20

    for tid in sorted(track_scores.keys()):
        scores = track_scores[tid]
        is_fall = track_falling.get(tid, False)
        state_str = scores.get('state', 'unknown')

        # 构建时序信息字符串
        temporal_info = ""
        if fusion_histories and tid in fusion_histories:
            history = fusion_histories[tid]
            if len(history) >= 3:
                # 显示最近3帧的融合分数
                recent_scores = history[-3:]
                temporal_info = f" H:[{recent_scores[0]:.2f},{recent_scores[1]:.2f},{recent_scores[2]:.2f}]"

        if is_fall:
            text = f"T{tid}: R={scores.get('rule', 0):.2f} C={scores.get('cls', 0):.2f} F={scores.get('final', 0):.2f} | {state_str}{temporal_info}"
            text_color = (0, 0, 255)  # 红色
            thickness = 2
        else:
            text = f"T{tid}: R={scores.get('rule', 0):.2f} C={scores.get('cls', 0):.2f} F={scores.get('final', 0):.2f} | {state_str}{temporal_info}"
            text_color = (0, 255, 0)  # 绿色
            thickness = 1

        cv2.putText(frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, thickness)
        y_offset += 22

    # 如果没有跟踪目标，显示提示
    if not tracks:
        cv2.putText(frame, "No targets detected", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    elif not track_scores:
        # 有跟踪目标但没有分数（非检测帧）
        for track in tracks:
            tid = track.track_id
            is_fall = track_falling.get(tid, False)
            status = "FALL!" if is_fall else "OK (skip frame)"
            color = (0, 0, 255) if is_fall else (0, 255, 0)
            cv2.putText(frame, f"T{tid}: {status}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 22

    return frame
