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
) -> np.ndarray:
    """在帧上绘制 bbox、track_id、关键点骨架和得分信息."""
    for track in tracks:
        tid = track.track_id
        x1, y1, x2, y2 = map(int, track.to_tlbr())
        color = (0, 0, 255) if track_falling.get(tid, False) else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID:{tid}",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        kpts = track_kpts.get(tid)
        if kpts is not None and kpts.shape == (17, 3):
            for (p1, p2) in COCO_SKELETON:
                x_a, y_a, c_a = kpts[p1]
                x_b, y_b, c_b = kpts[p2]
                if c_a > 0.1 and c_b > 0.1:
                    pt_a = (int(x_a), int(y_a))
                    pt_b = (int(x_b), int(y_b))
                    cv2.line(frame, pt_a, pt_b, color, 2)
            for i in range(17):
                x, y, c = kpts[i]
                if c > 0.1:
                    cv2.circle(frame, (int(x), int(y)), 3, color, -1)

    # overlay text
    y_offset = 20
    for tid in sorted(track_scores.keys()):
        scores = track_scores[tid]
        is_fall = track_falling.get(tid, False)
        text = (
            f"T{tid} R={scores.get('rule', 0):.2f} "
            f"C={scores.get('cls', 0):.2f} "
            f"F={scores.get('final', 0):.2f} "
            f"{'FALL' if is_fall else 'OK'}"
        )
        cv2.putText(
            frame,
            text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255) if is_fall else (0, 255, 0),
            2,
        )
        y_offset += 20

    return frame
