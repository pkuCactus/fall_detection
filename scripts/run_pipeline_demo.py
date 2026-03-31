import sys
import numpy as np
import cv2

sys.path.insert(0, "src")
from fall_detection.pipeline import FallDetectionPipeline
from fall_detection.utils import draw_results


def main():
    pipeline = FallDetectionPipeline("configs/default.yaml")
    cap = cv2.VideoCapture("data/sample.mp4")

    if not cap.isOpened():
        print("Warning: data/sample.mp4 not found, generating blank test frames.")
        # 生成 10fps 100 帧的空白测试序列
        for i in range(100):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            results = pipeline.process_frame(frame)
            frame = draw_results(
                frame,
                results["tracks"],
                results["track_kpts"],
                results["track_scores"],
                results["track_falling"],
            )
            cv2.imshow("demo", frame)
            if cv2.waitKey(100) == 27:
                break
        cv2.destroyAllWindows()
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay = int(1000 / fps)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = pipeline.process_frame(frame)
        frame = draw_results(
            frame,
            results["tracks"],
            results["track_kpts"],
            results["track_scores"],
            results["track_falling"],
        )
        cv2.imshow("demo", frame)
        if cv2.waitKey(delay) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
