#!/usr/bin/env python3
"""下载开源跌倒检测样本视频."""

import os
import urllib.request
import argparse


def download_file(url, output_path):
    """下载文件并显示进度."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
        print(f"\rProgress: {percent:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, output_path, progress_hook)
    print("\nDownload complete!")


def main():
    parser = argparse.ArgumentParser(description="Download sample video for fall detection")
    parser.add_argument("--output", default="data/sample_fall.mp4", help="Output video path")
    parser.add_argument("--dataset", choices=["urfall", "le2i", "synthetic"], default="synthetic",
                        help="Dataset to download from (default: synthetic)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.dataset == "urfall":
        # UR Fall Detection Dataset (需要手动下载)
        print("UR Fall Dataset requires manual download from:")
        print("http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html")
        print("Please download and place the video in data/ directory")

    elif args.dataset == "le2i":
        # Le2i Fall Detection Dataset (需要手动下载)
        print("Le2i Fall Dataset requires manual download from:")
        print("http://www.rip.ua.pt/wiki/ Fall_detection_based_on_computer_vision")
        print("Please download and place the video in data/ directory")

    else:
        # 创建合成跌倒检测视频用于演示
        print("Creating synthetic fall detection video...")
        create_synthetic_video(args.output)


def create_synthetic_video(output_path, fps=25, duration=10):
    """创建合成的跌倒检测演示视频."""
    import cv2
    import numpy as np

    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    num_frames = fps * duration

    for frame_idx in range(num_frames):
        # 创建背景
        frame = np.ones((height, width, 3), dtype=np.uint8) * 240

        # 添加地面线
        cv2.line(frame, (0, 400), (width, 400), (180, 180, 180), 2)

        # 模拟一个人行走的边界框
        progress = frame_idx / num_frames

        if progress < 0.3:
            # 正常行走阶段
            x = int(100 + progress * 3 * width)
            y = 250
            w, h = 60, 140
            color = (0, 255, 0)  # 绿色 - 正常
            label = "Walking"
        elif progress < 0.5:
            # 跌倒过程
            x = int(100 + 0.3 * 3 * width + (progress - 0.3) * 200)
            y = 250 + int((progress - 0.3) * 5 * 150)
            w = 60 + int((progress - 0.3) * 5 * 80)
            h = 140 - int((progress - 0.3) * 5 * 100)
            color = (0, 165, 255)  # 橙色 - 跌倒中
            label = "Falling"
        else:
            # 倒地状态
            x = int(100 + 0.3 * 3 * width + 0.2 * 200)
            y = 400 - 40
            w, h = 140, 40
            color = (0, 0, 255)  # 红色 - 跌倒
            label = "Fallen"

        # 确保框在画面内
        x = min(x, width - w - 10)

        # 绘制边界框
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"Person: {label}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 添加帧信息
        cv2.putText(frame, f"Frame: {frame_idx}/{num_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, "Synthetic Fall Detection Video", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # 添加时间戳
        seconds = frame_idx / fps
        cv2.putText(frame, f"Time: {seconds:.1f}s", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        out.write(frame)

    out.release()
    print(f"Created synthetic video: {output_path}")
    print(f"Duration: {duration}s, FPS: {fps}, Resolution: {width}x{height}")


if __name__ == "__main__":
    main()
