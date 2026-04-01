#!/usr/bin/env python3
"""从开源数据集下载真实跌倒检测视频样本."""

import os
import urllib.request
import zipfile
import argparse


def download_multicam_fall_dataset(output_dir="data"):
    """下载 Multi-Camera Fall Dataset (MCFD) 样本."""
    # 这个数据集有公开可用的样本视频
    # 主页: https://www.iro.umontreal.ca/~labimage/Dataset/

    os.makedirs(output_dir, exist_ok=True)

    # 尝试从GitHub上的镜像下载样本
    urls = [
        # 使用一个公开可用的跌倒检测样本视频 (来自GitHub上的开源项目)
        ("https://raw.githubusercontent.com/nttcom/WASB-SensorDataset/master/fall/sample_fall.mp4",
         os.path.join(output_dir, "real_fall.mp4")),
    ]

    for url, output_path in urls:
        try:
            print(f"Attempting to download from: {url}")
            print(f"This may take a few minutes...")

            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req, timeout=10) as response:
                print(f"URL accessible, size: {response.headers.get('Content-Length', 'unknown')} bytes")

            urllib.request.urlretrieve(url, output_path)
            print(f"Downloaded: {output_path}")
            return output_path

        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            continue

    return None


def create_demo_with_person_shape(output_path, fps=25, duration=10):
    """创建包含人形轮廓的合成视频（比简单方块更逼真）."""
    import cv2
    import numpy as np

    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    num_frames = fps * duration

    for frame_idx in range(num_frames):
        # 创建室内背景
        frame = np.ones((height, width, 3), dtype=np.uint8) * 220

        # 添加地板
        cv2.rectangle(frame, (0, 400), (width, height), (180, 180, 180), -1)
        cv2.line(frame, (0, 400), (width, 400), (150, 150, 150), 2)

        # 计算动画进度
        progress = frame_idx / num_frames

        # 基础位置
        base_x = 150 + int(progress * 300)

        if progress < 0.3:
            # 站立行走状态
            state = "walking"
            center_x = base_x
            center_y = 280

            # 绘制站立的人形（简笔画风格）
            # 头
            cv2.circle(frame, (center_x, center_y - 60), 20, (100, 150, 200), -1)
            # 身体
            cv2.line(frame, (center_x, center_y - 40), (center_x, center_y + 40), (100, 150, 200), 8)
            # 腿
            leg_offset = int(np.sin(frame_idx * 0.3) * 10)
            cv2.line(frame, (center_x, center_y + 40), (center_x - 15 + leg_offset, center_y + 100), (100, 150, 200), 6)
            cv2.line(frame, (center_x, center_y + 40), (center_x + 15 - leg_offset, center_y + 100), (100, 150, 200), 6)
            # 手臂
            cv2.line(frame, (center_x, center_y - 20), (center_x - 25, center_y + 20), (100, 150, 200), 5)
            cv2.line(frame, (center_x, center_y - 20), (center_x + 25, center_y + 20), (100, 150, 200), 5)

            bbox = [center_x - 40, center_y - 80, center_x + 40, center_y + 110]
            color = (0, 255, 0)

        elif progress < 0.5:
            # 跌倒过程
            state = "falling"
            fall_progress = (progress - 0.3) / 0.2

            center_x = base_x + int(fall_progress * 50)
            center_y = 280 + int(fall_progress * 100)

            # 绘制倒下的人形
            angle = fall_progress * 90  # 旋转角度

            # 头部位置（随跌倒移动）
            head_x = center_x + int(np.sin(np.radians(angle)) * 60)
            head_y = center_y - 60 + int(np.cos(np.radians(angle)) * 60)

            cv2.circle(frame, (head_x, head_y), 20, (100, 150, 200), -1)

            # 身体（倾斜）
            body_end_x = center_x - int(np.sin(np.radians(angle)) * 60)
            body_end_y = center_y + 40 - int(np.cos(np.radians(angle)) * 60)
            cv2.line(frame, (head_x, head_y), (body_end_x, body_end_y), (100, 150, 200), 8)

            # 四肢（展开）
            cv2.line(frame, (body_end_x, body_end_y), (body_end_x - 30, body_end_y + 40), (100, 150, 200), 5)
            cv2.line(frame, (body_end_x, body_end_y), (body_end_x + 30, body_end_y + 40), (100, 150, 200), 5)

            bbox = [center_x - 60, center_y - 80, center_x + 60, center_y + 120]
            color = (0, 165, 255)

        else:
            # 倒地状态
            state = "fallen"
            center_x = base_x + 50
            center_y = 380

            # 绘制躺在地上的人形
            # 头
            cv2.circle(frame, (center_x - 50, center_y), 20, (100, 150, 200), -1)
            # 身体（水平）
            cv2.line(frame, (center_x - 50, center_y), (center_x + 50, center_y), (100, 150, 200), 8)
            # 腿（弯曲）
            cv2.line(frame, (center_x + 50, center_y), (center_x + 80, center_y - 20), (100, 150, 200), 6)
            cv2.line(frame, (center_x + 80, center_y - 20), (center_x + 90, center_y + 10), (100, 150, 200), 6)
            # 手臂
            cv2.line(frame, (center_x, center_y), (center_x - 20, center_y - 30), (100, 150, 200), 5)
            cv2.line(frame, (center_x, center_y), (center_x + 10, center_y + 30), (100, 150, 200), 5)

            bbox = [center_x - 80, center_y - 40, center_x + 100, center_y + 60]
            color = (0, 0, 255)

        # 绘制边界框
        x1, y1, x2, y2 = [max(0, min(v, width-1)) for v in bbox]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # 添加标签
        label = f"Person: {state.upper()}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 添加帧信息
        cv2.putText(frame, f"Frame: {frame_idx}/{num_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
        cv2.putText(frame, f"Time: {frame_idx/fps:.1f}s", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

        out.write(frame)

    out.release()
    print(f"Created demo video with person shape: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download or create sample video")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--method", choices=["download", "create"], default="create",
                        help="Method to get sample video")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.method == "download":
        print("Attempting to download real fall detection video...")
        result = download_multicam_fall_dataset(args.output_dir)
        if result:
            print(f"Success! Video saved to: {result}")
        else:
            print("Download failed. Falling back to creating demo video...")
            output_path = os.path.join(args.output_dir, "demo_with_person.mp4")
            create_demo_with_person_shape(output_path)
    else:
        print("Creating demo video with person-like shape...")
        output_path = os.path.join(args.output_dir, "demo_with_person.mp4")
        create_demo_with_person_shape(output_path)
        print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    main()
