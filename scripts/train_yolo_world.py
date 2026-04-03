import argparse
import os

import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train person detector with YOLOWorld")
    parser.add_argument("--data", default="data/fall_detection.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", default="yolov8l-worldv2.pt")
    parser.add_argument("--project", default="train/yolo_world")
    parser.add_argument("--name", default="exp")
    args = parser.parse_args()

    # 从数据配置中读取类别名称作为文本提示
    with open(args.data, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    names = data_cfg.get("names", {})
    if isinstance(names, dict):
        classes = [names[i] for i in sorted(names)]
    elif isinstance(names, list):
        classes = list(names)
    else:
        classes = ["person"]

    model = YOLO(args.model)
    model.set_classes(classes)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )
    best_path = os.path.join(args.project, args.name, "weights", "best.pt")
    out_path = os.path.join(args.project, args.name, "best.pt")
    if os.path.exists(best_path):
        try:
            os.link(best_path, out_path)
        except OSError:
            import shutil
            shutil.copy2(best_path, out_path)
        print(f"Best weights saved to {out_path}")


if __name__ == "__main__":
    main()
