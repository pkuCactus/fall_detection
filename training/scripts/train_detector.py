import argparse
import os

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train person detector with YOLOv8")
    parser.add_argument("--data", default="data/fall_detection.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=832)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--project", default="train/detector")
    parser.add_argument("--name", default="exp")
    args = parser.parse_args()

    model = YOLO(args.model)
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
