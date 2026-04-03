"""Extract frames from video files.

Split video into images for annotation or training data preparation.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = None,
    frame_interval: int = 1,
    max_frames: int = None,
    resize: tuple = None,
    quality: int = 95,
    format: str = "jpg",
    prefix: str = None,
):
    """
    Extract frames from video file.

    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        fps: Target FPS (if None, use frame_interval)
        frame_interval: Extract every N frames (default: 1 = all frames)
        max_frames: Maximum number of frames to extract (None = unlimited)
        resize: Resize frames to (width, height) (None = original size)
        quality: JPEG quality (1-100)
        format: Output format (jpg, png)
        prefix: Filename prefix (default: video name)
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path.name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {video_fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames / video_fps:.2f}s")

    # Determine frame interval
    if fps is not None:
        frame_interval = max(1, int(video_fps / fps))
        print(f"  Extracting at {fps} FPS (every {frame_interval} frames)")
    else:
        print(f"  Extracting every {frame_interval} frame(s)")

    # Set filename prefix
    if prefix is None:
        prefix = video_path.stem

    # Extract frames
    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames based on interval
        if frame_count % frame_interval != 0:
            frame_count += 1
            continue

        # Check max frames limit
        if max_frames is not None and extracted_count >= max_frames:
            break

        # Resize if needed
        if resize is not None:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)

        # Generate filename
        timestamp = frame_count / video_fps
        filename = f"{prefix}_frame{frame_count:06d}_t{timestamp:.3f}.{format}"
        output_path = output_dir / filename

        # Save frame
        if format.lower() == "jpg" or format.lower() == "jpeg":
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            cv2.imwrite(str(output_path), frame, encode_param)
        else:
            cv2.imwrite(str(output_path), frame)

        extracted_count += 1
        frame_count += 1

        # Progress
        if extracted_count % 100 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"  Progress: {extracted_count} frames extracted ({progress:.1f}%)")

    cap.release()

    print(f"\nDone! Extracted {extracted_count} frames to: {output_dir}")
    return extracted_count


def process_directory(
    input_dir: str,
    output_dir: str,
    video_extensions: list = None,
    **kwargs
):
    """
    Process all videos in a directory.

    Args:
        input_dir: Directory containing video files
        output_dir: Base output directory
        video_extensions: List of video extensions to process
        **kwargs: Additional arguments for extract_frames
    """
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_dir.glob(f"*{ext}"))
        video_files.extend(input_dir.glob(f"*{ext.upper()}"))

    video_files = sorted(set(video_files))

    if not video_files:
        print(f"No video files found in: {input_dir}")
        return

    print(f"Found {len(video_files)} video(s) to process\n")

    total_frames = 0
    for i, video_path in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] Processing: {video_path.name}")

        # Create sub-directory for each video
        video_output_dir = output_dir / video_path.stem

        try:
            count = extract_frames(
                video_path=str(video_path),
                output_dir=str(video_output_dir),
                **kwargs
            )
            total_frames += count
            print()
        except Exception as e:
            print(f"  Error: {e}\n")

    print(f"All done! Total frames extracted: {total_frames}")


def create_image_list(image_dir: str, output_file: str):
    """
    Create a text file listing all images (for further processing).

    Args:
        image_dir: Directory containing images
        output_file: Output list file
    """
    image_dir = Path(image_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    images = []
    for ext in image_extensions:
        images.extend(image_dir.glob(f"*{ext}"))
        images.extend(image_dir.glob(f"*{ext.upper()}"))

    images = sorted(images)

    with open(output_file, 'w') as f:
        for img in images:
            f.write(f"{img}\n")

    print(f"Created image list: {output_file} ({len(images)} images)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all frames from a video
  python video_to_frames.py -i video.mp4 -o output/

  # Extract at 1 FPS (for annotation)
  python video_to_frames.py -i video.mp4 -o output/ --fps 1

  # Extract every 30 frames
  python video_to_frames.py -i video.mp4 -o output/ --interval 30

  # Extract first 1000 frames with resize
  python video_to_frames.py -i video.mp4 -o output/ --max-frames 1000 --resize 1920 1080

  # Process all videos in a directory
  python video_to_frames.py -i videos/ -o frames/ --fps 5
        """
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input video file or directory")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for frames")
    parser.add_argument("--fps", type=float, default=None,
                        help="Target FPS (default: use --interval)")
    parser.add_argument("--interval", type=int, default=1,
                        help="Extract every N frames (default: 1 = all frames)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames to extract per video")
    parser.add_argument("--resize", nargs=2, type=int, metavar=("W", "H"),
                        help="Resize frames to WxH")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG quality (1-100, default: 95)")
    parser.add_argument("--format", default="jpg", choices=["jpg", "png"],
                        help="Output format (default: jpg)")
    parser.add_argument("--prefix", default=None,
                        help="Filename prefix (default: video name)")
    parser.add_argument("--list", action="store_true",
                        help="Create image list file after extraction")

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        # Single video
        count = extract_frames(
            video_path=str(input_path),
            output_dir=args.output,
            fps=args.fps,
            frame_interval=args.interval,
            max_frames=args.max_frames,
            resize=tuple(args.resize) if args.resize else None,
            quality=args.quality,
            format=args.format,
            prefix=args.prefix,
        )

        # Create image list if requested
        if args.list and count > 0:
            list_file = Path(args.output) / "image_list.txt"
            create_image_list(args.output, str(list_file))

    elif input_path.is_dir():
        # Directory of videos
        process_directory(
            input_dir=str(input_path),
            output_dir=args.output,
            fps=args.fps,
            frame_interval=args.interval,
            max_frames=args.max_frames,
            resize=tuple(args.resize) if args.resize else None,
            quality=args.quality,
            format=args.format,
        )

        # Create image list if requested
        if args.list:
            list_file = Path(args.output) / "image_list.txt"
            create_image_list(args.output, str(list_file))

    else:
        print(f"Error: Input path not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
