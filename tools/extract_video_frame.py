import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--time", type=float, default=3.0, help="Timestamp in seconds")
    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = int(args.time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()

    if not ok:
        raise RuntimeError(f"Cannot read frame at {args.time}s from {video_path}")

    cv2.imwrite(str(output_path), frame)
    cap.release()

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()