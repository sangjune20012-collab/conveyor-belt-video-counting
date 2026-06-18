#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def color_for_id(obj_id: int):
    rng = np.random.RandomState(obj_id * 9973 + 12345)
    h = int(rng.randint(0, 180))
    hsv = np.uint8([[[h, 220, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    return int(b), int(g), int(r)


def mask_centroid(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def point_inside_polygon(point, polygon):
    x, y = point
    return cv2.pointPolygonTest(polygon.astype(np.float32), (float(x), float(y)), False) >= 0


def overlay_mask(frame, mask, color, alpha=0.45):
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    overlay = frame.copy()
    overlay[mask] = color
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    m8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, 2)

    return frame


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--sam-pkl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=25.0)

    # polygon: x1,y1 x2,y2 x3,y3 ...
    parser.add_argument(
        "--polygon",
        nargs="+",
        required=True,
        help='Example: --polygon "100,200" "1200,200" "1200,650" "100,650"',
    )

    parser.add_argument("--title", default="SAM segmentation polygon counting")
    parser.add_argument("--count-mode", choices=["enter", "inside_once"], default="enter")

    args = parser.parse_args()

    frames = sorted(Path(args.frames_dir).glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No frames found: {args.frames_dir}")

    with open(args.sam_pkl, "rb") as f:
        data = pickle.load(f)

    frame_idx_to_masks = data["frame_idx_to_masks"]
    prompt = data.get("text_prompt", "object")

    polygon = np.array(
        [[int(p.split(",")[0]), int(p.split(",")[1])] for p in args.polygon],
        dtype=np.int32,
    )

    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (w, h),
    )

    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter open failed: {args.output}")

    counted_ids = set()
    prev_inside = {}
    total_count = 0

    for idx, fp in enumerate(frames):
        frame = cv2.imread(str(fp))
        masks = frame_idx_to_masks.get(idx, {})

        current_inside_count = 0

        cv2.polylines(frame, [polygon], isClosed=True, color=(255, 255, 255), thickness=4)

        for obj_id, mask in masks.items():
            obj_id = int(obj_id)
            mask = mask.astype(bool)

            color = color_for_id(obj_id)
            frame = overlay_mask(frame, mask, color)

            centroid = mask_centroid(mask)
            if centroid is None:
                continue

            cx, cy = centroid
            inside = point_inside_polygon((cx, cy), polygon)

            if inside:
                current_inside_count += 1

            was_inside = prev_inside.get(obj_id, False)

            if args.count_mode == "enter":
                if (not was_inside) and inside and obj_id not in counted_ids:
                    total_count += 1
                    counted_ids.add(obj_id)

            elif args.count_mode == "inside_once":
                if inside and obj_id not in counted_ids:
                    total_count += 1
                    counted_ids.add(obj_id)

            prev_inside[obj_id] = inside

            cv2.circle(frame, (int(cx), int(cy)), 5, color, -1)

            status = "IN" if inside else "OUT"
            if obj_id in counted_ids:
                label = f"{prompt} #{obj_id} {status} counted"
            else:
                label = f"{prompt} #{obj_id} {status}"

            cv2.putText(
                frame,
                label,
                (int(cx) + 8, int(cy) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                label,
                (int(cx) + 8, int(cy) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)
        cv2.putText(
            frame,
            f"{args.title}",
            (15, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Total count: {total_count} | Current in zone: {current_inside_count} | Frame: {idx}",
            (15, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        if idx % 30 == 0:
            print(f"[frame {idx}] total={total_count}, in_zone={current_inside_count}")

    writer.release()

    print("[done]")
    print("saved:", args.output)
    print("total_count:", total_count)
    print("counted_ids:", sorted(counted_ids))


if __name__ == "__main__":
    main()
