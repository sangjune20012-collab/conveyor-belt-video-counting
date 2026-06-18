#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_numpy(x):
    import torch
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def binarize_mask(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == bool:
        return arr
    if arr.min() >= 0.0 and arr.max() <= 1.0 + 1e-4:
        return arr > 0.5
    return arr > 0.0


def color_for_id(obj_id: int) -> Tuple[int, int, int]:
    rng = np.random.RandomState(obj_id * 9973 + 12345)
    h = int(rng.randint(0, 180))
    hsv = np.uint8([[[h, 220, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    return int(b), int(g), int(r)


def preprocess_video(video_path: str, out_dir: str, scale: float = 1 / 3):
    out_dir = ensure_dir(out_dir)
    compare_dir = ensure_dir(out_dir / "compare")
    frames_dir = ensure_dir(out_dir / "frames")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))

    if new_w % 2:
        new_w -= 1
    if new_h % 2:
        new_h -= 1

    resized_video = compare_dir / f"video_{new_w}x{new_h}_{int(round(fps))}fps.mp4"

    existing_frames = sorted(frames_dir.glob("*.jpg"))
    if resized_video.exists() and len(existing_frames) == n_frames and n_frames > 0:
        print(f"[preprocess] skip existing: {resized_video}")
        cap.release()
    else:
        writer = cv2.VideoWriter(
            str(resized_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (new_w, new_h),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open writer: {resized_video}")

        print(f"[preprocess] source={src_w}x{src_h}, fps={fps:.2f}, frames={n_frames}")
        print(f"[preprocess] target={new_w}x{new_h}")

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            writer.write(small)
            cv2.imwrite(str(frames_dir / f"{idx:05d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 95])

            idx += 1
            if idx % 60 == 0:
                print(f"[preprocess] frame {idx}/{n_frames}")

        cap.release()
        writer.release()
        n_frames = idx
        print(f"[preprocess] done -> {resized_video}")

    meta = {
        "fps": fps,
        "width": new_w,
        "height": new_h,
        "frame_count": n_frames,
        "frames_dir": str(frames_dir),
        "video": str(resized_video),
    }

    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    return meta


def run_yolo_tracking(video_path: str, out_dir: str, yolo_model: str, tracker: str):
    from ultralytics import YOLO

    out_dir = ensure_dir(out_dir)
    out_video = out_dir / "yolo_track_car.mp4"
    out_pkl = out_dir / "yolo_track_car.pkl"

    model = YOLO(yolo_model)

    results = model.track(
        source=video_path,
        classes=[2],          # COCO car
        tracker=tracker,
        persist=True,
        stream=True,
        verbose=True,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    frame_idx_to_boxes = {}

    for idx, r in enumerate(results):
        frame = r.orig_img.copy()
        boxes_this = {}

        if r.boxes is not None and r.boxes.xyxy is not None:
            xyxy = r.boxes.xyxy.detach().cpu().numpy()
            conf = r.boxes.conf.detach().cpu().numpy() if r.boxes.conf is not None else np.zeros(len(xyxy))

            if r.boxes.id is not None:
                ids = r.boxes.id.detach().cpu().numpy().astype(int)
            else:
                ids = np.arange(len(xyxy))

            for box, score, tid in zip(xyxy, conf, ids):
                x1, y1, x2, y2 = box.astype(int).tolist()
                boxes_this[int(tid)] = {
                    "xyxy": [x1, y1, x2, y2],
                    "conf": float(score),
                }

                color = color_for_id(int(tid))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"YOLO car #{int(tid)} {score:.2f}",
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        frame_idx_to_boxes[idx] = boxes_this
        writer.write(frame)

    writer.release()

    payload = {
        "model": yolo_model,
        "type": "box_tracking",
        "frame_idx_to_boxes": frame_idx_to_boxes,
    }

    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f)

    print(f"[yolo] done -> {out_video}")
    return str(out_pkl), str(out_video)


def format_sam_outputs(outputs_per_frame):
    from sam3.visualization_utils import prepare_masks_for_visualization

    formatted = prepare_masks_for_visualization(outputs_per_frame)

    frame_idx_to_masks: Dict[int, Dict[int, np.ndarray]] = {}
    H = W = None

    for fi, per_frame in formatted.items():
        masks_dict = None

        if isinstance(per_frame, dict):
            if "masks" in per_frame and isinstance(per_frame["masks"], dict):
                masks_dict = per_frame["masks"]
            else:
                sample_key = next(iter(per_frame.keys()), None)
                if sample_key is not None and not isinstance(sample_key, str):
                    masks_dict = per_frame

        if masks_dict is None:
            masks_dict = {}

        cleaned = {}

        for obj_id, m in masks_dict.items():
            arr = to_numpy(m)

            while arr.ndim > 2:
                arr = arr[0]

            arr = binarize_mask(arr)

            if H is None:
                H, W = arr.shape[-2], arr.shape[-1]

            cleaned[int(obj_id)] = arr

        frame_idx_to_masks[int(fi)] = cleaned

    return frame_idx_to_masks, (H, W)


def run_sam3_tracking(
    frames_dir: str,
    ckpt: str,
    gpu: int,
    text: str,
    out_dir: str,
):
    out_dir = ensure_dir(out_dir)
    out_pkl = out_dir / "sam3_results.pkl"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    from sam3.model_builder import build_sam3_video_predictor

    print(f"[sam3] build predictor on GPU {gpu}")
    print(f"[sam3] checkpoint={ckpt}")

    predictor = build_sam3_video_predictor(
        gpus_to_use=[0],
        checkpoint_path=ckpt,
    )

    resp = predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": str(frames_dir),
        }
    )
    session_id = resp["session_id"]
    print(f"[sam3] session={session_id}")

    resp = predictor.handle_request(
        request={
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": text,
        }
    )

    outputs_per_frame = {0: resp["outputs"]}

    print("[sam3] propagate forward")

    for response in predictor.handle_stream_request(
        request={
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "forward",
        }
    ):
        fi = int(response["frame_index"])
        outputs_per_frame[fi] = response["outputs"]

        if fi % 30 == 0:
            print(f"[sam3] frame={fi}")

    predictor.handle_request(
        request={
            "type": "close_session",
            "session_id": session_id,
        }
    )

    frame_idx_to_masks, frame_shape = format_sam_outputs(outputs_per_frame)

    payload = {
        "model": "sam3",
        "type": "seg_tracking",
        "text_prompt": text,
        "frame_idx_to_masks": frame_idx_to_masks,
        "frame_shape": frame_shape,
    }

    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f)

    print(f"[sam3] done -> {out_pkl}")
    return str(out_pkl)


def overlay_masks(frame_bgr: np.ndarray, masks: Dict[int, np.ndarray], alpha: float = 0.45):
    if not masks:
        return frame_bgr.copy()

    overlay = frame_bgr.copy()
    contour_layer = frame_bgr.copy()
    h, w = frame_bgr.shape[:2]

    for obj_id, mask in masks.items():
        if mask is None or mask.size == 0:
            continue

        if mask.shape[:2] != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        color = color_for_id(int(obj_id))
        overlay[mask] = color

        m8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_layer, contours, -1, color, 2)

        ys, xs = np.where(mask)
        if xs.size:
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(
                contour_layer,
                f"#{int(obj_id)}",
                (cx - 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                contour_layer,
                f"#{int(obj_id)}",
                (cx - 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                cv2.LINE_AA,
            )

    blended = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)
    changed = (contour_layer != frame_bgr).any(axis=2)
    blended[changed] = contour_layer[changed]
    return blended


def draw_header(img: np.ndarray, title: str, count: int):
    h, w = img.shape[:2]
    bar = 44
    out = np.zeros((h + bar, w, 3), dtype=np.uint8)
    out[bar:] = img
    cv2.rectangle(out, (0, 0), (w, bar), (30, 30, 30), -1)
    cv2.putText(
        out,
        f"{title} | cars detected: {count}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def render_sam3_video(frames_dir: str, result_pkl: str, out_video: str, fps: float):
    frames = sorted(Path(frames_dir).glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No frames found: {frames_dir}")

    with open(result_pkl, "rb") as f:
        payload = pickle.load(f)

    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]

    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h + 44),
    )

    for idx, fp in enumerate(frames):
        frame = cv2.imread(str(fp))
        masks = payload["frame_idx_to_masks"].get(idx, {})

        vis = overlay_masks(frame, masks)
        vis = draw_header(vis, "SAM 3 seg tracking", len(masks))
        writer.write(vis)

        if idx % 60 == 0:
            print(f"[SAM3 render] frame={idx} objs={len(masks)}")

    writer.release()
    print(f"[SAM3 render] done -> {out_video}")
    return out_video


def load_yolo_boxes(yolo_pkl: str):
    with open(yolo_pkl, "rb") as f:
        return pickle.load(f)["frame_idx_to_boxes"]


def draw_yolo_panel(frame: np.ndarray, boxes: Dict[int, dict]):
    vis = frame.copy()

    for tid, info in boxes.items():
        x1, y1, x2, y2 = info["xyxy"]
        score = info.get("conf", 0.0)
        color = color_for_id(int(tid))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"#{tid} {score:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return draw_header(vis, "YOLOv8 box tracking", len(boxes))


def render_two_way(
    frames_dir: str,
    yolo_pkl: str,
    sam3_pkl: str,
    out_video: str,
    fps: float,
):
    frames = sorted(Path(frames_dir).glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No frames found: {frames_dir}")

    yolo_boxes = load_yolo_boxes(yolo_pkl)

    with open(sam3_pkl, "rb") as f:
        sam3 = pickle.load(f)

    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]

    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w * 2, h + 44),
    )

    for idx, fp in enumerate(frames):
        frame = cv2.imread(str(fp))

        y_panel = draw_yolo_panel(frame, yolo_boxes.get(idx, {}))

        m3 = sam3["frame_idx_to_masks"].get(idx, {})
        s3_panel = draw_header(overlay_masks(frame, m3), "SAM 3 seg tracking", len(m3))

        combined = np.concatenate([y_panel, s3_panel], axis=1)
        writer.write(combined)

        if idx % 60 == 0:
            print(f"[compare] frame={idx} YOLO={len(yolo_boxes.get(idx, {}))} SAM3={len(m3)}")

    writer.release()
    print(f"[compare] done -> {out_video}")


def run_all(args):
    out_dir = ensure_dir(args.output_dir)

    meta = preprocess_video(args.video, out_dir, scale=args.scale)

    frames_dir = meta["frames_dir"]
    resized_video = meta["video"]
    fps = meta["fps"]

    yolo_dir = ensure_dir(out_dir / "yolo")
    sam3_dir = ensure_dir(out_dir / "sam3")
    compare_dir = ensure_dir(out_dir / "compare")

    yolo_pkl, yolo_video = run_yolo_tracking(
        video_path=resized_video,
        out_dir=str(yolo_dir),
        yolo_model=args.yolo_model,
        tracker=args.yolo_tracker,
    )

    sam3_pkl = run_sam3_tracking(
        frames_dir=frames_dir,
        ckpt=args.sam3_ckpt,
        gpu=args.sam3_gpu,
        text=args.text,
        out_dir=str(sam3_dir),
    )

    sam3_video = render_sam3_video(
        frames_dir=frames_dir,
        result_pkl=sam3_pkl,
        out_video=str(sam3_dir / "sam3_seg_car.mp4"),
        fps=fps,
    )

    compare_video = str(compare_dir / "yolo_vs_sam3_compare.mp4")
    render_two_way(
        frames_dir=frames_dir,
        yolo_pkl=yolo_pkl,
        sam3_pkl=sam3_pkl,
        out_video=compare_video,
        fps=fps,
    )

    print("[done]")
    print(f"YOLO result: {yolo_video}")
    print(f"SAM3 result: {sam3_video}")
    print(f"Compare result: {compare_video}")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--video", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--sam3-ckpt", required=True)
    p.add_argument("--sam3-gpu", type=int, default=3)

    p.add_argument("--text", default="car")
    p.add_argument("--yolo-model", default="yolov8s.pt")
    p.add_argument("--yolo-tracker", default="bytetrack.yaml")

    p.add_argument("--scale", type=float, default=1 / 3)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(args)
