#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def preprocess_video(video_path: str, out_dir: str, scale: float = 1 / 3):
    out_dir = ensure_dir(out_dir)
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

    existing = sorted(frames_dir.glob("*.jpg"))
    if len(existing) == n_frames and n_frames > 0:
        print(f"[preprocess] skip existing frames: {len(existing)}")
    else:
        print(f"[preprocess] {src_w}x{src_h} -> {new_w}x{new_h}, frames={n_frames}")
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(frames_dir / f"{idx:05d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 95])
            idx += 1
            if idx % 60 == 0:
                print(f"[preprocess] frame {idx}/{n_frames}")
        n_frames = idx

    cap.release()

    meta = {
        "fps": fps,
        "width": new_w,
        "height": new_h,
        "frame_count": n_frames,
        "frames_dir": str(frames_dir),
    }

    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    return meta


def to_numpy(x):
    import torch
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def binarize(arr: np.ndarray):
    if arr.dtype == bool:
        return arr
    if arr.min() >= 0.0 and arr.max() <= 1.0 + 1e-4:
        return arr > 0.5
    return arr > 0.0


def format_outputs(outputs_per_frame):
    from sam3.visualization_utils import prepare_masks_for_visualization

    formatted = prepare_masks_for_visualization(outputs_per_frame)

    frame_idx_to_masks: Dict[int, Dict[int, np.ndarray]] = {}

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
            cleaned[int(obj_id)] = binarize(arr)

        frame_idx_to_masks[int(fi)] = cleaned

    return frame_idx_to_masks


def color_for_id(obj_id: int):
    rng = np.random.RandomState(obj_id * 9973 + 12345)
    h = int(rng.randint(0, 180))
    hsv = np.uint8([[[h, 220, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    return int(b), int(g), int(r)


def overlay_masks(frame, masks, alpha=0.45):
    if not masks:
        return frame.copy()

    overlay = frame.copy()
    contour_layer = frame.copy()
    h, w = frame.shape[:2]

    for obj_id, mask in masks.items():
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        color = color_for_id(obj_id)
        overlay[mask] = color

        m8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_layer, contours, -1, color, 2)

        ys, xs = np.where(mask)
        if xs.size:
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(contour_layer, f"#{obj_id}", (cx - 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    changed = (contour_layer != frame).any(axis=2)
    blended[changed] = contour_layer[changed]
    return blended


def draw_header(img, count):
    h, w = img.shape[:2]
    bar = 44
    out = np.zeros((h + bar, w, 3), dtype=np.uint8)
    out[bar:] = img
    cv2.rectangle(out, (0, 0), (w, bar), (30, 30, 30), -1)
    cv2.putText(out, f"SAM 3.1 | cars detected: {count}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def run_sam31(frames_dir: str, ckpt: str, gpu: int, text: str, out_dir: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    from sam3.model_builder import build_sam3_multiplex_video_predictor

    print(f"[sam3.1] build predictor on GPU {gpu}")
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=ckpt,
        use_fa3=False,
        use_rope_real=False,
    )

    resp = predictor.handle_request({
        "type": "start_session",
        "resource_path": str(frames_dir),
    })
    session_id = resp["session_id"]
    print(f"[sam3.1] session={session_id}")

    resp = predictor.handle_request({
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": 0,
        "text": text,
    })

    outputs_per_frame = {0: resp["outputs"]}

    print("[sam3.1] propagate forward")
    for response in predictor.handle_stream_request({
        "type": "propagate_in_video",
        "session_id": session_id,
        "propagation_direction": "forward",
    }):
        fi = int(response["frame_index"])
        outputs_per_frame[fi] = response["outputs"]
        if fi % 30 == 0:
            print(f"[sam3.1] frame={fi}")

    predictor.handle_request({
        "type": "close_session",
        "session_id": session_id,
    })

    masks = format_outputs(outputs_per_frame)

    out_dir = ensure_dir(out_dir)
    pkl_path = out_dir / "sam31_results.pkl"

    with open(pkl_path, "wb") as f:
        pickle.dump({
            "model": "sam3.1",
            "text_prompt": text,
            "frame_idx_to_masks": masks,
        }, f)

    print(f"[sam3.1] pkl saved -> {pkl_path}")
    return str(pkl_path)


def render_video(frames_dir: str, result_pkl: str, out_video: str, fps: float):
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
        vis = draw_header(vis, len(masks))
        writer.write(vis)

        if idx % 60 == 0:
            print(f"[render] frame={idx} objs={len(masks)}")

    writer.release()
    print(f"[render] done -> {out_video}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sam3p1-ckpt", required=True)
    parser.add_argument("--sam3p1-gpu", type=int, default=2)
    parser.add_argument("--text", default="car")
    parser.add_argument("--scale", type=float, default=1/3)
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)

    meta = preprocess_video(args.video, out_dir, scale=args.scale)

    sam31_dir = ensure_dir(out_dir / "sam3p1")

    pkl_path = run_sam31(
        frames_dir=meta["frames_dir"],
        ckpt=args.sam3p1_ckpt,
        gpu=args.sam3p1_gpu,
        text=args.text,
        out_dir=str(sam31_dir),
    )

    render_video(
        frames_dir=meta["frames_dir"],
        result_pkl=pkl_path,
        out_video=str(sam31_dir / "sam31_seg_car.mp4"),
        fps=meta["fps"],
    )

    print("[done]")
    print(f"result: {sam31_dir / 'sam31_seg_car.mp4'}")


if __name__ == "__main__":
    main()
