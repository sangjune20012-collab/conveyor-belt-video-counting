# Conveyor Belt Video Counting

Segmentation-based conveyor belt object counting using SAM3 and SAM3.1 mask tracking with polygon-zone counting.

## Overview

This repository provides a video counting pipeline for conveyor belt object counting.

The project started from vehicle video tracking experiments such as car, bike, and truck tracking. It was then extended to industrial conveyor belt videos, where object masks from SAM3 and SAM3.1 are used for polygon-zone counting.

## Main Purpose

The final goal of this project is conveyor belt video counting.

## Pipeline

Input conveyor belt video
SAM3 or SAM3.1 video segmentation tracking
Object mask extraction
Mask centroid calculation
Polygon-zone entry detection
Unique object counting
Count visualization video

## Main Scripts

- scripts/sam3_segment_video.py : SAM3 video segmentation and tracking.
- scripts/sam31_segment_video.py : SAM3.1 video segmentation and tracking.
- scripts/polygon_zone_counting.py : Polygon-zone counting using segmentation masks.
- prototypes/vehicle_tracking_prototype.py : Early vehicle tracking prototype before conveyor belt extension.

## Repository Notes

This repository does not include input videos, model checkpoints, the official SAM3 repository, generated output videos, or large experiment folders.

Please install the official SAM3 repository and download checkpoints separately.

## Limitations

- Counting currently depends on mask centroid stability.
- Tracking ID switches can cause duplicate counts.
- SAM3 and SAM3.1 video outputs do not expose YOLO-style class confidence scores in the current API path.

## Future Work

- Stable counting with minimum inside-frame threshold
- Mask-polygon overlap ratio filtering
- Line-crossing count mode
- Tracking ID recovery after short occlusion
- Quantitative comparison with YOLO box-based counting
