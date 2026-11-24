#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


TARGET_CLASS = "person"
OBSTACLE_CLASSES = ("chair", "table", "bottle", "backpack", "couch", "bench", "suitcase")

COCO_MAP = {
    "person": "person",
    "chair": "chair",
    "dining table": "table",
    "table": "table",
    "bottle": "bottle",
    "backpack": "backpack",
    "couch": "couch",
    "bench": "bench",
    "suitcase": "suitcase",
}

CLASS_HEIGHT = {
    "person": 1.70,
    "chair": 0.90,
    "table": 0.75,
    "bottle": 0.25,
    "backpack": 0.45,
    "couch": 0.85,
    "bench": 0.50,
    "suitcase": 0.65,
}


@dataclass
class Detection:
    cls: str
    bbox: Tuple[float, float, float, float]
    conf: float
    stamp: float


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    cam_height: float = 0.12

    @classmethod
    def from_resolution(cls, width: int, height: int, fov_deg: float = 62.0, cam_height: float = 0.12) -> "CameraIntrinsics":
        f = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
        return cls(
            width=width,
            height=height,
            fx=f,
            fy=f,
            cx=width / 2.0,
            cy=height / 2.0,
            cam_height=cam_height,
        )


def normalize_detections(raw_items, stamp: float) -> List[Detection]:
    """Chuan hoa danh sach phat hien tu YOLOv5/YOLOv8 ve schema Detection."""
    out: List[Detection] = []
    for item in raw_items:
        if isinstance(item, Detection):
            mapped = COCO_MAP.get(item.cls, item.cls)
            out.append(Detection(mapped, tuple(map(float, item.bbox)), float(item.conf), float(item.stamp)))
            continue
        if isinstance(item, dict):
            raw_cls = str(item.get("cls", item.get("name", ""))).lower()
            mapped = COCO_MAP.get(raw_cls)
            if mapped:
                out.append(Detection(mapped, tuple(map(float, item["bbox"])), float(item["conf"]), stamp))
        elif isinstance(item, (tuple, list)) and len(item) >= 3:
            if isinstance(item[0], str):
                mapped = COCO_MAP.get(item[0].lower())
                if mapped:
                    out.append(Detection(mapped, tuple(map(float, item[1])), float(item[2]), stamp))
            elif len(item) >= 6:
                mapped = COCO_MAP.get(str(item[5]).lower())
                if mapped:
                    bbox = (float(item[0]), float(item[1]), float(item[2]), float(item[3]))
                    out.append(Detection(mapped, bbox, float(item[4]), stamp))
    return out


def bearing_distance(
    bbox: Tuple[float, float, float, float],
    cam: CameraIntrinsics,
    cls: str = "person",
) -> Tuple[float, float]:
    """Tinh (theta_rad, Z_m) tu bounding box."""
    x1, y1, x2, y2 = bbox
    u = 0.5 * (x1 + x2)
    h = max(1.0, y2 - y1)
    theta = math.atan2(u - cam.cx, cam.fx)

    if y1 <= 2.0 and y2 > cam.cy + 2.0:
        z = cam.fy * cam.cam_height / max(1.0, (y2 - cam.cy))
        return theta, max(0.15, z)

    z = cam.fy * CLASS_HEIGHT.get(cls, 1.0) / h
    return theta, max(0.15, z)


def _iou_box(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0.0 else 0.0


def extract_visual_obstacles(
    detections: List[Detection],
    cam: CameraIntrinsics,
    target_bbox: Optional[Tuple[float, float, float, float]] = None,
    min_conf: float = 0.30,
) -> List[Tuple[float, float]]:
    """Tra danh sach (angle_rad, dist_m) cac vat can (hoac nguoi khac) truoc mat robot."""
    obstacles: List[Tuple[float, float]] = []
    for d in detections:
        if d.conf < min_conf:
            continue
        if d.cls == TARGET_CLASS:
            if target_bbox is not None and _iou_box(d.bbox, target_bbox) > 0.35:
                continue
        elif d.cls not in OBSTACLE_CLASSES:
            continue

        th, z = bearing_distance(d.bbox, cam, d.cls)
        if d.bbox[3] >= cam.height - 4.0 and d.bbox[3] > cam.cy + 4.0:
            z_floor = cam.fy * cam.cam_height / max(1.0, (d.bbox[3] - cam.cy))
            z = min(z, max(0.15, z_floor))
        obstacles.append((th, z))
    return obstacles
