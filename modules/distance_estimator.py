#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
distance_estimator.py - Uoc luong khoang cach Z hieu chuan da muc sang + chieu mat san cu ly gan
"""
from collections import deque
import json
import math
import os
import numpy as np
import config

OBSTACLE_HEIGHT_M = {
    "chair": 0.85,
    "table": 0.75,
    "couch": 0.85,
    "backpack": 0.45,
    "suitcase": 0.65,
    "bottle": 0.25,
}


class DistanceEstimator:
    def __init__(self):
        self.img_w = float(config.PROCESS_WIDTH)
        self.img_h = float(config.PROCESS_HEIGHT)
        hfov_rad = math.radians(config.CAMERA_HFOV_DEG)
        self.fx = (self.img_w / 2.0) / math.tan(hfov_rad / 2.0)
        self.fy = self.fx
        self.cam_height_m = 0.12
        self.ema_z = None
        self._z_window = deque(maxlen=3)
        self._ema_luma = 115.0
        self.last_u_comp = None

        self.calib_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "distance_calib.json"
        )
        self.c0 = 0.4350
        self.k_scale = 0.0790
        self._load_calib()

    def _load_calib(self):
        if os.path.exists(self.calib_path):
            try:
                with open(self.calib_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.c0 = float(data.get("c0", self.c0))
                self.k_scale = float(data.get("k_scale", self.k_scale))
            except Exception:
                pass

    def calibrate_known_distance(self, true_dist_m):
        if self.last_u_comp is None or true_dist_m <= 0.1:
            return False
        self.c0 = float(self.last_u_comp - (self.k_scale / float(true_dist_m)))
        self.ema_z = float(true_dist_m)
        self._z_window.clear()
        with open(self.calib_path, "w", encoding="utf-8") as f:
            json.dump({"c0": round(self.c0, 6), "k_scale": round(self.k_scale, 6)}, f, indent=2)
        return True

    def reset(self):
        self.ema_z = None
        self._z_window.clear()

    def _resolve_intrinsics(self, frame_shape=None, frame=None):
        if frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 2:
            img_h, img_w = float(frame.shape[0]), float(frame.shape[1])
            hfov_rad = math.radians(config.CAMERA_HFOV_DEG)
            fx = (img_w / 2.0) / math.tan(hfov_rad / 2.0)
            return img_w, img_h, fx, fx
        if frame_shape is not None and len(frame_shape) >= 2:
            img_h, img_w = float(frame_shape[0]), float(frame_shape[1])
            hfov_rad = math.radians(config.CAMERA_HFOV_DEG)
            fx = (img_w / 2.0) / math.tan(hfov_rad / 2.0)
            return img_w, img_h, fx, fx
        return self.img_w, self.img_h, self.fx, self.fy

    def estimate_obstacle(self, bbox, cls_name="chair", frame_shape=None):
        """Tinh (bearing_rad, dist_m) cho vat can phat hien boi YOLO."""
        img_w, img_h, fx, fy = self._resolve_intrinsics(frame_shape=frame_shape)
        cy = 0.5 * img_h
        x1, y1, x2, y2 = bbox
        cx = 0.5 * (x1 + x2)
        box_h = max(1.0, y2 - y1)

        bearing_rad = math.atan2(cx - (0.5 * img_w), fx)
        z_pinhole = (fy * OBSTACLE_HEIGHT_M.get(cls_name, 0.70)) / box_h

        # Neu canh day vat can cham san (y2 > cy + 4), ket hop phep chieu mat san
        if y2 > cy + 4.0:
            z_floor = (fy * self.cam_height_m) / max(1.0, (y2 - cy))
            z = min(z_pinhole, max(0.15, z_floor * 1.15))
        else:
            z = z_pinhole

        return float(bearing_rad), float(max(0.15, min(5.0, z)))

    def estimate(self, bbox, frame_shape=None, frame=None):
        img_w, img_h, fx, fy = self._resolve_intrinsics(frame_shape=frame_shape, frame=frame)
        cy = 0.5 * img_h

        if frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 2:
            y_floor_start = int(img_h * 0.45)
            luma_now = float(np.mean(frame[y_floor_start::6, ::8]))
            self._ema_luma = 0.25 * luma_now + 0.75 * self._ema_luma

        x1, y1, x2, y2 = bbox
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        cx = 0.5 * (x1 + x2)

        dx = cx - (img_w / 2.0)
        bearing_rad = math.atan2(dx, fx)
        bearing_deg = math.degrees(bearing_rad)

        top_margin = y1 / img_h
        bottom_margin = (img_h - y2) / img_h
        h_ratio = box_h / img_h
        is_clipped = bool(top_margin < 0.04 or bottom_margin < 0.04)

        u_raw = (0.85 * y2 + 0.15 * box_h) / fy

        dark_factor = max(0.0, min(1.0, (105.0 - self._ema_luma) / 35.0))
        close_factor = max(0.0, min(1.0, (u_raw - 0.540) / 0.045))
        shadow_offset = 0.0075 * dark_factor * close_factor

        u_comp = u_raw - shadow_offset
        self.last_u_comp = u_comp

        denom = max(0.020, u_comp - self.c0)
        z_ground = self.k_scale / denom

        z_width = (fx * 0.34) / box_w
        z_width_clamped = max(z_ground * 0.85, min(z_ground * 1.15, z_width))
        z_near_mid = 0.88 * z_ground + 0.12 * z_width_clamped

        # Nhanh chieu mat san khi dinh hop cham mep tren anh (nguoi dung rat sat camera)
        if y1 <= 4.0 and y2 > cy + 4.0:
            z_floor_pinhole = max(0.15, (fy * self.cam_height_m) / max(1.0, (y2 - cy)))
            z_raw = min(z_near_mid, z_floor_pinhole)
        elif top_margin < 0.32:
            z_raw = z_near_mid
        else:
            z_far = (fy * 1.65) / box_h
            blend = min(1.0, (top_margin - 0.32) / 0.10)
            z_raw = (1.0 - blend) * z_near_mid + blend * z_far

        z_raw = max(0.15, min(5.0, z_raw))

        self._z_window.append(z_raw)
        z_med = float(np.median(self._z_window))

        if self.ema_z is None:
            self.ema_z = z_med
        else:
            alpha = 0.55
            self.ema_z = alpha * z_med + (1.0 - alpha) * self.ema_z

        dist_m = float(self.ema_z)
        error_dist = dist_m - float(config.TARGET_DISTANCE_M)

        return {
            "distance_m": dist_m,
            "error_dist": float(error_dist),
            "bearing_deg": float(bearing_deg),
            "bearing_rad": float(bearing_rad),
            "box_h_ratio": float(h_ratio),
            "is_clipped": is_clipped,
        }