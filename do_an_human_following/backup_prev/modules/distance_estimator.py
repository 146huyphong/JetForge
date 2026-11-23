#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
distance_estimator.py - Ước lượng khoảng cách Z thích ứng theo mức độ cắt khung hình (toàn thân ở xa / cẳng chân ở gần)
"""
import math
import config


class DistanceEstimator:
    def __init__(self):
        self.img_w = float(config.PROCESS_WIDTH)
        self.img_h = float(config.PROCESS_HEIGHT)
        hfov_rad = math.radians(config.CAMERA_HFOV_DEG)
        self.fx = (self.img_w / 2.0) / math.tan(hfov_rad / 2.0)
        self.fy = self.fx
        self.ema_z = None

    def reset(self):
        self.ema_z = None

    def estimate(self, bbox, frame_shape=None):
        if frame_shape is not None and len(frame_shape) >= 2:
            img_h = float(frame_shape[0])
            img_w = float(frame_shape[1])
            hfov_rad = math.radians(config.CAMERA_HFOV_DEG)
            fx = (img_w / 2.0) / math.tan(hfov_rad / 2.0)
            fy = fx
        else:
            img_w, img_h = self.img_w, self.img_h
            fx, fy = self.fx, self.fy

        x1, y1, x2, y2 = bbox
        box_h = max(1.0, y2 - y1)
        cx = 0.5 * (x1 + x2)

        dx = cx - (img_w / 2.0)
        bearing_rad = math.atan2(dx, fx)
        bearing_deg = math.degrees(bearing_rad)

        top_margin = y1 / img_h
        bottom_margin = (img_h - y2) / img_h
        h_ratio = box_h / img_h
        is_clipped = bool(top_margin < 0.04 or bottom_margin < 0.04)

        # Chọn chiều cao vật thể thực tế (H_eff) dựa trên mức độ cắt khung hình:
        # 1. Ở xa (~1.8m - 3.0m): đầu người nằm trọn dưới mép trên (top_margin >= 0.06) -> H_eff = 1.55m
        # 2. Ở trung bình (~1.0m - 1.6m): đầu chạm mép trên nhưng bàn chân còn nằm trong ảnh -> H_eff = 1.00m
        # 3. Ở rất gần (< 0.8m): cắt cả mép trên lẫn mép dưới (chỉ thấy đoạn cẳng chân) -> H_eff = 0.48m
        if top_margin >= 0.06:
            effective_height_m = 1.55
        elif bottom_margin >= 0.05:
            effective_height_m = 1.00
        else:
            effective_height_m = 0.48

        z_raw = (fy * effective_height_m) / box_h

        # Giảm thêm nếu khung chân chiếm > 78% chiều cao ảnh khi đứng áp sát
        if bottom_margin < 0.04 and h_ratio > 0.78:
            z_raw *= max(0.30, 1.0 - (h_ratio - 0.72))

        z_raw = max(0.10, min(5.0, z_raw))

        if self.ema_z is None:
            self.ema_z = z_raw
        else:
            alpha = 0.45
            self.ema_z = alpha * z_raw + (1.0 - alpha) * self.ema_z

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
