#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
safety_layer.py - Lop loc an toan phan cung & tranh vat can thi giac truoc khi xuat PWM
"""
import math
import config


class SafetyLayer:
    def __init__(
        self,
        stop_dist_m=0.42,
        caution_dist_m=0.70,
        front_half_angle_deg=32.0,
    ):
        self.stop_dist_m = float(stop_dist_m)
        self.caution_dist_m = float(caution_dist_m)
        self.front_half_angle_rad = math.radians(float(front_half_angle_deg))

        self.state = "OK"
        self.reason = "OK"
        self.min_front_dist = math.inf

    def min_obstacle_range(self, obstacles):
        """obstacles: danh sach dict co keys ('bearing_rad', 'distance_m')"""
        m = math.inf
        for obs in obstacles:
            ang = abs(float(obs.get("bearing_rad", math.pi)))
            dist = float(obs.get("distance_m", math.inf))
            if ang <= self.front_half_angle_rad and dist < m:
                m = dist
        return m

    def filter(
        self,
        v_mps,
        w_rads,
        frame_age=0.0,
        estop=False,
        battery_v=None,
        obstacles=(),
    ):
        self.min_front_dist = self.min_obstacle_range(obstacles)

        if estop:
            return self._block(0.0, 0.0, "ESTOP")

        if frame_age > config.STALE_FRAME_SEC:
            return self._block(0.0, 0.0, "STALE_FRAME")

        if battery_v is not None and battery_v > 1.5 and battery_v < config.BATTERY_MIN_V:
            return self._block(0.0, 0.0, "BATTERY_CRITICAL")

        w_safe = max(-config.MAX_ANGULAR_SPEED, min(config.MAX_ANGULAR_SPEED, float(w_rads)))
        v_safe = max(-config.MAX_LINEAR_SPEED, min(config.MAX_LINEAR_SPEED, float(v_mps)))

        # Vat can ngay truoc mat: chan van toc tien v=0 nhung VAN CHO QUAY tai cho (w_safe)
        if v_safe > 0.0 and self.min_front_dist < self.stop_dist_m:
            return self._block(0.0, w_safe, "OBSTACLE_TOO_CLOSE")

        if v_safe > 0.0 and self.min_front_dist < self.caution_dist_m:
            span = max(0.05, self.caution_dist_m - self.stop_dist_m)
            ratio = max(0.35, (self.min_front_dist - self.stop_dist_m) / span)
            v_safe *= ratio
            self.state = "CAUTION"
            self.reason = "OBSTACLE_CAUTION"
            return v_safe, w_safe, self.reason

        self.state = "OK"
        self.reason = "OK"
        return v_safe, w_safe, "OK"

    def _block(self, v, w, reason):
        self.state = "BLOCKED"
        self.reason = reason
        return v, w, reason