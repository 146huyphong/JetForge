#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass
from typing import Sequence, Tuple


def to_wheels(
    v: float,
    w: float,
    wheelbase: float = 0.12,
    v_full: float = 0.40,
    max_duty: float = 0.40,
    deadzone: float = 0.085,
) -> Tuple[float, float]:
    """Quy doi (v [m/s], w [rad/s]) -> (left_pwm, right_pwm)."""
    vl = v - w * wheelbase / 2.0
    vr = v + w * wheelbase / 2.0
    l = vl / v_full
    r = vr / v_full
    k = max(1.0, abs(l) / max_duty, abs(r) / max_duty)
    l /= k
    r /= k

    def _apply_deadzone(pwm: float) -> float:
        if abs(pwm) < 1e-4:
            return 0.0
        scaled = math.copysign(deadzone + abs(pwm) * (1.0 - deadzone / max_duty), pwm)
        return max(-max_duty, min(max_duty, scaled))

    return round(_apply_deadzone(l), 4), round(_apply_deadzone(r), 4)


@dataclass
class SafetyConfig:
    cmd_timeout: float = 1.0
    stop_dist: float = 0.35
    caution_dist: float = 0.65
    front_half_angle: float = math.radians(35)
    v_max: float = 0.28
    w_max: float = 1.50
    battery_critical: float = 0.10


@dataclass
class SafetyInput:
    now: float
    last_cmd_t: float
    battery: float = 1.0
    camera_ok: bool = True
    need_camera: bool = True
    estop: bool = False
    visual_obstacles: Sequence[Tuple[float, float]] = ()


class SafetyLayer:
    def __init__(self, cfg: SafetyConfig = SafetyConfig()):
        self.cfg = cfg
        self.state = "OK"
        self.reason = ""
        self.min_front_dist = math.inf

    def min_obstacle_range(self, obstacles: Sequence[Tuple[float, float]]) -> float:
        m = math.inf
        for angle_rad, dist_m in obstacles:
            if abs(angle_rad) <= self.cfg.front_half_angle and dist_m < m:
                m = dist_m
        return m

    def filter(self, v: float, w: float, s: SafetyInput) -> Tuple[float, float, str]:
        c = self.cfg
        self.min_front_dist = self.min_obstacle_range(s.visual_obstacles)

        if s.estop:
            return self._block(0.0, 0.0, "ESTOP")
        if v == 0.0 and w == 0.0:
            self.state, self.reason = "OK", ""
            return 0.0, 0.0, "OK"
        if s.now - s.last_cmd_t > c.cmd_timeout:
            return self._block(0.0, 0.0, "CMD_TIMEOUT")
        if s.battery is not None and s.battery < c.battery_critical:
            return self._block(0.0, 0.0, "BATTERY_CRITICAL")
        if s.need_camera and not s.camera_ok:
            return self._block(0.0, 0.0, "CAMERA_LOST")

        w_clamped = max(-c.w_max, min(c.w_max, w))
        if v > 0.0 and self.min_front_dist < c.stop_dist:
            return self._block(0.0, w_clamped, "OBSTACLE_TOO_CLOSE")

        v_clamped = max(-c.v_max, min(c.v_max, v))
        if v_clamped > 0.0 and self.min_front_dist < c.caution_dist:
            ratio = max(0.35, (self.min_front_dist - c.stop_dist) / max(0.05, c.caution_dist - c.stop_dist))
            v_clamped *= ratio

        self.state, self.reason = "OK", ""
        return v_clamped, w_clamped, "OK"

    def _block(self, v: float, w: float, reason: str) -> Tuple[float, float, str]:
        self.state, self.reason = "BLOCKED", reason
        return v, w, reason
