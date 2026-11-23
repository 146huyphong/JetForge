#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
motion_controller.py - Bộ điều khiển bám PID (góc bearing theta) + PID khoảng cách
Tích hợp cos^2(theta) coupling, anti-windup, slew-rate limiter và động học vi sai hai bánh.
"""
import math
import config


class PIDController:
    def __init__(self, kp, ki, kd, out_min, out_max, integral_limit=0.45):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.out_min = float(out_min)
        self.out_max = float(out_max)
        self.integral_limit = float(integral_limit)

        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def update_gains(self, kp, ki, kd):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

    def compute(self, error, dt):
        dt = max(1e-3, min(0.25, dt))

        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))

        if self.prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / dt
        self.prev_error = error

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        return max(self.out_min, min(self.out_max, output))


class MotionController:
    def __init__(self):
        self.angular_pid = PIDController(
            config.ANGULAR_PID["kp"],
            config.ANGULAR_PID["ki"],
            config.ANGULAR_PID["kd"],
            -config.MAX_ANGULAR_SPEED,
            config.MAX_ANGULAR_SPEED,
            integral_limit=config.INTEGRAL_MAX,
        )
        self.linear_pid = PIDController(
            config.LINEAR_PID["kp"],
            config.LINEAR_PID["ki"],
            config.LINEAR_PID["kd"],
            -config.MAX_LINEAR_SPEED,
            config.MAX_LINEAR_SPEED,
            integral_limit=0.35,
        )
        self.theta_dead_rad = math.radians(config.THETA_DEADBAND_DEG)
        self.prev_linear = 0.0
        self.prev_angular = 0.0

    def _slew_limit(self, target, current, max_step):
        diff = target - current
        if diff > max_step:
            return current + max_step
        if diff < -max_step:
            return current - max_step
        return target

    def apply_slew_limit(self, target_v, target_w):
        smooth_v = self._slew_limit(target_v, self.prev_linear, config.MAX_ACCEL_STEP)
        smooth_w = self._slew_limit(target_w, self.prev_angular, config.MAX_ACCEL_STEP * 3.0)
        self.prev_linear = smooth_v
        self.prev_angular = smooth_w
        return smooth_v, smooth_w

    def compute(self, bearing_rad, error_dist, dt):
        """
        Quy ước chuẩn robot:
        - bearing_rad > 0 (người bên phải) -> omega < 0 (quay phải)
        - error_dist > 0 (người ở xa hơn d*) -> v > 0 (tiến tới)
        """
        if abs(bearing_rad) < self.theta_dead_rad:
            # Giữ thành phần tích phân I để triệt tiêu lệch tĩnh do 2 bánh không đều
            w_cmd = -(self.angular_pid.ki * self.angular_pid.integral)
            self.angular_pid.prev_error = bearing_rad
        else:
            w_cmd = -self.angular_pid.compute(bearing_rad, dt)

        if abs(error_dist) < config.Z_DEADBAND_M:
            v_cmd = 0.0
            self.linear_pid.prev_error = error_dist
        else:
            v_cmd = self.linear_pid.compute(error_dist, dt)

        # Lệch hướng nhiều -> ưu tiên xoay trước, giảm tốc độ tiến theo cos^2(theta)
        v_cmd *= max(0.0, math.cos(bearing_rad)) ** 2
        return v_cmd, w_cmd

    def to_wheels(self, v_mps, w_rads):
        """
        Quy đổi (v [m/s], omega [rad/s]) ra duty hai bánh [-1.0, 1.0]:
        v_left = v - w * L / 2 ; v_right = v + w * L / 2
        """
        half_l = 0.5 * config.WHEELBASE_M
        v_left = v_mps - w_rads * half_l
        v_right = v_mps + w_rads * half_l

        v_full = max(0.05, config.V_FULL_MPS)
        left_duty = v_left / v_full
        right_duty = v_right / v_full

        max_mag = max(abs(left_duty), abs(right_duty))
        if max_mag > 1.0:
            left_duty /= max_mag
            right_duty /= max_mag

        return float(left_duty), float(right_duty)

    def reset(self):
        self.angular_pid.reset()
        self.linear_pid.reset()
        self.prev_linear = 0.0
        self.prev_angular = 0.0

    def update_gains(self, pid_type, kp, ki, kd):
        if pid_type == "angular":
            self.angular_pid.update_gains(kp, ki, kd)
        elif pid_type == "linear":
            self.linear_pid.update_gains(kp, ki, kd)

    def get_gains(self):
        return {
            "angular": {
                "kp": self.angular_pid.kp,
                "ki": self.angular_pid.ki,
                "kd": self.angular_pid.kd,
            },
            "linear": {
                "kp": self.linear_pid.kp,
                "ki": self.linear_pid.ki,
                "kd": self.linear_pid.kd,
            },
        }