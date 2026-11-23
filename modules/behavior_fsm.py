#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
behavior_fsm.py - Máy trạng thái hành vi nâng cao:
Phanh sớm đón đầu quán tính + giảm tốc êm khi tiệm cận vùng giữ khoảng cách 0.80m.
"""
import time
import config


class BehaviorFSM:
    def __init__(self):
        self.mode = "IDLE"      # Chế độ điều hành: IDLE, AUTO, MANUAL, ESTOP
        self.state = "IDLE"     # Trạng thái thực thi thực tế
        self.reason = "INIT"
        self.lost_since = None
        self.last_z = None
        self.last_clipped = False
        self.manual_linear = 0.0
        self.manual_angular = 0.0
        self.manual_hold_until = 0.0

        self._smooth_v_cmd = 0.0
        self._pulse_end_ts = 0.0
        self._settle_end_ts = 0.0
        self._pulse_w_cmd = 0.0

        # Biến điều khiển xoay tại chỗ
        self._w_sign = None
        self._hold_ema_bearing = None
        self._align_confirm_frames = 0
        self._align_confirm_dir = 0
        self._last_turn_dir = 0
        self._last_turn_ts = 0.0

        # Trạng thái trễ Hysteresis khoảng cách: "BACKUP", "HOLD", "FOLLOW"
        self._long_mode = "HOLD"
        self._long_pending_mode = None
        self._long_confirm_frames = 0
        self._dist_settle_until = 0.0

    def set_mode(self, mode):
        valid_modes = {"IDLE", "AUTO", "MANUAL", "ESTOP"}
        if mode in valid_modes:
            self.mode = mode
            self.lost_since = None
            self._reset_motion_helpers()
            if mode != "MANUAL":
                self.manual_linear = 0.0
                self.manual_angular = 0.0

    def set_manual_cmd(self, linear, angular):
        lin = max(-config.MAX_LINEAR_SPEED, min(config.MAX_LINEAR_SPEED, float(linear)))
        ang = max(-config.MAX_ANGULAR_SPEED, min(config.MAX_ANGULAR_SPEED, float(angular)))
        now = time.time()
        if abs(lin) > 1e-4 or abs(ang) > 1e-4:
            self.manual_linear = lin
            self.manual_angular = ang
            self.manual_hold_until = now + 0.18
        elif now >= self.manual_hold_until:
            self.manual_linear = 0.0
            self.manual_angular = 0.0

    def _reset_motion_helpers(self):
        self._smooth_v_cmd = 0.0
        self._pulse_end_ts = 0.0
        self._settle_end_ts = 0.0
        self._pulse_w_cmd = 0.0
        self._hold_ema_bearing = None
        self._align_confirm_frames = 0
        self._align_confirm_dir = 0
        self._long_mode = "HOLD"
        self._long_pending_mode = None
        self._long_confirm_frames = 0
        self._dist_settle_until = 0.0

    def _update_longitudinal_mode(self, dist_now, now):
        """Phanh sớm đón đầu quán tính: dừng tiến tại 0.90m, dừng lùi tại 0.68m để xe tự trôi về đúng 0.76-0.84m."""
        if self._long_mode == "HOLD" and now < self._dist_settle_until:
            self._long_pending_mode = None
            self._long_confirm_frames = 0
            return "HOLD"

        desired = self._long_mode

        if self._long_mode == "HOLD":
            if dist_now < 0.56:
                desired = "BACKUP"
            elif dist_now > 1.02:
                desired = "FOLLOW"
        elif self._long_mode == "BACKUP":
            if dist_now >= 0.68:
                desired = "HOLD"
        elif self._long_mode == "FOLLOW":
            if dist_now < 0.56:
                desired = "BACKUP"
            elif dist_now <= 0.90:
                desired = "HOLD"

        if desired == self._long_mode:
            self._long_pending_mode = None
            self._long_confirm_frames = 0
            return self._long_mode

        if desired == "HOLD":
            self._long_mode = "HOLD"
            self._long_pending_mode = None
            self._long_confirm_frames = 0
            self._dist_settle_until = now + 0.85
        else:
            if self._long_pending_mode == desired:
                self._long_confirm_frames += 1
            else:
                self._long_pending_mode = desired
                self._long_confirm_frames = 1

            if self._long_confirm_frames >= 3:
                self._long_mode = desired
                self._long_pending_mode = None
                self._long_confirm_frames = 0

        return self._long_mode

    def step(
        self,
        locked_track,
        est_info,
        collision_info,
        motion_ctrl,
        frame_age=0.0,
        tracker_state="IDLE",
        last_side=0.0,
        dt=0.04,
        now=None,
    ):
        now = time.time() if now is None else now

        # 0. Lớp an toàn độ trễ khung hình
        if frame_age > config.STALE_FRAME_SEC:
            self.state = "STALE"
            self.reason = "STREAM_STALE"
            self._reset_motion_helpers()
            motion_ctrl.reset()
            return self.state, self.reason, 0.0, 0.0

        # 1. Dừng khẩn cấp hoặc chế độ chờ
        if self.mode in ("IDLE", "ESTOP"):
            self.state = "IDLE"
            self.reason = self.mode
            self._reset_motion_helpers()
            motion_ctrl.reset()
            return self.state, self.reason, 0.0, 0.0

        # 2. Chế độ lái tay qua Web Cockpit
        if self.mode == "MANUAL":
            self.state = "MANUAL"
            if abs(self.manual_linear) < 1e-4 and abs(self.manual_angular) < 1e-4 and now >= self.manual_hold_until:
                self.reason = "MANUAL_READY"
                return self.state, self.reason, 0.0, 0.0
            if collision_info["is_blocked"] and self.manual_linear > 0:
                self.reason = "MANUAL_BLOCKED"
                return self.state, self.reason, 0.0, self.manual_angular
            self.reason = "MANUAL_OK"
            return self.state, self.reason, self.manual_linear, self.manual_angular

        # 3. Chế độ bám tự động (AUTO) - Ưu tiên né vật cản
        if collision_info["is_blocked"]:
            self.state = "AVOIDING"
            self.reason = "OBSTACLE_FRONT"
            self._reset_motion_helpers()
            motion_ctrl.reset()
            turn_w = -config.AVOID_TURN_RAD_S if last_side >= 0 else config.AVOID_TURN_RAD_S
            v, w = motion_ctrl.apply_slew_limit(0.0, turn_w)
            return self.state, self.reason, v, w

        # 4. Đang khóa & nhìn thấy mục tiêu
        if locked_track is not None and est_info is not None:
            self.state = "TRACKING"
            self.lost_since = None
            dist_now = est_info["distance_m"]
            bearing_deg = est_info["bearing_deg"]
            self.last_z = dist_now
            self.last_clipped = est_info["is_clipped"]

            if self._w_sign is None:
                _, w_test = motion_ctrl.compute(0.10, 0.0, 0.04)
                motion_ctrl.reset()
                self._w_sign = -1.0 if w_test < 0 else 1.0
            turn_sign = self._w_sign if bearing_deg >= 0 else -self._w_sign

            abs_b = abs(bearing_deg)
            long_mode = self._update_longitudinal_mode(dist_now, now)

            if long_mode == "BACKUP":
                # Lùi nhẹ tốc độ cố định thấp để không trôi quá 0.80m
                self.reason = "CLOSE_BACKUP"
                self._pulse_end_ts = 0.0
                self._settle_end_ts = 0.0
                self._hold_ema_bearing = None
                self._align_confirm_frames = 0
                motion_ctrl.linear_pid.reset()
                target_v = -0.085
                w_cmd = 0.0 if abs_b <= 6.5 else turn_sign * 0.22

            elif long_mode == "HOLD":
                # Vùng giữ khoảng cách (0.56m - 1.02m): ngắt hẳn động cơ tiến/lùi
                target_v = 0.0
                self._smooth_v_cmd = 0.0
                motion_ctrl.linear_pid.reset()
                motion_ctrl.angular_pid.reset()

                if self._hold_ema_bearing is None:
                    self._hold_ema_bearing = bearing_deg
                else:
                    self._hold_ema_bearing = 0.55 * bearing_deg + 0.45 * self._hold_ema_bearing

                smooth_b = self._hold_ema_bearing
                abs_smooth_b = abs(smooth_b)
                smooth_dir = 1 if smooth_b > 0 else -1
                smooth_turn_sign = self._w_sign if smooth_b >= 0 else -self._w_sign

                if now < self._pulse_end_ts:
                    self.reason = "HOLD_TURN"
                    w_cmd = self._pulse_w_cmd
                    v, w = motion_ctrl.apply_slew_limit(0.0, w_cmd)
                    return self.state, self.reason, 0.0, w

                elif now < self._settle_end_ts:
                    self.reason = "HOLD_SETTLE"
                    self._align_confirm_frames = 0
                    motion_ctrl.reset()
                    return self.state, self.reason, 0.0, 0.0

                else:
                    motion_ctrl.reset()
                    if abs_smooth_b <= 5.5:
                        self.reason = "HOLD_ZONE"
                        self._align_confirm_frames = 0
                        self._align_confirm_dir = 0
                        return self.state, self.reason, 0.0, 0.0

                    is_reverse = (self._last_turn_dir != 0 and smooth_dir != self._last_turn_dir)
                    if is_reverse and (now - self._last_turn_ts) < 0.28 and abs_smooth_b < 13.0:
                        self.reason = "HOLD_VERIFY"
                        self._align_confirm_frames = 0
                        return self.state, self.reason, 0.0, 0.0

                    if smooth_dir == self._align_confirm_dir:
                        self._align_confirm_frames += 1
                    else:
                        self._align_confirm_dir = smooth_dir
                        self._align_confirm_frames = 1

                    req_frames = 1 if abs_smooth_b > 11.0 else 2
                    if self._align_confirm_frames < req_frames:
                        self.reason = "HOLD_VERIFY"
                        return self.state, self.reason, 0.0, 0.0

                    self.reason = "HOLD_TURN"
                    self._align_confirm_frames = 0
                    w_mag = min(0.78, 0.64 + max(0.0, abs_smooth_b - 5.5) * 0.008)
                    pulse_dur = max(0.045, min(0.095, 0.036 + abs_smooth_b * 0.0028))

                    self._pulse_w_cmd = smooth_turn_sign * w_mag
                    self._pulse_end_ts = now + pulse_dur
                    self._settle_end_ts = self._pulse_end_ts + 0.165
                    self._last_turn_dir = smooth_dir
                    self._last_turn_ts = now

                    v, w = motion_ctrl.apply_slew_limit(0.0, self._pulse_w_cmd)
                    return self.state, self.reason, 0.0, w

            else:
                # long_mode == "FOLLOW": giảm tốc dần khi dưới 1.15m và phanh sớm tại 0.90m để không vượt lố tới 0.55m
                self.reason = "FOLLOW"
                self._pulse_end_ts = 0.0
                self._settle_end_ts = 0.0
                self._hold_ema_bearing = None
                self._align_confirm_frames = 0

                dist_err = max(0.0, dist_now - 0.90)
                target_v = min(0.130, max(0.085, 0.085 + dist_err * 0.09))

                if abs_b <= 4.5:
                    w_cmd = 0.0
                else:
                    err_deg = abs_b - 4.5
                    w_cmd = turn_sign * min(0.36, 0.12 + err_deg * 0.011)

            # Lọc mượt vận tốc tiến cho FOLLOW và CLOSE_BACKUP
            if target_v == 0.0 and abs(self._smooth_v_cmd) < 0.03:
                v_cmd = 0.0
            else:
                alpha_v = 0.25 if target_v > self._smooth_v_cmd else 0.55
                v_cmd = alpha_v * target_v + (1.0 - alpha_v) * self._smooth_v_cmd
            self._smooth_v_cmd = v_cmd

            v, w = motion_ctrl.apply_slew_limit(v_cmd, w_cmd)
            return self.state, self.reason, v, w

        # 5. Mất dấu ngắn hạn (COASTING)
        if tracker_state == "COASTING":
            self.state = "COASTING"
            self.reason = "BRIEF_OCCLUSION"
            if self.last_z is not None and self.last_z < 1.30:
                motion_ctrl.reset()
                return self.state, self.reason, 0.0, 0.0
            else:
                v, w = motion_ctrl.apply_slew_limit(
                    motion_ctrl.prev_linear * 0.3, motion_ctrl.prev_angular * 0.3
                )
                return self.state, self.reason, v, w

        # Mất dấu hoàn toàn -> Xóa tích phân PID
        motion_ctrl.angular_pid.integral = 0.0
        motion_ctrl.linear_pid.integral = 0.0
        self._reset_motion_helpers()

        if tracker_state == "IDLE":
            self.state = "IDLE"
            self.reason = "NO_TARGET"
            v, w = motion_ctrl.apply_slew_limit(0.0, 0.0)
            return self.state, self.reason, v, w

        if self.lost_since is None:
            self.lost_since = now

        elapsed_lost = now - self.lost_since

        # 6. Chế độ BACKUP
        if (
            self.last_z is not None
            and (self.last_z < config.TARGET_DISTANCE_M * 0.85 or self.last_clipped)
            and elapsed_lost <= config.BACKUP_DURATION_SEC
        ):
            self.state = "BACKUP"
            self.reason = "CLOSE_RANGE_LOST"
            v, w = motion_ctrl.apply_slew_limit(-config.BACKUP_SPEED_MPS, 0.0)
            return self.state, self.reason, v, w

        # 7. Chế độ SEARCHING
        if elapsed_lost <= config.SEARCH_TIMEOUT_SEC:
            self.state = "SEARCHING"
            self.reason = "SPIN_SEARCH"
            spin_w = -0.55 if last_side >= 0 else 0.55
            v, w = motion_ctrl.apply_slew_limit(0.0, spin_w)
            return self.state, self.reason, v, w

        # 8. Hết thời gian tìm kiếm -> Dừng an toàn
        self.state = "IDLE"
        self.reason = "SEARCH_TIMEOUT"
        motion_ctrl.reset()
        return self.state, self.reason, 0.0, 0.0
