#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
behavior_fsm.py - Máy trạng thái hành vi nâng cao:
IDLE, TRACKING, COASTING, BACKUP (lùi khi mất mục tiêu ở gần), SEARCHING, AVOIDING, MANUAL, STALE
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

    def set_mode(self, mode):
        valid_modes = {"IDLE", "AUTO", "MANUAL", "ESTOP"}
        if mode in valid_modes:
            self.mode = mode
            self.lost_since = None
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

        # 0. Lớp an toàn độ trễ khung hình (Stale Frame Guard)
        if frame_age > config.STALE_FRAME_SEC:
            self.state = "STALE"
            self.reason = "STREAM_STALE"
            motion_ctrl.reset()
            return self.state, self.reason, 0.0, 0.0

        # 1. Dừng khẩn cấp hoặc chế độ chờ
        if self.mode in ("IDLE", "ESTOP"):
            self.state = "IDLE"
            self.reason = self.mode
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
            motion_ctrl.reset()
            # Xoay né theo hướng mục tiêu lần cuối xuất hiện (âm = quay phải, dương = quay trái)
            turn_w = -config.AVOID_TURN_RAD_S if last_side >= 0 else config.AVOID_TURN_RAD_S
            v, w = motion_ctrl.apply_slew_limit(0.0, turn_w)
            return self.state, self.reason, v, w

        # 4. Đang khóa & nhìn thấy mục tiêu
        if locked_track is not None and est_info is not None:
            self.state = "TRACKING"
            self.lost_since = None
            self.last_z = est_info["distance_m"]
            self.last_clipped = est_info["is_clipped"]

            v_cmd, w_cmd = motion_ctrl.compute(
                est_info["bearing_rad"], est_info["error_dist"], dt
            )

            dist_now = est_info["distance_m"]
            w_cmd = max(-0.22, min(0.22, w_cmd))

            if dist_now < 0.95:
                # Ở cự ly gần: chỉ lùi thẳng nếu quá sát (<0.75m), khóa quay ngang (w=0) tránh mất mục tiêu
                self.reason = "CLOSE_STRAIGHT"
                v_cmd = -0.04 if dist_now < 0.75 else 0.0
                w_cmd = 0.0
                motion_ctrl.angular_pid.integral = 0.0
            elif dist_now < 1.35:
                self.reason = "HOLD_ZONE"
                v_cmd = 0.0
                w_cmd = max(-0.18, min(0.18, w_cmd * 0.45))
            else:
                self.reason = "FOLLOW"

            v, w = motion_ctrl.apply_slew_limit(v_cmd, w_cmd)
            return self.state, self.reason, v, w

        # 5. Mất dấu ngắn hạn (COASTING): nếu đang ở gần thì đứng yên chờ bắt lại chân, không xoay lố
        if tracker_state == "COASTING":
            self.state = "COASTING"
            self.reason = "BRIEF_OCCLUSION"
            if self.last_z is not None and self.last_z < 1.30:
                v, w = motion_ctrl.apply_slew_limit(0.0, 0.0)
            else:
                v, w = motion_ctrl.apply_slew_limit(
                    motion_ctrl.prev_linear * 0.3, motion_ctrl.prev_angular * 0.3
                )
            return self.state, self.reason, v, w

        # Mất dấu hoàn toàn -> Xóa tích phân PID tránh giật khi thấy lại
        motion_ctrl.angular_pid.integral = 0.0
        motion_ctrl.linear_pid.integral = 0.0

        if tracker_state == "IDLE":
            self.state = "IDLE"
            self.reason = "NO_TARGET"
            v, w = motion_ctrl.apply_slew_limit(0.0, 0.0)
            return self.state, self.reason, v, w

        if self.lost_since is None:
            self.lost_since = now

        elapsed_lost = now - self.lost_since

        # 6. Chế độ BACKUP: Mất mục tiêu khi đang ở gần -> lùi nhẹ để đưa người vào lại góc nhìn camera
        if (
            self.last_z is not None
            and (self.last_z < config.TARGET_DISTANCE_M * 0.85 or self.last_clipped)
            and elapsed_lost <= config.BACKUP_DURATION_SEC
        ):
            self.state = "BACKUP"
            self.reason = "CLOSE_RANGE_LOST"
            v, w = motion_ctrl.apply_slew_limit(-config.BACKUP_SPEED_MPS, 0.0)
            return self.state, self.reason, v, w

        # 7. Chế độ SEARCHING: Xoay về phía người xuất hiện lần cuối
        if elapsed_lost <= config.SEARCH_TIMEOUT_SEC:
            self.state = "SEARCHING"
            self.reason = "SPIN_SEARCH"
            spin_w = -config.SEARCH_SPIN_RAD_S if last_side >= 0 else config.SEARCH_SPIN_RAD_S
            v, w = motion_ctrl.apply_slew_limit(0.0, spin_w)
            return self.state, self.reason, v, w

        # 8. Hết thời gian tìm kiếm -> Dừng an toàn
        self.state = "IDLE"
        self.reason = "SEARCH_TIMEOUT"
        motion_ctrl.reset()
        return self.state, self.reason, 0.0, 0.0