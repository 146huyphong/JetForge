#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - Diem khoi chay chinh dieu phoi toan bo he thong JetBot Human Follower (Giai doan 1)
"""
import argparse
import math
import time
import signal
import threading
import cv2

try:
    import rospy
    from geometry_msgs.msg import Twist
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

import config
from modules.motor_controller import MotorController
from modules.battery_monitor import BatteryMonitor
from modules.camera_streamer import CameraStreamer
from modules.human_detector import HumanDetector
from modules.human_tracker import HumanTracker
from modules.distance_estimator import DistanceEstimator
from modules.collision_guard import CollisionGuard
from modules.behavior_fsm import BehaviorFSM
from modules.motion_controller import MotionController
from modules.safety_layer import SafetyLayer
from modules.runlog import RunLogger
from modules.web_server import WebCockpitServer


class JetBotFollowerApp:
    def __init__(self, no_drive=False, show_window=False):
        self.no_drive = no_drive
        self.show_window = show_window
        self.running = False
        self.state_lock = threading.Lock()

        self.cmd_vel_pub = None
        if ROS_AVAILABLE:
            try:
                rospy.init_node("jetbot_human_follower", anonymous=True, disable_signals=True)
                self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
                rospy.loginfo("[Main] Da khoi tao ROS Node: jetbot_human_follower")
            except Exception as e:
                print(f"[Main] Khoi tao ROS that bai, chay che do Standalone: {e}")

        self.motors = MotorController()
        self.battery = BatteryMonitor()
        self.camera = CameraStreamer()
        self.detector = HumanDetector()
        self.tracker = HumanTracker()
        self.dist_estimator = DistanceEstimator()
        self.collision_guard = CollisionGuard()
        self.motion_ctrl = MotionController()
        self.fsm = BehaviorFSM()
        self.safety = SafetyLayer(stop_dist_m=0.42, caution_dist_m=0.70, front_half_angle_deg=32.0)
        self.logger = RunLogger(config.LOGS_DIR, enabled=config.ENABLE_LOGGING)

        self.latest_frame_shape = (config.PROCESS_HEIGHT, config.PROCESS_WIDTH, 3)
        self._last_safety_reason = "OK"

        self.shared_telemetry = {
            "state": "IDLE",
            "reason": "INIT",
            "mode": "IDLE",
            "no_drive": self.no_drive,
            "locked_id": None,
            "fps": 0.0,
            "frame_age_ms": 0.0,
            "distance_m": 0.0,
            "bearing_deg": 0.0,
            "error_dist": 0.0,
            "linear_v": 0.0,
            "angular_w": 0.0,
            "left_duty": 0.0,
            "right_duty": 0.0,
            "blocked_prob": 0.0,
            "is_blocked": False,
            "safety_state": "OK",
            "safety_reason": "OK",
            "min_obstacle_m": None,
            "battery": self.battery.get_status(),
            "pid": self.motion_ctrl.get_gains(),
        }

        callbacks = {
            "on_select_point": self._handle_select_point,
            "on_set_mode": self._handle_set_mode,
            "on_update_pid": self._handle_update_pid,
            "on_manual_cmd": self._handle_manual_cmd,
            "on_clear_target": self._handle_clear_target,
        }
        self.web_server = WebCockpitServer(self.shared_telemetry, callbacks)

    def _handle_select_point(self, nx, ny):
        with self.state_lock:
            tid = self.tracker.select_target_by_norm_point(nx, ny, self.latest_frame_shape)
            if tid is not None and self.fsm.mode == "IDLE":
                self.fsm.set_mode("AUTO")
            if tid is not None:
                self.logger.event(time.time(), "TARGET_LOCKED", track_id=tid)
            return tid

    def _handle_clear_target(self):
        with self.state_lock:
            self.tracker.clear_target()
            if self.fsm.mode == "AUTO":
                self.fsm.set_mode("IDLE")
            self.logger.event(time.time(), "TARGET_CLEARED")

    def _handle_set_mode(self, mode):
        with self.state_lock:
            self.fsm.set_mode(mode)
            if mode in ("IDLE", "ESTOP"):
                self.motors.stop()
                self.motion_ctrl.reset()
            self.logger.event(time.time(), "MODE_CHANGED", mode=mode)

    def _handle_update_pid(self, pid_type, kp, ki, kd):
        with self.state_lock:
            self.motion_ctrl.update_gains(pid_type, kp, ki, kd)

    def _handle_manual_cmd(self, linear, angular):
        with self.state_lock:
            if (abs(float(linear)) > 1e-4 or abs(float(angular)) > 1e-4) and self.fsm.mode != "ESTOP":
                self.fsm.set_mode("MANUAL")
            self.fsm.set_manual_cmd(linear, angular)

    def _draw_overlay(self, frame, tracks, obstacles, est_info, collision_info, state, reason, v_mps, w_rads, fps):
        vis = frame.copy()
        h, w = vis.shape[:2]

        cv2.line(vis, (w // 2, 0), (w // 2, h), (100, 100, 100), 1)

        # Ve vat can thi giac tu YOLO (ghe, ban, ba lo...)
        for obs in obstacles:
            ox1, oy1, ox2, oy2 = [int(v) for v in obs["bbox"]]
            odist = obs.get("distance_m", 0.0)
            is_danger = odist < self.safety.stop_dist_m
            ocolor = (0, 69, 255) if is_danger else (255, 165, 0)
            cv2.rectangle(vis, (ox1, oy1), (ox2, oy2), ocolor, 2)
            olabel = f"{obs['cls']} {odist:.2f}m"
            cv2.putText(vis, olabel, (ox1, max(16, oy1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ocolor, 1)

        for tr in tracks:
            x1, y1, x2, y2 = [int(v) for v in tr["bbox"]]
            tid = tr["track_id"]
            is_locked = tr["is_locked"]

            color = (0, 230, 118) if is_locked else (0, 195, 255)
            thickness = 5 if is_locked else 4
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            cx_box, cy_box = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(vis, (cx_box, cy_box), 6, color, -1)

            label = f"ID:{tid}"
            if is_locked:
                label += " [LOCKED]"
                if est_info is not None:
                    label += f" Z={est_info['distance_m']:.2f}m th={est_info['bearing_deg']:+.1f}d"
            else:
                label += " [CLICK]"

            cv2.rectangle(vis, (x1, max(0, y1 - 26)), (x1 + len(label) * 9 + 10, y1), color, -1)
            cv2.putText(vis, label, (x1 + 5, max(18, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2)

        if collision_info["is_blocked"] or self.safety.state == "BLOCKED":
            cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
            warn_msg = f"SAFETY BLOCKED: {self.safety.reason}"
            if not math.isinf(self.safety.min_front_dist):
                warn_msg += f" ({self.safety.min_front_dist:.2f}m)"
            cv2.putText(vis, warn_msg, (16, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 255), 2)

        drive_tag = " [NO-DRIVE]" if self.no_drive else ""
        hud_text = f"{state} ({reason}){drive_tag} | {fps:.1f}FPS | v={v_mps:+.2f} w={w_rads:+.2f}"
        cv2.putText(vis, hud_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
        return vis

    def run(self):
        self.running = True
        self.camera.start()
        self.web_server.start()

        target_dt = 1.0 / config.CONTROL_LOOP_HZ
        ema_fps = 0.0
        start_time = time.time()
        prev_loop_time = start_time

        print("[Main] He thong JetBot Human Follower (Giai doan 1) da khoi dong!")
        if self.no_drive:
            print("[Main] CHE DO AN TOAN (--no-drive): Dong co bi khoa, chi hien thi thi giac & tinh toan.")
        print(f"[Main] Web Cockpit: http://<IP_JETSON>:{config.WEB_PORT}")

        try:
            while self.running:
                if ROS_AVAILABLE and rospy.is_shutdown():
                    break

                now = time.time()
                dt = max(1e-4, now - prev_loop_time)
                prev_loop_time = now

                ret, frame, frame_stamp, _ = self.camera.read()
                if not ret or frame is None:
                    self.motors.drive_wheels(0.0, 0.0)
                    time.sleep(0.01)
                    continue

                frame_age = max(0.0, now - frame_stamp)
                self.latest_frame_shape = frame.shape

                # 1. Nhan dien da lop (nguoi + vat can) & CollisionGuard du phong
                person_dets, raw_obstacles = self.detector.detect_all(frame)
                collision_info = self.collision_guard.predict(frame)

                obstacles = []
                for obs in raw_obstacles:
                    b_rad, d_m = self.dist_estimator.estimate_obstacle(
                        obs["bbox"], obs["cls"], frame_shape=frame.shape
                    )
                    item = dict(obs)
                    item["bearing_rad"] = b_rad
                    item["distance_m"] = d_m
                    obstacles.append(item)

                bat_status = self.battery.get_status()
                bat_v = bat_status.get("voltage") if isinstance(bat_status, dict) else None

                with self.state_lock:
                    # 2. ByteTrack 2 vong + HSV Re-ID
                    tracks = self.tracker.update(person_dets, frame, now=now)

                    if self.fsm.mode == "AUTO" and self.tracker.locked_id is None and len(tracks) > 0:
                        tid = self.tracker.auto_lock_primary(tracks)
                        if tid is not None:
                            self.logger.event(now, "AUTO_LOCKED", track_id=tid)

                    locked_track = next((t for t in tracks if t["is_locked"]), None)

                    # 3. Uoc luong khoang cach (co chieu mat san khi dung sat camera)
                    est_info = None
                    if locked_track is not None:
                        est_info = self.dist_estimator.estimate(locked_track["bbox"], frame.shape, frame=frame)

                    # 4. May trang thai & Bo dieu khien
                    state, reason, v_raw, w_raw = self.fsm.step(
                        locked_track=locked_track,
                        est_info=est_info,
                        collision_info=collision_info,
                        motion_ctrl=self.motion_ctrl,
                        frame_age=frame_age,
                        tracker_state=self.tracker.state,
                        last_side=self.tracker.last_side,
                        dt=dt,
                        now=now,
                    )

                    # 5. Lop an toan cuoi (SafetyLayer) truoc khi xuat dong co
                    v_mps, w_rads, safety_reason = self.safety.filter(
                        v_raw,
                        w_raw,
                        frame_age=frame_age,
                        estop=(self.fsm.mode == "ESTOP"),
                        battery_v=bat_v,
                        obstacles=obstacles,
                    )
                    if safety_reason != "OK":
                        reason = safety_reason
                    if safety_reason != self._last_safety_reason:
                        self.logger.event(now, "SAFETY_TRANSITION", reason=safety_reason)
                        self._last_safety_reason = safety_reason

                    # 6. Quy doi dong hoc vi sai -> duty hai banh [-1, 1]
                    left_duty, right_duty = self.motion_ctrl.to_wheels(v_mps, w_rads)

                    if self.no_drive:
                        self.motors.drive_wheels(0.0, 0.0)
                    else:
                        self.motors.drive_wheels(left_duty, right_duty)

                    if self.cmd_vel_pub is not None:
                        twist = Twist()
                        twist.linear.x = 0.0 if self.no_drive else v_mps
                        twist.angular.z = 0.0 if self.no_drive else w_rads
                        self.cmd_vel_pub.publish(twist)

                loop_elapsed = max(1e-4, time.time() - now)
                inst_fps = 1.0 / loop_elapsed
                ema_fps = inst_fps if ema_fps == 0.0 else (0.85 * ema_fps + 0.15 * inst_fps)

                dist_m = est_info["distance_m"] if est_info else 0.0
                bearing_deg = est_info["bearing_deg"] if est_info else 0.0
                err_d = est_info["error_dist"] if est_info else 0.0
                min_obs = None if math.isinf(self.safety.min_front_dist) else round(self.safety.min_front_dist, 2)

                self.logger.csv_throttled(
                    now,
                    config.LOG_INTERVAL_SEC,
                    "commands.csv",
                    ["t", "mode", "state", "reason", "locked_id", "dist_m", "bearing_deg", "v_mps", "w_rads", "l_duty", "r_duty", "min_obs_m"],
                    [
                        round(now, 3),
                        self.fsm.mode,
                        state,
                        reason,
                        self.tracker.locked_id,
                        round(dist_m, 3),
                        round(bearing_deg, 1),
                        round(v_mps, 3),
                        round(w_rads, 3),
                        round(left_duty, 3),
                        round(right_duty, 3),
                        min_obs,
                    ],
                )

                with self.state_lock:
                    self.shared_telemetry.update({
                        "state": state,
                        "reason": reason,
                        "mode": self.fsm.mode,
                        "no_drive": self.no_drive,
                        "locked_id": self.tracker.locked_id,
                        "fps": round(ema_fps, 1),
                        "frame_age_ms": round(frame_age * 1000.0, 1),
                        "distance_m": round(dist_m, 2),
                        "bearing_deg": round(bearing_deg, 1),
                        "error_dist": round(err_d, 3),
                        "linear_v": round(v_mps, 3),
                        "angular_w": round(w_rads, 3),
                        "left_duty": round(left_duty, 3),
                        "right_duty": round(right_duty, 3),
                        "blocked_prob": round(collision_info["blocked_prob"], 3),
                        "is_blocked": bool(collision_info["is_blocked"] or self.safety.state == "BLOCKED"),
                        "safety_state": self.safety.state,
                        "safety_reason": safety_reason,
                        "min_obstacle_m": min_obs,
                        "battery": bat_status,
                        "pid": self.motion_ctrl.get_gains(),
                    })

                vis_frame = self._draw_overlay(
                    frame, tracks, obstacles, est_info, collision_info, state, reason, v_mps, w_rads, ema_fps
                )
                self.web_server.update_frame(vis_frame)

                if self.show_window:
                    cv2.imshow("JetBot Human Follower", vis_frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break

                sleep_time = target_dt - (time.time() - now)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[Main] Nhan tin hieu dung tu ban phim...")
        finally:
            self.shutdown()

    def shutdown(self):
        if not self.running:
            return
        self.running = False
        print("[Main] Dang tat an toan toan bo he thong...")
        self.motors.shutdown()
        self.camera.stop()
        self.web_server.stop()
        self.detector.cleanup()
        self.collision_guard.cleanup()
        self.battery.shutdown()
        self.logger.close()
        if self.show_window:
            cv2.destroyAllWindows()
        print("[Main] Da tat hoan tat.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JetBot Human Tracking & Following")
    parser.add_argument("--no-drive", action="store_true",
                        help="Che do an toan: khoa dong co, chi chay camera + AI + hien thi")
    parser.add_argument("--show", action="store_true",
                        help="Mo cua so OpenCV truc tiep tren man hinh Jetson")
    args, _ = parser.parse_known_args()

    app = JetBotFollowerApp(no_drive=args.no_drive, show_window=args.show)

    def _sig_handler(sig, frame):
        app.shutdown()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
    app.run()