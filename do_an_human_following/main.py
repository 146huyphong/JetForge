#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - Điểm khởi chạy chính điều phối toàn bộ hệ thống JetBot Human Follower
Hỗ trợ cờ --no-drive để kiểm thử thị giác & ước lượng khoảng cách an toàn trước khi cho xe chạy.
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
from modules.data_logger import DataLogger
from modules.web_server import WebCockpitServer


class JetBotFollowerApp:
    def __init__(self, no_drive=False, show_window=False):
        self.no_drive = no_drive
        self.show_window = show_window
        self.running = False
        self.state_lock = threading.Lock()

        # Khởi tạo ROS Node nếu chạy trong môi trường ROS
        self.cmd_vel_pub = None
        if ROS_AVAILABLE:
            try:
                rospy.init_node("jetbot_human_follower", anonymous=True, disable_signals=True)
                self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
                rospy.loginfo("[Main] Đã khởi tạo ROS Node: jetbot_human_follower")
            except Exception as e:
                print(f"[Main] Khởi tạo ROS thất bại, chạy chế độ Standalone: {e}")

        # Khởi tạo các modules thành phần
        self.motors = MotorController()
        self.battery = BatteryMonitor()
        self.camera = CameraStreamer()
        self.detector = HumanDetector()
        self.tracker = HumanTracker()
        self.dist_estimator = DistanceEstimator()
        self.collision_guard = CollisionGuard()
        self.motion_ctrl = MotionController()
        self.fsm = BehaviorFSM()
        self.logger = DataLogger()

        self.latest_frame_shape = (config.PROCESS_HEIGHT, config.PROCESS_WIDTH, 3)

        # Dữ liệu chia sẻ với Web Cockpit
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

    # ==================== CALLBACKS TỪ WEB COCKPIT ====================
    def _handle_select_point(self, nx, ny):
        with self.state_lock:
            tid = self.tracker.select_target_by_norm_point(nx, ny, self.latest_frame_shape)
            if tid is not None and self.fsm.mode == "IDLE":
                self.fsm.set_mode("AUTO")
            return tid

    def _handle_clear_target(self):
        with self.state_lock:
            self.tracker.clear_target()
            if self.fsm.mode == "AUTO":
                self.fsm.set_mode("IDLE")

    def _handle_set_mode(self, mode):
        with self.state_lock:
            self.fsm.set_mode(mode)
            if mode in ("IDLE", "ESTOP"):
                self.motors.stop()
                self.motion_ctrl.reset()

    def _handle_update_pid(self, pid_type, kp, ki, kd):
        with self.state_lock:
            self.motion_ctrl.update_gains(pid_type, kp, ki, kd)

    def _handle_manual_cmd(self, linear, angular):
        with self.state_lock:
            if (abs(float(linear)) > 1e-4 or abs(float(angular)) > 1e-4) and self.fsm.mode != "ESTOP":
                self.fsm.set_mode("MANUAL")
            self.fsm.set_manual_cmd(linear, angular)

    # ==================== VẼ HUD LÊN KHUNG HÌNH ====================
    def _draw_overlay(self, frame, tracks, est_info, collision_info, state, reason, v_mps, w_rads, fps):
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Đường tâm quang học
        cv2.line(vis, (w // 2, 0), (w // 2, h), (100, 100, 100), 1)

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

        if collision_info["is_blocked"]:
            cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
            cv2.putText(vis, f"OBSTACLE BLOCKED ({collision_info['blocked_prob']*100:.0f}%)",
                        (16, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        drive_tag = " [NO-DRIVE]" if self.no_drive else ""
        hud_text = f"{state} ({reason}){drive_tag} | {fps:.1f}FPS | v={v_mps:+.2f} w={w_rads:+.2f}"
        cv2.putText(vis, hud_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
        return vis

    # ==================== VÒNG LẶP ĐIỀU KHIỂN CHÍNH ====================
    def run(self):
        self.running = True
        self.camera.start()
        self.web_server.start()

        target_dt = 1.0 / config.CONTROL_LOOP_HZ
        ema_fps = 0.0
        start_time = time.time()
        prev_loop_time = start_time

        print("[Main] Hệ thống JetBot Human Follower đã khởi động!")
        if self.no_drive:
            print("[Main] CHẾ ĐỘ AN TOÀN (--no-drive): Động cơ bị khóa, chỉ hiển thị thị giác & tính toán.")
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

                # 1. Nhận diện người & Vật cản
                detections = self.detector.detect(frame)
                collision_info = self.collision_guard.predict(frame)

                with self.state_lock:
                    # 2. Tracking & Re-ID (hỗ trợ cổng khớp quay nhanh + Coasting)
                    tracks = self.tracker.update(detections, frame, now=now)

                    if self.fsm.mode == "AUTO" and self.tracker.locked_id is None and len(tracks) > 0:
                        self.tracker.auto_lock_primary(tracks)

                    locked_track = next((t for t in tracks if t["is_locked"]), None)

                    # 3. Ước lượng khoảng cách sàn (có bù góc nghiêng) & góc bearing
                    est_info = None
                    if locked_track is not None:
                        est_info = self.dist_estimator.estimate(locked_track["bbox"], frame.shape, frame=frame)

                    # 4. Máy trạng thái & Bộ điều khiển PID + P
                    state, reason, v_mps, w_rads = self.fsm.step(
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

                    # 5. Quy đổi động học vi sai -> duty hai bánh [-1, 1]
                    left_duty, right_duty = self.motion_ctrl.to_wheels(v_mps, w_rads)

                    # Gửi lệnh mỗi chu kỳ (kể cả lệnh 0) để nuôi Watchdog an toàn
                    if self.no_drive:
                        self.motors.drive_wheels(0.0, 0.0)
                    else:
                        self.motors.drive_wheels(left_duty, right_duty)

                    if self.cmd_vel_pub is not None:
                        twist = Twist()
                        twist.linear.x = 0.0 if self.no_drive else v_mps
                        twist.angular.z = 0.0 if self.no_drive else w_rads
                        self.cmd_vel_pub.publish(twist)

                # Tính FPS mượt
                loop_elapsed = max(1e-4, time.time() - now)
                inst_fps = 1.0 / loop_elapsed
                ema_fps = inst_fps if ema_fps == 0.0 else (0.85 * ema_fps + 0.15 * inst_fps)

                bat_status = self.battery.get_status()
                dist_m = est_info["distance_m"] if est_info else 0.0
                bearing_deg = est_info["bearing_deg"] if est_info else 0.0
                err_d = est_info["error_dist"] if est_info else 0.0

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
                        "is_blocked": collision_info["is_blocked"],
                        "battery": bat_status,
                        "pid": self.motion_ctrl.get_gains(),
                    })

                self.logger.log({
                    "elapsed_s": round(now - start_time, 3),
                    "state": state,
                    "reason": reason,
                    "fps": round(ema_fps, 2),
                    "frame_age_ms": round(frame_age * 1000.0, 1),
                    "target_id": self.tracker.locked_id if self.tracker.locked_id is not None else -1,
                    "distance_m": round(dist_m, 3),
                    "bearing_deg": round(bearing_deg, 2),
                    "error_dist": round(err_d, 4),
                    "linear_v": round(v_mps, 4),
                    "angular_w": round(w_rads, 4),
                    "battery_v": bat_status["voltage"],
                    "battery_pct": bat_status["percentage"],
                    "blocked_prob": round(collision_info["blocked_prob"], 4),
                })

                vis_frame = self._draw_overlay(
                    frame, tracks, est_info, collision_info, state, reason, v_mps, w_rads, ema_fps
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
            print("\n[Main] Nhận tín hiệu dừng từ bàn phím...")
        finally:
            self.shutdown()

    def shutdown(self):
        if not self.running:
            return
        self.running = False
        print("[Main] Đang tắt an toàn toàn bộ hệ thống...")
        self.motors.shutdown()
        self.camera.stop()
        self.web_server.stop()
        self.detector.cleanup()
        self.collision_guard.cleanup()
        self.battery.shutdown()
        self.logger.close()
        if self.show_window:
            cv2.destroyAllWindows()
        print("[Main] Đã tắt hoàn tất.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JetBot Human Tracking & Following")
    parser.add_argument("--no-drive", action="store_true",
                        help="Chế độ an toàn: khóa động cơ, chỉ chạy camera + AI + hiển thị")
    parser.add_argument("--show", action="store_true",
                        help="Mở cửa sổ OpenCV trực tiếp trên màn hình Jetson")
    args, _ = parser.parse_known_args()

    app = JetBotFollowerApp(no_drive=args.no_drive, show_window=args.show)

    def _sig_handler(sig, frame):
        app.shutdown()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
    app.run()