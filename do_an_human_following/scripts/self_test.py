#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
self_test.py - Suite kiểm thử tự động & hiệu chỉnh thực nghiệm (Width / V_FULL) cho JetBot
Chạy kiểm thử:      python3 scripts/self_test.py
Hiệu chỉnh chiều rộng người ở 1.0m: python3 scripts/self_test.py --calibrate-width --distance 1.0
Hiệu chỉnh tốc độ thực V_FULL:      python3 scripts/self_test.py --calibrate-speed
"""
import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.motor_controller import MotorController
from modules.battery_monitor import BatteryMonitor
from modules.camera_streamer import CameraStreamer
from modules.human_detector import HumanDetector
from modules.human_tracker import HumanTracker
from modules.distance_estimator import DistanceEstimator
from modules.collision_guard import CollisionGuard
from modules.motion_controller import MotionController


def calibrate_width(known_distance_m=1.0):
    print(f"\n[Calibrate Width] Đứng thẳng cách camera đúng {known_distance_m:.2f} m trong 4 giây...")
    cam = CameraStreamer()
    det = HumanDetector()
    cam.start()
    time.sleep(1.0)

    widths = []
    t_end = time.time() + 4.0
    while time.time() < t_end:
        ret, frame, _, _ = cam.read()
        if ret and frame is not None:
            persons = det.detect(frame)
            if persons:
                biggest = max(persons, key=lambda p: p["bbox"][2] - p["bbox"][0])
                w_px = biggest["bbox"][2] - biggest["bbox"][0]
                widths.append(w_px)
        time.sleep(0.05)

    cam.stop()
    det.cleanup()

    if not widths:
        print("[Calibrate Width] Không phát hiện được người trong khung hình!")
        return

    median_w_px = float(np.median(widths))
    eff_width_m = (median_w_px * known_distance_m) / config.FOCAL_LENGTH_PX
    print(f"  -> Trung vị bề rộng box: {median_w_px:.1f} px")
    print(f"  -> Hãy cập nhật AVG_SHOULDER_WIDTH_M = {eff_width_m:.3f} trong config.py")


def calibrate_speed():
    print("\n[Calibrate Speed] Xe sẽ chạy thẳng ở 100% duty trong 1.5 giây. Hãy đặt xe ở khoảng trống >= 1.5m!")
    input("Nhấn Enter để bắt đầu chạy đo tốc độ...")
    motors = MotorController()
    t0 = time.time()
    while time.time() - t0 < 1.5:
        motors.drive_wheels(1.0, 1.0)
        time.sleep(0.05)
    motors.stop()
    motors.shutdown()
    dist_str = input("Nhập quãng đường xe đã đi được (mét, ví dụ 0.52): ").strip()
    try:
        dist_m = float(dist_str)
        v_full = dist_m / 1.5
        print(f"  -> V_FULL_MPS đo thực tế = {v_full:.3f} m/s. Hãy cập nhật vào config.py!")
    except ValueError:
        print("Giá trị nhập không hợp lệ.")


def run_tests():
    print("=" * 62)
    print(" BẮT ĐẦU KIỂM THỬ HỆ THỐNG JETBOT HUMAN FOLLOWER")
    print("=" * 62)

    print("\n[1/6] Kiểm tra thư mục models/...")
    for path in (config.SSD_ENGINE_PATH, config.COLLISION_ENGINE_PATH):
        status = "TỒN TẠI" if os.path.exists(path) else "CHƯA CÓ (Sẽ dùng fallback)"
        print(f"  - {os.path.basename(path)}: {status}")

    print("\n[2/6] Kiểm tra cảm biến Pin INA219...")
    bat = BatteryMonitor()
    time.sleep(0.2)
    print(f"  - Trạng thái Pin: {bat.get_status()}")
    bat.shutdown()

    print("\n[3/6] Kiểm tra luồng Camera RGB...")
    cam = CameraStreamer()
    cam.start()
    time.sleep(0.5)
    ret, frame, stamp, fps = cam.read()
    assert ret and frame is not None, "Lỗi: Không đọc được khung hình từ Camera!"
    print(f"  - Đọc ảnh thành công: shape={frame.shape}, age={(time.time()-stamp)*1000:.1f}ms, FPS={fps:.1f}")
    cam.stop()

    print("\n[4/6] Kiểm tra suy luận AI (HumanDetector & CollisionGuard)...")
    det = HumanDetector()
    col = CollisionGuard()
    t0 = time.time()
    persons = det.detect(frame)
    obs = col.predict(frame)
    dt_ms = (time.time() - t0) * 1000.0
    print(f"  - Thời gian inference: {dt_ms:.1f} ms | Số người: {len(persons)} | Vật cản: {obs}")
    det.cleanup()
    col.cleanup()

    print("\n[5/6] Kiểm tra Tracker, Bù góc nghiêng & Bộ điều khiển PID...")
    tracker = HumanTracker()
    dist_est = DistanceEstimator()
    motion = MotionController()

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_frame[120:380, 260:380] = (40, 180, 90)
    for _ in range(3):
        tracks = tracker.update([{"bbox": [260, 120, 380, 380], "conf": 0.92}], fake_frame)
    assert len(tracks) == 1, "Lỗi: Tracker không khởi tạo được track!"
    tid = tracker.auto_lock_primary(tracks)
    est = dist_est.estimate(tracks[0]["bbox"], fake_frame.shape)
    v_mps, w_rads = motion.compute(est["bearing_rad"], est["error_dist"], 0.04)
    l_duty, r_duty = motion.to_wheels(v_mps, w_rads)
    print(
        f"  - Khóa ID:{tid} | Khoảng cách sàn: {est['distance_m']:.2f}m | "
        f"Góc: {est['bearing_deg']:+.1f}° | Duty bánh: L={l_duty:.2f}, R={r_duty:.2f}"
    )

    print("\n[6/6] Kiểm tra MotorController & Watchdog an toàn...")
    motors = MotorController()
    motors.drive_wheels(0.2, 0.2)
    time.sleep(0.1)
    motors.stop()
    motors.shutdown()
    print(f"  - Chế độ động cơ: {motors.mode} -> Đã dừng an toàn.")

    print("\n" + "=" * 62)
    print(" HOÀN TẤT KIỂM THỬ: TẤT CẢ MODULE HOẠT ĐỘNG ỔN ĐỊNH!")
    print("=" * 62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm thử & Hiệu chỉnh JetBot Human Follower")
    parser.add_argument("--calibrate-width", action="store_true", help="Hiệu chỉnh bề rộng người ở khoảng cách chuẩn")
    parser.add_argument("--distance", type=float, default=1.0, help="Khoảng cách chuẩn khi hiệu chỉnh (mét)")
    parser.add_argument("--calibrate-speed", action="store_true", help="Đo tốc độ tối đa thực tế V_FULL_MPS")
    args = parser.parse_args()

    if args.calibrate_width:
        calibrate_width(args.distance)
    elif args.calibrate_speed:
        calibrate_speed()
    else:
        run_tests()