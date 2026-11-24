#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from perception import CameraIntrinsics, Detection, bearing_distance
from tracking import Tracker
from safety import SafetyInput, SafetyLayer, to_wheels
from follow_controller import HumanFollowingPipeline


def test_close_range_ground_projection():
    cam = CameraIntrinsics.from_resolution(640, 480, fov_deg=62.0, cam_height=0.12)
    th, z = bearing_distance((280, 0, 360, 420), cam, "person")
    expected_z = cam.fy * 0.12 / (420 - cam.cy)
    assert abs(z - expected_z) < 1e-6, f"Sai khoang cach mat san: {z} vs {expected_z}"


def test_bytetrack_2pass():
    tr = Tracker(high=0.5, low=0.12)
    for k in range(4):
        out = tr.update([Detection("person", (100 + 4 * k, 50, 160 + 4 * k, 300), 0.85, float(k))])
    assert len(out) == 1 and out[0].track_id == 1
    out = tr.update([Detection("person", (116, 50, 176, 300), 0.22, 5.0)])
    assert len(out) == 1 and out[0].track_id == 1 and out[0].lost_frames == 0


def test_safety_and_to_wheels():
    l, r = to_wheels(0.0, 1.0, max_duty=0.40)
    assert l < 0.0 < r

    s = SafetyLayer()
    v, w, reason = s.filter(
        0.20,
        0.60,
        SafetyInput(now=10.2, last_cmd_t=10.0, visual_obstacles=[(0.0, 0.22)]),
    )
    assert v == 0.0 and w == 0.60 and reason == "OBSTACLE_TOO_CLOSE"


def test_pipeline_end_to_end():
    pipe = HumanFollowingPipeline(width=320, height=240, enable_log=False)
    for t in [1.0, 1.1, 1.2]:
        l, r, info = pipe.step([("person", (130, 40, 190, 210), 0.88)], frame_shape=(240, 320), now=t)
    assert info["state"] == "TRACKING" and info["track_id"] == 1
    assert info["safety_reason"] == "OK"


if __name__ == "__main__":
    test_close_range_ground_projection()
    test_bytetrack_2pass()
    test_safety_and_to_wheels()
    test_pipeline_end_to_end()
    print("[OK] Toan bo 4 bai kiem thu Giai doan 1 da dat!")
