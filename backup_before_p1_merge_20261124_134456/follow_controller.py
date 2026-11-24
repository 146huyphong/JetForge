#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import time
from typing import Dict, List, Optional, Tuple

from perception import (
    CameraIntrinsics,
    Detection,
    TARGET_CLASS,
    bearing_distance,
    extract_visual_obstacles,
    normalize_detections,
)
from runlog import RunLogger
from safety import SafetyConfig, SafetyInput, SafetyLayer, to_wheels
from tracking import Track, Tracker


class HumanFollower:
    """Bo dieu khien bam nguoi: khoa muc tieu + Re-ID gan + chu dong xoay tim khi mat dau."""

    def __init__(
        self,
        z_ref: float = 1.15,
        k_v: float = 0.55,
        k_w: float = 1.75,
        deadband_z: float = 0.10,
        lost_timeout: float = 10.0,
    ):
        self.z_ref = z_ref
        self.k_v = k_v
        self.k_w = k_w
        self.deadband_z = deadband_z
        self.lost_timeout = lost_timeout

        self.track_id: Optional[int] = None
        self.last_seen: Optional[float] = None
        self.last_theta: float = 0.0
        self.last_bbox: Optional[Tuple[float, float, float, float]] = None
        self.state: str = "IDLE"

    def reset(self) -> None:
        self.track_id = None
        self.last_seen = None
        self.last_theta = 0.0
        self.last_bbox = None
        self.state = "IDLE"

    def lock_target(self, track_id: Optional[int]) -> None:
        self.track_id = track_id
        self.last_seen = None
        self.last_bbox = None

    def _control_law(self, theta: float, z: float) -> Tuple[float, float]:
        v = self.k_v * (z - self.z_ref) if abs(z - self.z_ref) > self.deadband_z else 0.0
        w = -self.k_w * theta
        return max(-0.15, min(0.28, v)), max(-1.50, min(1.50, w))

    def step(
        self,
        tracks: List[Track],
        tracker: Tracker,
        cam: CameraIntrinsics,
        now: float,
        logger: Optional[RunLogger] = None,
    ) -> Tuple[float, float, Dict]:
        people = [t for t in tracks if t.cls == TARGET_CLASS]

        if self.track_id is None:
            if not people:
                return self._handle_lost(now, logger)
            chosen = min(people, key=lambda t: abs(0.5 * (t.bbox[0] + t.bbox[2]) - cam.cx))
            self.track_id = chosen.track_id
            if logger:
                logger.event(now, "TARGET_LOCKED", track_id=self.track_id)

        tr = tracker.get(self.track_id)
        if (tr is None or tr.lost_frames > 0) and self.last_bbox is not None and people:
            old_cx = 0.5 * (self.last_bbox[0] + self.last_bbox[2])
            best = min(people, key=lambda t: abs(0.5 * (t.bbox[0] + t.bbox[2]) - old_cx))
            if abs(0.5 * (best.bbox[0] + best.bbox[2]) - old_cx) < 120.0:
                if logger:
                    logger.event(now, "TARGET_REACQUIRED", old_id=self.track_id, new_id=best.track_id)
                self.track_id = best.track_id
                tr = best

        if tr is None or tr.lost_frames > 0:
            return self._handle_lost(now, logger)

        theta, z = bearing_distance(tr.bbox, cam, TARGET_CLASS)
        self.last_seen = now
        self.last_theta = theta
        self.last_bbox = tr.bbox
        self.state = "TRACKING"

        v, w = self._control_law(theta, z)
        return v, w, {
            "state": self.state,
            "track_id": self.track_id,
            "distance_m": round(z, 2),
            "bearing_deg": round(math.degrees(theta), 1),
            "target_bbox": tr.bbox,
        }

    def _handle_lost(self, now: float, logger: Optional[RunLogger]) -> Tuple[float, float, Dict]:
        if self.last_seen is None:
            self.state = "SEARCHING"
            return 0.0, 0.0, {"state": self.state, "track_id": self.track_id, "distance_m": None, "bearing_deg": 0.0, "target_bbox": None}

        if now - self.last_seen > self.lost_timeout:
            self.state = "LOST_STOP"
            if logger:
                logger.event(now, "TARGET_LOST_TIMEOUT", track_id=self.track_id)
            self.track_id = None
            return 0.0, 0.0, {"state": self.state, "track_id": None, "distance_m": None, "bearing_deg": 0.0, "target_bbox": None}

        self.state = "REACQUIRING_SPIN"
        spin_w = -0.70 if self.last_theta > 0.0 else 0.70
        return 0.0, spin_w, {
            "state": self.state,
            "track_id": self.track_id,
            "distance_m": None,
            "bearing_deg": round(math.degrees(self.last_theta), 1),
            "target_bbox": None,
        }


class HumanFollowingPipeline:
    """Lop tich hop Giai doan 1: Detection -> Tracker -> HumanFollower -> SafetyLayer -> PWM."""

    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        fov_deg: float = 62.0,
        cam_height: float = 0.12,
        z_ref: float = 1.15,
        max_duty: float = 0.40,
        enable_log: bool = False,
        log_root: str = "runs",
    ):
        self.cam = CameraIntrinsics.from_resolution(width, height, fov_deg=fov_deg, cam_height=cam_height)
        self.tracker = Tracker(high=0.50, low=0.12, iou_min=0.20, max_lost=15)
        self.follower = HumanFollower(z_ref=z_ref)
        self.safety = SafetyLayer(SafetyConfig())
        self.max_duty = max_duty
        self.logger = RunLogger(root=log_root, enabled=enable_log)

    def reset(self) -> None:
        self.tracker.reset()
        self.follower.reset()

    def step(
        self,
        raw_detections,
        frame_shape: Optional[Tuple[int, int]] = None,
        now: Optional[float] = None,
        camera_ok: bool = True,
        battery_pct: Optional[float] = 1.0,
        estop: bool = False,
        manual_vw: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float, Dict]:
        now = time.time() if now is None else now
        if frame_shape is not None:
            fh, fw = int(frame_shape[0]), int(frame_shape[1])
            if fw != self.cam.width or fh != self.cam.height:
                self.cam = CameraIntrinsics.from_resolution(fw, fh, cam_height=self.cam.cam_height)

        dets: List[Detection] = normalize_detections(raw_detections, now) if camera_ok else []
        tracks: List[Track] = self.tracker.update(dets) if camera_ok else []

        if manual_vw is not None:
            v_cmd, w_cmd = float(manual_vw[0]), float(manual_vw[1])
            follow_info = {"state": "MANUAL", "track_id": self.follower.track_id, "distance_m": None, "bearing_deg": 0.0, "target_bbox": None}
            obstacles = extract_visual_obstacles(dets, self.cam, target_bbox=None)
        else:
            v_cmd, w_cmd, follow_info = self.follower.step(tracks, self.tracker, self.cam, now, self.logger)
            obstacles = extract_visual_obstacles(dets, self.cam, target_bbox=follow_info.get("target_bbox"))

        v_safe, w_safe, safety_reason = self.safety.filter(
            v_cmd,
            w_cmd,
            SafetyInput(
                now=now,
                last_cmd_t=now,
                battery=battery_pct if battery_pct is not None else 1.0,
                camera_ok=camera_ok,
                need_camera=(manual_vw is None),
                estop=estop,
                visual_obstacles=obstacles,
            ),
        )
        left_pwm, right_pwm = to_wheels(v_safe, w_safe, max_duty=self.max_duty)

        self.logger.csv(
            "commands.csv",
            ["t", "state", "track_id", "dist_m", "v_cmd", "w_cmd", "v_safe", "w_safe", "safety", "l_pwm", "r_pwm"],
            [
                round(now, 2),
                follow_info["state"],
                follow_info["track_id"],
                follow_info["distance_m"],
                round(v_cmd, 3),
                round(w_cmd, 3),
                round(v_safe, 3),
                round(w_safe, 3),
                safety_reason,
                left_pwm,
                right_pwm,
            ],
        )

        info = {
            **follow_info,
            "v_cmd": round(v_cmd, 3),
            "w_cmd": round(w_cmd, 3),
            "v_safe": round(v_safe, 3),
            "w_safe": round(w_safe, 3),
            "safety_state": self.safety.state,
            "safety_reason": safety_reason,
            "min_obstacle_m": None if math.isinf(self.safety.min_front_dist) else round(self.safety.min_front_dist, 2),
            "left_pwm": left_pwm,
            "right_pwm": right_pwm,
            "tracks": tracks,
        }
        return left_pwm, right_pwm, info
