#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
human_tracker.py - Khoa ID muc tieu 2D:
Kalman Filter + ByteTrack 2 vong (high/low conf) + HSV Histogram Re-ID + Fast-Turn Rescue
"""
import math
import time
import cv2
import numpy as np
import config


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def extract_torso_histogram(frame, bbox):
    img_h, img_w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    tx1 = max(0, x1 + int(0.20 * w))
    tx2 = min(img_w, x1 + int(0.80 * w))
    ty1 = max(0, y1 + int(0.20 * h))
    ty2 = min(img_h, y1 + int(0.68 * h))

    if tx2 <= tx1 or ty2 <= ty1:
        return None

    crop = frame[ty1:ty2, tx1:tx2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()


def histogram_similarity(hist_a, hist_b):
    if hist_a is None or hist_b is None:
        return 0.5
    corr = cv2.compareHist(hist_a.astype(np.float32), hist_b.astype(np.float32), cv2.HISTCMP_CORREL)
    return float(max(0.0, min(1.0, (corr + 1.0) / 2.0)))


class KalmanBoxTracker:
    _id_counter = 1

    def __init__(self, bbox, hist):
        self.id = KalmanBoxTracker._id_counter
        KalmanBoxTracker._id_counter += 1

        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0
        self.Q = np.eye(8, dtype=np.float32) * 0.05
        self.Q[4:, 4:] *= 0.2
        self.R = np.eye(4, dtype=np.float32) * 2.0

        self.x = np.zeros((8, 1), dtype=np.float32)
        self.x[:4, 0] = self._bbox_to_z(bbox)

        self.hist = hist
        self.time_since_update = 0
        self.hits = 1
        self.conf = 1.0

    @staticmethod
    def _bbox_to_z(bbox):
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        cx = bbox[0] + 0.5 * w
        cy = bbox[1] + 0.5 * h
        return np.array([cx, cy, w, h], dtype=np.float32)

    def get_bbox(self):
        cx, cy, w, h = self.x[:4, 0]
        w = max(4.0, w)
        h = max(8.0, h)
        return [float(cx - 0.5 * w), float(cy - 0.5 * h), float(cx + 0.5 * w), float(cy + 0.5 * h)]

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        return self.get_bbox()

    def update(self, bbox, hist, conf):
        self.time_since_update = 0
        self.hits += 1
        self.conf = conf

        z = self._bbox_to_z(bbox).reshape(4, 1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(8, dtype=np.float32) - K @ self.H) @ self.P

        if hist is not None:
            if self.hist is None:
                self.hist = hist
            else:
                alpha = config.HIST_UPDATE_ALPHA
                self.hist = (1.0 - alpha) * self.hist + alpha * hist


class HumanTracker:
    IDLE = "IDLE"
    LOCKED = "LOCKED"
    COASTING = "COASTING"
    LOST = "LOST"

    def __init__(self):
        self.trackers = []
        self.locked_id = None
        self.locked_hist_gallery = None
        self.last_seen_time = 0.0
        self.last_side = 0.0
        self.state = self.IDLE

    def update(self, detections, frame, now=None):
        now = time.time() if now is None else now
        img_w = frame.shape[1]

        for trk in self.trackers:
            trk.predict()

        # Tach 2 nhom tin cay theo ByteTrack: high >= DETECTION_CONF_THRESHOLD, low < threshold
        high_dets = [d for d in detections if d["conf"] >= config.DETECTION_CONF_THRESHOLD]
        low_dets = [d for d in detections if d["conf"] < config.DETECTION_CONF_THRESHOLD]

        high_hists = [extract_torso_histogram(frame, d["bbox"]) for d in high_dets]
        low_hists = [extract_torso_histogram(frame, d["bbox"]) for d in low_dets]

        num_trk = len(self.trackers)
        num_high = len(high_dets)
        matched_high = []
        unmatched_trks = set(range(num_trk))
        unmatched_high = set(range(num_high))

        # Vong 1A: Khop tin cay cao bang IoU + HSV Histogram
        if num_trk > 0 and num_high > 0:
            score_matrix = np.zeros((num_trk, num_high), dtype=np.float32)
            for t_idx, trk in enumerate(self.trackers):
                pred_box = trk.get_bbox()
                for d_idx, det in enumerate(high_dets):
                    iou = compute_iou(pred_box, det["bbox"])
                    h_sim = histogram_similarity(trk.hist, high_hists[d_idx])
                    score_matrix[t_idx, d_idx] = (
                        config.IOU_WEIGHT * iou + config.HIST_WEIGHT * h_sim
                    )

            while True:
                t_idx, d_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
                if score_matrix[t_idx, d_idx] < config.MATCH_THRESHOLD:
                    break
                matched_high.append((t_idx, d_idx))
                unmatched_trks.discard(t_idx)
                unmatched_high.discard(d_idx)
                score_matrix[t_idx, :] = -1.0
                score_matrix[:, d_idx] = -1.0

        # Vong 1B (Fast-Turn Rescue): Robot quay nhanh -> khop theo tam gan + mau ao
        if unmatched_trks and unmatched_high:
            for t_idx in list(unmatched_trks):
                trk = self.trackers[t_idx]
                px1, py1, px2, py2 = trk.get_bbox()
                pw = max(1.0, px2 - px1)
                pcx, pcy = 0.5 * (px1 + px2), 0.5 * (py1 + py2)

                best_d = None
                best_dist = float("inf")
                for d_idx in unmatched_high:
                    dx1, dy1, dx2, dy2 = high_dets[d_idx]["bbox"]
                    dw = max(1.0, dx2 - dx1)
                    dcx, dcy = 0.5 * (dx1 + dx2), 0.5 * (dy1 + dy2)
                    center_dist = math.hypot(dcx - pcx, dcy - pcy)
                    scale_ratio = dw / pw
                    h_sim = histogram_similarity(trk.hist, high_hists[d_idx])

                    if (
                        center_dist < config.FAST_TURN_CENTER_RATIO * pw
                        and config.FAST_TURN_SCALE_MIN < scale_ratio < config.FAST_TURN_SCALE_MAX
                        and h_sim >= 0.45
                    ):
                        if center_dist < best_dist:
                            best_dist = center_dist
                            best_d = d_idx

                if best_d is not None:
                    matched_high.append((t_idx, best_d))
                    unmatched_trks.discard(t_idx)
                    unmatched_high.discard(best_d)

        for t_idx, d_idx in matched_high:
            self.trackers[t_idx].update(
                high_dets[d_idx]["bbox"], high_hists[d_idx], high_dets[d_idx]["conf"]
            )

        # Vong 2 (ByteTrack Low-Conf Association): Giu track khi nguoi vao vung toi / bi che khuat
        num_low = len(low_dets)
        if unmatched_trks and num_low > 0:
            rem_trks = sorted(list(unmatched_trks))
            low_matrix = np.zeros((len(rem_trks), num_low), dtype=np.float32)
            for row_i, t_idx in enumerate(rem_trks):
                pred_box = self.trackers[t_idx].get_bbox()
                for d_idx, det in enumerate(low_dets):
                    iou = compute_iou(pred_box, det["bbox"])
                    h_sim = histogram_similarity(self.trackers[t_idx].hist, low_hists[d_idx])
                    low_matrix[row_i, d_idx] = 0.70 * iou + 0.30 * h_sim if iou >= 0.15 else -1.0

            while True:
                r_i, d_idx = np.unravel_index(np.argmax(low_matrix), low_matrix.shape)
                if low_matrix[r_i, d_idx] < 0.25:
                    break
                t_idx = rem_trks[r_i]
                self.trackers[t_idx].update(
                    low_dets[d_idx]["bbox"], low_hists[d_idx], low_dets[d_idx]["conf"]
                )
                unmatched_trks.discard(t_idx)
                low_matrix[r_i, :] = -1.0
                low_matrix[:, d_idx] = -1.0

        # Chi tao track moi tu detection tin cay cao (high_dets) de loai nhieu gia
        for d_idx in unmatched_high:
            new_trk = KalmanBoxTracker(high_dets[d_idx]["bbox"], high_hists[d_idx])
            if (
                self.locked_id is not None
                and self.locked_hist_gallery is not None
                and not any(t.id == self.locked_id for t in self.trackers)
            ):
                reid_sim = histogram_similarity(self.locked_hist_gallery, high_hists[d_idx])
                if reid_sim >= 0.72:
                    new_trk.id = self.locked_id
                    print(f"[Tracker] Re-ID tai khoa thanh cong muc tieu ID:{self.locked_id} (sim={reid_sim:.2f})")
            self.trackers.append(new_trk)

        active_outputs = []
        surviving_trackers = []
        locked_visible = False

        for trk in self.trackers:
            if trk.time_since_update <= config.TRACKER_MAX_AGE:
                surviving_trackers.append(trk)
                bbox = trk.get_bbox()
                if trk.id == self.locked_id and trk.time_since_update == 0:
                    locked_visible = True
                    self.last_seen_time = now
                    self.last_side = (0.5 * (bbox[0] + bbox[2])) - (0.5 * img_w)
                    self.locked_hist_gallery = trk.hist.copy() if trk.hist is not None else None

                if trk.hits >= config.TRACKER_MIN_HITS and trk.time_since_update <= 2:
                    active_outputs.append({
                        "track_id": trk.id,
                        "bbox": bbox,
                        "conf": trk.conf,
                        "is_locked": (trk.id == self.locked_id),
                        "hits": trk.hits,
                    })

        self.trackers = surviving_trackers

        if self.locked_id is None:
            self.state = self.IDLE
        elif locked_visible:
            self.state = self.LOCKED
        elif (now - self.last_seen_time) <= config.COASTING_MAX_SEC:
            self.state = self.COASTING
        else:
            self.state = self.LOST

        return active_outputs

    def select_target_by_norm_point(self, nx, ny, frame_shape):
        h, w = frame_shape[:2]
        px, py = nx * w, ny * h

        best_id = None
        best_dist = float("inf")
        for trk in self.trackers:
            if trk.time_since_update > 3:
                continue
            x1, y1, x2, y2 = trk.get_bbox()
            if (x1 - 25) <= px <= (x2 + 25) and (y1 - 25) <= py <= (y2 + 25):
                cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                dist = (cx - px) ** 2 + (cy - py) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_id = trk.id

        if best_id is not None:
            self.locked_id = best_id
            self.last_seen_time = time.time()
            self.state = self.LOCKED
            print(f"[Tracker] Da khoa muc tieu nguoi dieu khien ID:{best_id}")
        return best_id

    def auto_lock_primary(self, active_tracks):
        if not active_tracks:
            return None
        largest = max(
            active_tracks,
            key=lambda t: (t["bbox"][2] - t["bbox"][0]) * (t["bbox"][3] - t["bbox"][1]),
        )
        self.locked_id = largest["track_id"]
        self.last_seen_time = time.time()
        self.state = self.LOCKED
        return self.locked_id

    def clear_target(self):
        self.locked_id = None
        self.locked_hist_gallery = None
        self.state = self.IDLE