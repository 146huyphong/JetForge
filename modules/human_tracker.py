#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
human_tracker.py - Khóa ID mục tiêu 2D (Kalman Filter + HSV Histogram Re-ID + Cổng khớp quay nhanh)
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
    """Trích xuất biểu đồ màu HSV vùng thân áo để Re-ID chống nhảy ID"""
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
    """Bộ lọc Kalman 8 trạng thái [cx, cy, w, h, vx, vy, vw, vh]"""
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
        self.last_side = 0.0   # Dương: mục tiêu lệch bên phải tâm ảnh; Âm: bên trái
        self.state = self.IDLE

    def update(self, detections, frame, now=None):
        now = time.time() if now is None else now
        img_w = frame.shape[1]

        for trk in self.trackers:
            trk.predict()

        det_hists = [extract_torso_histogram(frame, d["bbox"]) for d in detections]

        num_trk = len(self.trackers)
        num_det = len(detections)
        matched_pairs = []
        unmatched_trks = set(range(num_trk))
        unmatched_dets = set(range(num_det))

        # Cổng 1: Khớp kết hợp IoU + HSV Histogram
        if num_trk > 0 and num_det > 0:
            score_matrix = np.zeros((num_trk, num_det), dtype=np.float32)
            for t_idx, trk in enumerate(self.trackers):
                pred_box = trk.get_bbox()
                for d_idx, det in enumerate(detections):
                    iou = compute_iou(pred_box, det["bbox"])
                    h_sim = histogram_similarity(trk.hist, det_hists[d_idx])
                    score_matrix[t_idx, d_idx] = (
                        config.IOU_WEIGHT * iou + config.HIST_WEIGHT * h_sim
                    )

            while True:
                t_idx, d_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
                if score_matrix[t_idx, d_idx] < config.MATCH_THRESHOLD:
                    break
                matched_pairs.append((t_idx, d_idx))
                unmatched_trks.discard(t_idx)
                unmatched_dets.discard(d_idx)
                score_matrix[t_idx, :] = -1.0
                score_matrix[:, d_idx] = -1.0

        # Cổng 2 (Fast-Turn Rescue): Robot quay nhanh -> mặt/người trượt ngang nhiều pixel (IoU = 0).
        # Khớp theo tâm gần (< 1.5 bề rộng box) + kích thước tương đồng + màu áo.
        if unmatched_trks and unmatched_dets:
            for t_idx in list(unmatched_trks):
                trk = self.trackers[t_idx]
                px1, py1, px2, py2 = trk.get_bbox()
                pw = max(1.0, px2 - px1)
                pcx, pcy = 0.5 * (px1 + px2), 0.5 * (py1 + py2)

                best_d = None
                best_dist = float("inf")
                for d_idx in unmatched_dets:
                    dx1, dy1, dx2, dy2 = detections[d_idx]["bbox"]
                    dw = max(1.0, dx2 - dx1)
                    dcx, dcy = 0.5 * (dx1 + dx2), 0.5 * (dy1 + dy2)
                    center_dist = math.hypot(dcx - pcx, dcy - pcy)
                    scale_ratio = dw / pw
                    h_sim = histogram_similarity(trk.hist, det_hists[d_idx])

                    if (
                        center_dist < config.FAST_TURN_CENTER_RATIO * pw
                        and config.FAST_TURN_SCALE_MIN < scale_ratio < config.FAST_TURN_SCALE_MAX
                        and h_sim >= 0.45
                    ):
                        if center_dist < best_dist:
                            best_dist = center_dist
                            best_d = d_idx

                if best_d is not None:
                    matched_pairs.append((t_idx, best_d))
                    unmatched_trks.discard(t_idx)
                    unmatched_dets.discard(best_d)

        for t_idx, d_idx in matched_pairs:
            self.trackers[t_idx].update(
                detections[d_idx]["bbox"], det_hists[d_idx], detections[d_idx]["conf"]
            )

        # Tạo track mới hoặc tái nhận diện (Re-ID) mục tiêu đã khóa bị mất dấu
        for d_idx in unmatched_dets:
            new_trk = KalmanBoxTracker(detections[d_idx]["bbox"], det_hists[d_idx])
            if (
                self.locked_id is not None
                and self.locked_hist_gallery is not None
                and not any(t.id == self.locked_id for t in self.trackers)
            ):
                reid_sim = histogram_similarity(self.locked_hist_gallery, det_hists[d_idx])
                if reid_sim >= 0.72:
                    new_trk.id = self.locked_id
                    print(f"[Tracker] Re-ID tái khóa thành công mục tiêu ID:{self.locked_id} (sim={reid_sim:.2f})")
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

        # Cập nhật trạng thái Tracker (IDLE / LOCKED / COASTING / LOST)
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
            print(f"[Tracker] Đã khóa mục tiêu người điều khiển ID:{best_id}")
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