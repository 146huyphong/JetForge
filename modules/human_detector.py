#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
human_detector.py - YOLOv5s 320 FP16 da lop (person + vat can trong nha)
"""
import os
import sys
import time
import threading
import cv2
import numpy as np
import torch
import config

YOLO_REPO_DIR = os.path.join(config.BASE_DIR, "yolov5_repo")
if YOLO_REPO_DIR not in sys.path:
    sys.path.insert(0, YOLO_REPO_DIR)

import types
try:
    import seaborn  # noqa: F401
except Exception:
    sys.modules['seaborn'] = types.ModuleType('seaborn')

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords

WEIGHTS_PATH = os.path.join(config.BASE_DIR, "models", "yolov5s.pt")

# COCO class IDs: 0=person, 24=backpack, 28=suitcase, 39=bottle, 56=chair, 57=couch, 60=dining table
TARGET_CLASSES = [0, 24, 28, 39, 56, 57, 60]
OBSTACLE_NAMES = {
    24: "backpack",
    28: "suitcase",
    39: "bottle",
    56: "chair",
    57: "couch",
    60: "table",
}


class HumanDetector:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.half = self.device.type == "cuda"
        self.img_size = 320
        self.low_conf_thres = 0.15
        self.obstacle_conf_thres = 0.35

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.model = DetectMultiBackend(WEIGHTS_PATH, device=self.device, fp16=self.half)
        self.backend = "yolov5s_320_multiclass_{}".format(self.device.type)

        dummy = torch.zeros(
            (1, 3, self.img_size, self.img_size),
            device=self.device,
            dtype=torch.half if self.half else torch.float32,
        )
        for _ in range(2):
            _ = self.model(dummy)

        self._lock = threading.Lock()
        self._pending_frame = None
        self._latest_persons = []
        self._latest_obstacles = []
        self._last_infer_ms = 0.0
        self._running = True
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        print("[Detector] Nap thanh cong YOLOv5s 320 FP16 da lop tren {}.".format(self.device))

    def _letterbox_fast(self, frame):
        h, w = frame.shape[:2]
        r = min(float(self.img_size) / h, float(self.img_size) / w)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        dw = (self.img_size - new_w) // 2
        dh = (self.img_size - new_h) // 2
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        canvas[dh:dh + new_h, dw:dw + new_w] = resized
        return canvas

    def _run_inference(self, frame):
        t0 = time.time()
        img_h, img_w = frame.shape[:2]
        padded = self._letterbox_fast(frame)

        im = torch.from_numpy(padded).to(self.device, non_blocking=True)
        im = im[:, :, [2, 1, 0]].permute(2, 0, 1).contiguous()
        im = im.half() if self.half else im.float()
        im.mul_(1.0 / 255.0)
        im = im.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(im)
            pred = non_max_suppression(
                pred,
                conf_thres=self.low_conf_thres,
                iou_thres=config.NMS_IOU_THRESHOLD,
                classes=TARGET_CLASSES,
            )

        persons = []
        obstacles = []
        det = pred[0]
        if det is not None and len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls_id in det.tolist():
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                bw = x2 - x1
                bh = y2 - y1
                c_val = float(conf)
                cid = int(cls_id)

                if cid == 0:
                    aspect = bh / max(1.0, bw)
                    if bw < 28.0 or bh < 55.0 or not (0.70 <= aspect <= 4.2):
                        continue
                    if (bw / float(img_w)) > 0.85:
                        continue
                    if y2 < (img_h * 0.52) and c_val < 0.62:
                        continue
                    persons.append({"cls": "person", "bbox": [x1, y1, x2, y2], "conf": c_val})
                elif cid in OBSTACLE_NAMES and c_val >= self.obstacle_conf_thres:
                    if bw < 20.0 or bh < 20.0:
                        continue
                    obstacles.append({
                        "cls": OBSTACLE_NAMES[cid],
                        "bbox": [x1, y1, x2, y2],
                        "conf": c_val,
                    })

        self._last_infer_ms = (time.time() - t0) * 1000.0
        return persons, obstacles

    def _worker_loop(self):
        while self._running:
            frame = None
            with self._lock:
                if self._pending_frame is not None:
                    frame = self._pending_frame.copy()
                    self._pending_frame = None
            if frame is not None:
                persons, obstacles = self._run_inference(frame)
                with self._lock:
                    self._latest_persons = persons
                    self._latest_obstacles = obstacles
            else:
                time.sleep(0.003)

    def detect_all(self, frame):
        with self._lock:
            self._pending_frame = frame
            return list(self._latest_persons), list(self._latest_obstacles)

    def detect(self, frame):
        persons, _ = self.detect_all(frame)
        return persons

    def cleanup(self):
        self._running = False