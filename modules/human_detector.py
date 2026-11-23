#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import types
import time
import threading
import cv2
import torch
import numpy as np
import config

if "seaborn" not in sys.modules:
    sys.modules["seaborn"] = types.ModuleType("seaborn")

YOLO_REPO = os.path.join(config.BASE_DIR, "yolov5_repo")
if YOLO_REPO not in sys.path:
    sys.path.insert(0, YOLO_REPO)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords

WEIGHTS_PATH = os.path.join(config.BASE_DIR, "models", "yolov5s.pt")


class HumanDetector:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.half = self.device.type == "cuda"
        self.img_size = 320  # Tối ưu tốc độ ~35 FPS trên Jetson Nano FP16 mà vẫn giữ độ nét YOLOv5s

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.model = DetectMultiBackend(WEIGHTS_PATH, device=self.device, fp16=self.half)
        self.backend = "yolov5s_320_{}".format(self.device.type)

        dummy = torch.zeros(
            (1, 3, self.img_size, self.img_size),
            device=self.device,
            dtype=torch.half if self.half else torch.float32,
        )
        for _ in range(2):
            _ = self.model(dummy)

        self._lock = threading.Lock()
        self._pending_frame = None
        self._latest_results = []
        self._last_infer_ms = 0.0
        self._running = True
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        print("[Detector] Nap thanh cong YOLOv5s 320 FP16 (GPU preprocess) tren {}.".format(self.device))

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

        # Đẩy uint8 lên GPU trước rồi mới chuyển BGR->RGB, CHW và FP16 trên CUDA
        im = torch.from_numpy(padded).to(self.device, non_blocking=True)
        im = im[:, :, [2, 1, 0]].permute(2, 0, 1).contiguous()
        im = im.half() if self.half else im.float()
        im.mul_(1.0 / 255.0)
        im = im.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(im)
            pred = non_max_suppression(
                pred,
                conf_thres=config.DETECTION_CONF_THRESHOLD,
                iou_thres=config.NMS_IOU_THRESHOLD,
                classes=[0],
            )

        results = []
        det = pred[0]
        if det is not None and len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, _ in det.tolist():
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                bw = x2 - x1
                bh = y2 - y1
                aspect = bh / max(1.0, bw)

                # Bộ lọc hình học & vị trí sàn cho góc nhìn camera thấp của JetBot:
                # 1. Kích thước tối thiểu & tỷ lệ chiều cao/rộng hợp lý (loại chân bàn mảnh & sofa bè ngang)
                if bw < 30.0 or bh < 60.0 or not (0.70 <= aspect <= 4.2):
                    continue
                # 2. Loại khung nhiễu choán > 85% bề ngang do nhòe chuyển động khi xoay
                if (bw / float(img_w)) > 0.85:
                    continue
                # 3. Nếu hộp nằm lơ lửng ở nửa trên khung hình (không chạm vùng sàn y2 < 52% ảnh), yêu cầu conf >= 0.62
                if y2 < (img_h * 0.52) and float(conf) < 0.62:
                    continue

                results.append({"bbox": [x1, y1, x2, y2], "conf": float(conf)})

        self._last_infer_ms = (time.time() - t0) * 1000.0
        return results

    def _worker_loop(self):
        while self._running:
            frame = None
            with self._lock:
                if self._pending_frame is not None:
                    frame = self._pending_frame.copy()
                    self._pending_frame = None
            if frame is not None:
                res = self._run_inference(frame)
                with self._lock:
                    self._latest_results = res
            else:
                time.sleep(0.003)

    def detect(self, frame):
        with self._lock:
            self._pending_frame = frame
            return list(self._latest_results)

    def cleanup(self):
        self._running = False
