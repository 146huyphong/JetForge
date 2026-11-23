#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, types, time, threading, cv2, torch, numpy as np, config

if "seaborn" not in sys.modules:
    sys.modules["seaborn"] = types.ModuleType("seaborn")

YOLO_REPO = os.path.join(config.BASE_DIR, "yolov5_repo")
if YOLO_REPO not in sys.path:
    sys.path.insert(0, YOLO_REPO)

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords

WEIGHTS_PATH = os.path.join(config.BASE_DIR, "models", "yolov5s.pt")


class HumanDetector:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.half = self.device.type == "cuda"
        self.img_size = 416
        self.model = DetectMultiBackend(WEIGHTS_PATH, device=self.device, fp16=self.half)
        self.backend = f"yolov5s_{self.device.type}"

        dummy = torch.zeros(
            (1, 3, self.img_size, self.img_size),
            device=self.device,
            dtype=torch.half if self.half else torch.float32,
        )
        _ = self.model(dummy)

        self._lock = threading.Lock()
        self._pending_frame = None
        self._latest_results = []
        self._running = True
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        print(f"[Detector] Nap thanh cong YOLOv5s 416 FP16 tren {self.device}.")

    def _run_inference(self, frame):
        img = letterbox(frame, self.img_size, stride=32, auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        im = torch.from_numpy(img).to(self.device)
        im = im.half() if self.half else im.float()
        im /= 255.0
        if im.ndim == 3:
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
                bw_box, bh_box = (x2 - x1), (y2 - y1)
                aspect = bh_box / max(1.0, bw_box)
                if bw_box > 28 and bh_box > 55 and 0.65 <= aspect <= 4.5:
                    results.append({"bbox": [x1, y1, x2, y2], "conf": float(conf)})
        return results

    def _worker_loop(self):
        while self._running:
            frame = None
            with self._lock:
                if self._pending_frame is not None:
                    frame = self._pending_frame
                    self._pending_frame = None
            if frame is not None:
                res = self._run_inference(frame)
                with self._lock:
                    self._latest_results = res
            else:
                time.sleep(0.005)

    def detect(self, frame):
        with self._lock:
            self._pending_frame = frame.copy()
            return list(self._latest_results)

    def cleanup(self):
        self._running = False
