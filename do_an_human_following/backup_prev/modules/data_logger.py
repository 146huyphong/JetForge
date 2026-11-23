#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_logger.py - Ghi log sai số PID, độ trễ ảnh, quỹ đạo & FPS ra CSV phục vụ vẽ biểu đồ báo cáo
"""
import os
import csv
import time
from datetime import datetime
import config


class DataLogger:
    FIELDS = [
        "timestamp",
        "elapsed_s",
        "state",
        "reason",
        "fps",
        "frame_age_ms",
        "target_id",
        "distance_m",
        "bearing_deg",
        "error_dist",
        "linear_v",
        "angular_w",
        "battery_v",
        "battery_pct",
        "blocked_prob",
    ]

    def __init__(self):
        self.enabled = config.ENABLE_LOGGING
        self.file = None
        self.writer = None
        self.last_log_time = 0.0

        if self.enabled:
            os.makedirs(config.LOGS_DIR, exist_ok=True)
            fname = f"jetbot_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            fpath = os.path.join(config.LOGS_DIR, fname)
            self.file = open(fpath, mode="w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
            self.writer.writeheader()
            print(f"[Logger] Đang ghi dữ liệu thực nghiệm tại: {fpath}")

    def log(self, row_dict):
        if not self.enabled or self.writer is None:
            return
        now = time.time()
        if (now - self.last_log_time) < config.LOG_INTERVAL_SEC:
            return
        self.last_log_time = now

        data = {"timestamp": datetime.now().isoformat(timespec="milliseconds")}
        data.update({k: row_dict.get(k, "") for k in self.FIELDS if k != "timestamp"})
        self.writer.writerow(data)

    def close(self):
        if self.file is not None:
            try:
                self.file.flush()
                self.file.close()
            except Exception:
                pass
            self.file = None