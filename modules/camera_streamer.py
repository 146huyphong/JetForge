#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_streamer.py - Luồng nền luôn giữ khung hình MỚI NHẤT kèm dấu thời gian (LatestFrameSource)
"""
import time
import threading
import cv2
import numpy as np
import config

try:
    import rospy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    ROS_IMG_AVAILABLE = True
except Exception:
    ROS_IMG_AVAILABLE = False


def build_gstreamer_pipeline():
    return (
        f"nvarguscamerasrc sensor-id={config.SENSOR_ID} ! "
        f"video/x-raw(memory:NVMM), width=(int){config.CAPTURE_WIDTH}, "
        f"height=(int){config.CAPTURE_HEIGHT}, format=(string)NV12, "
        f"framerate=(fraction){config.CAMERA_FPS}/1 ! "
        f"nvvidconv flip-method={config.FLIP_METHOD} ! "
        f"video/x-raw, width=(int){config.PROCESS_WIDTH}, "
        f"height=(int){config.PROCESS_HEIGHT}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1"
    )


class CameraStreamer:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.stamp = 0.0
        self.ret = False
        self.fps = 0.0
        self.running = False
        self.cap = None
        self.bridge = CvBridge() if ROS_IMG_AVAILABLE else None
        self.ros_sub = None

        self.map1, self.map2 = None, None
        if config.ENABLE_UNDISTORT:
            size = (config.PROCESS_WIDTH, config.PROCESS_HEIGHT)
            new_mtx, _ = cv2.getOptimalNewCameraMatrix(
                config.CAMERA_MATRIX, config.DIST_COEFFS, size, 1, size
            )
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                config.CAMERA_MATRIX, config.DIST_COEFFS, None, new_mtx, size, cv2.CV_16SC2
            )

    def _preprocess(self, img):
        if img is None:
            return None
        if img.shape[1] != config.PROCESS_WIDTH or img.shape[0] != config.PROCESS_HEIGHT:
            img = cv2.resize(img, (config.PROCESS_WIDTH, config.PROCESS_HEIGHT))
        if self.map1 is not None and self.map2 is not None:
            img = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
        return img

    def _ros_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            processed = self._preprocess(cv_img)
            with self.lock:
                self.frame = processed
                self.stamp = time.time()
                self.ret = True
        except Exception as e:
            print(f"[Camera] Lỗi giải mã ROS Image: {e}")

    def start(self):
        self.running = True

        if config.USE_ROS_IMAGE_TOPIC and ROS_IMG_AVAILABLE:
            self.ros_sub = rospy.Subscriber(
                config.ROS_IMAGE_TOPIC, Image, self._ros_callback, queue_size=1
            )
            print(f"[Camera] Đang nhận ảnh qua ROS topic: {config.ROS_IMAGE_TOPIC}")
            return

        if config.CAMERA_TYPE == "csi":
            pipeline = build_gstreamer_pipeline()
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                print("[Camera] Không mở được CSI GStreamer, thử chuyển sang USB camera...")
                self.cap = cv2.VideoCapture(config.SENSOR_ID)
        else:
            self.cap = cv2.VideoCapture(config.SENSOR_ID)

        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.PROCESS_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.PROCESS_HEIGHT)
            print("[Camera] Đã mở luồng camera phần cứng thành công.")
        else:
            print("[Camera] CẢNH BÁO: Không tìm thấy camera thật, tạo luồng ảnh giả lập để test.")

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        prev_t = time.time()
        while self.running:
            if self.cap is not None and self.cap.isOpened():
                ok, raw = self.cap.read()
                if ok and raw is not None:
                    processed = self._preprocess(raw)
                    now = time.time()
                    dt = max(1e-4, now - prev_t)
                    prev_t = now
                    with self.lock:
                        self.frame = processed
                        self.stamp = now
                        self.ret = True
                        self.fps = 1.0 / dt
                else:
                    time.sleep(0.01)
            else:
                dummy = np.zeros((config.PROCESS_HEIGHT, config.PROCESS_WIDTH, 3), dtype=np.uint8)
                cv2.putText(dummy, "NO CAMERA - SIMULATION MODE", (130, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                now = time.time()
                with self.lock:
                    self.frame = dummy
                    self.stamp = now
                    self.ret = True
                    self.fps = 30.0
                time.sleep(1.0 / 30.0)

    def read(self):
        """Trả về (ret, frame_copy, capture_timestamp, camera_fps)."""
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None, 0.0, 0.0
            return True, self.frame.copy(), self.stamp, self.fps

    def stop(self):
        self.running = False
        if self.ros_sub is not None:
            self.ros_sub.unregister()
        if self.cap is not None:
            self.cap.release()