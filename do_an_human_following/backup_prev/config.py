#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py - Cấu hình trung tâm cho hệ thống JetBot Human Follower
"""
import os
import math
import numpy as np

# ==================== ĐƯỜNG DẪN HỆ THỐNG ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

SSD_ENGINE_PATH = os.path.join(MODELS_DIR, "ssd_mobilenet_v2_coco.engine")
COLLISION_ENGINE_PATH = os.path.join(MODELS_DIR, "collision_resnet18.engine")

# ==================== CẤU HÌNH CAMERA RGB & GÓC NGHIÊNG ====================
CAMERA_TYPE = "csi"          # "csi" (GStreamer nvarguscamerasrc) hoặc "usb"
SENSOR_ID = 0
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 480
CAMERA_FPS = 30
FLIP_METHOD = 0              # 0: bình thường, 2: xoay 180 độ nếu lắp ngược camera

# Thông số quang học & góc nghiêng vật lý của camera trên JetBot
CAMERA_HFOV_DEG = 62.2       # Góc nhìn ngang camera IMX219 chuẩn (độ)
CAMERA_TILT_DEG = 20.0
FOCAL_LENGTH_PX = (PROCESS_WIDTH / 2.0) / math.tan(math.radians(CAMERA_HFOV_DEG / 2.0))

# Nếu True, camera_streamer nhận ảnh từ ROS topic thay vì mở trực tiếp GStreamer
USE_ROS_IMAGE_TOPIC = False
ROS_IMAGE_TOPIC = "/camera/image_raw"

# Hiệu chỉnh méo ống kính (Camera Calibration)
ENABLE_UNDISTORT = False
CAMERA_MATRIX = np.array([
    [FOCAL_LENGTH_PX, 0.0,             PROCESS_WIDTH / 2.0],
    [0.0,             FOCAL_LENGTH_PX, PROCESS_HEIGHT / 2.0],
    [0.0,             0.0,             1.0]
], dtype=np.float32)
DIST_COEFFS = np.array([-0.28, 0.08, 0.0, 0.0, 0.0], dtype=np.float32)

# ==================== CẤU HÌNH NHẬN DIỆN NGƯỜI (SSD-MOBILENET-V2) ====================
DETECTOR_INPUT_SIZE = (300, 300)
PERSON_CLASS_ID = 1          # Class ID của 'person' trong bộ dữ liệu COCO
DETECTION_CONF_THRESHOLD = 0.45
NMS_IOU_THRESHOLD = 0.45

# ==================== CẤU HÌNH TRACKING & RE-ID ====================
TRACKER_MAX_AGE = 25         # Số frame tối đa giữ bộ nhớ track
TRACKER_MIN_HITS = 1         # Số frame liên tiếp để xác nhận track hợp lệ
COASTING_MAX_SEC = 1.80
IOU_WEIGHT = 0.55            # Trọng số khớp vị trí Kalman IoU
HIST_WEIGHT = 0.45           # Trọng số khớp màu áo HSV Histogram
MATCH_THRESHOLD = 0.22       # Ngưỡng tương đồng tối thiểu để ghép cặp
HIST_UPDATE_ALPHA = 0.15     # Tốc độ cập nhật đặc trưng màu áo (EMA)

# Cổng khớp dự phòng khi robot xoay nhanh làm IoU rơi về 0
FAST_TURN_CENTER_RATIO = 1.5 # Khoảng cách tâm tối đa theo bề rộng box dự đoán
FAST_TURN_SCALE_MIN = 0.60
FAST_TURN_SCALE_MAX = 1.60

# ==================== CẤU HÌNH ƯỚC LƯỢNG KHOẢNG CÁCH ====================
TARGET_DISTANCE_M = 1.50
TOO_CLOSE_DISTANCE_M = 0.65  # Ngưỡng quá gần: cấm tiến thêm để tránh va chạm
AVG_HUMAN_HEIGHT_M = 1.68    # Chiều cao trung bình người trưởng thành (mét)
AVG_SHOULDER_WIDTH_M = 0.46  # Bề rộng vai/thân hiệu dụng (có thể đo lại bằng self_test.py)
EDGE_MARGIN_PX = 16          # Biên nhận biết box chạm mép trên/dưới khung hình

# ==================== CẤU HÌNH TRÁNH VẬT CẢN (RESNET-18) ====================
ENABLE_COLLISION_GUARD = False
COLLISION_INPUT_SIZE = (224, 224)
COLLISION_BLOCKED_THRESHOLD = 0.65
COLLISION_EMA_ALPHA = 0.40

# ==================== CẤU HÌNH ĐIỀU KHIỂN PID & ĐỘNG HỌC VI SAI ====================
# Angular PID tính theo góc lệch bearing theta (radian) -> tốc độ góc omega (rad/s)
ANGULAR_PID = {"kp": 0.08, "ki": 0.00, "kd": 0.03}
# Linear PID tính theo sai số khoảng cách ez (mét) -> vận tốc tiến v (m/s)
LINEAR_PID = {"kp": 0.28, "ki": 0.00, "kd": 0.04}
INTEGRAL_MAX = 0.45          # Chống bão hòa tích phân (Anti-windup, rad*s)

THETA_DEADBAND_DEG = 3.0     # Vùng chết góc lệch (độ) tránh rung lắc tại chỗ
Z_DEADBAND_M = 0.14          # Vùng chết khoảng cách (mét)

# Thông số cơ khí JetBot (cần hiệu chỉnh V_FULL_MPS thực tế trên mặt sàn của bạn)
WHEELBASE_M = 0.118          # Khoảng cách giữa 2 bánh xe JetBot (~11.8 cm)
V_FULL_MPS = 0.85            # Tốc độ tịnh tiến thực tế ở 100% PWM (m/s)
MAX_LINEAR_SPEED = 0.20      # Giới hạn vận tốc tiến tối đa (m/s)
MAX_ANGULAR_SPEED = 1.40     # Giới hạn vận tốc quay tối đa (rad/s)

MIN_MOTOR_PWM = 0.11         # Bù vùng chết (deadband) thắng ma sát tĩnh hộp số
MAX_ACCEL_STEP = 0.035        # Giới hạn bước nhảy duty mỗi chu kỳ tránh sụt áp
CONTROL_LOOP_HZ = 25

# ==================== AN TOÀN & MÁY TRẠNG THÁI (FSM) ====================
STALE_FRAME_SEC = 0.45       # Dừng khẩn cấp nếu ảnh camera cũ hơn ngưỡng này
BACKUP_DURATION_SEC = 0.0    # Thời gian lùi tối đa khi mất mục tiêu ở cự ly gần
BACKUP_SPEED_MPS = 0.05      # Tốc độ lùi tìm lại mục tiêu (m/s)
SEARCH_TIMEOUT_SEC = 0.0     # Thời gian xoay tìm kiếm trước khi về IDLE
SEARCH_SPIN_RAD_S = 0.35     # Tốc độ xoay tìm kiếm (rad/s)
AVOID_TURN_RAD_S = 0.95      # Tốc độ xoay né vật cản (rad/s)

# ==================== CẤU HÌNH PHẦN CỨNG I2C (JETBOT) ====================
I2C_BUS = 1
MOTOR_DRIVER_TYPE = "adafruit_motorkit"  # "adafruit_motorkit" hoặc "pca9685_raw"
PCA9685_ADDRESS = 0x60
LEFT_MOTOR_TRIM = 1.00
RIGHT_MOTOR_TRIM = 1.00
MOTOR_WATCHDOG_TIMEOUT = 0.40  # Tự ngắt động cơ nếu quá 0.40s không nhận lệnh mới

INA219_ADDRESS = 0x41
BATTERY_MIN_V = 9.6          # Pin 3S cạn (3.2V/cell)
BATTERY_MAX_V = 12.6         # Pin 3S đầy (4.2V/cell)

# ==================== WEB COCKPIT SERVER ====================
WEB_HOST = "0.0.0.0"
WEB_PORT = 8080
STREAM_WIDTH = 480
STREAM_HEIGHT = 360
STREAM_FPS = 15
STREAM_JPEG_QUALITY = 65

# ==================== GHI LOG BÁO CÁO ====================
ENABLE_LOGGING = True
LOG_INTERVAL_SEC = 0.10