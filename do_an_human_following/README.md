# JetBot Human Tracking & Following (ROS Melodic + TensorRT)

Hệ thống robot tự hành **NVIDIA JetBot (Jetson Nano)** nhận diện, khóa định danh (ID) và bám theo người điều khiển theo thời gian thực sử dụng Camera RGB đơn.

## 1. Tính năng kỹ thuật chính
- **`human_detector.py`**: Nhận diện người bằng `SSD-MobileNet-V2` tối ưu TensorRT FP16 (~25–30 FPS).
- **`human_tracker.py`**: Kalman Filter 8 trạng thái + biểu đồ màu áo HSV (Re-ID) + cổng khớp quay nhanh (Fast-Turn Rescue) chống mất dấu khi robot xoay gắt.
- **`distance_estimator.py`**: Mô hình hình học Pinhole tính góc bearing $\theta$ và khoảng cách ngang trên sàn có **bù góc nghiêng camera** (`CAMERA_TILT_DEG`).
- **`behavior_fsm.py` & `motion_controller.py`**: Máy trạng thái 8 trạng thái (`IDLE`, `TRACKING`, `COASTING`, `BACKUP`, `SEARCHING`, `AVOIDING`, `MANUAL`, `STALE`) kết hợp quy luật điều khiển $\cos^2(\theta)$ và Watchdog ngắt động cơ an toàn.
- **`web_server.py` & `cockpit.html`**: Bảng điều khiển Web thời gian thực hỗ trợ click chọn mục tiêu trực tiếp trên video và tinh chỉnh hệ số PID.

## 2. Quy trình chạy an toàn từng bước
```bash
# 1. Cài đặt thư viện phụ trợ trên Jetson Nano
pip3 install flask smbus2 numpy opencv-python

# 2. Kiểm thử phần cứng & hiệu chỉnh thông số thực tế
python3 scripts/self_test.py
python3 scripts/self_test.py --calibrate-speed
python3 scripts/self_test.py --calibrate-width --distance 1.0

# 3. Bước an toàn bắt buộc: chạy --no-drive để kiểm tra nhận diện & khoảng cách Z trước
python3 main.py --no-drive

# 4. Khởi chạy chính thức (Standalone hoặc qua ROS launch)
python3 main.py
# Hoặc: roslaunch jetbot_human_follower follower.launch