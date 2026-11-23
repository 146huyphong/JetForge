# KẾ HOẠCH THỰC HIỆN & CHECKLIST ĐỒ ÁN JETBOT HUMAN FOLLOWER

## Giai đoạn 1: Kiểm thử phần cứng & Hiệu chỉnh cơ khí (Tuần 1)
- [x] Thiết lập workspace ROS Melodic (`catkin_ws/src/jetbot_human_follower`).
- [ ] Cấp quyền thực thi: `chmod +x main.py scripts/*.py`.
- [ ] Chạy `python3 scripts/self_test.py` kiểm tra I2C PCA9685, INA219 và luồng Camera CSI GStreamer.
- [ ] Đo góc ngước camera thực tế và cập nhật `CAMERA_TILT_DEG` trong `config.py`.
- [ ] Chạy `python3 scripts/self_test.py --calibrate-speed` để đo `V_FULL_MPS` thực tế trên mặt sàn.

## Giai đoạn 2: Chuẩn bị Model TensorRT (Tuần 2)
- [ ] Xuất model `ssd_mobilenet_v2_coco.engine` (FP16) từ `jetson-inference` vào thư mục `models/`.
- [ ] Thu thập 600–800 ảnh thực tế (`blocked` / `free`) từ camera JetBot, fine-tune ResNet-18 và xuất sang `models/collision_resnet18.engine`.
- [ ] Chạy `python3 scripts/self_test.py --calibrate-width --distance 1.0` để hiệu chỉnh bề rộng hiệu dụng của detector.

## Giai đoạn 3: Chạy thử an toàn (`--no-drive`) & Tinh chỉnh PID (Tuần 3)
- [ ] Chạy `python3 main.py --no-drive`, mở Web Cockpit kiểm tra bounding box khóa đúng người và khoảng cách `Z` khớp thước đo thực tế.
- [ ] Bỏ `--no-drive`, tinh chỉnh PID góc (`Angular`) và PID khoảng cách (`Linear`) qua thanh trượt trên Web Cockpit.
- [ ] Kiểm thử các tình huống biên: người đứng quá gần (`BACKUP`), người quay lưng/khuất tạm thời (`COASTING` -> `SEARCHING`), và ngắt kết nối (Watchdog dừng sau 0.4s).

## Giai đoạn 4: Thu thập dữ liệu thực nghiệm & Viết báo cáo (Tuần 4)
- [ ] Trích xuất file CSV trong `logs/` để vẽ đồ thị đáp ứng góc lệch $\theta$, sai số khoảng cách $e_z$, độ trễ khung hình và FPS.