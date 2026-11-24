#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import threading
import config

try:
    from jetbot import Robot as JetBotDriver
    JETBOT_LIB_AVAILABLE = True
except Exception:
    JETBOT_LIB_AVAILABLE = False

try:
    import smbus2
    SMBUS_AVAILABLE = True
except Exception:
    SMBUS_AVAILABLE = False


class _RawPCA9685Driver:
    """Driver giao tiếp trực tiếp thanh ghi PCA9685 đúng sơ đồ chân MotorHAT JetBot"""
    MODE1 = 0x00
    PRESCALE = 0xFE
    LED0_ON_L = 0x06

    def __init__(self, bus_id=1, address=0x60):
        self.bus = smbus2.SMBus(bus_id)
        self.address = address
        self.bus.write_byte_data(self.address, self.MODE1, 0x00)
        self._set_pwm_freq(1000)

    def _set_pwm_freq(self, freq_hz):
        prescale = int(round(25000000.0 / (4096.0 * freq_hz)) - 1.0)
        old_mode = self.bus.read_byte_data(self.address, self.MODE1)
        self.bus.write_byte_data(self.address, self.MODE1, (old_mode & 0x7F) | 0x10)
        self.bus.write_byte_data(self.address, self.PRESCALE, prescale)
        self.bus.write_byte_data(self.address, self.MODE1, old_mode)
        time.sleep(0.005)
        self.bus.write_byte_data(self.address, self.MODE1, old_mode | 0xA1)

    def _set_pwm(self, channel, on, off):
        reg = self.LED0_ON_L + 4 * channel
        self.bus.write_byte_data(self.address, reg, on & 0xFF)
        self.bus.write_byte_data(self.address, reg + 1, on >> 8)
        self.bus.write_byte_data(self.address, reg + 2, off & 0xFF)
        self.bus.write_byte_data(self.address, reg + 3, off >> 8)

    def set_motor(self, pwm_ch, in1_ch, in2_ch, speed):
        pwm_val = int(min(4095, max(0, abs(speed) * 4095)))
        if speed > 0:
            self._set_pwm(in1_ch, 0, 4095)
            self._set_pwm(in2_ch, 0, 0)
        elif speed < 0:
            self._set_pwm(in1_ch, 0, 0)
            self._set_pwm(in2_ch, 0, 4095)
        else:
            self._set_pwm(in1_ch, 0, 0)
            self._set_pwm(in2_ch, 0, 0)
        self._set_pwm(pwm_ch, 0, pwm_val)


class MotorController:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_cmd_time = time.time()
        self.left_speed = 0.0
        self.right_speed = 0.0
        self.watchdog_trips = 0
        self.mode = "mock"
        self.robot = None
        self.raw_driver = None

        if JETBOT_LIB_AVAILABLE:
            try:
                self.robot = JetBotDriver()
                self.mode = "jetbot_native"
                print("[Motor] Kết nối thành công driver gốc jetbot.Robot.")
            except Exception as e:
                print(f"[Motor] Lỗi khởi tạo jetbot.Robot ({e}), thử smbus2...")

        if self.mode == "mock" and SMBUS_AVAILABLE:
            try:
                self.raw_driver = _RawPCA9685Driver(config.I2C_BUS, config.PCA9685_ADDRESS)
                self.mode = "pca9685_raw"
                print("[Motor] Kết nối thành công Raw PCA9685 (kênh 8-13).")
            except Exception as e:
                print(f"[Motor] Không tìm thấy PCA9685 ({e}), chạy chế độ Mock.")

        self._watchdog_running = True
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def _apply_deadband_and_clamp(self, val, trim):
        val = val * trim
        if abs(val) < 1e-3:
            return 0.0
        sign = 1.0 if val > 0 else -1.0
        scaled = config.MIN_MOTOR_PWM + (1.0 - config.MIN_MOTOR_PWM) * min(1.0, abs(val))
        return max(-1.0, min(1.0, sign * scaled))

    def drive_wheels(self, left_duty, right_duty):
        with self.lock:
            self.last_cmd_time = time.time()
            l_out = self._apply_deadband_and_clamp(left_duty, config.LEFT_MOTOR_TRIM)
            r_out = self._apply_deadband_and_clamp(right_duty, config.RIGHT_MOTOR_TRIM)
            self.left_speed = l_out
            self.right_speed = r_out

            if self.mode == "jetbot_native" and self.robot is not None:
                self.robot.set_motors(l_out, r_out)
            elif self.mode == "pca9685_raw" and self.raw_driver is not None:
                self.raw_driver.set_motor(8, 10, 9, l_out)
                self.raw_driver.set_motor(13, 11, 12, r_out)

    def stop(self):
        self.drive_wheels(0.0, 0.0)

    def _watchdog_loop(self):
        while self._watchdog_running:
            time.sleep(0.05)
            with self.lock:
                elapsed = time.time() - self.last_cmd_time
                if elapsed > config.MOTOR_WATCHDOG_TIMEOUT and (self.left_speed != 0.0 or self.right_speed != 0.0):
                    self.left_speed = 0.0
                    self.right_speed = 0.0
                    self.watchdog_trips += 1
                    if self.mode == "jetbot_native" and self.robot is not None:
                        self.robot.stop()
                    elif self.mode == "pca9685_raw" and self.raw_driver is not None:
                        self.raw_driver.set_motor(8, 10, 9, 0.0)
                        self.raw_driver.set_motor(13, 11, 12, 0.0)

    def shutdown(self):
        self._watchdog_running = False
        self.stop()
