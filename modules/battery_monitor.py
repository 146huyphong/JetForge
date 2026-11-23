#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
battery_monitor.py - Giám sát điện áp, dòng điện, công suất & % Pin qua cảm biến INA219
"""
import time
import threading
import config

try:
    import smbus2
    SMBUS_AVAILABLE = True
except Exception:
    SMBUS_AVAILABLE = False


class BatteryMonitor:
    REG_CONFIG = 0x00
    REG_SHUNT_V = 0x01
    REG_BUS_V = 0x02
    REG_POWER = 0x03
    REG_CURRENT = 0x04
    REG_CALIBRATION = 0x05

    def __init__(self):
        self.lock = threading.Lock()
        self.bus = None
        self.available = False
        self.status = {
            "voltage": 11.8,
            "current_a": 0.45,
            "power_w": 5.31,
            "percentage": 73,
            "is_low": False,
        }

        if SMBUS_AVAILABLE:
            try:
                self.bus = smbus2.SMBus(config.I2C_BUS)
                self._write_word(self.REG_CALIBRATION, 4096)
                self._write_word(self.REG_CONFIG, 0x399F)
                self.available = True
                print(f"[Battery] Đã kết nối INA219 tại địa chỉ 0x{config.INA219_ADDRESS:02X}")
            except Exception as e:
                print(f"[Battery] Không tìm thấy INA219 ({e}), dùng giá trị mô phỏng.")

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _write_word(self, reg, val):
        data = [(val >> 8) & 0xFF, val & 0xFF]
        self.bus.write_i2c_block_data(config.INA219_ADDRESS, reg, data)

    def _read_word(self, reg):
        data = self.bus.read_i2c_block_data(config.INA219_ADDRESS, reg, 2)
        return (data[0] << 8) | data[1]

    def _read_signed_word(self, reg):
        val = self._read_word(reg)
        return val - 65536 if val > 32767 else val

    def _poll_loop(self):
        while self._running:
            if self.available and self.bus is not None:
                try:
                    raw_bus = self._read_word(self.REG_BUS_V)
                    voltage = ((raw_bus >> 3) * 4) * 0.001

                    raw_curr = self._read_signed_word(self.REG_CURRENT)
                    current_a = raw_curr * 0.0001

                    raw_pwr = self._read_word(self.REG_POWER)
                    power_w = raw_pwr * 0.002

                    pct = int(round(
                        (voltage - config.BATTERY_MIN_V) /
                        max(0.1, config.BATTERY_MAX_V - config.BATTERY_MIN_V) * 100.0
                    ))
                    pct = max(0, min(100, pct))

                    with self.lock:
                        self.status = {
                            "voltage": round(voltage, 2),
                            "current_a": round(abs(current_a), 2),
                            "power_w": round(power_w, 2),
                            "percentage": pct,
                            "is_low": voltage <= (config.BATTERY_MIN_V + 0.3),
                        }
                except Exception:
                    pass
            time.sleep(2.0)

    def get_status(self):
        with self.lock:
            return dict(self.status)

    def shutdown(self):
        self._running = False