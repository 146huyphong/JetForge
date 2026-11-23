#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collision_guard.py - Tránh vật cản bằng AI thị giác ResNet-18 (Blocked / Free)
"""
import os
import cv2
import numpy as np
import config

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    TRT_AVAILABLE = True
except Exception:
    TRT_AVAILABLE = False


class CollisionGuard:
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self):
        self.use_trt = False
        self.ema_prob = 0.0
        self.cuda_ctx = None
        self.engine = None
        self.context = None
        self.stream = None
        self.inputs = []
        self.outputs = []
        self.bindings = []

        if not config.ENABLE_COLLISION_GUARD:
            return

        if TRT_AVAILABLE and os.path.exists(config.COLLISION_ENGINE_PATH):
            try:
                self._init_tensorrt(config.COLLISION_ENGINE_PATH)
                self.use_trt = True
                print(f"[Collision] Nạp thành công TensorRT engine: {config.COLLISION_ENGINE_PATH}")
            except Exception as e:
                print(f"[Collision] Lỗi nạp engine ({e}), dùng bộ kiểm tra heuristic dự phòng.")
        else:
            print("[Collision] Chưa có collision_resnet18.engine -> Dùng heuristic dự phòng.")

    def _init_tensorrt(self, engine_path):
        self.cuda_ctx = cuda.Device(0).make_context()
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            entry = {"host": host_mem, "device": device_mem}
            if self.engine.binding_is_input(binding):
                self.inputs.append(entry)
            else:
                self.outputs.append(entry)

    def _preprocess(self, frame):
        w, h = config.COLLISION_INPUT_SIZE
        resized = cv2.resize(frame, (w, h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = (rgb - self.MEAN) / self.STD
        chw = np.transpose(norm, (2, 0, 1))
        return np.ascontiguousarray(chw[np.newaxis, ...])

    def predict(self, frame):
        if not config.ENABLE_COLLISION_GUARD:
            return {"is_blocked": False, "blocked_prob": 0.0}

        if not self.use_trt:
            h, w = frame.shape[:2]
            roi = frame[int(h * 0.78):h, int(w * 0.25):int(w * 0.75)]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 180)
            edge_density = float(np.mean(edges > 0))
            raw_prob = min(1.0, edge_density * 2.2)
        else:
            self.cuda_ctx.push()
            try:
                blob = self._preprocess(frame)
                np.copyto(self.inputs[0]["host"], blob.ravel())
                cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
                self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
                cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
                self.stream.synchronize()
            finally:
                self.cuda_ctx.pop()

            logits = self.outputs[0]["host"][:2]
            exp_l = np.exp(logits - np.max(logits))
            probs = exp_l / np.sum(exp_l)
            raw_prob = float(probs[0])

        alpha = config.COLLISION_EMA_ALPHA
        self.ema_prob = alpha * raw_prob + (1.0 - alpha) * self.ema_prob
        is_blocked = self.ema_prob >= config.COLLISION_BLOCKED_THRESHOLD

        return {"is_blocked": bool(is_blocked), "blocked_prob": float(self.ema_prob)}

    def cleanup(self):
        if self.cuda_ctx is not None:
            try:
                self.cuda_ctx.detach()
            except Exception:
                pass