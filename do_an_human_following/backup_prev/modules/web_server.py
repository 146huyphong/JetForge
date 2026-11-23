#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
web_server.py - Server Web Cockpit giám sát luồng MJPEG, click chọn người & chỉnh PID trực tiếp
"""
import os
import time
import threading
import logging
import cv2
from flask import Flask, render_template, Response, request, jsonify
import config


class WebCockpitServer:
    def __init__(self, shared_telemetry, callbacks):
        self.shared_telemetry = shared_telemetry
        self.callbacks = callbacks
        self.frame_lock = threading.Lock()
        self.jpeg_bytes = None
        self.running = False

        template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        self.app = Flask(__name__, template_folder=template_dir)

        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

        self._register_routes()

    def update_frame(self, bgr_frame):
        if bgr_frame is None:
            return
        resized = cv2.resize(bgr_frame, (config.STREAM_WIDTH, config.STREAM_HEIGHT))
        ok, buf = cv2.imencode(
            ".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), config.STREAM_JPEG_QUALITY]
        )
        if ok:
            with self.frame_lock:
                self.jpeg_bytes = buf.tobytes()

    def _mjpeg_generator(self):
        interval = 1.0 / max(1, config.STREAM_FPS)
        while self.running:
            with self.frame_lock:
                frame_data = self.jpeg_bytes
            if frame_data is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"
                )
            time.sleep(interval)

    def _register_routes(self):
        @self.app.route("/")
        def index():
            return render_template("cockpit.html")

        @self.app.route("/video_feed")
        def video_feed():
            return Response(
                self._mjpeg_generator(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @self.app.route("/api/status", methods=["GET"])
        def api_status():
            return jsonify(self.shared_telemetry)

        @self.app.route("/api/select", methods=["POST"])
        def api_select():
            data = request.get_json(force=True)
            nx = float(data.get("nx", 0.5))
            ny = float(data.get("ny", 0.5))
            tid = self.callbacks["on_select_point"](nx, ny)
            return jsonify({"ok": tid is not None, "locked_id": tid})

        @self.app.route("/api/clear_target", methods=["POST"])
        def api_clear_target():
            self.callbacks["on_clear_target"]()
            return jsonify({"ok": True})

        @self.app.route("/api/mode", methods=["POST"])
        def api_mode():
            data = request.get_json(force=True)
            mode = str(data.get("mode", "IDLE")).upper()
            self.callbacks["on_set_mode"](mode)
            return jsonify({"ok": True, "mode": mode})

        @self.app.route("/api/pid", methods=["POST"])
        def api_pid():
            data = request.get_json(force=True)
            pid_type = data.get("type", "angular")
            kp = float(data.get("kp", 0.5))
            ki = float(data.get("ki", 0.0))
            kd = float(data.get("kd", 0.0))
            self.callbacks["on_update_pid"](pid_type, kp, ki, kd)
            return jsonify({"ok": True})

        @self.app.route("/api/manual", methods=["POST"])
        def api_manual():
            data = request.get_json(force=True)
            linear = float(data.get("linear", 0.0))
            angular = float(data.get("angular", 0.0))
            self.callbacks["on_manual_cmd"](linear, angular)
            return jsonify({"ok": True})

    def start(self):
        self.running = True
        self._thread = threading.Thread(
            target=lambda: self.app.run(
                host=config.WEB_HOST,
                port=config.WEB_PORT,
                debug=False,
                use_reloader=False,
                threaded=True,
            ),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self.running = False