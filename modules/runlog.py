#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import json
import os
import time


class RunLogger:
    def __init__(self, root_dir, enabled=True):
        self.enabled = bool(enabled)
        self.run_id = time.strftime("run_%Y%m%d_%H%M%S")
        self.dir = os.path.join(root_dir, self.run_id)
        self._files = {}
        self._last_csv_ts = 0.0
        if self.enabled:
            os.makedirs(self.dir, exist_ok=True)

    def _f(self, name, header=None):
        if name not in self._files:
            path = os.path.join(self.dir, name)
            is_new = not os.path.exists(path)
            fh = open(path, "a", newline="", encoding="utf-8")
            writer = csv.writer(fh) if name.endswith(".csv") else None
            self._files[name] = (fh, writer)
            if is_new and header and writer is not None:
                writer.writerow(header)
        return self._files[name]

    def csv_throttled(self, now, interval_sec, name, header, row):
        if not self.enabled:
            return
        if (now - self._last_csv_ts) < interval_sec:
            return
        self._last_csv_ts = now
        self._f(name, header)[1].writerow(row)

    def event(self, t, kind, **kw):
        if not self.enabled:
            return
        payload = {"t": round(t, 3), "event": kind}
        payload.update(kw)
        fh, _ = self._f("events.jsonl")
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def close(self):
        for fh, _ in self._files.values():
            try:
                fh.close()
            except Exception:
                pass
        self._files.clear()