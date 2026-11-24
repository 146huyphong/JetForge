#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import json
import os
import time
from typing import Optional


class RunLogger:
    def __init__(self, root: str = "runs", run_id: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        self.run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
        self.dir = os.path.join(root, self.run_id)
        self._files = {}
        if enabled:
            os.makedirs(self.dir, exist_ok=True)

    def _f(self, name: str, header=None):
        if name not in self._files:
            path = os.path.join(self.dir, name)
            new = not os.path.exists(path)
            fh = open(path, "a", newline="", encoding="utf-8")
            self._files[name] = (fh, csv.writer(fh) if name.endswith(".csv") else None)
            if new and header and name.endswith(".csv"):
                self._files[name][1].writerow(header)
        return self._files[name]

    def csv(self, name: str, header, row) -> None:
        if self.enabled:
            self._f(name, header)[1].writerow(row)

    def jsonl(self, name: str, obj: dict) -> None:
        if self.enabled:
            fh, _ = self._f(name)
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def event(self, t: float, kind: str, **kw) -> None:
        self.jsonl("events.jsonl", {"t": round(t, 3), "event": kind, **kw})

    def close(self) -> None:
        for fh, _ in self._files.values():
            fh.close()
        self._files.clear()
