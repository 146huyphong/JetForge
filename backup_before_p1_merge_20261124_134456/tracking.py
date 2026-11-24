#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from perception import Detection


def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0.0 else 0.0


@dataclass
class Track:
    track_id: int
    cls: str
    bbox: Tuple[float, float, float, float]
    conf: float
    velocity: Tuple[float, float] = (0.0, 0.0)
    age: int = 1
    hits: int = 1
    lost_frames: int = 0
    truth_ids: Dict[str, int] = field(default_factory=dict)

    def predict(self) -> Tuple[float, float, float, float]:
        vx, vy = self.velocity
        x1, y1, x2, y2 = self.bbox
        return (x1 + vx, y1 + vy, x2 + vx, y2 + vy)

    @property
    def confirmed(self) -> bool:
        return self.hits >= 3


class Tracker:
    """ByteTrack-lite: ghep 2 vong IoU (tin cay cao -> tin cay thap) giu track khi bi che khuat."""

    def __init__(self, high: float = 0.50, low: float = 0.12, iou_min: float = 0.20, max_lost: int = 15):
        self.high = high
        self.low = low
        self.iou_min = iou_min
        self.max_lost = max_lost
        self.tracks: List[Track] = []
        self._next = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next = 1

    def _match(self, tracks: List[Track], dets: List[Detection]):
        pairs = sorted(
            (
                (iou(t.predict(), d.bbox), ti, di)
                for ti, t in enumerate(tracks)
                for di, d in enumerate(dets)
                if t.cls == d.cls
            ),
            reverse=True,
        )
        used_t, used_d, out = set(), set(), []
        for s, ti, di in pairs:
            if s < self.iou_min:
                break
            if ti not in used_t and di not in used_d:
                used_t.add(ti)
                used_d.add(di)
                out.append((ti, di))
        unmatched_t = [i for i in range(len(tracks)) if i not in used_t]
        unmatched_d = [i for i in range(len(dets)) if i not in used_d]
        return out, unmatched_t, unmatched_d

    def _update_track(self, t: Track, d: Detection) -> None:
        ocx = 0.5 * (t.bbox[0] + t.bbox[2])
        ocy = 0.5 * (t.bbox[1] + t.bbox[3])
        ncx = 0.5 * (d.bbox[0] + d.bbox[2])
        ncy = 0.5 * (d.bbox[1] + d.bbox[3])
        t.velocity = (
            0.6 * t.velocity[0] + 0.4 * (ncx - ocx),
            0.6 * t.velocity[1] + 0.4 * (ncy - ocy),
        )
        t.bbox = d.bbox
        t.conf = d.conf
        t.hits += 1
        t.lost_frames = 0

    def update(self, dets: List[Detection]) -> List[Track]:
        high = [d for d in dets if d.conf >= self.high]
        low = [d for d in dets if self.low <= d.conf < self.high]

        for t in self.tracks:
            t.age += 1

        m1, ut, ud = self._match(self.tracks, high)
        for ti, di in m1:
            self._update_track(self.tracks[ti], high[di])

        rest = [self.tracks[i] for i in ut]
        m2, ut2, _ = self._match(rest, low)
        for ti, di in m2:
            self._update_track(rest[ti], low[di])

        for i in ut2:
            t = rest[i]
            t.lost_frames += 1
            t.bbox = t.predict()

        for di in ud:
            d = high[di]
            self.tracks.append(Track(self._next, d.cls, d.bbox, d.conf))
            self._next += 1

        self.tracks = [t for t in self.tracks if t.lost_frames <= self.max_lost]
        return [t for t in self.tracks if t.confirmed and t.lost_frames == 0]

    def get(self, track_id: int) -> Optional[Track]:
        return next((t for t in self.tracks if t.track_id == track_id), None)
