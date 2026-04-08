"""Classical CV per-frame analysis: dark-blob lumen + HSV stone hint."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class FrameAnalysis:
    dark_mask: np.ndarray
    dark_centroid: Optional[tuple[float, float]]  # (x, y) in pixels, or None
    dark_area_frac: float
    n_dark_blobs: int
    classification_hint: str  # "lumen" | "possible_junction" | "possible_dead_end"
    stone_hint: bool
    stone_mask: np.ndarray
    confirmed: bool  # hint stable across recent frames


class FrameAnalyzer:
    """Stateful analyzer that smooths hints across frames."""

    def __init__(
        self,
        dark_v_thresh: int = 60,
        min_blob_frac: float = 0.005,
        junction_blob_count: int = 2,
        dead_end_area_frac: float = 0.02,
        stone_min_frac: float = 0.003,
        stable_window: int = 3,
    ) -> None:
        self.dark_v_thresh = dark_v_thresh
        self.min_blob_frac = min_blob_frac
        self.junction_blob_count = junction_blob_count
        self.dead_end_area_frac = dead_end_area_frac
        self.stone_min_frac = stone_min_frac
        self._hist: deque[str] = deque(maxlen=stable_window)
        self._stone_hist: deque[bool] = deque(maxlen=stable_window)

    def analyze(self, frame_rgb: np.ndarray) -> FrameAnalysis:
        h, w = frame_rgb.shape[:2]
        total_px = float(h * w)
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]

        # Dark mask: openings into deeper lumen
        dark = (v < self.dark_v_thresh).astype(np.uint8) * 255
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
        # Filter blobs by min size
        min_area = self.min_blob_frac * total_px
        big_idxs = [
            i for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area
        ]
        n_blobs = len(big_idxs)

        if big_idxs:
            # Largest blob's centroid
            largest = max(big_idxs, key=lambda i: stats[i, cv2.CC_STAT_AREA])
            cx, cy = centroids[largest]
            dark_centroid: Optional[tuple[float, float]] = (float(cx), float(cy))
            area = sum(stats[i, cv2.CC_STAT_AREA] for i in big_idxs)
            area_frac = area / total_px
        else:
            dark_centroid = None
            area_frac = 0.0

        if n_blobs >= self.junction_blob_count:
            hint = "possible_junction"
        elif area_frac < self.dead_end_area_frac:
            hint = "possible_dead_end"
        else:
            hint = "lumen"

        # Stone mask: yellow/brown against pink
        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]
        stone_mask = (
            (h_ch >= 15) & (h_ch <= 30) & (s_ch >= 60) & (v_ch < 200) & (v_ch > 40)
        ).astype(np.uint8) * 255
        stone_mask = cv2.morphologyEx(stone_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        stone_area_frac = float(stone_mask.sum() / 255) / total_px
        stone_hint_now = stone_area_frac > self.stone_min_frac

        self._hist.append(hint)
        self._stone_hist.append(stone_hint_now)
        confirmed = len(self._hist) == self._hist.maxlen and all(
            x == hint for x in self._hist
        )

        return FrameAnalysis(
            dark_mask=dark,
            dark_centroid=dark_centroid,
            dark_area_frac=area_frac,
            n_dark_blobs=n_blobs,
            classification_hint=hint,
            stone_hint=stone_hint_now and any(self._stone_hist),
            stone_mask=stone_mask,
            confirmed=confirmed,
        )
