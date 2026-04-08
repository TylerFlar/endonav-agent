"""Place recognition via HSV histograms."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class PlaceMatch:
    place_id: str
    score: float


class PlaceMemory:
    def __init__(self, threshold: float = 0.80) -> None:
        self.threshold = threshold
        self._places: dict[str, np.ndarray] = {}
        self._counter = 0

    @staticmethod
    def _hist(frame_rgb: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(h, h)
        return h.flatten().astype(np.float32)

    def add(self, frame_rgb: np.ndarray, place_id: Optional[str] = None) -> str:
        if place_id is None:
            place_id = f"place_{self._counter:03d}"
            self._counter += 1
        self._places[place_id] = self._hist(frame_rgb)
        return place_id

    def match(self, frame_rgb: np.ndarray) -> Optional[PlaceMatch]:
        if not self._places:
            return None
        h = self._hist(frame_rgb)
        best_id, best = None, -1.0
        for pid, ph in self._places.items():
            denom = float(np.linalg.norm(h) * np.linalg.norm(ph)) + 1e-9
            score = float(np.dot(h, ph) / denom)
            if score > best:
                best, best_id = score, pid
        if best_id is None or best < self.threshold:
            return None
        return PlaceMatch(place_id=best_id, score=best)
