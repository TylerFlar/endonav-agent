"""Stateless single-frame analysis for the endonav agent.

FrameAnalyzer takes raw RGB frames from the simulator and produces:
- Dark blobs (lumen openings) via adaptive percentile thresholding
- Scene classification: tube / junction_candidate / dead_end_candidate
- Stone hints via HSV color masking against pink mucosa
- Image-space steering commands (roll + deflection deltas)
- HSV histogram place signatures

Every CV operation is masked to the active region to exclude the camera
letterbox. All thresholds are tunable from constants.py or the FrameAnalyzer
constructor.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .constants import (
    ACTIVE_AREA_PX,
    ACTIVE_MASK,
    CX,
    CY,
    DEFL_SIGN,
    IMG_W,
    MAX_DEFL_STEP_DEG,
    MAX_ROLL_DEG,
    ROLL_SIGN,
)
from .types import Blob, Scene, StoneHint


def _wrap180(angle_deg: float) -> float:
    """Wrap an angle in degrees into the range [-180, 180)."""
    return ((angle_deg + 180.0) % 360.0) - 180.0


# HSV color ranges per stone composition. Hue is OpenCV's [0, 180].
# Refined empirically against the EndoPBR shader's stone palette.
_STONE_HSV_RANGES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "uric_acid": ((10, 80, 100), (30, 220, 230)),
    "struvite": ((10, 30, 120), (30, 120, 220)),
    "calcium_oxalate": ((0, 0, 170), (40, 60, 240)),
    "cystine": ((20, 40, 170), (40, 140, 240)),
}

# Pink mucosa hue ranges (red wraps in HSV).
_MUCOSA_RANGES: tuple[
    tuple[tuple[int, int, int], tuple[int, int, int]],
    tuple[tuple[int, int, int], tuple[int, int, int]],
] = (
    ((0, 80, 60), (15, 255, 220)),
    ((160, 80, 60), (180, 255, 220)),
)

# Specular highlight (white-ish, very bright, low saturation).
_SPEC_RANGE: tuple[tuple[int, int, int], tuple[int, int, int]] = (
    (0, 0, 245),
    (180, 40, 255),
)


class FrameAnalyzer:
    """Stateless image analysis. One instance per agent is fine."""

    def __init__(
        self,
        dark_percentile: float = 8.0,
        min_blob_area_px: int = 400,
        max_blob_area_frac: float = 0.6,
        min_circularity: float = 0.25,
        # Mucosa is bright (200+); a real lumen blob's mean intensity tops
        # out around 130 even with gradient inclusion.
        dead_end_intensity: float = 140.0,
        min_stone_area_px: int = 600,
        plaque_std_max: float = 8.0,
        plaque_max_area_px: int = 1500,
    ) -> None:
        self.dark_percentile = dark_percentile
        self.min_blob_area_px = min_blob_area_px
        self.max_blob_area_px = int(max_blob_area_frac * ACTIVE_AREA_PX)
        self.min_circularity = min_circularity
        self.dead_end_intensity = dead_end_intensity
        self.min_stone_area_px = min_stone_area_px
        self.plaque_std_max = plaque_std_max
        self.plaque_max_area_px = plaque_max_area_px

    # ------------------------------------------------------------------
    # Dark blob detection
    # ------------------------------------------------------------------
    def detect_dark_blobs(self, rgb: np.ndarray) -> list[Blob]:
        """Find dark connected regions inside the active view.

        Adaptive percentile thresholding handles the coaxial-light vignette:
        as the tube gets brighter or dimmer the threshold moves with it.
        """
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2.0)

        active_vals = gray[ACTIVE_MASK > 0]
        if active_vals.size == 0:
            return []
        thr = float(np.percentile(active_vals, self.dark_percentile))
        _, bw = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
        bw = cv2.bitwise_and(bw, ACTIVE_MASK)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs: list[Blob] = []
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < self.min_blob_area_px or area > self.max_blob_area_px:
                continue
            perim = float(cv2.arcLength(c, True))
            if perim < 1e-6:
                continue
            circularity = float(4.0 * math.pi * area / (perim * perim))
            if circularity < self.min_circularity:
                continue
            M = cv2.moments(c)
            if M["m00"] < 1e-6:
                continue
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(c)
            # Mean intensity inside this contour only.
            blob_mask = np.zeros_like(gray)
            cv2.drawContours(blob_mask, [c], -1, 255, -1)
            mean_intensity = float(cv2.mean(gray, mask=blob_mask)[0])
            blobs.append(
                Blob(
                    center_xy=(cx, cy),
                    area=area,
                    bbox=(int(x), int(y), int(w), int(h)),
                    mean_intensity=mean_intensity,
                    circularity=circularity,
                )
            )
        # Darkest first.
        blobs.sort(key=lambda b: b.mean_intensity)
        return blobs

    # ------------------------------------------------------------------
    # Scene classification
    # ------------------------------------------------------------------
    def classify_scene(self, rgb: np.ndarray, blobs: list[Blob]) -> Scene:
        """Classify the frame as tube / junction / dead end."""
        if not blobs:
            return Scene(
                type="dead_end_candidate",
                confidence=0.7,
                darkest_blob_idx=None,
                blob_count=0,
                reason="no_dark_blobs",
            )

        darkest = blobs[0]
        if darkest.mean_intensity > self.dead_end_intensity:
            return Scene(
                type="dead_end_candidate",
                confidence=0.65,
                darkest_blob_idx=0,
                blob_count=len(blobs),
                reason=f"darkest_blob_too_bright_{darkest.mean_intensity:.0f}",
            )

        # Junction check: 2+ comparably-sized blobs with angular separation
        # AND a brighter septum between them.
        if len(blobs) >= 2:
            second = blobs[1]
            ratio = darkest.area / max(second.area, 1.0)
            if 1.0 / 3.0 <= ratio <= 3.0:
                # Angular separation from image center.
                ang0 = math.atan2(darkest.center_xy[1] - CY, darkest.center_xy[0] - CX)
                ang1 = math.atan2(second.center_xy[1] - CY, second.center_xy[0] - CX)
                ang_sep_deg = abs(math.degrees(_wrap180(math.degrees(ang0 - ang1))))
                if ang_sep_deg >= 25.0:
                    if self._has_septum(rgb, darkest, second):
                        return Scene(
                            type="junction_candidate",
                            confidence=0.85,
                            darkest_blob_idx=0,
                            blob_count=len(blobs),
                            reason=f"two_blobs_sep_{ang_sep_deg:.0f}deg",
                        )

        # Single huge central blob = wide lumen, still classify as tube.
        if darkest.area > 0.4 * ACTIVE_AREA_PX:
            return Scene(
                type="tube",
                confidence=0.6,
                darkest_blob_idx=0,
                blob_count=len(blobs),
                reason="wide_lumen",
            )

        return Scene(
            type="tube",
            confidence=0.85,
            darkest_blob_idx=0,
            blob_count=len(blobs),
            reason="single_dominant_blob",
        )

    def _has_septum(self, rgb: np.ndarray, a: Blob, b: Blob) -> bool:
        """Sample 20 points on the line between two blob centroids; the mean
        intensity must be at least 1.4x the average of the two blob means.
        """
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        xs = np.linspace(a.center_xy[0], b.center_xy[0], 20)
        ys = np.linspace(a.center_xy[1], b.center_xy[1], 20)
        xs = np.clip(xs.astype(int), 0, w - 1)
        ys = np.clip(ys.astype(int), 0, h - 1)
        line_mean = float(np.mean(gray[ys, xs]))
        blob_mean = (a.mean_intensity + b.mean_intensity) / 2.0
        return line_mean >= 1.4 * blob_mean

    # ------------------------------------------------------------------
    # Stone hint detection
    # ------------------------------------------------------------------
    def detect_stone_hints(self, rgb: np.ndarray) -> list[StoneHint]:
        """Detect stone-colored regions, rejecting mucosa, plaque, and specular."""
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Mucosa mask = red ranges.
        mucosa_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in _MUCOSA_RANGES:
            mucosa_mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

        # Specular highlight mask.
        spec_mask = cv2.inRange(hsv, np.array(_SPEC_RANGE[0]), np.array(_SPEC_RANGE[1]))

        not_mucosa = cv2.bitwise_not(mucosa_mask)
        not_spec = cv2.bitwise_not(spec_mask)

        hints: list[StoneHint] = []
        for name, (lo, hi) in _STONE_HSV_RANGES.items():
            m = cv2.inRange(hsv, np.array(lo), np.array(hi))
            m = cv2.bitwise_and(m, not_mucosa)
            m = cv2.bitwise_and(m, not_spec)
            m = cv2.bitwise_and(m, ACTIVE_MASK)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = float(cv2.contourArea(c))
                if area < self.min_stone_area_px:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                roi = gray[y : y + h, x : x + w]
                if roi.size == 0:
                    continue
                # Randall's plaque rejection: textureless flat off-white blobs.
                if (
                    name == "calcium_oxalate"
                    and float(roi.std()) < self.plaque_std_max
                    and area < self.plaque_max_area_px
                ):
                    continue
                M = cv2.moments(c)
                if M["m00"] < 1e-6:
                    continue
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
                conf = float(min(1.0, area / 5000.0 + 0.3))
                hints.append(
                    StoneHint(
                        center_xy=(cx, cy),
                        area=area,
                        composition_guess=name,
                        confidence=conf,
                    )
                )
        hints.sort(key=lambda s: -s.confidence)
        return hints

    # ------------------------------------------------------------------
    # Steering
    # ------------------------------------------------------------------
    def compute_steering(
        self,
        target_xy: tuple[float, float],
    ) -> dict[str, float]:
        """Convert image-space target offset to (roll_delta_deg, defl_delta_deg).

        Coupled proportional control: every frame, roll a bit and deflect a bit
        simultaneously. Empirical findings (probed against the renderer):

        - +deflection tilts the camera UP, which makes the lumen appear LOWER
          in the image. So to bring a lumen below center back to center we
          need -deflection. Result: ``defl_delta = -k_d * dy``.
        - Roll rotates the image around the shaft axis. We use ``+k_r * dx``
          and let ROLL_SIGN handle empirical inversion.

        Returns DELTA values; the caller converts to absolute by adding to
        the current deflection (and clamping to the absolute envelope).
        """
        tx, ty = target_xy
        dx = float(tx - CX)
        dy = float(ty - CY)
        defl_delta = float(
            np.clip(-0.04 * dy * DEFL_SIGN, -MAX_DEFL_STEP_DEG, MAX_DEFL_STEP_DEG)
        )
        roll_delta = float(
            np.clip(0.02 * dx * ROLL_SIGN, -MAX_ROLL_DEG, MAX_ROLL_DEG)
        )
        return {"roll_delta_deg": roll_delta, "defl_delta_deg": defl_delta}

    # ------------------------------------------------------------------
    # Place signature
    # ------------------------------------------------------------------
    def compute_place_signature(self, rgb: np.ndarray) -> np.ndarray:
        """HSV 8x8x8 normalized histogram, flattened to 512-D float32."""
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            ACTIVE_MASK,
            [8, 8, 8],
            [0, 180, 0, 256, 0, 256],
        )
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.flatten().astype(np.float32)
