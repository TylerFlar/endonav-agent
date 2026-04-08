"""Classical lumen-following PD controller. Free, runs every frame."""
from __future__ import annotations

from typing import Optional

from .perception import FrameAnalysis


class Autopilot:
    """Steers toward the dark blob (lumen opening) and advances when centered."""

    def __init__(
        self,
        image_width: int = 1024,
        image_height: int = 768,
        center_tol_frac: float = 0.08,
        kp_deflection: float = 30.0,
        max_step_advance_mm: float = 1.0,
        retract_area_frac: float = 0.03,
    ) -> None:
        self.cx = image_width / 2.0
        self.cy = image_height / 2.0
        self.center_tol = center_tol_frac * image_width
        self.kp_deflection = kp_deflection
        self.max_step_advance_mm = max_step_advance_mm
        self.retract_area_frac = retract_area_frac
        self._cur_deflection_deg: float = 0.0
        self._cur_roll_deg: float = 0.0

    def reset(self) -> None:
        self._cur_deflection_deg = 0.0
        self._cur_roll_deg = 0.0
        self._buckle_recovery = 0

    def notify_feedback(self, feedback) -> None:
        """Agent calls this with CommandFeedback after each command."""
        if getattr(feedback, "buckled", False) or getattr(feedback, "collided", False):
            # Cable is loaded — straighten fully and retract
            self._cur_deflection_deg = 0.0
            self._buckle_recovery = 10

    def compute_command(self, analysis: FrameAnalysis) -> Optional[dict]:
        if self._buckle_recovery > 0:
            self._buckle_recovery -= 1
            self._cur_deflection_deg *= 0.6
            return {
                "advance_mm": -0.6,
                "roll_deg": 0.0,
                "deflection_deg": self._cur_deflection_deg,
            }
        # Ambiguous: no clear opening AND not obviously a wall
        if analysis.dark_centroid is None and analysis.dark_area_frac > 0.005:
            return None

        if analysis.dark_centroid is None:
            # Wall straight ahead — back off
            return {"advance_mm": -0.5, "roll_deg": 0.0, "deflection_deg": self._cur_deflection_deg}

        x, y = analysis.dark_centroid
        dx = x - self.cx
        dy = y - self.cy

        # Roll image so vertical component is mostly horizontal-handled by deflection.
        # Simple: convert vertical pixel error into deflection_deg adjustment.
        # Use dy/img_height as normalized error.
        norm_err_y = dy / self.cy  # [-1, 1]
        norm_err_x = dx / self.cx

        # If lumen is far off-center horizontally, roll to bring it onto bending plane.
        roll_cmd = 0.0
        if abs(dx) > self.center_tol:
            # Positive dx => roll positive (CW) to put opening "above" bending plane
            roll_cmd = max(-10.0, min(10.0, norm_err_x * 12.0))

        # In an open chamber the centroid jumps wildly — reduce gain so we don't
        # slam the tip into a wall while waiting for the VLM to take over.
        chamber = analysis.dark_area_frac > 0.30
        gain_scale = 0.3 if chamber else 1.0

        # Deflection follows vertical error — slow slew, low gain
        delta = self.kp_deflection * norm_err_y * 0.06 * gain_scale
        delta = max(-1.5, min(1.5, delta))
        target_deflection = self._cur_deflection_deg + delta
        target_deflection = max(-15.0, min(15.0, target_deflection))
        self._cur_deflection_deg = target_deflection

        # Advance — slow down in chambers, don't ram things
        centered = abs(dx) < self.center_tol * 1.5 and abs(dy) < self.center_tol * 1.5
        if chamber:
            advance = 0.2
        elif centered and analysis.dark_area_frac > self.retract_area_frac:
            advance = self.max_step_advance_mm
        elif analysis.dark_area_frac < 0.005:
            advance = -0.5
        else:
            advance = 0.2

        return {
            "advance_mm": advance,
            "roll_deg": roll_cmd,
            "deflection_deg": target_deflection,
        }
