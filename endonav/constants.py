"""Shared constants for endonav-agent.

Image geometry, scope rate limits, sign conventions, and the cached active-region
mask used by every CV operation. Keeping these in one place lets the empirical
tuning knobs (ROLL_SIGN, DEFL_SIGN, blob percentile, etc.) be flipped from a
single file.
"""

from __future__ import annotations

import cv2
import numpy as np

# ----- Image geometry (matches endonav_sim CoaxialRenderer defaults) -----
IMG_W: int = 1024
IMG_H: int = 768
ACTIVE_W: int = 870
ACTIVE_H: int = 760
ACTIVE_X0: int = (IMG_W - ACTIVE_W) // 2  # 77
ACTIVE_Y0: int = (IMG_H - ACTIVE_H) // 2  # 4
CX: int = IMG_W // 2  # 512
CY: int = IMG_H // 2  # 384

# ----- Scope rate limits (must match endonav_sim ScopeLimits defaults) -----
MAX_ADV_MM: float = 2.0
MAX_ROLL_DEG: float = 15.0
# Per-step deflection-change cap. The sim's ScopeLimits has this field but
# does not enforce it; we enforce it in the autopilot to keep motion smooth.
MAX_DEFL_STEP_DEG: float = 20.0
# Absolute deflection envelope used by the agent. Sim allows ±270° but
# anything beyond ±60° is rarely useful and risks tangling the shaft.
MAX_DEFL_ABS_DEG: float = 60.0
DEFL_NEUTRAL_DEG: float = 0.0

# ----- Sign conventions (flip if empirical tests show inverted behaviour) -----
ROLL_SIGN: int = +1
DEFL_SIGN: int = +1


def _build_active_mask() -> np.ndarray:
    """Build the active-region mask once at import time."""
    mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    cv2.rectangle(
        mask,
        (ACTIVE_X0, ACTIVE_Y0),
        (ACTIVE_X0 + ACTIVE_W, ACTIVE_Y0 + ACTIVE_H),
        255,
        -1,
    )
    return mask


ACTIVE_MASK: np.ndarray = _build_active_mask()
ACTIVE_AREA_PX: int = ACTIVE_W * ACTIVE_H
