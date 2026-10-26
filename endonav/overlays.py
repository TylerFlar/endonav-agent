"""Agent-side HUD compositor.

The agent owns ONE thing on the realtime feed: a heads-up display showing
its own state (calyx counts, stones destroyed, frame counter, REC indicator,
last command feedback). Everything physical (laser flash, smoke, basket
cage, camera micro-motion) lives in ``endonav_sim.visuals`` and is composed
by the sim's ``compose_visuals``.

The top-level ``compose_display_frame`` here:
  1. Calls ``endonav_sim.visuals.compose_visuals`` to apply physical effects
  2. Draws the HUD on top of that cinematic frame
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from endonav_sim.visuals import IMG_H, IMG_W, ToolVisualState, compose_visuals

# ----------------------------------------------------------------------
# HUD constants
# ----------------------------------------------------------------------
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_FONT_THICK = 1


def _put_text_with_shadow(
    rgb: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int] = (240, 240, 240),
) -> None:
    cv2.putText(
        rgb,
        text,
        (org[0] + 1, org[1] + 1),
        _FONT,
        _FONT_SCALE,
        (0, 0, 0),
        _FONT_THICK + 1,
        cv2.LINE_AA,
    )
    cv2.putText(rgb, text, org, _FONT, _FONT_SCALE, color, _FONT_THICK, cv2.LINE_AA)


def _draw_text_bar(
    rgb: np.ndarray,
    rect: tuple[int, int, int, int],
    alpha: float = 0.5,
) -> None:
    """Translucent dark bar behind a text region for legibility on pink mucosa."""
    x, y, w, h = rect
    sub = rgb[y : y + h, x : x + w]
    if sub.size == 0:
        return
    dark = np.zeros_like(sub)
    rgb[y : y + h, x : x + w] = cv2.addWeighted(sub, 1.0 - alpha, dark, alpha, 0)


def draw_hud(
    rgb: np.ndarray,
    mission_status: dict,
    frame_idx: int,
    start_ts: float,
) -> None:
    """Top-bar mission status, bottom-bar frame/clock/REC + scope feedback."""
    calyces_visited = mission_status.get("calyces_visited", 0)
    total_calyces = mission_status.get("total_calyces", 0)
    calyces_clear = mission_status.get("calyces_verified_clear", 0)
    stones_destroyed = mission_status.get("stones_destroyed", 0)
    fragments = mission_status.get("fragments_remaining", 0)
    current_node = mission_status.get("current_node_id", "—")
    feedback = mission_status.get("last_feedback") or {}
    wall_mm = feedback.get("wall_clearance_mm")
    defl_deg = feedback.get("actual_deflection_deg")
    total_commands = mission_status.get("total_commands", 0)

    # Top bar
    _draw_text_bar(rgb, (0, 0, IMG_W, 36), alpha=0.55)
    _put_text_with_shadow(
        rgb,
        f"Calyces {calyces_clear}/{total_calyces} clear  |  Visited {calyces_visited}  |  Stones destroyed {stones_destroyed}  |  Fragments remaining {fragments}",
        (12, 24),
    )

    # Bottom bar
    _draw_text_bar(rgb, (0, IMG_H - 36, IMG_W, 36), alpha=0.55)
    elapsed = max(0.0, time.time() - start_ts)
    hh = int(elapsed // 3600)
    mm = int((elapsed % 3600) // 60)
    ss = int(elapsed % 60)
    clock = f"{hh:02d}:{mm:02d}:{ss:02d}"
    bottom_left = f"Frame {frame_idx}  |  Cmd {total_commands}  |  {clock}"
    _put_text_with_shadow(rgb, bottom_left, (12, IMG_H - 12))

    # Blinking REC dot — placed AFTER the bottom_left text using its measured width.
    bl_size = cv2.getTextSize(bottom_left, _FONT, _FONT_SCALE, _FONT_THICK)[0]
    rec_x = 12 + bl_size[0] + 16
    if int(elapsed * 2) % 2 == 0:
        cv2.circle(rgb, (rec_x, IMG_H - 18), 6, (255, 60, 60), -1, lineType=cv2.LINE_AA)
        _put_text_with_shadow(
            rgb, "REC", (rec_x + 10, IMG_H - 12), color=(255, 100, 100)
        )

    # Bottom-right scope feedback
    parts = [f"node {current_node}"]
    if wall_mm is not None:
        try:
            parts.append(f"wall {float(wall_mm):.1f}mm")
        except (TypeError, ValueError):
            pass
    if defl_deg is not None:
        try:
            parts.append(f"defl {float(defl_deg):+.0f}deg")
        except (TypeError, ValueError):
            pass
    right_text = "  ".join(parts)
    text_size = cv2.getTextSize(right_text, _FONT, _FONT_SCALE, _FONT_THICK)[0]
    _put_text_with_shadow(rgb, right_text, (IMG_W - text_size[0] - 12, IMG_H - 12))


def compose_display_frame(
    canonical_rgb: np.ndarray,
    mission_status: dict,
    tool_state: ToolVisualState | None,
    frame_idx: int,
    rng: np.random.Generator,
    start_ts: float,
) -> np.ndarray:
    """Build a single display frame: sim cinematic → HUD on top.

    The cinematic pass (jitter + laser/smoke/basket) is owned by the sim.
    The HUD pass (mission metadata) is owned by the agent.
    """
    if canonical_rgb is None:
        return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    cinematic = compose_visuals(canonical_rgb, tool_state, jitter_rng=rng)
    draw_hud(cinematic, mission_status, frame_idx, start_ts)
    return cinematic
