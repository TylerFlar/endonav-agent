"""Classical autopilot routines for the endonav agent.

These functions are pure subroutines that the LLM-driven agent calls when it
just needs to "follow the tube for a while," "look around this spot," or
"reverse until I see something familiar." They take the sim object directly
because they need to issue many commands per call without round-tripping
through MissionTools, but they NEVER read ground-truth fields from the sim
(no sim.pose, sim.path_node, sim.stones, etc.). The only sim attribute they
read is sim.current_deflection — which is just the commanded post-dynamics
state the agent already has via CommandFeedback, not ground truth.

All three routines stop early on "interesting" events and return control to
the LLM with a summary plus the last frame.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Callable

import numpy as np

from .constants import (
    MAX_DEFL_ABS_DEG,
    MAX_DEFL_STEP_DEG,
    MAX_ROLL_DEG,
)
from .memory import PlaceMemory
from .perception import FrameAnalyzer
from .types import AutopilotResult, ScanFrame

# Optional callback signature: on_frame(rgb) -> None. Used by MissionTools to
# push intermediate frames into the LiveBuffer so the realtime view shows
# smooth lumen-following instead of jumping to the last frame of each batch.
FrameCallback = Callable[[np.ndarray], None]


def _current_defl_deg(sim) -> float:
    """Read the sim's current absolute deflection in degrees.

    sim.current_deflection is in radians (post-dynamics commanded state). This
    is NOT ground truth — it's the same value the agent already has from the
    most recent CommandFeedback.actual_deflection_deg, just accessed directly
    so we can chain commands without buffering it.
    """
    return float(math.degrees(sim.current_deflection))


def _adaptive_advance(blob_area: float | None, nearest_wall_mm: float) -> float:
    """Compute a safe per-step advance length in mm.

    Larger lumen up ahead = faster advance. Tight clearance = slower.
    """
    if blob_area is None:
        adv = 0.6
    else:
        adv = float(np.clip(0.5 + 1.5 * min(blob_area / 20000.0, 1.0), 0.5, 2.0))
    if nearest_wall_mm is not None and nearest_wall_mm < 2.0:
        adv = min(adv, 0.6)
    if nearest_wall_mm is not None and nearest_wall_mm < 1.0:
        adv = 0.3
    return adv


# ----------------------------------------------------------------------
# advance
# ----------------------------------------------------------------------
def advance(
    sim,
    perception: FrameAnalyzer,
    max_frames: int = 30,
    stop_on: str = "any",
    on_frame: FrameCallback | None = None,
) -> AutopilotResult:
    """Follow the lumen forward until something interesting happens.

    stop_on: "any" | "junction" | "dead_end" | "stone"
    on_frame: optional callback invoked with each rendered RGB frame so the
        caller can push it to a live display buffer.
    """
    valid_stop = {"any", "junction", "dead_end", "stone"}
    if stop_on not in valid_stop:
        raise ValueError(f"stop_on must be in {valid_stop}, got {stop_on!r}")

    recent_adv: deque[float] = deque(maxlen=3)
    buckle_hist: deque[int] = deque(maxlen=3)
    collisions = 0
    buckles = 0
    path_mm = 0.0
    last_rgb = None
    last_scene = None
    last_hints: list = []
    frames_executed = 0
    reason = "max_frames"

    for _ in range(max_frames):
        out = sim.render()
        rgb = out["rgb"]
        nearest_wall_mm = float(out.get("nearest_wall_mm", 0.0))
        last_rgb = rgb
        if on_frame is not None:
            try:
                on_frame(rgb)
            except Exception:
                pass

        blobs = perception.detect_dark_blobs(rgb)
        scene = perception.classify_scene(rgb, blobs)
        hints = perception.detect_stone_hints(rgb)
        last_scene = scene
        last_hints = hints

        # ---- stop checks (priority order) ----
        if stop_on in ("any", "stone") and hints and hints[0].confidence >= 0.6:
            reason = "stone_hint"
            break
        if (
            stop_on in ("any", "junction")
            and scene.type == "junction_candidate"
        ):
            reason = "junction_candidate"
            break
        if (
            stop_on in ("any", "dead_end")
            and scene.type == "dead_end_candidate"
        ):
            reason = "dead_end_candidate"
            break
        if len(recent_adv) == 3 and all(a < 0.3 for a in recent_adv):
            reason = "stuck"
            break

        # ---- pick a target ----
        target_xy: tuple[float, float] | None = None
        target_area: float | None = None
        if scene.darkest_blob_idx is not None and scene.darkest_blob_idx < len(blobs):
            tb = blobs[scene.darkest_blob_idx]
            target_xy = tb.center_xy
            target_area = tb.area

        # ---- compute steering ----
        if target_xy is not None:
            steer = perception.compute_steering(target_xy)
            roll_cmd = steer["roll_delta_deg"]
            defl_delta = steer["defl_delta_deg"]
        else:
            roll_cmd = 0.0
            defl_delta = 0.0

        # Convert deflection delta to absolute by adding to current.
        cur_defl = _current_defl_deg(sim)
        new_defl = float(
            np.clip(cur_defl + defl_delta, -MAX_DEFL_ABS_DEG, MAX_DEFL_ABS_DEG)
        )

        adv_mm = _adaptive_advance(target_area, nearest_wall_mm)

        cmd = sim.command(
            advance_mm=float(adv_mm),
            roll_deg=float(np.clip(roll_cmd, -MAX_ROLL_DEG, MAX_ROLL_DEG)),
            deflection_deg=float(new_defl),
        )
        frames_executed += 1
        recent_adv.append(float(cmd.actual_advance_mm))
        path_mm += float(cmd.actual_advance_mm)
        buckle_hist.append(1 if cmd.buckled else 0)
        if cmd.collided:
            collisions += 1
        if cmd.buckled:
            buckles += 1

        # ---- post-command stop checks ----
        if cmd.collided:
            reason = "collision"
            break
        if sum(buckle_hist) >= 2:
            reason = "buckled"
            break

    # Final render to make sure last_rgb reflects the post-stop state.
    if frames_executed > 0 and reason in ("collision", "buckled"):
        out = sim.render()
        last_rgb = out["rgb"]
        if on_frame is not None:
            try:
                on_frame(last_rgb)
            except Exception:
                pass

    summary = (
        f"advance: {reason} after {frames_executed} frames, "
        f"path={path_mm:+.1f}mm, blobs={last_scene.blob_count if last_scene else 0}, "
        f"collisions={collisions}, buckles={buckles}"
    )

    return AutopilotResult(
        reason=reason,
        frames_executed=frames_executed,
        last_rgb=last_rgb,
        last_scene=last_scene,
        stone_hints=last_hints,
        collisions=collisions,
        buckles=buckles,
        path_mm=path_mm,
        summary=summary,
    )


# ----------------------------------------------------------------------
# rotate_scan
# ----------------------------------------------------------------------
def rotate_scan(
    sim,
    perception: FrameAnalyzer,
    degrees: float = 360.0,
    step_deg: float = 30.0,
    on_frame: FrameCallback | None = None,
) -> list[ScanFrame]:
    """Sweep roll through `degrees`, capturing frames every `step_deg`.

    Holds the current deflection (does NOT zero it). Each scan step may
    require multiple micro-commands because the per-step roll cap is 15°.

    on_frame: optional callback called once per scan position with the
        captured RGB so the caller can push it to a live display buffer.
    """
    if degrees <= 0 or step_deg <= 0:
        raise ValueError("degrees and step_deg must be positive")
    n_steps = int(round(degrees / step_deg))
    held_defl = _current_defl_deg(sim)
    scans: list[ScanFrame] = []

    for i in range(n_steps):
        # Apply step_deg of roll using as many micro-commands as needed.
        remaining = float(step_deg)
        sign = 1.0 if remaining >= 0 else -1.0
        remaining = abs(remaining)
        while remaining > 1e-3:
            chunk = min(remaining, MAX_ROLL_DEG) * sign
            sim.command(
                advance_mm=0.0,
                roll_deg=float(chunk),
                deflection_deg=float(held_defl),
            )
            remaining -= MAX_ROLL_DEG

        out = sim.render()
        rgb = out["rgb"]
        if on_frame is not None:
            try:
                on_frame(rgb)
            except Exception:
                pass
        blobs = perception.detect_dark_blobs(rgb)
        hints = perception.detect_stone_hints(rgb)
        scans.append(
            ScanFrame(
                angle_deg=float((i + 1) * step_deg),
                rgb=rgb,
                dark_blob_count=len(blobs),
                stone_hint=hints[0] if hints else None,
            )
        )

    return scans


# ----------------------------------------------------------------------
# backtrack
# ----------------------------------------------------------------------
def backtrack(
    sim,
    perception: FrameAnalyzer,
    place_memory: PlaceMemory,
    max_frames: int = 100,
    place_match_threshold: float = 0.80,
    on_frame: FrameCallback | None = None,
) -> AutopilotResult:
    """Reverse along the lumen until a junction reappears or place memory matches.

    Steering is intentionally disabled while reversing — the tube geometry
    constrains the path; any attempt to steer scrapes the wall.

    on_frame: optional callback called with each rendered RGB during the
        reverse so the caller can push it to a live display buffer.
    """
    last_rgb = None
    last_scene = None
    matched_id: str | None = None
    frames_executed = 0
    collisions = 0
    buckles = 0
    path_mm = 0.0
    stuck_count = 0
    reason = "max_frames"

    for _ in range(max_frames):
        out = sim.render()
        rgb = out["rgb"]
        last_rgb = rgb
        if on_frame is not None:
            try:
                on_frame(rgb)
            except Exception:
                pass

        blobs = perception.detect_dark_blobs(rgb)
        scene = perception.classify_scene(rgb, blobs)
        last_scene = scene

        # Place recognition: strict bar (above the default query threshold).
        q = place_memory.query(rgb)
        if q["all_matches"]:
            best = max(q["all_matches"].values())
            if best >= place_match_threshold:
                matched_id = max(q["all_matches"], key=q["all_matches"].get)
                reason = "place_match"
                break

        if scene.type == "junction_candidate":
            reason = "junction_detected"
            break

        cur_defl = _current_defl_deg(sim)
        cmd = sim.command(
            advance_mm=-0.8,
            roll_deg=0.0,
            deflection_deg=float(cur_defl),
        )
        frames_executed += 1
        path_mm += float(cmd.actual_advance_mm)
        if cmd.collided:
            collisions += 1
            reason = "collision"
            break
        if cmd.buckled:
            buckles += 1
        if abs(cmd.actual_advance_mm) < 0.2:
            stuck_count += 1
        else:
            stuck_count = 0
        if stuck_count >= 3:
            reason = "stuck"
            break

    summary = (
        f"backtrack: {reason} after {frames_executed} frames, "
        f"reversed={-path_mm:.1f}mm"
    )
    if matched_id:
        summary += f", matched={matched_id}"

    return AutopilotResult(
        reason=reason,
        frames_executed=frames_executed,
        last_rgb=last_rgb,
        last_scene=last_scene,
        stone_hints=[],
        collisions=collisions,
        buckles=buckles,
        path_mm=path_mm,
        summary=summary,
        matched_node_id=matched_id,
    )
