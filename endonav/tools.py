"""MissionTools: the public interface the autonomous agent calls.

This is the ONLY module that imports endonav_sim. Every other module
(perception, memory, topo_graph, viz, autopilot) is sim-agnostic and takes
the sim instance as a parameter. This keeps the "ground-truth cheat surface"
auditable in a single file.

The agent never reads sim.pose, sim.path_node, sim.stones, or any
ground-truth field on sim.anatomy_meta. The only allowed metadata fields
(treated as preop CT) are: variant, seed, minor_calyces_total, n_dead_ends,
infundibulopelvic_angle_deg.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

from endonav_sim import (
    AnatomyParams,
    KidneySimulator,
    StoneParams,
    ToolMode,
)
from endonav_sim.visuals import compose_visuals

from . import autopilot as autopilot_mod
from .live_buffer import LiveBuffer
from .memory import PlaceMemory
from .perception import FrameAnalyzer
from .topo_graph import TopoGraph
from .types import ScanFrame

# Allowed anatomy_meta fields the agent is permitted to know (preop CT analog).
_ALLOWED_META_FIELDS = (
    "variant",
    "seed",
    "minor_calyces_total",
    "n_dead_ends",
    "infundibulopelvic_angle_deg",
)


def _np_to_jsonable(obj):
    """Recursively convert numpy types and arrays to JSON-friendly Python."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _np_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_np_to_jsonable(v) for v in obj]
    return obj


class MissionTools:
    """The public surface the LLM agent invokes via the HTTP shim."""

    def __init__(
        self,
        seed: int = 42,
        realistic: bool = True,
        output_dir: str | Path = "runs/latest",
        jpeg_quality: int = 85,
        thumb_size: tuple[int, int] = (256, 192),
    ) -> None:
        self.seed = int(seed)
        self.realistic = bool(realistic)
        self.jpeg_quality = int(jpeg_quality)
        self.thumb_size = tuple(thumb_size)

        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "frames"
        self.thumbs_dir = self.output_dir / "thumbs"
        self.scans_dir = self.output_dir / "scans"
        for d in (self.frames_dir, self.thumbs_dir, self.scans_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Boot the sim.
        self.sim = KidneySimulator(
            anatomy_params=AnatomyParams(seed=self.seed),
            stone_params=StoneParams(seed=self.seed),
            seed=self.seed,
            realistic=self.realistic,
        )
        self.sim.reset()
        self._initial_stone_count = len(self.sim.stones)

        # Internal state.
        self.frame_counter: int = 0
        self.command_counter: int = 0
        self.last_feedback_dict: dict | None = None
        self.stones_destroyed: int = 0
        self.fragments_remaining: int = 0
        self._recent_frames: deque = deque(maxlen=5)
        # Independent RNG for cinematic jitter so it doesn't perturb the
        # sim's own dynamics RNG (which is responsible for command noise,
        # buckling, etc.).
        self._jitter_rng = np.random.default_rng(self.seed + 1)

        # Subsystems.
        self.perception = FrameAnalyzer()
        self.memory = PlaceMemory(analyzer=self.perception)
        self.topo = TopoGraph()
        self.topo.seed_root("ureter")
        # Shared state with the realtime display thread (which is owned by
        # the server, not by MissionTools — the server starts/stops it around
        # this object's lifetime).
        self.live_buffer = LiveBuffer()

        # Snapshot the start time and the briefing for diagnostics.
        self._start_ts = time.time()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _feedback_to_dict(self, fb) -> dict:
        d = asdict(fb)
        # tip_pose_estimate is a 4x4 numpy array; flatten to nested list.
        if isinstance(d.get("tip_pose_estimate"), np.ndarray):
            d["tip_pose_estimate"] = d["tip_pose_estimate"].tolist()
        return _np_to_jsonable(d)

    def _save_frame(self, rgb: np.ndarray) -> Path:
        """Write the cinematic version of a frame to disk for the LLM to read.

        ``rgb`` is the clean canonical frame from ``sim.render()``. We layer
        cardiac/respiratory jitter and the active tool overlays on top
        before writing, so the agent looking at this JPEG sees what a real
        operator would see through the eyepiece — including their own laser
        flash, the smoke from a fragmentation, and the basket cage when
        deployed.

        The clean rgb is what gets cached in ``_recent_frames`` for place-
        signature averaging (HSV histograms are best computed from a
        consistent baseline, not jittered frames).
        """
        idx = self.frame_counter
        self.frame_counter += 1
        name = f"frame_{idx:06d}.jpg"
        full_path = self.frames_dir / name
        thumb_path = self.thumbs_dir / name
        cinematic = compose_visuals(
            rgb,
            self.sim.tool_visual_state,
            jitter_rng=self._jitter_rng,
        )
        iio.imwrite(str(full_path), cinematic, quality=self.jpeg_quality)
        thumb = cv2.resize(cinematic, self.thumb_size, interpolation=cv2.INTER_AREA)
        iio.imwrite(str(thumb_path), thumb, quality=80)
        # Cache the CLEAN frame for place-signature averaging.
        self._recent_frames.append(rgb.copy())
        return full_path

    def _publish_frame(self, rgb: np.ndarray) -> None:
        """Push a fresh canonical RGB + sim tool state to the LiveBuffer."""
        self.live_buffer.set_frame(
            rgb,
            self._compact_status(),
            tool_state=self.sim.tool_visual_state,
        )

    def _compact_status(self) -> dict:
        """The minimal status dict the HUD reads each tick. Cheap to build."""
        meta = self.sim.anatomy_meta
        return {
            "calyces_visited": sum(
                1 for n in self.topo.nodes.values() if n.type == "calyx"
            ),
            "calyces_verified_clear": sum(
                1
                for n in self.topo.nodes.values()
                if n.type == "calyx" and n.status == "verified_clear"
            ),
            "total_calyces": int(getattr(meta, "minor_calyces_total", 0)),
            "stones_destroyed": int(self.stones_destroyed),
            "fragments_remaining": int(self.fragments_remaining),
            "current_node_id": self.topo.current_node_id,
            "total_commands": int(self.command_counter),
            "last_feedback": self.last_feedback_dict,
        }

    # ------------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------------
    def get_mission_briefing(self) -> str:
        """Pretty-printed preop CT analog. Uses only allowed anatomy fields."""
        meta = self.sim.anatomy_meta
        lines = ["=== KIDNEY MISSION BRIEFING ==="]
        for f in _ALLOWED_META_FIELDS:
            if hasattr(meta, f):
                val = getattr(meta, f)
                if isinstance(val, float):
                    lines.append(f"  {f}: {val:.2f}")
                else:
                    lines.append(f"  {f}: {val}")
        lines.append(f"  total_stones (preop CT): {self._initial_stone_count}")
        lines.append("")
        lines.append("Mission: visit every calyx, destroy every stone, verify clear.")
        lines.append("Use only the camera + proprioceptive feedback. No ground truth.")
        return "\n".join(lines)

    def render_and_save(self) -> dict:
        """Render the current view, save full + thumb to disk, return metadata."""
        out = self.sim.render()
        rgb = out["rgb"]
        path = self._save_frame(rgb)
        # tip_pose_estimate must come from the last command feedback (sim.render
        # does not produce one). If we have no prior command, fall back to the
        # sim's clean pose only here, treated as the dead-reckoned start point.
        if self.last_feedback_dict is not None:
            tip_pose = self.last_feedback_dict.get("tip_pose_estimate")
        else:
            tip_pose = (
                self.sim.dynamics.pose_estimate.tolist()
                if self.sim.dynamics.pose_estimate is not None
                else None
            )
        result = {
            "image_path": str(path),
            "thumb_path": str(self.thumbs_dir / path.name),
            "nearest_wall_mm": float(out["nearest_wall_mm"]),
            "tip_pose_estimate": tip_pose,
            "frame_index": self.frame_counter - 1,
        }
        self._publish_frame(rgb)
        return result

    def get_mission_status(self) -> dict:
        """Snapshot of mission bookkeeping for the LLM to reason over."""
        meta = self.sim.anatomy_meta
        return {
            "total_calyces": int(getattr(meta, "minor_calyces_total", 0)),
            "calyces_visited": sum(
                1 for n in self.topo.nodes.values() if n.type == "calyx"
            ),
            "calyces_verified_clear": sum(
                1
                for n in self.topo.nodes.values()
                if n.type == "calyx" and n.status == "verified_clear"
            ),
            "stones_destroyed": int(self.stones_destroyed),
            "fragments_remaining": int(self.fragments_remaining),
            "topo_graph": self.topo.snapshot(),
            "topo_stats": self.topo.stats(),
            "dfs_stack": list(self.topo.dfs_stack),
            "current_node_id": self.topo.current_node_id,
            "total_steps": int(self.frame_counter),
            "total_commands": int(self.command_counter),
            "elapsed_seconds": round(time.time() - self._start_ts, 1),
        }

    def get_feedback(self) -> dict:
        """Return the most recent CommandFeedback dict (or None)."""
        return self.last_feedback_dict or {}

    def check_place_recognition(self) -> dict:
        """Compare the current view against every stored place signature."""
        out = self.sim.render()
        rgb = out["rgb"]
        return _np_to_jsonable(self.memory.query(rgb))

    # ------------------------------------------------------------------
    # NAVIGATION
    # ------------------------------------------------------------------
    def send_command(
        self,
        advance_mm: float = 0.0,
        roll_deg: float = 0.0,
        deflection_deg: float = 0.0,
    ) -> dict:
        """Issue a single motion command.

        advance_mm  : signed mm along shaft (sim caps at ±2)
        roll_deg    : INCREMENTAL axial rotation (sim caps at ±15)
        deflection_deg : ABSOLUTE bending angle, 0 = straight (sim caps at ±270;
                         agent's MISSION.md recommends staying within ±45)
        """
        fb = self.sim.command(
            advance_mm=float(advance_mm),
            roll_deg=float(roll_deg),
            deflection_deg=float(deflection_deg),
        )
        self.command_counter += 1
        self.last_feedback_dict = self._feedback_to_dict(fb)
        # Refresh the live display: the scope just moved, so the canonical
        # frame should reflect the new pose. This is a single render() — does
        # NOT save to disk (only render_and_save does that).
        try:
            out = self.sim.render()
            self._publish_frame(out["rgb"])
        except Exception:
            pass
        return self.last_feedback_dict

    def autopilot_advance(self, max_frames: int = 30, stop_on: str = "any") -> dict:
        """Delegate forward navigation to the classical lumen-following autopilot."""
        result = autopilot_mod.advance(
            self.sim,
            self.perception,
            max_frames=int(max_frames),
            stop_on=stop_on,
            on_frame=self._publish_frame,
        )
        self.command_counter += result.frames_executed
        # Save the final frame and publish it to the live buffer.
        path: Path | None = None
        if result.last_rgb is not None:
            path = self._save_frame(result.last_rgb)
            self._publish_frame(result.last_rgb)
        # The autopilot already issued real commands; we just expose the result.
        scene = result.last_scene
        return {
            "frames_executed": result.frames_executed,
            "stop_reason": result.reason,
            "final_image_path": str(path) if path else None,
            "summary": result.summary,
            "collisions": result.collisions,
            "buckles": result.buckles,
            "path_mm": float(result.path_mm),
            "scene": (
                {
                    "type": scene.type,
                    "confidence": scene.confidence,
                    "blob_count": scene.blob_count,
                    "reason": scene.reason,
                }
                if scene
                else None
            ),
            "stone_hints": [
                {
                    "center_xy": list(h.center_xy),
                    "area": h.area,
                    "composition_guess": h.composition_guess,
                    "confidence": h.confidence,
                }
                for h in result.stone_hints
            ],
        }

    def autopilot_rotate_scan(self, degrees: float = 360.0, step_deg: float = 30.0) -> dict:
        """Sweep roll, capturing frames at each step."""
        scans: list[ScanFrame] = autopilot_mod.rotate_scan(
            self.sim,
            self.perception,
            degrees=float(degrees),
            step_deg=float(step_deg),
            on_frame=self._publish_frame,
        )
        # Persist scan frames separately for later inspection.
        frame_list = []
        for s in scans:
            self.command_counter += 1
            full_path = self._save_frame(s.rgb)
            s.image_path = str(full_path)
            self._publish_frame(s.rgb)
            frame_list.append(
                {
                    "angle_deg": s.angle_deg,
                    "image_path": s.image_path,
                    "dark_blob_count": s.dark_blob_count,
                    "stone_hint": (
                        {
                            "composition_guess": s.stone_hint.composition_guess,
                            "confidence": s.stone_hint.confidence,
                            "area": s.stone_hint.area,
                        }
                        if s.stone_hint
                        else None
                    ),
                }
            )
        n_openings = sum(1 for f in frame_list if f["dark_blob_count"] >= 1)
        n_hints = sum(1 for f in frame_list if f["stone_hint"] is not None)
        summary = (
            f"scan {degrees:.0f}° in {step_deg:.0f}° steps: "
            f"{len(frame_list)} frames, {n_openings} with openings, "
            f"{n_hints} with stone hints"
        )
        return {"frames": frame_list, "summary": summary}

    def autopilot_backtrack(self, max_frames: int = 100) -> dict:
        result = autopilot_mod.backtrack(
            self.sim,
            self.perception,
            self.memory,
            max_frames=int(max_frames),
            on_frame=self._publish_frame,
        )
        self.command_counter += result.frames_executed
        path: Path | None = None
        if result.last_rgb is not None:
            path = self._save_frame(result.last_rgb)
            self._publish_frame(result.last_rgb)
        return {
            "frames_executed": result.frames_executed,
            "stop_reason": result.reason,
            "matched_junction_id": result.matched_node_id,
            "final_image_path": str(path) if path else None,
            "summary": result.summary,
            "collisions": result.collisions,
            "buckles": result.buckles,
            "path_mm": float(result.path_mm),
        }

    # ------------------------------------------------------------------
    # TREATMENT
    # ------------------------------------------------------------------
    def fire_laser(self) -> dict:
        """Attempt to laser the nearest reachable stone.

        The sim records its own tool visual event (hit / dust / miss) inside
        ``attempt_capture``; we just publish a fresh frame so the realtime
        view sees the post-capture mesh + the new tool state immediately.
        """
        cr = self.sim.attempt_capture(ToolMode.LASER)
        n_frags = len(cr.fragments_produced)
        if cr.success:
            self.stones_destroyed += 1
            self.fragments_remaining += n_frags
        try:
            out = self.sim.render()
            self._publish_frame(out["rgb"])
        except Exception:
            pass
        return {
            "hit": bool(cr.success),
            "stone_id": cr.stone_id,
            "stone_size_mm": float(cr.stone_size_mm),
            "stone_composition": cr.stone_composition,
            "fragments_produced": int(n_frags),
            "failure_reason": cr.failure_reason,
        }

    def use_basket(self) -> dict:
        """Attempt to basket-capture the nearest reachable fragment."""
        cr = self.sim.attempt_capture(ToolMode.BASKET)
        if cr.success:
            self.fragments_remaining = max(0, self.fragments_remaining - 1)
        try:
            out = self.sim.render()
            self._publish_frame(out["rgb"])
        except Exception:
            pass
        return {
            "success": bool(cr.success),
            "stone_id": cr.stone_id,
            "stone_size_mm": float(cr.stone_size_mm),
            "stone_composition": cr.stone_composition,
            "failure_reason": cr.failure_reason,
        }

    # ------------------------------------------------------------------
    # MAP MANAGEMENT
    # ------------------------------------------------------------------
    def register_junction(self, branch_count: int) -> str:
        return self.topo.register_junction(int(branch_count))

    def register_dead_end(self) -> str:
        return self.topo.register_dead_end()

    def mark_branch_explored(self) -> dict:
        return self.topo.mark_branch_explored()

    def mark_calyx_status(self, calyx_id: str, status: str) -> dict:
        self.topo.set_status(calyx_id, status)
        return {"node_id": calyx_id, "status": status}

    def store_place_signature(self, node_id: str) -> dict:
        """Store an averaged signature from the last few rendered frames.

        If fewer than 1 frame has been rendered, render one now.
        """
        if not self._recent_frames:
            out = self.sim.render()
            self._save_frame(out["rgb"])
        frames = list(self._recent_frames)
        self.memory.add(node_id, frames)
        return {"node_id": node_id, "frames_used": len(frames)}

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------
    def finalize(self) -> dict:
        """End-of-mission summary + summary.json dump.

        The mp4 writer is owned by the RealtimeRenderer (which the server
        starts/stops); finalizing it is the server's job, not ours.
        """
        meta = self.sim.anatomy_meta
        summary = {
            "seed": self.seed,
            "elapsed_seconds": round(time.time() - self._start_ts, 1),
            "total_steps": self.frame_counter,
            "total_commands": self.command_counter,
            "stones_destroyed": self.stones_destroyed,
            "fragments_remaining": self.fragments_remaining,
            "initial_stone_count": self._initial_stone_count,
            "topo_stats": self.topo.stats(),
            "topo_graph": self.topo.snapshot(),
            "anatomy_briefing": {
                f: getattr(meta, f, None) for f in _ALLOWED_META_FIELDS
            },
        }
        summary = _np_to_jsonable(summary)
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        # Mark the live buffer closed so the realtime thread stops accepting
        # new frames once this object is being torn down.
        self.live_buffer.close()
        return summary
