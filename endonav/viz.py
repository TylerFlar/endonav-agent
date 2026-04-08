"""4-panel mission visualizer + metrics recorder."""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .state import CalyxStatus, MissionStateMachine


class MissionVisualizer:
    def __init__(self, panel_w: int = 512, panel_h: int = 384) -> None:
        self.panel_w = panel_w
        self.panel_h = panel_h
        self._frames: list[np.ndarray] = []
        self._log: list[dict] = []

    def _resize(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (self.panel_w, self.panel_h))

    def _to_bgr(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def log_step(
        self,
        step: int,
        frame_rgb: np.ndarray,
        analysis: Any,
        vlm_decision: Any,
        action: dict,
        sm: MissionStateMachine,
        feedback: Any,
        gt_node: str,
    ) -> None:
        # Build 4-panel composite
        cam = self._resize(self._to_bgr(frame_rgb))
        # Overlay centroid + state
        if analysis.dark_centroid is not None:
            x, y = analysis.dark_centroid
            sx = int(x * self.panel_w / frame_rgb.shape[1])
            sy = int(y * self.panel_h / frame_rgb.shape[0])
            cv2.circle(cam, (sx, sy), 8, (0, 255, 0), 2)
        cv2.putText(cam, f"step {step}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        dark = self._resize(self._to_bgr(analysis.dark_mask))

        # Topo placeholder: black panel with calyx progress
        topo = np.zeros((self.panel_h, self.panel_w, 3), dtype=np.uint8)
        for i, (cid, status) in enumerate(sorted(sm.calyces.items())):
            color = {
                CalyxStatus.UNVISITED: (128, 128, 128),
                CalyxStatus.VISITED_STONES_FOUND: (0, 165, 255),
                CalyxStatus.TREATMENT_IN_PROGRESS: (0, 255, 255),
                CalyxStatus.VERIFIED_CLEAR: (0, 200, 0),
            }[status]
            y = 30 + 22 * i
            if y > self.panel_h - 10:
                break
            cv2.circle(topo, (20, y), 8, color, -1)
            cv2.putText(topo, cid, (38, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Status text panel
        status = np.zeros((self.panel_h, self.panel_w, 3), dtype=np.uint8)
        lines = [
            f"State: {sm.current_state.name}",
            f"Calyces: {sm.calyces_verified}/{sm.expected_calyces}",
            f"Stones destroyed: {sm.stones_destroyed}/{sm.expected_stones}",
            f"Fragments pending: {sm.fragments_pending}",
            f"VLM calls: {sm.vlm_calls}",
            f"GT node: {gt_node}",
            f"Action: a={action.get('advance_mm', 0):.2f} r={action.get('roll_deg', 0):.1f} d={action.get('deflection_deg', 0):.1f}",
            f"Buckled: {getattr(feedback, 'buckled', False)}  Collided: {getattr(feedback, 'collided', False)}",
            f"Wall: {getattr(feedback, 'wall_clearance_mm', 0):.1f} mm",
        ]
        if vlm_decision is not None:
            lines.append(f"VLM: {vlm_decision.get('classification', '?')}")
        for i, line in enumerate(lines):
            cv2.putText(status, line, (8, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        top = np.hstack([cam, dark])
        bot = np.hstack([topo, status])
        composite = np.vstack([top, bot])
        self._frames.append(composite)

        # Per-step log entry
        self._log.append(
            {
                "step": step,
                "state": sm.current_state.name,
                "action": action,
                "vlm_called": vlm_decision is not None,
                "vlm_classification": vlm_decision.get("classification") if vlm_decision else None,
                "buckled": bool(getattr(feedback, "buckled", False)),
                "collided": bool(getattr(feedback, "collided", False)),
                "actual_advance": float(getattr(feedback, "actual_advance_mm", 0.0)),
                "wall_clearance_mm": float(getattr(feedback, "wall_clearance_mm", 0.0)),
                "gt_tree_node": gt_node,
            }
        )

    def save_video(self, path: str, fps: int = 10) -> None:
        if not self._frames:
            return
        h, w = self._frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for f in self._frames:
            out.write(f)
        out.release()

    def save_log(self, path: str) -> None:
        Path(path).write_text(json.dumps(self._log, indent=2))

    def save_metrics(
        self,
        path: str,
        sm: MissionStateMachine,
        sim,
        total_steps: int,
        collisions: int,
        buckles: int,
    ) -> dict:
        original_stones = [s for s in sim.stones if not s.is_fragment]
        removed_originals = sum(1 for s in original_stones if s.removed)
        fragments_remaining = sum(1 for s in sim.stones if s.is_fragment and not s.removed)
        per_calyx = [
            {"calyx": cid, "status": status.value}
            for cid, status in sorted(sm.calyces.items())
        ]
        metrics = {
            "total_steps": total_steps,
            "vlm_calls": sm.vlm_calls,
            "calyces_visited": len(sm.calyces),
            "calyces_total": sm.expected_calyces,
            "calyces_verified_clear": sm.calyces_verified,
            "stone_free_rate": sm.calyces_verified / max(1, sm.expected_calyces),
            "stones_total": len(original_stones),
            "stones_destroyed": removed_originals,
            "fragments_remaining": fragments_remaining,
            "collisions": collisions,
            "buckles": buckles,
            "per_calyx_detail": per_calyx,
            "treatment_history": sm.history,
            "completed": sm.current_state.name == "MISSION_COMPLETE",
        }
        Path(path).write_text(json.dumps(metrics, indent=2))
        return metrics
