"""Thread-safe shared state between MissionTools (writer) and the realtime
display thread (reader).

The agent's command-issuing path produces canonical RGB frames + HUD status
+ a snapshot of the sim's tool visual state. The display thread consumes
them at a fixed cadence (e.g. 30 Hz) and composes the live view + recorded
video. The two sides communicate ONLY through this buffer, so the display
thread never touches the sim or the shim Lock.

The buffer carries no domain logic of its own — it's a typed slot. Tool
event tracking lives in ``endonav_sim.visuals.ToolVisualState`` (owned by
the sim); HUD metadata lives in the ``mission_status`` dict (owned by the
agent). The display thread just reads both and hands them to the
compositor.
"""

from __future__ import annotations

import threading
import time

import numpy as np
from endonav_sim.visuals import ToolVisualState


class LiveBuffer:
    """Single-writer (per-call), multi-reader thread-safe frame store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rgb: np.ndarray | None = None
        self._version: int = 0
        self._mission_status: dict = {}
        self._tool_state: ToolVisualState | None = None
        self._last_command_ts: float = 0.0
        self._closed: bool = False

    # ------------------------------------------------------------------
    def set_frame(
        self,
        rgb: np.ndarray,
        mission_status: dict | None = None,
        tool_state: ToolVisualState | None = None,
    ) -> None:
        """Publish a new canonical RGB frame and bump the version counter."""
        if rgb is None:
            return
        with self._lock:
            if self._closed:
                return
            # Store a reference to the caller's frame; the reader copies on
            # get(). Callers must not mutate the frame after handing it over.
            self._rgb = rgb
            self._version += 1
            if mission_status is not None:
                self._mission_status = dict(mission_status)
            if tool_state is not None:
                self._tool_state = tool_state
            self._last_command_ts = time.monotonic()

    # ------------------------------------------------------------------
    def update_status(self, status: dict) -> None:
        """Update the HUD status dict without changing the frame."""
        with self._lock:
            if self._closed:
                return
            self._mission_status = dict(status)

    # ------------------------------------------------------------------
    def get(self) -> tuple[np.ndarray | None, int, dict, ToolVisualState | None]:
        """Snapshot the current state for the display thread.

        Returns ``(rgb_copy, version, status_copy, tool_state_copy)``. The
        frame is copied so the display thread can safely mutate it (apply
        jitter, draw overlays) without racing the writer. ``rgb_copy`` is
        None until the first ``set_frame`` call.
        """
        with self._lock:
            rgb = None if self._rgb is None else self._rgb.copy()
            version = self._version
            status = dict(self._mission_status)
            # ToolVisualState is a frozen-ish dataclass; the writer always
            # passes us a snapshot, so a shallow reference is safe.
            tool_state = self._tool_state
        return rgb, version, status, tool_state

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Mark the buffer closed; further writes are silently dropped."""
        with self._lock:
            self._closed = True
