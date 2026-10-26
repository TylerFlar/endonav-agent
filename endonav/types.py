"""Shared dataclasses across endonav-agent modules.

Kept in their own file so perception, autopilot, memory and topo_graph can
import them without creating cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Blob:
    """A connected dark region in the endoscope view."""

    center_xy: tuple[float, float]
    area: float
    bbox: tuple[int, int, int, int]  # x, y, w, h
    mean_intensity: float
    circularity: float


@dataclass
class StoneHint:
    """A possible stone detection in the endoscope view."""

    center_xy: tuple[float, float]
    area: float
    composition_guess: str
    confidence: float


@dataclass
class Scene:
    """High-level classification of a single frame."""

    type: str  # "tube" | "junction_candidate" | "dead_end_candidate"
    confidence: float
    darkest_blob_idx: int | None
    blob_count: int
    reason: str = ""


@dataclass
class AutopilotResult:
    """Outcome of one autopilot subroutine call."""

    reason: str
    frames_executed: int
    last_rgb: Any  # np.ndarray
    last_scene: Scene | None
    stone_hints: list[StoneHint] = field(default_factory=list)
    collisions: int = 0
    buckles: int = 0
    path_mm: float = 0.0
    summary: str = ""
    matched_node_id: str | None = None  # only set by backtrack on place match


@dataclass
class ScanFrame:
    """One sample from a rotate_scan sweep."""

    angle_deg: float
    rgb: Any  # np.ndarray
    dark_blob_count: int
    stone_hint: StoneHint | None
    image_path: str | None = None  # filled in by MissionTools after saving
