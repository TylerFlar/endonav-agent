"""Mission state machine: navigation + stone treatment."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MissionState(Enum):
    ADVANCING = "advancing"
    AT_JUNCTION = "at_junction"
    ENTERING_BRANCH = "entering_branch"
    AT_DEAD_END = "at_dead_end"
    STONE_APPROACH = "stone_approach"
    LASER_FIRING = "laser_firing"
    FRAGMENT_CLEANUP = "fragment_cleanup"
    CALYX_VERIFICATION = "calyx_verification"
    BACKTRACKING = "backtracking"
    MISSION_COMPLETE = "mission_complete"


class CalyxStatus(Enum):
    UNVISITED = "unvisited"
    VISITED_STONES_FOUND = "stones_found"
    TREATMENT_IN_PROGRESS = "treating"
    VERIFIED_CLEAR = "verified_clear"


@dataclass
class MissionStateMachine:
    expected_calyces: int
    expected_stones: int
    current_state: MissionState = MissionState.ADVANCING
    calyces: dict[str, CalyxStatus] = field(default_factory=dict)
    stones_destroyed: int = 0
    fragments_pending: int = 0
    vlm_calls: int = 0
    steps_since_progress: int = 0
    history: list[dict] = field(default_factory=list)
    current_calyx_id: Optional[str] = None
    scan_steps: int = 0  # how long we've been scanning current calyx

    @property
    def calyces_verified(self) -> int:
        return sum(1 for s in self.calyces.values() if s == CalyxStatus.VERIFIED_CLEAR)

    def enter_calyx(self, calyx_id: str) -> None:
        self.current_calyx_id = calyx_id
        if calyx_id not in self.calyces:
            self.calyces[calyx_id] = CalyxStatus.UNVISITED
        self.current_state = MissionState.AT_DEAD_END
        self.scan_steps = 0

    def found_stone(self) -> None:
        if self.current_calyx_id is not None:
            self.calyces[self.current_calyx_id] = CalyxStatus.VISITED_STONES_FOUND
        self.current_state = MissionState.STONE_APPROACH

    def mark_calyx_clear(self) -> None:
        if self.current_calyx_id is not None:
            self.calyces[self.current_calyx_id] = CalyxStatus.VERIFIED_CLEAR
        self.current_state = MissionState.BACKTRACKING

    def record_treatment(self, step: int, tool: str, result: Any) -> None:
        self.history.append(
            {
                "step": step,
                "tool": tool,
                "success": getattr(result, "success", False),
                "stone_id": getattr(result, "stone_id", None),
                "size_mm": getattr(result, "stone_size_mm", 0.0),
                "fragments": len(getattr(result, "fragments_produced", []) or []),
                "failure_reason": getattr(result, "failure_reason", None),
            }
        )
        if not getattr(result, "success", False):
            return
        if tool == "laser":
            n_frags = len(result.fragments_produced or [])
            self.fragments_pending += n_frags
            if self.current_calyx_id is not None:
                self.calyces[self.current_calyx_id] = CalyxStatus.TREATMENT_IN_PROGRESS
            self.current_state = MissionState.FRAGMENT_CLEANUP
        elif tool == "basket":
            self.stones_destroyed += 1
            self.fragments_pending = max(0, self.fragments_pending - 1)
            if self.fragments_pending == 0:
                self.current_state = MissionState.CALYX_VERIFICATION
                self.scan_steps = 0

    def is_complete(self) -> bool:
        return (
            self.calyces_verified >= self.expected_calyces
            and self.current_state != MissionState.MISSION_COMPLETE
            or self.current_state == MissionState.MISSION_COMPLETE
        )

    def build_context(self, analysis, place_match, trigger: str) -> str:
        cs = self
        lines = [
            f"Mission progress: {cs.calyces_verified}/{cs.expected_calyces} calyces verified clear.",
            f"Stones destroyed so far: {cs.stones_destroyed}. Fragments pending cleanup: {cs.fragments_pending}.",
            f"Current state: {cs.current_state.name}. Trigger: {trigger}.",
            f"Classical CV: dark_blobs={analysis.n_dark_blobs}, dark_area={analysis.dark_area_frac:.3f}, hint={analysis.classification_hint}, stone_hint={analysis.stone_hint}.",
        ]
        if place_match is not None:
            lines.append(f"Place memory match: {place_match.place_id} (score {place_match.score:.2f}).")

        state_prompts = {
            "stuck": "You haven't made progress in many steps. Look at the frame and propose a recovery move (small retraction, roll to find an opening).",
            "stone_candidate": "Classical CV detected a possible stone (yellow/brown patch). Confirm: is there really a stone visible? If yes, set stone_visible=true and aim toward it.",
            "junction_candidate": "Classical CV thinks this is a branching point. Confirm: is it really a junction? How many branches do you see?",
            "dead_end_candidate": "Classical CV thinks we've reached a dead-end calyx. Confirm and plan a rotational scan.",
            "branch_selection": "We are at a junction. Pick a branch (describe which one and why) and steer toward its dark opening.",
            "calyx_scan": "We just entered a dead-end calyx. Scan it: are there any stones visible? Look carefully for yellow/brown lumps against pink tissue.",
            "stone_approach": "A stone is visible. Center it in the frame and approach to within ~4 mm. The capture range is only 4 mm.",
            "fire_check": "Stone should be centered and very close. Should we fire the laser now? Set fire_laser=true ONLY if you're confident.",
            "fragment_scan": "We just lasered a stone. Find each fragment. Small fragments (<3.5mm diameter) → set use_basket=true when one is centered. Larger fragments → fire_laser again.",
            "verify_clear": "All known stones treated. Final scan: confirm the calyx is now stone-free.",
            "backtrack_arrival": "While backtracking, we may have arrived at a previous junction. Confirm and select an untried branch.",
        }
        if trigger in state_prompts:
            lines.append(state_prompts[trigger])
        return "\n".join(lines)
