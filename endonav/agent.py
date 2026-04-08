"""AutonomousSurgeon main loop. Sees only what a real ureteroscope would see."""
from __future__ import annotations

from typing import Optional

from endonav_sim import KidneySimulator, ToolMode

from .autopilot import Autopilot
from .memory import PlaceMemory
from .perception import FrameAnalysis, FrameAnalyzer
from .state import CalyxStatus, MissionState, MissionStateMachine
from .topo_graph import TopoGraph
from .viz import MissionVisualizer
from .vlm import query_vlm


class AutonomousSurgeon:
    def __init__(self, sim: KidneySimulator, use_vlm: bool = True) -> None:
        self.sim = sim
        self.use_vlm = use_vlm
        self.analyzer = FrameAnalyzer()
        self.autopilot = Autopilot(image_width=sim.renderer.width, image_height=sim.renderer.height)
        self.state = MissionStateMachine(
            expected_calyces=sim.anatomy_meta.n_dead_ends,
            expected_stones=len([s for s in sim.stones if not s.is_fragment]),
        )
        self.memory = PlaceMemory()
        self.topo = TopoGraph()
        self.viz = MissionVisualizer()
        self._collisions = 0
        self._buckles = 0
        self._last_gt_node: Optional[str] = None

    # --------------- triggers ---------------
    def _check_trigger(self, analysis: FrameAnalysis, place_match) -> Optional[str]:
        sm = self.state
        if sm.steps_since_progress > 25:
            sm.steps_since_progress = 0
            return "stuck"
        if analysis.stone_hint and sm.current_state in (
            MissionState.AT_DEAD_END,
            MissionState.CALYX_VERIFICATION,
            MissionState.FRAGMENT_CLEANUP,
        ):
            return "stone_candidate"
        s = sm.current_state
        if s == MissionState.ADVANCING:
            if analysis.classification_hint == "possible_junction" and analysis.confirmed:
                return "junction_candidate"
            if analysis.classification_hint == "possible_dead_end" and analysis.confirmed:
                return "dead_end_candidate"
        elif s == MissionState.AT_JUNCTION:
            return "branch_selection"
        elif s == MissionState.AT_DEAD_END:
            return "calyx_scan"
        elif s == MissionState.STONE_APPROACH:
            return "stone_approach"
        elif s == MissionState.LASER_FIRING:
            return "fire_check"
        elif s == MissionState.FRAGMENT_CLEANUP:
            return "fragment_scan"
        elif s == MissionState.CALYX_VERIFICATION:
            return "verify_clear"
        elif s == MissionState.BACKTRACKING:
            if place_match is not None and place_match.score > 0.80:
                return "backtrack_arrival"
            if analysis.classification_hint == "possible_junction" and analysis.confirmed:
                return "backtrack_arrival"
        return None

    # --------------- decision processing ---------------
    def _process_decision(self, decision: dict, frame_rgb, step: int) -> None:
        sm = self.state
        cls = decision.get("classification", "uncertain")

        if cls == "junction":
            n_branches = max(2, int(decision.get("branch_count", 2) or 2))
            self.topo.add_node("junction", parent=self.topo.current, branches_total=n_branches)
            self.memory.add(frame_rgb, place_id=self.topo.current)
            sm.current_state = MissionState.AT_JUNCTION
        elif cls == "dead_end":
            cid = self.topo.new_id("calyx")
            self.topo.add_node("calyx", parent=self.topo.current)
            self.memory.add(frame_rgb, place_id=self.topo.current)
            sm.enter_calyx(self.topo.current or cid)
        elif cls == "stone_visible" or decision.get("stone_visible"):
            if sm.current_state in (MissionState.AT_DEAD_END, MissionState.CALYX_VERIFICATION):
                sm.found_stone()
        elif cls == "fragment_visible":
            sm.current_state = MissionState.FRAGMENT_CLEANUP
        elif cls == "clear_calyx":
            if sm.current_state in (MissionState.CALYX_VERIFICATION, MissionState.AT_DEAD_END):
                sm.mark_calyx_clear()

        # Treatment intent transitions
        if sm.current_state == MissionState.STONE_APPROACH and decision.get("fire_laser"):
            sm.current_state = MissionState.LASER_FIRING

    # --------------- main loop ---------------
    def run(self, max_steps: int = 5000) -> dict:
        sim = self.sim
        sim.reset()
        self.autopilot.reset()

        last_step = 0
        for step in range(max_steps):
            last_step = step
            out = sim.render(with_depth=False, with_stones_visible=True)
            frame = out["rgb"]
            gt_node = out.get("current_tree_node", "?")
            if gt_node != self._last_gt_node:
                self.state.steps_since_progress = 0
                self._last_gt_node = gt_node
            else:
                self.state.steps_since_progress += 1

            analysis = self.analyzer.analyze(frame)
            place_match = self.memory.match(frame)

            trigger = self._check_trigger(analysis, place_match)
            vlm_decision = None
            if self.use_vlm and trigger is not None:
                ctx = self.state.build_context(analysis, place_match, trigger)
                vlm_decision = query_vlm(frame, ctx, step)
                self.state.vlm_calls += 1
                self._process_decision(vlm_decision, frame, step)

            # Pick action
            if vlm_decision is not None:
                action = vlm_decision.get("action") or {}
                should_laser = bool(vlm_decision.get("fire_laser", False))
                should_basket = bool(vlm_decision.get("use_basket", False))
            else:
                a = self.autopilot.compute_command(analysis)
                action = a or {"advance_mm": 0.0, "roll_deg": 0.0, "deflection_deg": 0.0}
                should_laser = False
                should_basket = False

            fb = sim.command(
                advance_mm=float(action.get("advance_mm", 0.0) or 0.0),
                roll_deg=float(action.get("roll_deg", 0.0) or 0.0),
                deflection_deg=float(action.get("deflection_deg", 0.0) or 0.0),
            )

            if should_laser:
                res = sim.attempt_capture(ToolMode.LASER)
                self.state.record_treatment(step, "laser", res)
            elif should_basket:
                res = sim.attempt_capture(ToolMode.BASKET)
                self.state.record_treatment(step, "basket", res)

            if fb.buckled:
                self._buckles += 1
            if fb.collided:
                self._collisions += 1
            self.autopilot.notify_feedback(fb)

            self.viz.log_step(step, frame, analysis, vlm_decision, action, self.state, fb, gt_node)

            if self.state.calyces_verified >= self.state.expected_calyces:
                self.state.current_state = MissionState.MISSION_COMPLETE
                print(f"MISSION COMPLETE at step {step}")
                break

            if step % 100 == 0:
                cs = self.state
                print(
                    f"[{step}] state={cs.current_state.name} "
                    f"calyces={cs.calyces_verified}/{cs.expected_calyces} "
                    f"stones={cs.stones_destroyed}/{cs.expected_stones} "
                    f"vlm={cs.vlm_calls}"
                )

        self.viz.save_video("mission.mp4")
        self.viz.save_log("mission_log.json")
        metrics = self.viz.save_metrics(
            "mission_metrics.json",
            self.state,
            sim,
            total_steps=last_step + 1,
            collisions=self._collisions,
            buckles=self._buckles,
        )
        return metrics
