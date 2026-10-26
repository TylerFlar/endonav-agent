"""Place memory: HSV histogram signatures keyed by topo-graph node id.

Used by autopilot_backtrack to recognize when the scope has returned to a
previously-visited junction. HSV histograms are rotation-invariant by design,
so the same chamber seen at different roll angles still matches.
"""

from __future__ import annotations

import cv2
import numpy as np

from .perception import FrameAnalyzer


class PlaceMemory:
    """Stores HSV-histogram fingerprints by node id."""

    def __init__(
        self,
        match_threshold: float = 0.72,
        analyzer: FrameAnalyzer | None = None,
    ) -> None:
        self.match_threshold = float(match_threshold)
        self._analyzer = analyzer or FrameAnalyzer()
        self.signatures: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    def add(self, node_id: str, frames: list[np.ndarray]) -> None:
        """Compute and store an averaged signature from one or more frames.

        Multi-frame averaging is recommended at junctions: 3-5 frames at
        slightly different roll angles produce a signature that's robust to
        small tip-pose flicker. Single-frame is fine for static views.
        """
        if not frames:
            raise ValueError("PlaceMemory.add requires at least one frame")
        sigs = [self._analyzer.compute_place_signature(f) for f in frames]
        avg = np.mean(np.stack(sigs, axis=0), axis=0).astype(np.float32)
        norm = float(np.linalg.norm(avg))
        if norm > 1e-9:
            avg /= norm
        self.signatures[node_id] = avg

    # ------------------------------------------------------------------
    def query(self, frame: np.ndarray) -> dict:
        """Compare the current frame against every stored signature.

        Returns a dict with the best match id (or None if below threshold),
        the raw correlation score, and a full id->score map.
        """
        if not self.signatures:
            return {
                "best_match_id": None,
                "score": 0.0,
                "all_matches": {},
                "matched": False,
            }
        q = self._analyzer.compute_place_signature(frame)
        scores: dict[str, float] = {}
        for node_id, sig in self.signatures.items():
            # cv2.compareHist expects 1-D float32 arrays for non-image hists.
            scores[node_id] = float(cv2.compareHist(q, sig, cv2.HISTCMP_CORREL))
        best_id = max(scores, key=scores.get)
        best_score = scores[best_id]
        matched = best_score >= self.match_threshold
        return {
            "best_match_id": best_id if matched else None,
            "score": best_score,
            "all_matches": scores,
            "matched": matched,
        }

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.signatures)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self.signatures
