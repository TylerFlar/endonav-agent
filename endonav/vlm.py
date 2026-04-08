"""VLM bridge: spawn `claude` CLI as subprocess for visual reasoning.

Cost-control: only call this for triggered events. Saves frame to disk and
references the absolute path in the prompt so Claude Code reads it via Read.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

FRAME_DIR = Path("./vlm_frames")
FRAME_DIR.mkdir(exist_ok=True)


SYSTEM_PROMPT = """You are an autonomous ureteroscope controller inside a kidney. Your mission: DFS-explore every calyx and destroy all kidney stones.

## CAMERA
Coaxial endoscope - light at the camera:
- BRIGHT/WHITE = tissue very close (wall)
- DARK = openings/passages leading deeper
- PINK/RED = tissue at moderate distance
- YELLOW/BROWN/TAN irregular lumps = KIDNEY STONES (distinctly different from pink tissue)
- Flat white specks on tissue = Randall's plaque (NOT stones, ignore)
- Small dark dots in clusters = cribriform openings on papillae (NOT passages, ignore)

## ANATOMY
Variable tree: Ureter -> Pelvis -> Major calyces -> Minor calyces (dead ends).
Total calyces varies (7-13 per kidney). You'll be told the expected count.
Stones are biased toward lower-pole calyces but can be anywhere.

## MOTION (3 DOFs of a flexible ureteroscope)
- advance_mm: forward/back along shaft. Range: -3 to 3.
- roll_deg: axial rotation - rotates bending plane AND camera image. Range: -30 to 30. INCREMENTAL.
- deflection_deg: single-plane tip bend. 0 = straight. Range: -45 to 45. ABSOLUTE angle.

IMPORTANT: The scope has realistic dynamics - commands may not execute exactly as requested.
There's backlash, dead zones, and the shaft can buckle if you push into a wall.
Make small moves and reassess.

## STONE TREATMENT
Capture range is ONLY ~4 mm and a 30-degree cone. You must get the tip very close.
1. Center stone in view and approach to within ~4 mm
2. Fire the laser when centered and close
3. Laser fragments stones into 2-6 smaller pieces
4. Fragments <=3.5 mm diameter: capture with basket
5. Fragments >3.5 mm: laser again, then basket
6. After treatment: scan the calyx to verify stone-free

## RESPONSE FORMAT
Respond with ONLY a JSON object. No markdown fences, no extra text:
{
    "observation": "what you see",
    "classification": "lumen | junction | dead_end | stone_visible | fragment_visible | clear_calyx | uncertain",
    "branch_count": 0,
    "stone_visible": false,
    "stone_description": "color, size estimate, position in frame",
    "reasoning": "why this action",
    "action": {"advance_mm": 0.0, "roll_deg": 0.0, "deflection_deg": 0.0},
    "fire_laser": false,
    "use_basket": false,
    "confidence": 0.0
}"""


def _safe_default() -> dict[str, Any]:
    return {
        "observation": "parse_error",
        "classification": "uncertain",
        "branch_count": 0,
        "stone_visible": False,
        "stone_description": "",
        "reasoning": "vlm call failed",
        "action": {"advance_mm": 0.0, "roll_deg": 0.0, "deflection_deg": 0.0},
        "fire_laser": False,
        "use_basket": False,
        "confidence": 0.0,
    }


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # Drop first fence line
        parts = text.split("\n", 1)
        if len(parts) == 2:
            text = parts[1]
        # Drop trailing fence
        if "```" in text:
            text = text.rsplit("```", 1)[0]
    return text.strip()


def query_vlm(frame_rgb: np.ndarray, context: str, step: int, model: str = "sonnet") -> dict[str, Any]:
    frame_path = FRAME_DIR / f"step_{step:05d}.jpg"
    cv2.imwrite(
        str(frame_path),
        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )

    prompt = f"""{SYSTEM_PROMPT}

---

Look at the endoscope camera image saved at: {frame_path.resolve()}

{context}

Respond with ONLY a JSON object. No markdown fences, no extra text."""

    # Strip CLAUDE_* env vars so the child uses the user's global config,
    # not the parent Claude Code session's environment.
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE") and k != "CLAUDECODE"}

    cmd = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "json",
        "--permission-mode",
        "bypassPermissions",
        "--model",
        model,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return _safe_default()

    if result.returncode != 0:
        return _safe_default()

    try:
        outer = json.loads(result.stdout)
        text = outer.get("result", result.stdout) if isinstance(outer, dict) else result.stdout
    except json.JSONDecodeError:
        text = result.stdout

    text = _strip_fences(str(text))

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Last resort: find first { ... } block
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return _safe_default()
        else:
            return _safe_default()

    # Fill missing keys with defaults
    default = _safe_default()
    for k, v in default.items():
        parsed.setdefault(k, v)
    if not isinstance(parsed.get("action"), dict):
        parsed["action"] = default["action"]
    else:
        for k in ("advance_mm", "roll_deg", "deflection_deg"):
            parsed["action"].setdefault(k, 0.0)
    return parsed
