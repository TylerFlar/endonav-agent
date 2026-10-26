"""Mission briefing pre-flight check.

This script boots a MissionTools just long enough to print the deterministic
preop briefing for the given seed, then exits. The actual mission server is
launched separately by serve_mission.py.

Why two scripts? The briefing is fully deterministic from the seed
(generate_anatomy is a pure function), so this lets you preview the kidney
without leaving a server running. The Claude Code session that drives the
mission talks to serve_mission.py via the endonav.cli HTTP wrapper.

Usage:
    python scripts/run_mission.py --seed 42
"""

from __future__ import annotations

import argparse
import sys

from endonav.tools import MissionTools


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"Booting sim for seed={args.seed} (this takes a few seconds) ...")
    tools = MissionTools(seed=args.seed, record_video=False, output_dir="runs/preview")
    print()
    print(tools.get_mission_briefing())
    print()
    print("To run the autonomous mission:")
    print("  Terminal 1:  uv run python scripts/serve_mission.py --seed", args.seed)
    print("  Terminal 2:  cd into this repo, run `claude`, paste MISSION.md")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
