"""Multi-seed evaluation sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from endonav_sim import AnatomyParams, KidneySimulator, StoneParams

from endonav.agent import AutonomousSurgeon


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--no-vlm", action="store_true")
    p.add_argument("--out", type=str, default="sweep_results.json")
    args = p.parse_args()

    results = []
    for seed in range(args.seeds):
        print(f"\n=== seed {seed} ===")
        sim = KidneySimulator(
            anatomy_params=AnatomyParams(seed=seed),
            stone_params=StoneParams(seed=seed),
            seed=seed,
            realistic=True,
        )
        agent = AutonomousSurgeon(sim, use_vlm=not args.no_vlm)
        m = agent.run(max_steps=args.max_steps)
        results.append({"seed": seed, **{k: v for k, v in m.items() if k not in ("per_calyx_detail", "treatment_history")}})

    n = len(results)
    success_rate = sum(1 for r in results if r["completed"]) / n
    mean_steps = sum(r["total_steps"] for r in results) / n
    mean_vlm = sum(r["vlm_calls"] for r in results) / n
    stones_missed = sum(r["stones_total"] - r["stones_destroyed"] for r in results)
    summary = {
        "n": n,
        "success_rate": success_rate,
        "mean_steps": mean_steps,
        "mean_vlm_calls": mean_vlm,
        "stones_missed_total": stones_missed,
        "results": results,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nSWEEP: success={success_rate:.0%}, mean_steps={mean_steps:.0f}, mean_vlm={mean_vlm:.0f}")


if __name__ == "__main__":
    main()
