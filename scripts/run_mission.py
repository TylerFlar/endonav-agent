"""Single-seed autonomous mission entry point."""
from __future__ import annotations

import argparse

from endonav_sim import AnatomyParams, KidneySimulator, StoneParams

from endonav.agent import AutonomousSurgeon


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--no-vlm", action="store_true", help="Run autopilot only")
    args = p.parse_args()

    sim = KidneySimulator(
        anatomy_params=AnatomyParams(seed=args.seed),
        stone_params=StoneParams(seed=args.seed),
        seed=args.seed,
        realistic=True,
    )

    print(
        f"Kidney: {sim.anatomy_meta.n_dead_ends} calyces, "
        f"IPA={sim.anatomy_meta.infundibulopelvic_angle_deg:.0f} deg"
    )
    print(f"Stones: {len(sim.stones)}")
    for s in sim.stones:
        print(f"  {s.id}: {s.composition}, r={s.radius:.1f}mm in {s.node_id}")

    agent = AutonomousSurgeon(sim, use_vlm=not args.no_vlm)
    metrics = agent.run(max_steps=args.max_steps)
    print("\nMETRICS:")
    for k, v in metrics.items():
        if k not in ("per_calyx_detail", "treatment_history"):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
