# endonav-agent

> **Status:** paused, last worked on in April 2026. Archived on GitHub; unarchive it to pick it up again.
>
> Part of my endoscopic-navigation experiments at UC San Diego's ARCLab: [all the repos](https://github.com/search?q=user%3ATylerFlar+topic%3Aucsd-arclab&type=repositories).

An autonomous agent for robotic ureteroscopy in the [endonav-sim](https://github.com/TylerFlar/endonav-sim)
kidney simulator. The goal is a fully autonomous controller that:

1. Enters a procedurally generated kidney through the ureter
2. DFS-explores every calyx
3. Finds kidney stones, fragments them with the laser, extracts fragments with the basket
4. Verifies each calyx is stone-free before backtracking
5. Exits when the entire collecting system has been cleared

The agent sees only what a real ureteroscope would: the camera frame and noisy
proprioceptive feedback. Ground-truth fields from the simulator are reserved
for evaluation only.
