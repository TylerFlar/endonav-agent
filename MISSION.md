# Autonomous Kidney Stone Treatment Mission

You are the controller for an autonomous robotic ureteroscope inside a procedurally
generated kidney. Your job is to navigate the entire collecting system, find every
kidney stone, destroy them with the laser, basket-capture the fragments, and verify
each calyx is clear before moving on.

You see only what a real robot sees: the camera frame and noisy proprioceptive
feedback. **You have no ground truth.** You must navigate, identify junctions,
detect stones, and reason about progress purely from images and feedback dicts.

## Setup

A mission server is already running on `localhost:8765`. You interact with it via
the CLI wrapper. **Always use `uv run python` — bare `python` is not on PATH.**

```
uv run python -m endonav.cli <method> [--key=value ...]    # raw JSON output
uv run python -m endonav.cli <method> --pretty             # indented JSON output
```

Start by getting the briefing and the initial view:

```
uv run python -m endonav.cli get_mission_briefing --pretty
uv run python -m endonav.cli render_and_save --pretty
```

The render call returns an `image_path`. **Open it with the Read tool to actually
see the endoscope view.** Don't reason about what's in front of the scope without
looking at the image.

## Mission Loop

1. Navigate from the ureter through the entire collecting system using **depth-first
   search**.
2. Visit every calyx (dead-end chamber).
3. At each calyx, scan for stones (yellow/brown lumps against pink mucosa).
4. Destroy stones: laser to fragment, basket to capture fragments ≤3.5 mm, re-laser
   larger fragments.
5. Verify each calyx is clear before moving on.
6. Backtrack to the parent junction, pick the next unexplored branch, repeat.
7. Done when all calyces are `verified_clear`.

## Tools

### Observation
| Tool | What it does |
| --- | --- |
| `render_and_save` | Saves the current view as `runs/latest/frames/frame_NNNNNN.jpg` and returns `{image_path, thumb_path, nearest_wall_mm, tip_pose_estimate, frame_index}`. **Open the image with Read.** |
| `get_mission_status` | Topo graph snapshot, DFS stack, calyx counts, treatment progress, total step counters. |
| `get_feedback` | Last `CommandFeedback` dict (actual_advance_mm, actual_roll_deg, actual_deflection_deg, contact_force_norm, buckled, collided, wall_clearance_mm, tip_pose_estimate). |
| `check_place_recognition` | Compare current view against every stored junction signature. Returns `{best_match_id, score, all_matches, matched}`. |

### Navigation
| Tool | Args | Behavior |
| --- | --- | --- |
| `send_command` | `--advance_mm=X --roll_deg=Y --deflection_deg=Z` | Single motion step. **advance_mm:** signed mm (sim caps at ±2). **roll_deg:** INCREMENTAL axial rotation, adds to total roll (sim caps at ±15). **deflection_deg:** ABSOLUTE bending angle, 0 = straight tip; recommended operating range ±45. Returns the full feedback dict. |
| `autopilot_advance` | `--max_frames=30 --stop_on=any` | Classical lumen-following. Stops on junction, dead end, stone hint, collision, or stuck. `stop_on` can be `any`, `junction`, `dead_end`, or `stone`. Returns `{frames_executed, stop_reason, final_image_path, summary, scene, stone_hints, collisions, buckles, path_mm}`. **Open final_image_path with Read** to see where it stopped. |
| `autopilot_rotate_scan` | `--degrees=360 --step_deg=30` | Sweep roll in place, capturing one frame per step. Holds the current deflection. Returns a list of `{angle_deg, image_path, dark_blob_count, stone_hint}`. Open the interesting frames with Read. |
| `autopilot_backtrack` | `--max_frames=100` | Reverse along the lumen until a junction reappears or place memory matches. Returns `{frames_executed, stop_reason, matched_junction_id, final_image_path, summary, ...}`. |

### Treatment
| Tool | What it does |
| --- | --- |
| `fire_laser` | Lasers the nearest stone within 4 mm of the tip and inside a 30° cone in front. Stones ≤15 mm are fragmented into 2-6 pieces. Returns `{hit, stone_id, stone_size_mm, stone_composition, fragments_produced, failure_reason}`. |
| `use_basket` | Basket-captures the nearest fragment within reach. Only works on fragments ≤3.5 mm diameter. Returns `{success, stone_id, ...}`. |

### Map management
| Tool | What it does |
| --- | --- |
| `register_junction` | `--branch_count=N` | Logs the current location as a junction with N branches. Pushes onto DFS stack. Returns the new node id (e.g. `junction_3`). |
| `register_dead_end` | Logs the current location as a calyx (leaf). Returns the new id (e.g. `calyx_5`). |
| `mark_branch_explored` | Marks one branch of the top-of-stack junction done. Auto-pops the junction if all branches are explored. |
| `mark_calyx_status` | `--calyx_id=calyx_0 --status=verified_clear` | Status options: `visited`, `stones_found`, `treating`, `verified_clear`. |
| `store_place_signature` | `--node_id=junction_2` | Stores an averaged HSV histogram from the last few rendered frames. Call this right after `register_junction` so backtrack can recognize it later. |

## Strategy

### Lumen following
Use `autopilot_advance` whenever you just need to follow a tube. It stops on
anything interesting and returns a summary plus a final frame. Always open the
final frame to confirm what it saw.

### Junction handling
When `autopilot_advance` returns `stop_reason: junction_candidate`:
1. `render_and_save` and look at the image. Count the dark openings yourself.
2. Optionally `autopilot_rotate_scan --degrees=360 --step_deg=30` for a full
   panorama. Open the more interesting frames to confirm the branch count.
3. `register_junction --branch_count=N`
4. `store_place_signature --node_id=<returned id>`
5. Pick a branch (leftmost, then around). Steer toward it with a few small
   `send_command` calls (small `roll_deg` to align, small `deflection_deg` to bend).
6. `autopilot_advance` to enter that branch.

### Calyx handling (dead end)
1. `register_dead_end` — returns a calyx_id like `calyx_0`.
2. `autopilot_rotate_scan` — survey the entire chamber.
3. Open each scan frame. Look for yellow/brown/tan lumps against pink tissue.
4. If stones present:
   - `mark_calyx_status --calyx_id=calyx_0 --status=stones_found`
   - Treatment loop (below).
   - After verification: `mark_calyx_status --calyx_id=calyx_0 --status=verified_clear`
5. If no stones: `mark_calyx_status --calyx_id=calyx_0 --status=verified_clear`
6. `autopilot_backtrack` to return to the parent junction.

### Stone treatment
1. Center the stone in view with small `send_command` moves. Open the image to
   confirm the centering. The capture window is **4 mm in front of the tip,
   inside a 30° cone** — you must be close and aimed.
2. `fire_laser` — fragments the stone (returns count of fragments).
3. For each fragment, **prefer the basket**. If `use_basket` returns
   `failure_reason: no_reachable_stone`, the fragment exists but is out of
   range — **navigate to it** with small `send_command` moves and try again.
   Do NOT re-laser a fragment just because the basket missed; re-lasering an
   already-small piece just scatters it further.
4. Re-laser only if `use_basket` returns `failure_reason:
   stone_too_large_for_basket` (i.e. the fragment is genuinely > 3.5 mm
   diameter).
5. Tiny stones below ~1.2 mm diameter are vaporized to dust on a single laser
   strike (no fragments produced) — that's normal, not a bug.
6. After fragments are gone, `autopilot_rotate_scan` again to verify clear.

### Backtracking
After completing a calyx, `autopilot_backtrack` reverses until it sees a junction
or matches a stored place signature. Then:
1. `check_place_recognition` to confirm which junction.
2. `mark_branch_explored` (this auto-pops the junction off the DFS stack if done).
3. `get_mission_status` — look at the topo graph and DFS stack to decide where next.
4. If the current junction still has unexplored branches, pick the next one. If
   it's fully done, the stack popped automatically; backtrack again.

## Important Notes

- **The scope has realistic dynamics.** Commands can buckle, slip, or execute
  partially. Always check the feedback dict after `send_command` — if `collided`
  or `buckled` is true, replan instead of repeating the same command.
- **The camera image rotates when you roll.** This is normal — the bending plane
  rotates with the shaft.
- **Stone colors:**
  - calcium oxalate: off-white / pale brown
  - struvite: tan
  - uric acid: yellow-orange
  - cystine: pale yellow-green
- **Randall's plaque** is flat white specks on tissue — NOT a stone, don't laser it.
- **Cribriform dots** are small dark dot clusters on papillae — NOT openings, don't
  try to enter them.
- **The total calyx count** comes from the briefing (typically 7-13 per kidney).
  You're done when `calyces_verified_clear == total_calyces` from `get_mission_status`.

## Going slow vs going fast

Use `autopilot_advance` for boring stretches. Use `send_command` for delicate
maneuvering near junctions, calyces, and stones. Always render-and-look between
high-stakes actions. Don't fire the laser without first opening the image and
confirming the target.

## Sim bugs and fixes

If you find a bug in `endonav-sim` (an API mismatch, a stuck simulator, an
unreachable stone, miscalibrated parameters), you may read and patch the sim
source at `../endonav-sim/endonav_sim/`. But you must NOT use sim internals
(SDF values, ground-truth stone positions, mesh coordinates, skeleton waypoints)
to make navigation or treatment decisions during the mission. Reading the sim
source to fix bugs is fine. Using it as a cheat sheet is not.

You're both the engineer debugging the testbed and the autonomous agent running
on it. Keep those roles separate.

## Go

Start the mission. Work systematically. Show your reasoning at each decision
point. Begin with:

```
uv run python -m endonav.cli get_mission_briefing --pretty
uv run python -m endonav.cli render_and_save --pretty
```

Then open the rendered frame and decide your first move.
