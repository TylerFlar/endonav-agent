"""Tiny HTTP client used by the LLM agent's Bash calls.

Usage:
    python -m endonav.cli render_and_save
    python -m endonav.cli send_command --advance_mm=1.5 --roll_deg=0 --deflection_deg=0
    python -m endonav.cli autopilot_advance --max_frames=20 --stop_on=any
    python -m endonav.cli mark_calyx_status --calyx_id=calyx_0 --status=verified_clear
    python -m endonav.cli get_mission_status --pretty

POSTs `localhost:8765/<method>` with JSON body of kwargs, prints the JSON
response. Add `--pretty` for indented output. The server (scripts/serve_mission.py)
must already be running.
"""

from __future__ import annotations

import argparse
import json
import sys

import requests

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _coerce(val: str):
    """Convert CLI string args to int/float/bool/null where it makes sense."""
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val.lower() in ("null", "none"):
        return None
    try:
        if "." in val or "e" in val.lower():
            return float(val)
        return int(val)
    except ValueError:
        return val


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="endonav.cli", description=__doc__)
    p.add_argument("method", help="MissionTools method name")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--pretty", action="store_true", help="indent JSON output")
    p.add_argument(
        "--timeout", type=float, default=120.0, help="HTTP timeout (seconds)"
    )
    args, remaining = p.parse_known_args(argv)

    # Parse remaining as --key=value or --key value pairs.
    payload: dict = {}
    i = 0
    while i < len(remaining):
        tok = remaining[i]
        if not tok.startswith("--"):
            print(f"unexpected positional arg: {tok}", file=sys.stderr)
            return 2
        key = tok[2:]
        if "=" in key:
            k, v = key.split("=", 1)
            payload[k] = _coerce(v)
            i += 1
        else:
            if i + 1 >= len(remaining):
                print(f"missing value for --{key}", file=sys.stderr)
                return 2
            payload[key] = _coerce(remaining[i + 1])
            i += 2

    url = f"http://{args.host}:{args.port}/{args.method}"
    try:
        r = requests.post(url, json=payload, timeout=args.timeout)
    except requests.exceptions.ConnectionError as e:
        print(
            f"connection error: {e}\n"
            f"is the mission server running? "
            f"start it with: python scripts/serve_mission.py",
            file=sys.stderr,
        )
        return 3
    except requests.exceptions.Timeout:
        print(f"request timed out after {args.timeout}s", file=sys.stderr)
        return 4

    if r.status_code != 200:
        print(f"HTTP {r.status_code}: {r.text}", file=sys.stderr)
        return 5

    try:
        data = r.json()
    except ValueError:
        print(r.text)
        return 0

    if args.pretty:
        print(json.dumps(data, indent=2, default=str))
    else:
        print(json.dumps(data, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
