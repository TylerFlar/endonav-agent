"""Mission server: long-lived process that owns the sim + MissionTools.

Run from a separate terminal BEFORE starting the Claude Code session:

    uv run python scripts/serve_mission.py --seed 42

The server exposes every public method on MissionTools as ``POST /<method>``,
JSON body = kwargs, JSON response = method return value. The sim and ModernGL
renderer are not thread-safe, so all sim operations are serialized via a
single shim Lock.

Also serves a realtime endoscope feed:
- ``GET /viewer``  — small HTML page that embeds the live MJPEG stream
- ``GET /stream``  — multipart/x-mixed-replace MJPEG stream
- ``GET /snapshot`` — single JPEG of the current view (poll-based fallback)

Open ``http://localhost:8765/viewer`` in a browser to watch the mission live.

Graceful shutdown (Ctrl+C) stops the realtime renderer (closes mp4) and calls
``MissionTools.finalize`` to dump ``summary.json``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import signal
import sys
import threading
import time

from flask import Flask, Response, jsonify, request

from endonav.realtime import RealtimeRenderer
from endonav.tools import MissionTools

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger("endonav.serve")


_VIEWER_HTML = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>endonav mission viewer</title>
  <style>
    body {
      margin: 0;
      background: #0a0a0c;
      color: #d8d8d8;
      font-family: -apple-system, Segoe UI, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 12px 0 24px;
    }
    h1 { font-size: 14px; letter-spacing: 0.05em; margin: 8px 0 12px; opacity: 0.7; }
    .frame {
      max-width: 95vw;
      max-height: 85vh;
      border: 1px solid #2a2a30;
      border-radius: 4px;
      box-shadow: 0 12px 48px rgba(0, 0, 0, 0.6);
    }
    a { color: #6cc4ff; text-decoration: none; }
    .footer { font-size: 11px; opacity: 0.5; margin-top: 10px; }
  </style>
</head>
<body>
  <h1>ENDONAV MISSION — LIVE FEED</h1>
  <img class=\"frame\" src=\"/stream\" alt=\"endoscope feed\">
  <div class=\"footer\">
    Snapshot: <a href=\"/snapshot\">/snapshot</a> &middot;
    Status: <a href=\"/\">/</a>
  </div>
</body>
</html>
"""


def build_app(tools: MissionTools, realtime: RealtimeRenderer) -> Flask:
    app = Flask("endonav.serve")
    # The sim isn't thread-safe; serialize all method invocations.
    sim_lock = threading.Lock()

    # Discover public methods on MissionTools.
    method_names = {
        name
        for name, obj in inspect.getmembers(tools, predicate=inspect.ismethod)
        if not name.startswith("_")
    }
    log.info("registering %d methods: %s", len(method_names), sorted(method_names))

    @app.get("/")
    def index():
        return jsonify(
            {
                "service": "endonav.serve",
                "seed": tools.seed,
                "methods": sorted(method_names),
                "realtime_fps": realtime.fps,
                "endpoints": {
                    "viewer": "/viewer",
                    "stream": "/stream",
                    "snapshot": "/snapshot",
                },
            }
        )

    @app.post("/<method>")
    def call(method: str):
        if method not in method_names:
            return jsonify({"error": f"unknown method: {method}"}), 404
        try:
            payload = request.get_json(silent=True) or {}
        except Exception as e:
            return jsonify({"error": f"bad json: {e}"}), 400
        if not isinstance(payload, dict):
            return jsonify({"error": "request body must be a JSON object"}), 400
        log.info("call %s(%s)", method, payload)
        with sim_lock:
            try:
                fn = getattr(tools, method)
                result = fn(**payload)
            except TypeError as e:
                return jsonify({"error": f"bad args: {e}"}), 400
            except Exception as e:
                log.exception("error in %s", method)
                return jsonify({"error": str(e), "type": type(e).__name__}), 500
        return app.response_class(
            response=_safe_dumps(result),
            mimetype="application/json",
        )

    # ------------------------------------------------------------------
    # Realtime endpoints
    # ------------------------------------------------------------------
    @app.get("/viewer")
    def viewer():
        return Response(_VIEWER_HTML, mimetype="text/html")

    @app.get("/snapshot")
    def snapshot():
        jpeg = realtime.get_latest_jpeg()
        if jpeg is None:
            return Response(b"", status=503, mimetype="image/jpeg")
        return Response(jpeg, mimetype="image/jpeg")

    @app.get("/stream")
    def stream():
        boundary = b"--frame"
        period = 1.0 / max(realtime.fps, 1)

        def gen():
            last_jpeg: bytes | None = None
            while True:
                jpeg = realtime.get_latest_jpeg()
                if jpeg is None:
                    time.sleep(period)
                    continue
                # If the underlying frame hasn't actually changed, still emit
                # the same JPEG so the browser keeps the connection alive.
                last_jpeg = jpeg
                yield (
                    boundary
                    + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode("ascii")
                    + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
                time.sleep(period)

        return Response(
            gen(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def _safe_dumps(obj) -> str:
    return json.dumps(obj, default=str)


def main() -> int:
    p = argparse.ArgumentParser(description="endonav mission server")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--realistic", type=int, default=1, help="1 = realistic dynamics")
    p.add_argument("--output-dir", default="runs/latest")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--fps", type=int, default=30, help="realtime / recording fps")
    p.add_argument(
        "--no-record",
        action="store_true",
        help="disable mp4 recording (still serves /stream)",
    )
    args = p.parse_args()

    log.info("booting MissionTools (seed=%d) ...", args.seed)
    tools = MissionTools(
        seed=args.seed,
        realistic=bool(args.realistic),
        output_dir=args.output_dir,
    )
    log.info("\n%s", tools.get_mission_briefing())

    realtime = RealtimeRenderer(
        live_buffer=tools.live_buffer,
        output_dir=tools.output_dir,
        fps=args.fps,
        record_video=not args.no_record,
    )
    realtime.start()

    app = build_app(tools, realtime)

    def _shutdown(signum, frame):
        log.info("shutdown signal %s — finalizing mission", signum)
        try:
            realtime.stop()
        except Exception:
            log.exception("error stopping realtime renderer")
        try:
            summary = tools.finalize()
            log.info(
                "finalized: %s",
                {k: summary.get(k) for k in ("seed", "stones_destroyed", "topo_stats")},
            )
        except Exception:
            log.exception("error during finalize")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info("listening on http://%s:%d  (viewer: http://%s:%d/viewer)", args.host, args.port, args.host, args.port)
    # threaded=True so the streaming endpoint doesn't block command requests.
    # The sim Lock still serializes sim operations.
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
