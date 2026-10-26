"""Background thread that turns the LiveBuffer into a continuous video feed.

Owns:
- The display thread (30 Hz by default).
- The mp4 writer (`runs/latest/mission.mp4`).
- The latest-JPEG byte buffer that the Flask /stream endpoint serves.

The thread NEVER touches the sim or the shim Lock. It reads canonical RGB
frames from the LiveBuffer (which has its own lock) and composes the live
view + recorded video locally. The shim Lock is held only by agent commands.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from .live_buffer import LiveBuffer
from .overlays import compose_display_frame
from endonav_sim.visuals import ToolVisualState

log = logging.getLogger("endonav.realtime")


class RealtimeRenderer:
    def __init__(
        self,
        live_buffer: LiveBuffer,
        output_dir: Path | str,
        fps: int = 30,
        jpeg_quality: int = 80,
        record_video: bool = True,
    ) -> None:
        self.live_buffer = live_buffer
        self.fps = int(fps)
        self.period = 1.0 / float(self.fps)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jpeg_quality = int(jpeg_quality)
        self.record_video = bool(record_video)
        self.video_path = self.output_dir / "mission.mp4"

        self._writer = None
        if self.record_video:
            self._writer = imageio.get_writer(
                str(self.video_path),
                fps=self.fps,
                codec="libx264",
                quality=7,
                macro_block_size=1,
            )

        self._latest_jpeg: bytes | None = None
        self._latest_jpeg_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="endonav.realtime",
            daemon=True,
        )
        self._frame_idx = 0
        self._rng = np.random.default_rng(0)
        self._start_ts = time.time()

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._start_ts = time.time()
        self._thread.start()
        log.info("realtime renderer started @ %d fps", self.fps)

    def stop(self, timeout: float = 2.0) -> None:
        if not self._thread.is_alive():
            return
        self._stop.set()
        self._thread.join(timeout=timeout)
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                log.exception("error closing mp4 writer")
            self._writer = None
        log.info("realtime renderer stopped (frames=%d)", self._frame_idx)

    # ------------------------------------------------------------------
    def get_latest_jpeg(self) -> bytes | None:
        with self._latest_jpeg_lock:
            return self._latest_jpeg

    # ------------------------------------------------------------------
    def _run(self) -> None:
        next_tick = time.monotonic()
        cached_rgb: np.ndarray | None = None
        cached_status: dict = {}
        cached_tool_state: ToolVisualState | None = None
        last_seen_version = -1

        while not self._stop.is_set():
            now = time.monotonic()
            sleep = next_tick - now
            if sleep > 0:
                # Use the stop event so shutdown is responsive.
                if self._stop.wait(sleep):
                    break
            next_tick += self.period
            # If we've fallen way behind (slow JPEG encode, etc) reset the
            # cadence so we don't burn CPU catching up.
            if time.monotonic() - next_tick > 0.5:
                next_tick = time.monotonic() + self.period

            try:
                rgb, version, status, tool_state = self.live_buffer.get()
            except Exception:
                log.exception("live_buffer.get failed")
                continue

            if rgb is not None and version != last_seen_version:
                cached_rgb = rgb
                last_seen_version = version
            # Status + tool_state always refresh: HUD timestamps tick at the
            # display rate, and tool overlays animate from sim event times
            # even if the underlying RGB hasn't changed.
            if status:
                cached_status = status
            if tool_state is not None:
                cached_tool_state = tool_state

            if cached_rgb is None:
                continue

            try:
                display_rgb = compose_display_frame(
                    cached_rgb,
                    cached_status,
                    cached_tool_state,
                    self._frame_idx,
                    self._rng,
                    self._start_ts,
                )
            except Exception:
                log.exception("compose_display_frame failed")
                continue
            self._frame_idx += 1

            if self._writer is not None:
                try:
                    self._writer.append_data(display_rgb)
                except Exception:
                    log.exception("video append_data failed")

            try:
                bgr = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(
                    ".jpg",
                    bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                if ok:
                    jpeg_bytes = buf.tobytes()
                    with self._latest_jpeg_lock:
                        self._latest_jpeg = jpeg_bytes
            except Exception:
                log.exception("jpeg encode failed")
