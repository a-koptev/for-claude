"""Realtime frame capture with a bounded latest-frame buffer.

The capture thread owns VideoCapture and publishes only the newest decoded
frame. Processing never waits for old frames, so latency cannot grow without
bound when inference is slower than the source.

For local video files, pacing is done in the capture thread using the source
FPS. This simulates a camera clock without throttling inference. Live/RTSP
sources are consumed as frames arrive.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Optional

import cv2


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    frame: object
    captured_at: float
    source_timestamp_ms: float | None


class RealtimeFrameSource:
    def __init__(self, source: str | int, file_pacing: bool = True):
        self.source = source
        self.file_pacing = file_pacing
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть источник видео: {source}")

        reported_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.source_fps = reported_fps if reported_fps > 0.0 else None
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.is_file = isinstance(source, (str, bytes))
        self.duration_s = (
            self.total_frames / self.source_fps
            if self.is_file and self.source_fps and self.total_frames > 0
            else None
        )

        self._latest: Optional[FramePacket] = None
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ended = False

        self._captured_frames = 0
        self._dropped_frames = 0
        self._processed_frames = 0
        self._first_capture_at: float | None = None
        self._last_capture_at: float | None = None
        self._first_process_at: float | None = None
        self._last_process_at: float | None = None
        self._latency_sum_s = 0.0
        self._latency_samples = 0

    @property
    def ended(self) -> bool:
        with self._condition:
            return self._ended and self._latest is None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="video-capture",
            daemon=True,
        )
        self._thread.start()

        if self.duration_s is not None:
            print(
                f"[VIDEO] source={self.source} fps={self.source_fps:.2f} "
                f"frames={self.total_frames} duration={self.duration_s:.2f}s"
            )
        else:
            print(
                f"[VIDEO] source={self.source} fps={self.source_fps or 'unknown'} "
                f"frames={self.total_frames or 'unknown'}"
            )

    def get_latest(self, timeout: float | None = None) -> Optional[FramePacket]:
        """Return the newest packet; older pending packets are overwritten."""
        deadline = None if timeout is None else time.monotonic() + timeout

        with self._condition:
            while self._latest is None and not self._ended:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)

            packet = self._latest
            self._latest = None
            return packet

    def record_processed(self, packet: FramePacket) -> None:
        now = time.monotonic()
        with self._condition:
            self._processed_frames += 1
            if self._first_process_at is None:
                self._first_process_at = now
            self._last_process_at = now
            latency = max(0.0, now - packet.captured_at)
            self._latency_sum_s += latency
            self._latency_samples += 1

    def performance_line(self) -> str:
        with self._condition:
            arrival_fps = self._rate(
                self._captured_frames, self._first_capture_at, self._last_capture_at
            )
            processing_fps = self._rate(
                self._processed_frames, self._first_process_at, self._last_process_at
            )
            avg_latency_ms = (
                self._latency_sum_s / self._latency_samples * 1000.0
                if self._latency_samples
                else 0.0
            )
            return (
                f"[PERF] source={arrival_fps:.1f} FPS "
                f"processing={processing_fps:.1f} FPS "
                f"latency={avg_latency_ms:.0f} ms "
                f"dropped={self._dropped_frames} "
                f"captured={self._captured_frames} "
                f"processed={self._processed_frames}"
            )

    @staticmethod
    def _rate(count: int, started: float | None, ended: float | None) -> float:
        if started is None or ended is None or ended <= started:
            return 0.0
        return max(0.0, (count - 1) / (ended - started))

    def _capture_loop(self) -> None:
        frame_id = 0
        wall_start = time.monotonic()
        source_fps = self.source_fps if self.is_file else None

        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break

            now = time.monotonic()

            # Local files are paced here, not in the inference loop.
            # If inference is slower, old frames are replaced by newer ones.
            if self.file_pacing and source_fps and source_fps > 0:
                target = wall_start + frame_id / source_fps
                wait_s = target - now
                if wait_s > 0:
                    self._stop.wait(wait_s)
                    if self._stop.is_set():
                        break
                    now = time.monotonic()

            source_timestamp_ms = (
                frame_id / source_fps * 1000.0 if source_fps else None
            )
            packet = FramePacket(
                frame_id=frame_id,
                frame=frame,
                captured_at=now,
                source_timestamp_ms=source_timestamp_ms,
            )

            with self._condition:
                if self._latest is not None:
                    self._dropped_frames += 1
                self._latest = packet
                self._captured_frames += 1
                if self._first_capture_at is None:
                    self._first_capture_at = now
                self._last_capture_at = now
                self._condition.notify()

            frame_id += 1

        with self._condition:
            self._ended = True
            self._condition.notify_all()

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self.cap.release()
        self._thread = None
