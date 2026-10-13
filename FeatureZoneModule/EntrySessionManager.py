"""Ordered entry-session coordination for preproduction feature capture.

This coordinator is intentionally independent from video capture, OCR, and DB I/O.
The caller supplies validated trigger events and local track IDs from each camera.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any


@dataclass
class EntrySession:
    session_id: str
    entry_order: int
    created_frame: int
    camera_tracks: dict[int, int] = field(default_factory=dict)
    galleries: dict[int, dict[str, Any]] = field(default_factory=dict)
    status: str = "active"
    finish_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "entry_order": self.entry_order,
            "created_frame": self.created_frame,
            "camera_tracks": dict(self.camera_tracks),
            "galleries": dict(self.galleries),
            "status": self.status,
            "finish_reason": self.finish_reason,
        }


class EntrySessionManager:
    """Keep ordered entry sessions without blocking later triggers.

    Camera 1 creates entry sessions. Tracks from cameras 2/3 are attached only
    when the caller has confirmed they belong to a registered entrant. The FIFO
    helper is therefore not a generic "first car seen on a camera" matcher.
    Camera 2 track loss is the global session-completion signal for this first
    preproduction iteration; OCR completion is deliberately not involved.
    """

    FEATURE_CAMERAS = (1, 2, 3)

    def __init__(self) -> None:
        self._lock = RLock()
        self._sessions: list[EntrySession] = []
        self._by_id: dict[str, EntrySession] = {}
        self._active_track_to_session: dict[tuple[int, int], str] = {}
        self._seen_camera1_tracks: set[int] = set()
        self._next_order = 1

    def register_entry(self, camera_id: int, track_id: int, frame_id: int) -> EntrySession:
        """Create one ordered session for a newly triggered camera-1 track.

        Repeated zone hits for the same still-known local track return its
        existing session instead of creating a session on every frame.
        """
        if int(camera_id) != 1:
            raise ValueError("Entry sessions must be triggered by camera 1")
        track_id = int(track_id)
        with self._lock:
            existing_id = self._active_track_to_session.get((1, track_id))
            if existing_id is not None:
                existing = self._by_id[existing_id]
                if existing.status == "active":
                    return existing
            if track_id in self._seen_camera1_tracks:
                # The previous session may have completed while this track is
                # still visible. Do not re-trigger until the caller releases it.
                for session in reversed(self._sessions):
                    if session.camera_tracks.get(1) == track_id:
                        return session
            order = self._next_order
            self._next_order += 1
            session = EntrySession(
                session_id=f"entry-{order:06d}",
                entry_order=order,
                created_frame=int(frame_id),
                camera_tracks={1: track_id},
            )
            self._sessions.append(session)
            self._by_id[session.session_id] = session
            self._active_track_to_session[(1, track_id)] = session.session_id
            self._seen_camera1_tracks.add(track_id)
            return session

    def release_camera1_track(self, track_id: int) -> None:
        """Allow a tracker-reused local ID to trigger a future entry after it vanishes."""
        with self._lock:
            self._seen_camera1_tracks.discard(int(track_id))
            key = (1, int(track_id))
            session_id = self._active_track_to_session.get(key)
            if session_id is not None and self._by_id[session_id].status != "active":
                self._active_track_to_session.pop(key, None)

    def bind_track(self, session_id: str, camera_id: int, track_id: int) -> EntrySession:
        """Attach a known corresponding local track to a specific session."""
        camera_id, track_id = int(camera_id), int(track_id)
        if camera_id not in self.FEATURE_CAMERAS:
            raise ValueError("Feature-session tracks are supported only on cameras 1, 2, and 3")
        with self._lock:
            session = self._require_session(session_id)
            if session.status != "active":
                raise ValueError(f"Cannot bind a track to completed session {session_id}")
            owner = self._active_track_to_session.get((camera_id, track_id))
            if owner is not None and owner != session_id:
                raise ValueError(
                    f"Camera {camera_id} track {track_id} is already assigned to session {owner}"
                )
            previous_track = session.camera_tracks.get(camera_id)
            if previous_track is not None and previous_track != track_id:
                self._active_track_to_session.pop((camera_id, previous_track), None)
            session.camera_tracks[camera_id] = track_id
            self._active_track_to_session[(camera_id, track_id)] = session_id
            return session

    def bind_next_in_order(
        self, camera_id: int, track_id: int, *, confirmed_registered_entry: bool
    ) -> EntrySession | None:
        """Bind a confirmed entrant to the oldest session missing this camera.

        Do not call this for every detected car. A non-entering/interloper vehicle
        must pass confirmed_registered_entry=False and cannot consume a queue slot.
        """
        camera_id = int(camera_id)
        if camera_id not in (2, 3):
            raise ValueError("FIFO camera binding is only for cameras 2 and 3")
        if not confirmed_registered_entry:
            return None
        with self._lock:
            for session in self._sessions:
                if session.status == "active" and camera_id not in session.camera_tracks:
                    return self.bind_track(session.session_id, camera_id, track_id)
            return None

    def attach_gallery(
        self, session_id: str, camera_id: int, gallery: dict[str, Any]
    ) -> None:
        camera_id = int(camera_id)
        if camera_id not in self.FEATURE_CAMERAS:
            raise ValueError("Feature galleries are supported only on cameras 1, 2, and 3")
        with self._lock:
            session = self._require_session(session_id)
            session.galleries[camera_id] = gallery

    def finish_on_camera2_track_lost(self, camera2_track_id: int) -> dict[str, Any] | None:
        """Complete the owning entry session when its camera-2 track leaves the frame."""
        with self._lock:
            session_id = self._active_track_to_session.get((2, int(camera2_track_id)))
            if session_id is None:
                return None
            session = self._by_id[session_id]
            if session.status != "active":
                return None
            session.status = "completed"
            session.finish_reason = "camera_2_track_left_frame"
            for camera_id, track_id in list(session.camera_tracks.items()):
                self._active_track_to_session.pop((camera_id, track_id), None)
            return session.as_dict()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            session = self._by_id.get(session_id)
            return session.as_dict() if session is not None else None

    def active_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [session.as_dict() for session in self._sessions if session.status == "active"]

    def _require_session(self, session_id: str) -> EntrySession:
        session = self._by_id.get(str(session_id))
        if session is None:
            raise KeyError(f"Unknown entry session: {session_id}")
        return session
