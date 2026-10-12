"""PyCharm-friendly debugger for polygon-based Re-ID feature zones.

Run directly from PyCharm. No database connection or writes are performed.

Mouse / keyboard:
  e - toggle polygon edit mode; left-click adds polygon vertices
  1 - select LEFT zone (cameras 1 and 2)
  2 - select RIGHT zone (cameras 1 and 2)
  d - select DEFAULT zone (camera 3 and cameras 4-9)
  s - save the current polygon
  c - clear the current polygon
  space - pause/resume video
  q / Esc - quit

Coordinates use the resized 1280x720 frame.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from Config import Config
from FeaturesModule.FeaturesManager import FeaturesManager
from FeatureZoneModule import FeatureZoneManager
from TopologyModule.CameraNetwork import CameraNetwork
from TopologyModule.CrossingDetector import CrossingDetector


# =============================================================================
# SETTINGS — change these in PyCharm, then press Run. No CLI arguments needed.
# =============================================================================
CAMERA_ID = 1
VIDEO_SOURCE = r"../step_1/test_videos/1/side.ts"  # video path, webcam index as string ("0"), or RTSP URL
ENTRY_GATE = "LEFT"  # for camera 2/3: LEFT or RIGHT; persistent gate from camera 1
FRAME_SIZE = (1280, 720)
SAMPLE_EVERY_N_FRAMES = 3
SIMILARITY_THRESHOLD = 0.90
LOST_TTL_FRAMES = 30
SHOW_TRANSITION_LINES = True
RUN_OSNET_FEATURES = True  # False = inspect tracking/markup without loading OSNet
WINDOW_NAME = "Barrier + ReID feature-zone debugger"

ROOT = Path(__file__).resolve().parent
ZONES_PATH = ROOT / "feature_zones.yaml"
BARRIER_LINES_PATH = ROOT / "barrier_lines.yaml"
LOG_PATH = ROOT / "feature_zone_debug.log"
REPORT_PATH = ROOT / "feature_zone_debug_report.json"


def choose_video_if_missing(source: str) -> str:
    """Open a file picker when the configured local video path does not exist."""
    if source.isdigit() or source.lower().startswith(("rtsp://", "http://", "https://")):
        return source
    candidate = Path(source)
    if candidate.exists():
        return str(candidate)
    root = tk.Tk()
    root.withdraw()
    try:
        selected = filedialog.askopenfilename(
            title="Select a camera video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.ts *.m4v"), ("All files", "*.*")]
        )
    finally:
        root.destroy()
    if not selected:
        raise FileNotFoundError(
            f"Video not found: {candidate.resolve()}. Select a video in the file dialog "
            "or update VIDEO_SOURCE at the top of this file."
        )
    return selected


def load_yaml(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or default


class Debugger:
    def __init__(self):
        self.zones_cfg = load_yaml(ZONES_PATH, {
            "sample_every_n_frames": SAMPLE_EVERY_N_FRAMES,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "min_crop_size": 8,
            "cameras": {},
        })
        self.zones_cfg.setdefault("cameras", {})
         self.profile = ENTRY_GATE.upper() if CAMERA_ID in (1, 2) else "DEFAULT"
        self.edit_polygon = False
        self.paused = False
        self.pending_points: list[list[int]] = []
        self.last_frame: np.ndarray | None = None
        self.frame_id = 0
        self.last_seen: dict[int, int] = {}
        self.accepted = Counter()
        self.rejected = Counter()
        self.transitions = Counter()
        self.start_time = time.time()
        self.log = logging.getLogger("feature-zone-debug")

        self.network = CameraNetwork.from_yaml(Config.TOPOLOGY_CONFIG_PATH)
        self.crossing = CrossingDetector(self.network.camera(CAMERA_ID))
        self.detector = YOLO(Config.YOLO_CAR_MODEL_PATH)
        self.features = None
        self.zones = None
        if RUN_OSNET_FEATURES:
            self.features = FeaturesManager(Config.OSNET_MODEL_PATH)
            self.zones = FeatureZoneManager(self.features, str(ZONES_PATH))

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

    def profile_points(self) -> list:
        cam = self.zones_cfg["cameras"].get(CAMERA_ID,
              self.zones_cfg["cameras"].get(str(CAMERA_ID), {})) or {}
        return cam.get(self.profile, []) or []

    def set_profile(self, profile: str):
        if CAMERA_ID in (1, 2) and profile not in ("LEFT", "RIGHT"):
            print("Для камер 1/2 выберите профиль LEFT или RIGHT.")
            return
        if CAMERA_ID not in (1, 2) and profile != "DEFAULT":
            print("Камеры 3-9 используют профиль DEFAULT.")
            return
        self.profile = profile
        self.pending_points.clear()
        print(f"Активный профиль зоны: camera={CAMERA_ID}, profile={self.profile}")

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.edit_polygon:
            self.pending_points.append([int(x), int(y)])
            return
 
    def save_config(self):
        cameras_cfg = self.zones_cfg.setdefault("cameras", {})
        cam_key = str(CAMERA_ID)
        # YAML may parse unquoted camera IDs as integers. Normalize this key before saving,
        # otherwise both 1 and "1" could coexist and the old polygon might win on reload.
        integer_entry = cameras_cfg.pop(CAMERA_ID, {}) if CAMERA_ID in cameras_cfg else {}
        string_entry = cameras_cfg.get(cam_key, {}) or {}
        merged_entry = dict(integer_entry or {})
        merged_entry.update(string_entry)
        cameras_cfg[cam_key] = merged_entry
        self.zones_cfg["cameras"][cam_key][self.profile] = self.pending_points.copy() \
            if self.edit_polygon and self.pending_points else self.profile_points()
        # If not actively drawing, preserve the last stored polygon.
        if self.edit_polygon:
            self.zones_cfg["cameras"][cam_key][self.profile] = self.pending_points.copy()
        with ZONES_PATH.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.zones_cfg, stream, allow_unicode=True, sort_keys=False)
        print(f"Saved zones: {ZONES_PATH}")
        if RUN_OSNET_FEATURES:
            self.zones = FeatureZoneManager(self.features, str(ZONES_PATH))

    def clear_current(self):
        self.pending_points.clear()
        cam = self.zones_cfg["cameras"].setdefault(str(CAMERA_ID), {})
        cam[self.profile] = []
        print(f"Cleared polygon: camera={CAMERA_ID}, profile={self.profile}")
        if RUN_OSNET_FEATURES:
            self.zones = FeatureZoneManager(self.features, str(ZONES_PATH))

    def draw_line(self, image, p1, p2, color, label, thickness=2):
        a, b = tuple(map(int, p1)), tuple(map(int, p2))
        cv2.line(image, a, b, color, thickness, cv2.LINE_AA)
        midpoint = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        cv2.putText(image, label, midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 2, cv2.LINE_AA)

    def render(self, frame, tracks, boxes, ids, decisions, events):
        view = frame.copy()

        # Show both zones on cameras 1/2; selected zone is highlighted.
        camera_cfg = self.zones_cfg.get("cameras", {}).get(
            str(CAMERA_ID), self.zones_cfg.get("cameras", {}).get(CAMERA_ID, {})
        ) or {}
        if not self.edit_polygon and CAMERA_ID in (1, 2):
            for side, color in (("LEFT", (255, 120, 0)), ("RIGHT", (0, 150, 255))):
                side_points = camera_cfg.get(side, []) or []
                if len(side_points) >= 3:
                    pts = np.asarray(side_points, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(view, [pts], True, color, 3 if side == self.profile else 1, cv2.LINE_AA)
                    cv2.putText(view, side, tuple(side_points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        points = self.pending_points if self.edit_polygon else self.profile_points()
        if len(points) >= 2:
            pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(view, [pts], self.edit_polygon and len(points) >= 3,
                          (0, 255, 255), 2, cv2.LINE_AA)
            for i, point in enumerate(points):
                cv2.circle(view, tuple(point), 5, (0, 255, 255), -1)
                cv2.putText(view, str(i + 1), (point[0] + 5, point[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        elif not self.edit_polygon:
            cv2.putText(view, "ZONE NOT CONFIGURED - press E and click polygon vertices",
                        (18, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        if SHOW_TRANSITION_LINES:
            for line in self.network.lines(CAMERA_ID):
                self.draw_line(view, line.p1, line.p2, (180, 0, 220),
                               f"TRANSITION L{line.line_id}->C{line.peer_camera}", 1)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 220, 0) if decisions.get(int(track_id)) else (255, 255, 255)
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
            cv2.putText(view, f"ID {track_id}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if CAMERA_ID == 2:
                anchor = (x2, (y1 + y2) // 2)
            elif CAMERA_ID == 3:
                anchor = ((x1 + x2) // 2, y1)
            else:
                anchor = ((x1 + x2) // 2, y2)
            cv2.circle(view, anchor, 5, (0, 255, 255), -1)

        for event in events:
            cv2.putText(view, f"{event.role.value} track={event.track_id} line={event.line_id}",
                        (18, 125 + 24 * (self.transitions.total() % 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

        active_mode = "POLYGON EDIT" if self.edit_polygon else "VIEW / FEATURE TEST"
        gate_text = f"selected zone={self.profile}" if CAMERA_ID in (1, 2) else "single DEFAULT zone"
        lines = [
            f"Camera {CAMERA_ID} | profile={self.profile} | frame={self.frame_id} | {active_mode}",
            gate_text,
            f"Accepted={sum(self.accepted.values())} rejected_similar={sum(self.rejected.values())}",
            "E edit polygon | 1 LEFT | 2 RIGHT | D DEFAULT | S save | C clear | SPACE pause | Q quit",
        ]
        for i, text in enumerate(lines):
            cv2.putText(view, text, (12, 25 + i * 23), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (255, 255, 255), 2, cv2.LINE_AA)
        return view

    def run(self):
        source = choose_video_if_missing(VIDEO_SOURCE)
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        self.log.info("START camera=%d video=%s entry_gate=%s db_writes=0",
                      CAMERA_ID, source, ENTRY_GATE)
        print("Debugger started. Click video to mark polygons/lines; press Q to quit.")
        print("IMPORTANT: camera-1 barrier lines are only saved as markup; zone polygons drive feature sampling.")
        try:
            while True:
                if not self.paused:
                    ok, raw = cap.read()
                    if not ok:
                        break
                    frame = cv2.resize(raw, FRAME_SIZE)
                    result = self.detector.track(
                        frame, persist=True, tracker=Config.YOLO_TRACK_CONFIG_PATH,
                        conf=Config.YOLO_TRACK_CONF_THRESHOLD,
                        iou=Config.YOLO_TRACK_IOU_THRESHOLD, verbose=False
                    )[0]
                    if result.boxes is not None and result.boxes.id is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        ids = result.boxes.id.cpu().numpy().astype(int)
                        tracks = [(int(tid), box.tolist()) for tid, box in zip(ids, boxes)]
                    else:
                        boxes = np.empty((0, 4), dtype=np.float32)
                        ids = np.empty((0,), dtype=int)
                        tracks = []

                    events = self.crossing.update(self.frame_id, tracks)
                    self.transitions.update(f"{e.role.value}:{e.line_id}->{e.peer_camera}" for e in events)
                    decisions = {}
                    if self.zones is not None:
                        decisions = self.zones.process_frame(
                            camera_id=CAMERA_ID, frame_id=self.frame_id, frame=frame,
                            tracks=tracks, entry_gate=self.profile if CAMERA_ID in (2, 3) else None
                        )
                        for track_id, accepted in decisions.items():
                            if accepted:
                                self.accepted[int(track_id)] += 1
                                self.log.info("FEATURE_ACCEPT frame=%d camera=%d track=%d",
                                              self.frame_id, CAMERA_ID, track_id)
                            else:
                                self.rejected[int(track_id)] += 1
                                self.log.info("FEATURE_REJECT_SIMILAR frame=%d camera=%d track=%d",
                                              self.frame_id, CAMERA_ID, track_id)
                    for event in events:
                        self.log.info("TRANSITION frame=%d track=%d role=%s line=%d peer=%d",
                                      event.frame_id, event.track_id, event.role.value,
                                      event.line_id, event.peer_camera)
                        if self.zones is not None and event.role.value == "EXIT":
                            gallery = self.zones.end_track(CAMERA_ID, event.track_id)
                            if gallery:
                                self.log.info("GALLERY_CLOSED track=%d samples=%d",
                                              event.track_id, gallery["sample_count"])
                    present = set(map(int, ids))
                    for track_id in list(self.last_seen):
                        if track_id not in present and self.frame_id - self.last_seen[track_id] >= LOST_TTL_FRAMES:
                            if self.zones is not None:
                                gallery = self.zones.end_track(CAMERA_ID, track_id)
                                if gallery:
                                    self.log.info("GALLERY_CLOSED reason=LOST track=%d samples=%d",
                                                  track_id, gallery["sample_count"])
                            self.last_seen.pop(track_id, None)
                    for track_id in present:
                        self.last_seen[track_id] = self.frame_id
                    self.last_frame = self.render(frame, tracks, boxes, ids, decisions, events)
                    self.frame_id += 1

                if self.last_frame is not None:
                    cv2.imshow(WINDOW_NAME, self.last_frame)
                key = cv2.waitKey(25 if not self.paused else 80) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord(" "):
                    self.paused = not self.paused
                elif key == ord("e"):
                    self.edit_polygon = not self.edit_polygon
                    self.pending_points = self.profile_points().copy() if self.edit_polygon else []
                    print(f"Polygon edit: {self.edit_polygon}. Click vertices in order; press S to save.")
                elif key == ord("1"):
                    self.set_profile("LEFT")
                elif key == ord("2"):
                    self.set_profile("RIGHT")
                elif key == ord("d"):
                    self.set_profile("DEFAULT")
                elif key == ord("s"):
                    self.save_config()
                elif key == ord("c"):
                    self.clear_current()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.zones is not None:
                self.zones.close_all()
            report = {
                "camera_id": CAMERA_ID, "video": source, "entry_gate": self.profile if CAMERA_ID in (2, 3) else None,
                "frames_processed": self.frame_id,
                "elapsed_seconds": round(time.time() - self.start_time, 2),
                "accepted_samples": sum(self.accepted.values()),
                "rejected_similar": sum(self.rejected.values()),
                "transitions": dict(self.transitions),
                "database_writes": 0,
            }
            REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            self.log.info("REPORT %s", REPORT_PATH)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    Debugger().run()


if __name__ == "__main__":
    main()
