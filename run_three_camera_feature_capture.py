"""Run feature-zone capture on three local videos (cameras 1, 2, and 3).

Edit VIDEO_PATHS below, then run:
    python run_three_camera_feature_capture.py

The script processes the videos side by side, displays the active zones / track IDs,
and prints accepted/rejected feature-sample events. It does not write to the database.
Press Q or Esc to stop. Camera 1 uses the bottom-center anchor, camera 2 the
right-center anchor, and camera 3 the top-center anchor.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from Config import Config
from FeaturesModule.FeaturesManager import FeaturesManager
from FeatureZoneModule import FeatureZoneManager


# ---------------------------------------------------------------------------
# EDIT THESE THREE PATHS. Use absolute paths if videos are outside the project.
# Example: r"D:\\parking\\camera1.mp4"
# ---------------------------------------------------------------------------
VIDEO_PATHS = {
    1: r"videos/test_videos/1/side.ts",
    2: r"videos/test_videos/2/side.ts",
    3: r"videos/test_videos/3/side.ts",
}

# Each camera is displayed at this size; the combined window is 2x2 tiles.
TILE_SIZE = (640, 360)
WINDOW_NAME = "Feature capture test: cameras 1, 2, 3"
REPORT_PATH = Path(__file__).resolve().parent / "three_camera_feature_report.json"
ZONES_PATH = Path(__file__).resolve().parent / "feature_zones.yaml"
LOG_PATH = Path(__file__).resolve().parent / "three_camera_feature_capture.log"
CAMERA_IDS = (1, 2, 3)


def validate_paths() -> None:
    missing = []
    for camera_id in CAMERA_IDS:
        value = str(VIDEO_PATHS.get(camera_id, "")).strip()
        if not value:
            missing.append(f"camera {camera_id}: path is empty")
        elif not Path(value).expanduser().exists():
            missing.append(f"camera {camera_id}: file not found: {value}")
    if missing:
        raise FileNotFoundError(
            "Update VIDEO_PATHS at the top of this file.\n  - " + "\n  - ".join(missing)
        )


def draw_polygon(tile: np.ndarray, points, color, label: str) -> None:
    if not points or len(points) < 3:
        return
    polygon = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
    # Zone coordinates are authored on 1280x720 frames; scale for display.
    polygon[:, 0, 0] = (polygon[:, 0, 0] * tile.shape[1] / 1280).astype(np.int32)
    polygon[:, 0, 1] = (polygon[:, 0, 1] * tile.shape[0] / 720).astype(np.int32)
    cv2.polylines(tile, [polygon], True, color, 2, cv2.LINE_AA)
    x, y = polygon[0, 0]
    cv2.putText(tile, label, (int(x), max(18, int(y) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def make_tile(camera_id, frame, boxes, ids, active_ids, decisions, zones_cfg, frame_id):
    tile = cv2.resize(frame, TILE_SIZE)
    scale_x = TILE_SIZE[0] / frame.shape[1]
    scale_y = TILE_SIZE[1] / frame.shape[0]
    cam_cfg = zones_cfg.get("cameras", {})
    cam_cfg = cam_cfg.get(str(camera_id), cam_cfg.get(camera_id, {})) or {}

    profiles = (("LEFT", (255, 140, 0)), ("RIGHT", (0, 165, 255))) if camera_id in (1, 2) else (("DEFAULT", (255, 200, 0)),)
    for profile, color in profiles:
        draw_polygon(tile, cam_cfg.get(profile, []) or [], color, profile)

    for box, track_id in zip(boxes, ids):
        x1, y1, x2, y2 = map(float, box)
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
        active = int(track_id) in active_ids
        accepted = decisions.get(int(track_id))
        color = (0, 220, 0) if accepted is True else ((0, 180, 255) if active else (235, 235, 235))
        cv2.rectangle(tile, (x1, y1), (x2, y2), color, 2)
        status = "CAPTURING" if active else "not activated"
        if accepted is True:
            status = "FEATURE +"
        elif accepted is False:
            status = "similarity >= threshold (skip)"
        cv2.putText(tile, f"ID {track_id} | {status}", (x1, max(16, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        if camera_id == 2:
            anchor = (x2, (y1 + y2) // 2)
        elif camera_id == 3:
            anchor = ((x1 + x2) // 2, y1)
        else:
            anchor = ((x1 + x2) // 2, y2)
        cv2.circle(tile, anchor, 4, (0, 255, 255), -1)

    cv2.rectangle(tile, (0, 0), (tile.shape[1], 48), (25, 25, 25), -1)
    cv2.putText(tile, f"CAMERA {camera_id} | frame {frame_id} | active tracks: {len(active_ids)}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(tile, "green=accepted feature; orange=activated track",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    return tile


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    validate_paths()

    # A separate YOLO tracker per video is important: tracker state must not be shared.
    captures = {}
    detectors = {}
    managers = {}
    frame_ids = Counter()
    accepted = Counter()
    rejected = Counter()
    ended = set()
    started_at = time.time()
    features = None

    try:
        features = FeaturesManager(Config.OSNET_MODEL_PATH)
        for camera_id in CAMERA_IDS:
            source = str(Path(VIDEO_PATHS[camera_id]).expanduser())
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {camera_id} video: {source}")
            captures[camera_id] = cap
            detectors[camera_id] = YOLO(Config.YOLO_CAR_MODEL_PATH)
            managers[camera_id] = FeatureZoneManager(features, str(ZONES_PATH))
            print(f"Camera {camera_id}: {source}")

        zones_cfg = managers[1].config
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)
        print("Started. Press Q or Esc to stop. No database writes are performed.")

        while len(ended) < len(CAMERA_IDS):
            tiles = {}
            for camera_id in CAMERA_IDS:
                if camera_id in ended:
                    blank = np.zeros((TILE_SIZE[1], TILE_SIZE[0], 3), dtype=np.uint8)
                    cv2.putText(blank, f"CAMERA {camera_id}: video finished",
                                (25, TILE_SIZE[1] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (230, 230, 230), 2)
                    tiles[camera_id] = blank
                    continue

                ok, raw = captures[camera_id].read()
                if not ok:
                    ended.add(camera_id)
                    logging.info("VIDEO_FINISHED camera=%d frame=%d", camera_id, frame_ids[camera_id])
                    galleries = managers[camera_id].close_all()
                    for (cam, track_id), gallery in galleries.items():
                        logging.info("GALLERY_END camera=%d track=%d samples=%d reason=VIDEO_END",
                                     cam, track_id, gallery["sample_count"])
                    blank = np.zeros((TILE_SIZE[1], TILE_SIZE[0], 3), dtype=np.uint8)
                    cv2.putText(blank, f"CAMERA {camera_id}: video finished",
                                (25, TILE_SIZE[1] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (230, 230, 230), 2)
                    tiles[camera_id] = blank
                    continue

                frame = cv2.resize(raw, (1280, 720))
                result = detectors[camera_id].track(
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

                frame_id = frame_ids[camera_id]
                decisions = managers[camera_id].process_frame(
                    camera_id=camera_id, frame_id=frame_id, frame=frame, tracks=tracks
                )
                for track_id, was_accepted in decisions.items():
                    if was_accepted:
                        accepted[camera_id] += 1
                        logging.info("FEATURE_ACCEPT camera=%d frame=%d track=%d total=%d",
                                     camera_id, frame_id, track_id, accepted[camera_id])
                        print(f"[CAM {camera_id}][frame {frame_id}] track {track_id}: FEATURE ACCEPTED")
                    else:
                        rejected[camera_id] += 1
                        logging.info("FEATURE_REJECT_SIMILAR camera=%d frame=%d track=%d total=%d",
                                     camera_id, frame_id, track_id, rejected[camera_id])
                        print(f"[CAM {camera_id}][frame {frame_id}] track {track_id}: skipped (similarity >= threshold)")

                active_ids = {track_id for cam, track_id in managers[camera_id].active_track_keys() if cam == camera_id}
                tiles[camera_id] = make_tile(camera_id, frame, boxes, ids, active_ids,
                                              decisions, zones_cfg, frame_id)
                frame_ids[camera_id] += 1

            top = np.hstack((tiles[1], tiles[2]))
            bottom = np.hstack((tiles[3], np.zeros_like(tiles[3])))
            combined = np.vstack((top, bottom))
            cv2.imshow(WINDOW_NAME, combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        for cap in captures.values():
            cap.release()
        cv2.destroyAllWindows()
        gallery_summary = []
        for camera_id, manager in managers.items():
            for (cam, track_id), gallery in manager.close_all().items():
                gallery_summary.append({
                    "camera_id": cam,
                    "track_id": track_id,
                    "zone_profile": gallery["zone_profile"],
                    "accepted_original_features": gallery["sample_count"],
                    "flipped_features": len(gallery["flipped"]),
                })
        report = {
            "videos": {str(k): str(v) for k, v in VIDEO_PATHS.items()},
            "frames_processed": {str(k): frame_ids[k] for k in CAMERA_IDS},
            "accepted_samples": {str(k): accepted[k] for k in CAMERA_IDS},
            "rejected_similar": {str(k): rejected[k] for k in CAMERA_IDS},
            "galleries_at_shutdown": gallery_summary,
            "elapsed_seconds": round(time.time() - started_at, 2),
            "database_writes": 0,
        }
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("REPORT %s", REPORT_PATH)
        print("\n=== FEATURE CAPTURE SUMMARY ===")
        for camera_id in CAMERA_IDS:
            print(f"Camera {camera_id}: frames={frame_ids[camera_id]}, "
                  f"accepted={accepted[camera_id]}, rejected_similar={rejected[camera_id]}")
        print(f"Report: {REPORT_PATH}")
        print("Note: gallery embeddings are kept in memory only; no database writes were performed.")


if __name__ == "__main__":
    main()
