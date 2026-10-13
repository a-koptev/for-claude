"""Triggered Re-ID feature capture runner for local camera videos.

Run: python run_feature_capture.py

Camera 1 is shown alone until a tracked vehicle enters one of its configured
feature zones. Then collection starts on cameras 1-3 and a 2x2 debug window
shows all three streams. Camera 2 EXIT line-crossing finalizes a camera-2
gallery and calls a mock persistence function (no database connection).
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from Config import Config
from FeaturesModule.FeaturesManager import FeaturesManager
from PlateOCRModule import PlateOCRTracker
from FeatureZoneModule import FeatureZoneManager
from TopologyModule.CameraNetwork import CameraNetwork
from TopologyModule.CrossingDetector import CrossingDetector


ROOT = Path(__file__).resolve().parent
FEATURE_CAMERAS = (1, 2, 3)
TILE_SIZE = (640, 360)
WINDOW_NAME = "Parking feature capture"
ZONES_PATH = ROOT / "feature_zones.yaml"
REPORT_PATH = ROOT / "feature_capture_run_report.json"
LOG_PATH = ROOT / "feature_capture_run.log"
# Ultralytics accepts device index 0 for the first CUDA GPU, or "cpu".
INFERENCE_DEVICE = 0 if torch.cuda.is_available() else "cpu"


def source_for(camera_id: int) -> str:
    """Resolve configured paths relative to the repository root."""
    value = str(Config.CAMERA_VIDEO_PATHS.get(camera_id, "") or "").strip()
    if not value:
        raise ValueError(
            f"Camera {camera_id} has no video source. Set Config.CAMERA_VIDEO_PATHS[{camera_id}] "
            f"or PARKING_CAMERA_{camera_id}_VIDEO."
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return str(path)


def draw_zones(image: np.ndarray, camera_id: int, zones_cfg: dict) -> None:
    camera_cfg = zones_cfg.get("cameras", {}).get(str(camera_id), {}) or {}
    profiles = (("LEFT", (255, 140, 0)), ("RIGHT", (0, 165, 255))) if camera_id in (1, 2) else (("DEFAULT", (255, 200, 0)),)
    for profile, color in profiles:
        points = camera_cfg.get(profile) or []
        if len(points) < 3:
            continue
        polygon = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [polygon], True, color, 2, cv2.LINE_AA)
        x, y = polygon[0, 0]
        cv2.putText(image, profile, (int(x), max(18, int(y) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_tracks(image: np.ndarray, boxes, track_ids, active_ids, comparisons=None) -> None:
    comparisons = comparisons or {}
    for box, track_id in zip(boxes, track_ids):
        x1, y1, x2, y2 = map(int, box)
        track_id = int(track_id)
        active = track_id in active_ids
        color = (0, 220, 0) if active else (230, 230, 230)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"ID {track_id} | {'CAPTURING' if active else 'waiting for zone'}",
                    (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        comparison = comparisons.get(track_id)
        if comparison:
            label = f"cos dist: {comparison['cosine_distance']:.3f} | sim: {comparison['cosine_similarity']:.3f}"
            y = min(image.shape[0] - 6, y2 + 20)
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(image, (x1, max(0, y - th - 4)),
                          (min(image.shape[1] - 1, x1 + tw + 8), min(image.shape[0] - 1, y + baseline + 2)),
                          (20, 20, 20), -1)
            cv2.putText(image, label, (x1 + 3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)




def best_plate(votes: dict[str, list[float]]) -> tuple[str, float]:
    """Choose highest mean confidence; vote count breaks ties."""
    if not votes:
        return "UNKNOWN", 0.0
    plate, scores = max(
        votes.items(),
        key=lambda item: (sum(item[1]) / len(item[1]), len(item[1]), item[0]),
    )
    return plate, sum(scores) / len(scores)


def draw_plate_labels(image, camera_id, boxes, track_ids, active_ids, votes_by_track):
    for box, track_id in zip(boxes, track_ids):
        track_id = int(track_id)
        if track_id not in active_ids:
            continue
        plate, confidence = best_plate(votes_by_track.get((camera_id, track_id), {}))
        label = f"plate: {plate} ({confidence:.2f})" if plate != "UNKNOWN" else "plate: reading..."
        x1, y1, x2, y2 = map(int, box)
        y = min(image.shape[0] - 6, y2 + 38)
        cv2.putText(image, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 255, 255), 1, cv2.LINE_AA)


def mock_send_to_database(car_id: str, camera_id: int, track_id: int, gallery: dict) -> dict:
    """Mock persistence only: deliberately does not connect to PostgreSQL."""
    original_count = len(gallery.get("original", []))
    flipped_count = len(gallery.get("flipped", []))
    record = {
        "car_id": car_id,
        "source_camera": camera_id,
        "source_track_id": track_id,
        "original_features": original_count,
        "flipped_features": flipped_count,
        "total_features": original_count + flipped_count,
        "status": "MOCK_ONLY_NOT_WRITTEN",
    }
    return record


def unpack_tracks(result):
    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        return boxes, ids, [(int(tid), box.tolist()) for tid, box in zip(ids, boxes)]
    return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=int), []


def main() -> None:
    summary_logger = logging.getLogger("feature_capture_summary")
    summary_logger.setLevel(logging.INFO)
    summary_logger.propagate = False
    summary_logger.handlers.clear()
    summary_formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(summary_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(summary_formatter)
    summary_logger.addHandler(file_handler)
    summary_logger.addHandler(console_handler)
    with ZONES_PATH.open("r", encoding="utf-8") as stream:
        zones_cfg = yaml.safe_load(stream) or {}

    sources = {camera_id: source_for(camera_id) for camera_id in FEATURE_CAMERAS}
    captures = {}
    models = {}
    managers = {}
    frame_ids = Counter()
    ended = set()
    mock_records = []
    plate_votes = defaultdict(lambda: defaultdict(list))
    car_sequence = 0
    accepted = Counter()
    rejected = Counter()
    transitions = Counter()
    galleries_closed = Counter()
    trigger_fired = False
    trigger_track_id = None
    started_at = time.time()
    network = CameraNetwork.from_yaml(Config.TOPOLOGY_CONFIG_PATH)
    crossings = {cid: CrossingDetector(network.camera(cid)) for cid in FEATURE_CAMERAS}

    try:
        # Start only camera 1. Cameras 2 and 3 are opened lazily after the trigger,
        # so their video streams and per-camera trackers are not initialized up front.
        first_source = sources[1]
        first_capture = cv2.VideoCapture(first_source)
        if not first_capture.isOpened():
            raise RuntimeError(f"Cannot open camera 1 video source: {first_source}")
        captures[1] = first_capture
        models[1] = YOLO(Config.YOLO_CAR_MODEL_PATH)
        features = FeaturesManager(Config.OSNET_MODEL_PATH)
        plate_ocr = PlateOCRTracker(
            plate_model_path=Config.YOLO_PLATE_MODEL_PATH,
            ocr_checkpoint_path=Config.OCR_MODEL_PATH,
            plate_conf=Config.OCR_PLATE_CONF_THRESHOLD,
            plate_iou=Config.OCR_PLATE_IOU_THRESHOLD,
            plate_imgsz=Config.OCR_PLATE_IMGSZ,
            ocr_device=Config.OCR_DEVICE,
            ocr_batch_size=Config.OCR_BATCH_SIZE,
            plate_device=INFERENCE_DEVICE,
        )
        for camera_id in FEATURE_CAMERAS:
            managers[camera_id] = FeatureZoneManager(features, str(ZONES_PATH))

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)
        if torch.cuda.is_available():
            print(f"Inference device: CUDA:0 ({torch.cuda.get_device_name(0)})")
        else:
            print("Inference device: CPU (CUDA is unavailable in this PyTorch environment)")
        print("Waiting for camera 1 zone trigger. Press Q or Esc to stop.")
        print("Configured feature-capture cameras:", ", ".join(map(str, FEATURE_CAMERAS)))
        print("Camera sources 4-9 are configurable in Config.py but are not used for feature capture.")

        # Trigger phase: only camera 1 is opened and advanced. The other videos
        # will be opened after the trigger and seeked to camera 1's current frame index.
        while not trigger_fired:
            ok, raw = captures[1].read()
            if not ok:
                raise RuntimeError("Camera 1 video ended before a vehicle entered a feature zone.")

            frame = cv2.resize(raw, (1280, 720))
            result = models[1].track(
                frame, persist=True, tracker=Config.YOLO_TRACK_CONFIG_PATH,
                conf=Config.YOLO_TRACK_CONF_THRESHOLD, iou=Config.YOLO_TRACK_IOU_THRESHOLD,
                verbose=False, device=INFERENCE_DEVICE,
            )[0]
            boxes, ids, tracks = unpack_tracks(result)
            manager = managers[1]
            trigger_decisions = manager.process_frame(1, frame_ids[1], frame, tracks)
            for tid, was_accepted in trigger_decisions.items():
                if was_accepted:
                    accepted[1] += 1
                else:
                    rejected[1] += 1
            active_ids = {tid for cid, tid in manager.active_track_keys() if cid == 1}
            if frame_ids[1] % max(1, Config.OCR_EVERY_N_FRAMES) == 0 and active_ids:
                readings = plate_ocr.recognize(
                    frame, [(int(tid), box) for box, tid in zip(boxes, ids) if int(tid) in active_ids]
                )
                for tid, reading in readings.items():
                    if reading.ocr_confidence >= Config.OCR_MIN_CHAR_CONFIDENCE:
                        plate = reading.text
                        if plate:
                            plate_votes[(1, int(tid))][plate].append(float(reading.ocr_confidence))
            display = frame.copy()
            draw_zones(display, 1, zones_cfg)
            draw_tracks(display, boxes, ids, active_ids, manager.last_comparison_by_track)
            draw_plate_labels(display, 1, boxes, ids, active_ids, plate_votes)
            status = "TRIGGER FIRED - starting capture" if active_ids else "WAITING: car must enter LEFT/RIGHT zone"
            cv2.rectangle(display, (0, 0), (1280, 42), (25, 25, 25), -1)
            cv2.putText(display, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255) if active_ids else (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, cv2.resize(display, TILE_SIZE))
            frame_ids[1] += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return
            if active_ids:
                trigger_fired = True
                trigger_track_id = min(active_ids)
                print(f"\n[TRIGGER] Camera 1 track={trigger_track_id} entered a feature zone. Capture started.")

        while len(ended) < len(FEATURE_CAMERAS):
            tiles = {}
            for camera_id in FEATURE_CAMERAS:
                if camera_id in ended:
                    blank = np.zeros((TILE_SIZE[1], TILE_SIZE[0], 3), dtype=np.uint8)
                    cv2.putText(blank, f"CAMERA {camera_id}: video finished", (20, TILE_SIZE[1] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
                    tiles[camera_id] = blank
                    continue

                # Lazily open each secondary video only when capture begins.
                # Camera 1 is processed first in this loop; frame_ids[1] has
                # advanced by one, so frame_ids[1] - 1 is its current frame index.
                if camera_id not in captures:
                    source = sources[camera_id]
                    cap = cv2.VideoCapture(source)
                    if not cap.isOpened():
                        raise RuntimeError(
                            f"Cannot open camera {camera_id} video source: {source}"
                        )
                    target_frame = max(0, frame_ids[1] - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    captures[camera_id] = cap
                    frame_ids[camera_id] = target_frame
                    print(
                        f"[LAZY START] Camera {camera_id}: seeking to frame "
                        f"{target_frame} to align with camera 1."
                    )
                    models[camera_id] = YOLO(Config.YOLO_CAR_MODEL_PATH)

                ok, raw = captures[camera_id].read()
                if not ok:
                    ended.add(camera_id)
                    # Keep unfinished galleries open until shutdown so they are
                    # included in the final report rather than silently discarded.
                    tiles[camera_id] = np.zeros((TILE_SIZE[1], TILE_SIZE[0], 3), dtype=np.uint8)
                    continue

                frame = cv2.resize(raw, (1280, 720))
                result = models[camera_id].track(
                    frame, persist=True, tracker=Config.YOLO_TRACK_CONFIG_PATH,
                    conf=Config.YOLO_TRACK_CONF_THRESHOLD, iou=Config.YOLO_TRACK_IOU_THRESHOLD,
                    verbose=False, device=INFERENCE_DEVICE,
                )[0]
                boxes, ids, tracks = unpack_tracks(result)
                frame_id = frame_ids[camera_id]
                events = crossings[camera_id].update(frame_id, tracks)
                decisions = managers[camera_id].process_frame(camera_id, frame_id, frame, tracks)
                active_before_exit = {tid for cam, tid in managers[camera_id].active_track_keys() if cam == camera_id}
                if frame_id % max(1, Config.OCR_EVERY_N_FRAMES) == 0 and active_before_exit:
                    readings = plate_ocr.recognize(
                        frame,
                        [(int(tid), box) for box, tid in zip(boxes, ids) if int(tid) in active_before_exit],
                    )
                    for tid, reading in readings.items():
                        if reading.ocr_confidence >= Config.OCR_MIN_CHAR_CONFIDENCE:
                            plate = reading.text
                            if plate:
                                plate_votes[(camera_id, int(tid))][plate].append(float(reading.ocr_confidence))

                for tid, was_accepted in decisions.items():
                    if was_accepted:
                        accepted[camera_id] += 1
                    else:
                        rejected[camera_id] += 1

                for event in events:
                    role = getattr(event.role, "value", str(event.role)).upper()
                    transitions[f"{camera_id}:{role}->{event.peer_camera}"] += 1
                    if role != "EXIT":
                        continue
                    gallery = managers[camera_id].end_track(camera_id, int(event.track_id))
                    if gallery is None:
                        continue
                    galleries_closed[camera_id] += 1
                    plate, plate_confidence = best_plate(plate_votes.get((camera_id, int(event.track_id)), {}))
                    original_count = len(gallery.get("original", []))
                    flipped_count = len(gallery.get("flipped", []))
                    summary_logger.info(
                        "track=%s:%d plate=%s plate_conf=%.3f features=%d (original=%d, flipped=%d) crossed_line=%d next_camera=%d",
                        camera_id, int(event.track_id), plate, plate_confidence,
                        original_count + flipped_count, original_count, flipped_count,
                        event.line_id, event.peer_camera,
                    )
                    if camera_id == 2:
                        car_sequence += 1
                        record = mock_send_to_database(
                            f"MOCK-CAR-{car_sequence:04d}", camera_id, int(event.track_id), gallery
                        )
                        record["plate"] = plate
                        record["plate_confidence"] = plate_confidence
                        mock_records.append(record)
                    plate_votes.pop((camera_id, int(event.track_id)), None)

                active_ids = {tid for cid, tid in managers[camera_id].active_track_keys() if cid == camera_id}
                display = frame.copy()
                draw_zones(display, camera_id, zones_cfg)
                draw_tracks(display, boxes, ids, active_ids, managers[camera_id].last_comparison_by_track)
                draw_plate_labels(display, camera_id, boxes, ids, active_ids, plate_votes)
                cv2.rectangle(display, (0, 0), (1280, 45), (25, 25, 25), -1)
                cv2.putText(display, f"CAMERA {camera_id} | frame {frame_id} | active galleries: {len(active_ids)}",
                            (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                tiles[camera_id] = cv2.resize(display, TILE_SIZE)
                frame_ids[camera_id] += 1

            combined = np.vstack((
                np.hstack((tiles[1], tiles[2])),
                np.hstack((tiles[3], np.zeros_like(tiles[3]))),
            ))
            cv2.imshow(WINDOW_NAME, combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        for cap in captures.values():
            cap.release()
        cv2.destroyAllWindows()
        remaining = []
        for camera_id, manager in managers.items():
            for (cam, track_id), gallery in manager.close_all().items():
                remaining.append({
                    "camera_id": cam,
                    "track_id": track_id,
                    "zone_profile": gallery["zone_profile"],
                    "original_features": len(gallery["original"]),
                    "flipped_features": len(gallery["flipped"]),
                    "total_features": len(gallery["original"]) + len(gallery["flipped"]),
                })
        report = {
            "video_sources": {str(k): v for k, v in sources.items()},
            "trigger_fired": trigger_fired,
            "trigger_camera1_track_id": trigger_track_id,
            "frames_processed": {str(k): int(frame_ids[k]) for k in FEATURE_CAMERAS},
            "accepted_original_samples": {str(k): int(accepted[k]) for k in FEATURE_CAMERAS},
            "rejected_as_similar": {str(k): int(rejected[k]) for k in FEATURE_CAMERAS},
            "transitions": dict(transitions),
            "galleries_closed_on_exit": {str(k): int(galleries_closed[k]) for k in FEATURE_CAMERAS},
            "mock_database_records": mock_records,
            "galleries_still_open_at_shutdown": remaining,
            "elapsed_seconds": round(time.time() - started_at, 2),
            "real_database_writes": 0,
        }
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nReport: {REPORT_PATH}")


if __name__ == "__main__":
    main()
