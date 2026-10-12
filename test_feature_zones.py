"""Diagnostic runner for FeatureZoneModule. No database writes are performed.

Run from the repository root:
  python test_feature_zones.py --video ../step_1/test_videos/1/side.ts --camera-id 1 --show

Configure real polygons in feature_zones.yaml first; empty polygons intentionally disable sampling.
"""
from __future__ import annotations

import argparse
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
from TopologyModule.CameraNetwork import CameraNetwork
from TopologyModule.CrossingDetector import CrossingDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Log-only diagnostic run for zone-based Re-ID feature collection")
    parser.add_argument("--video", default="../step_1/test_videos/1/side.ts", help="Video file or RTSP/other OpenCV source")
    parser.add_argument("--camera-id", type=int, default=1, help="Camera ID from topology.yaml")
    parser.add_argument("--entry-gate", choices=("LEFT", "RIGHT"), default=None,
                        help="Persistent registration gate; required for camera 2/3 zones")
    parser.add_argument("--zones", default="feature_zones.yaml", help="Path to feature_zones.yaml")
    parser.add_argument("--log", default="feature_zone_test.log", help="Diagnostic log path")
    parser.add_argument("--report", default="feature_zone_test_report.json", help="Summary JSON path")
    parser.add_argument("--show", action="store_true", help="Show annotated preview; press Esc to stop")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 means until end")
    parser.add_argument("--lost-ttl", type=int, default=30,
                        help="Close gallery after a track is absent for this many frames")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.log, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger("feature-zone-test")

    zone_path = Path(args.zones)
    if not zone_path.exists():
        raise FileNotFoundError(f"Zone config not found: {zone_path.resolve()}")

    with zone_path.open("r", encoding="utf-8") as stream:
        import yaml
        config = yaml.safe_load(stream) or {}
    camera_cfg = (config.get("cameras", {}) or {}).get(args.camera_id,
                  (config.get("cameras", {}) or {}).get(str(args.camera_id), {})) or {}
    if args.camera_id in (2, 3):
        selected = camera_cfg.get(args.entry_gate, []) if args.entry_gate else []
        if not selected:
            log.warning("No polygon for camera %s / gate %s: feature sampling is disabled.",
                        args.camera_id, args.entry_gate or "(not supplied)")
    elif not camera_cfg.get("DEFAULT"):
        log.warning("No DEFAULT polygon configured for camera %s: feature sampling is disabled.", args.camera_id)

    video_source = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {args.video}")

    log.info("Initializing YOLO and OSNet. No database will be accessed or modified.")
    detector = YOLO(Config.YOLO_CAR_MODEL_PATH)
    feature_manager = FeaturesManager(Config.OSNET_MODEL_PATH)
    zones = FeatureZoneManager(feature_manager, str(zone_path))
    network = CameraNetwork.from_yaml(Config.TOPOLOGY_CONFIG_PATH)
    crossing = CrossingDetector(network.camera(args.camera_id))

    accepted_total = Counter()
    rejected_total = Counter()
    last_seen: dict[int, int] = {}
    frame_id = 0
    start = time.time()

    try:
        while True:
            ok, raw_frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(raw_frame, (1280, 720))
            result = detector.track(
                frame,
                persist=True,
                tracker=Config.YOLO_TRACK_CONFIG_PATH,
                conf=Config.YOLO_TRACK_CONF_THRESHOLD,
                iou=Config.YOLO_TRACK_IOU_THRESHOLD,
                verbose=False,
            )[0]

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                tracks = [(int(track_id), box.tolist()) for track_id, box in zip(ids, boxes)]
            else:
                boxes = np.empty((0, 4), dtype=np.float32)
                ids = np.empty((0,), dtype=int)
                tracks = []

            events = crossing.update(frame_id, tracks)
            for event in events:
                log.info("TRANSITION frame=%d track=%d role=%s line=%d %d->%d",
                         event.frame_id, event.track_id, event.role.value, event.line_id,
                         event.from_camera, event.to_camera)
                if event.camera_id == args.camera_id and event.role.value == "EXIT":
                    gallery = zones.end_track(args.camera_id, event.track_id)
                    if gallery is not None:
                        log.info("GALLERY_CLOSED reason=EXIT camera=%d track=%d originals=%d flips=%d",
                                 args.camera_id, event.track_id, len(gallery["original"]), len(gallery["flipped"]))

            decisions = zones.process_frame(
                camera_id=args.camera_id,
                frame_id=frame_id,
                frame=frame,
                tracks=tracks,
                entry_gate=args.entry_gate,
            )
            for track_id, accepted in decisions.items():
                gallery = next((g for key, g in zones._galleries.items()
                                if key == (args.camera_id, track_id)), None)
                count = len(gallery.original) if gallery else 0
                if accepted:
                    accepted_total[track_id] += 1
                    log.info("FEATURE_ACCEPT frame=%d camera=%d track=%d originals=%d flips=%d",
                             frame_id, args.camera_id, track_id, count, count)
                else:
                    rejected_total[track_id] += 1
                    log.info("FEATURE_REJECT_SIMILAR frame=%d camera=%d track=%d gallery_size=%d",
                             frame_id, args.camera_id, track_id, count)

            present = set(map(int, ids))
            for track_id in list(last_seen):
                if track_id not in present and frame_id - last_seen[track_id] >= max(1, args.lost_ttl):
                    gallery = zones.end_track(args.camera_id, track_id)
                    last_seen.pop(track_id, None)
                    if gallery is not None:
                        log.info("GALLERY_CLOSED reason=TRACK_LOST camera=%d track=%d originals=%d flips=%d",
                                 args.camera_id, track_id, len(gallery["original"]), len(gallery["flipped"]))
            for track_id in present:
                last_seen[track_id] = frame_id

            if args.show:
                preview = frame.copy()
                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"track={track_id}"
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 220, 0), 2)
                    cv2.putText(preview, label, (x1, max(20, y1 - 7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
                cv2.putText(preview, f"frame={frame_id} sampled={sum(accepted_total.values())}",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Feature-zone diagnostic (Esc to stop)", preview)
                if cv2.waitKey(1) == 27:
                    break

            frame_id += 1
            if args.max_frames and frame_id >= args.max_frames:
                break
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    for (camera_id, track_id), gallery in zones.close_all().items():
        log.info("GALLERY_CLOSED reason=END_OF_RUN camera=%d track=%d originals=%d flips=%d",
                 camera_id, track_id, len(gallery["original"]), len(gallery["flipped"]))

    report = {
        "video": args.video,
        "camera_id": args.camera_id,
        "entry_gate": args.entry_gate,
        "frames_processed": frame_id,
        "elapsed_seconds": round(time.time() - start, 3),
        "accepted_samples_by_track": dict(accepted_total),
        "rejected_similar_by_track": dict(rejected_total),
        "accepted_total": sum(accepted_total.values()),
        "rejected_total": sum(rejected_total.values()),
        "database_writes": 0,
        "note": "Diagnostic only; feature_zones.yaml must contain real polygons for sampling to occur.",
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("REPORT_WRITTEN path=%s", Path(args.report).resolve())
    log.info("SUMMARY frames=%d accepted=%d rejected_similar=%d elapsed=%.1fs",
             frame_id, report["accepted_total"], report["rejected_total"], report["elapsed_seconds"])


if __name__ == "__main__":
    main()
