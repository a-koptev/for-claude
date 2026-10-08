from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

from Config import Config
from TopologyModule.CameraNetwork import CameraNetwork
from TopologyModule.CrossingDetector import CrossingDetector
from TrackletModule.Tracklet import TrackletState
from TrackletModule.TrackletManager import TrackletManager
from PlateOCRModule import PlateOCRTracker, PlateReading, levenshtein_similarity

VIDEO_PATH = "../step_1/test_videos/1/side.ts"
OUTPUT_PATH = "5009.mp4"
CAMERA_ID = 1
TRANSITION_LOG_PATH = "transitions.log"

model = YOLO(Config.YOLO_CAR_MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

SHOW_VIDEO = True          # True - показывать на экране, False - только запись в файл
USE_TRACKLET_MANAGER = False  # False - текущий тест без Re-ID, только BoT-SORT + topology

network = CameraNetwork.from_yaml(Config.TOPOLOGY_CONFIG_PATH)
camera = network.camera(CAMERA_ID)
crossing_detector = CrossingDetector(camera)

# Plate OCR is intentionally independent from Re-ID for this test:
# vehicle track bbox -> plate detector inside vehicle crop -> PARSeq OCR.
plate_ocr = PlateOCRTracker(
    plate_model_path=Config.YOLO_PLATE_MODEL_PATH,
    ocr_checkpoint_path=Config.OCR_MODEL_PATH,
    plate_conf=Config.OCR_PLATE_CONF_THRESHOLD,
    plate_iou=Config.OCR_PLATE_IOU_THRESHOLD,
    plate_imgsz=Config.OCR_PLATE_IMGSZ,
    ocr_device=Config.OCR_DEVICE,
    ocr_batch_size=Config.OCR_BATCH_SIZE,
)
ocr_cache: dict[int, PlateReading] = {}
# Все принятые OCR-результаты по локальному track ID.
# Нужны для отладки стабильности распознавания и анализа ошибок.
ocr_history: dict[int, Counter[str]] = {}
ocr_last_text: dict[int, str] = {}
ocr_similarity_cache: dict[int, float | None] = {}


def draw_plate_label(image, box, track_id, reading):
    """Draw the last successful plate OCR result above the vehicle track."""
    x1, y1, x2, y2 = map(int, box)
    label = f"ID {track_id} | {reading.text} OCR {reading.ocr_confidence * 100:.0f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    text_y = y1 - 8
    if text_y - th - baseline < 0:
        text_y = min(image.shape[0] - baseline - 4, y1 + th + baseline + 8)
    bg_x1 = max(0, x1)
    bg_y1 = max(0, text_y - th - baseline - 5)
    bg_x2 = min(image.shape[1] - 1, x1 + tw + 10)
    bg_y2 = min(image.shape[0] - 1, text_y + baseline + 5)
    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.putText(image, label, (x1 + 5, text_y), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)

def draw_transition_lines(image):
    """Рисует линии переходов текущей камеры и стрелки направления EXIT."""
    for line in camera.lines:
        p1 = tuple(map(int, line.p1))
        p2 = tuple(map(int, line.p2))
        cv2.line(image, p1, p2, (0, 0, 255), 3)

        direction = line.exit_direction.value
        if direction == "RIGHT":
            ex, ey = 1.0, 0.0
        elif direction == "LEFT":
            ex, ey = -1.0, 0.0
        elif direction == "DOWN":
            ex, ey = 0.0, 1.0
        else:
            ex, ey = 0.0, -1.0

        cx = int((p1[0] + p2[0]) / 2)
        cy = int((p1[1] + p2[1]) / 2)
        cv2.arrowedLine(
            image,
            (int(cx - ex * 30), int(cy - ey * 30)),
            (int(cx + ex * 30), int(cy + ey * 30)),
            (0, 255, 255),
            3,
            tipLength=0.35,
        )

        cv2.putText(
            image,
            f"L{line.line_id} -> Cam {line.peer_camera}",
            (p1[0] + 5, p1[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )


if network.warnings:
    print("Предупреждения топологии:")
    for warning in network.warnings:
        print(f"  - {warning}")

transition_log = open(TRANSITION_LOG_PATH, "w", encoding="utf-8")

# Маршрут относится к локальному BoT-SORT track ID этой камеры.
# Один и тот же numeric track ID на другой камере не считается тем же автомобилем.
track_routes: dict[int, list[int]] = {}

if not SHOW_VIDEO:
    output_size = (1280, 720)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 10
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, output_size)

    print(f"Запись видео в файл: {OUTPUT_PATH}")

traclet_manager = None
if USE_TRACKLET_MANAGER:
    traclet_manager = TrackletManager(
        cam_id=CAMERA_ID,
        similarity_threshold=0.63,
        frames_for_confirm=3,
        frames_for_lost=5
    )
    traclet_manager.update_searched_vehicle(["22"])

frame_id = 0
frame_counter_for_fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    # ВАЖНО: YOLO получает чистый кадр. Линии перехода рисуются только на
    # отдельной копии для отображения, чтобы красные линии не влияли на детектор.
    tracking_frame = frame
    display_frame = frame.copy()

    results = model.track(
        tracking_frame,
        persist=True,
        tracker=Config.YOLO_TRACK_CONFIG_PATH,
        conf=Config.YOLO_TRACK_CONF_THRESHOLD,
        iou=Config.YOLO_TRACK_IOU_THRESHOLD,
        verbose=False
    )

    r = results[0]

    if r.boxes is None or r.boxes.id is None:
        ocr_cache.clear()
        ocr_history.clear()
        events = crossing_detector.update(frame_id, [])

        for event in events:
            route = track_routes.setdefault(event.track_id, [event.from_camera])
            if route[-1] != event.to_camera:
                route.append(event.to_camera)
            message = (
                f"[TRANSITION] frame={event.frame_id} "
                f"track={event.track_id} "
                f"{event.from_camera} -> {event.to_camera} "
                f"role={event.role.value} line={event.line_id} "
                f"motion=({event.motion[0]:.1f},{event.motion[1]:.1f}) "
                f"route={route}"
            )
            print(message)
            transition_log.write(message + "\n")
            transition_log.flush()

        if USE_TRACKLET_MANAGER:
            traclet_manager.update(
                byte_track_bboxes=np.empty((0, 4)),
                byte_track_ids=np.empty((0,), dtype=int),
                frame_id=frame_id,
                frame=tracking_frame
            )

        draw_transition_lines(display_frame)

        if SHOW_VIDEO:
            cv2.imshow("Tracking", display_frame)
        else:
            out.write(display_frame)

        frame_id += 1
        if cv2.waitKey(1) == 27:
            break
        continue

    boxes = r.boxes.xyxy.cpu().numpy()
    ids = r.boxes.id.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()

    # Never display a cached plate for a track that is no longer present.
    present_track_ids = {int(track_id) for track_id in ids}
    for cached_track_id in list(ocr_cache):
        if cached_track_id not in present_track_ids:
            del ocr_cache[cached_track_id]
    for history_track_id in list(ocr_history):
        if history_track_id not in present_track_ids:
            del ocr_history[history_track_id]

    events = crossing_detector.update(
        frame_id,
        zip(ids, boxes)
    )
    for event in events:
        route = track_routes.setdefault(event.track_id, [event.from_camera])
        if route[-1] != event.to_camera:
            route.append(event.to_camera)
        message = (
            f"[TRANSITION] frame={event.frame_id} "
            f"track={event.track_id} "
            f"{event.from_camera} -> {event.to_camera} "
            f"role={event.role.value} line={event.line_id} "
            f"motion=({event.motion[0]:.1f},{event.motion[1]:.1f}) "
            f"route={route}"
        )
        print(message)
        transition_log.write(message + "\n")
        transition_log.flush()

    # OCR only for moving vehicles. The vehicle bbox comes directly from
    # the existing YOLO + BoT-SORT tracker; Re-ID is not involved here.
    if frame_id % max(1, Config.OCR_EVERY_N_FRAMES) == 0:
        moving_tracks = [
            (int(track_id), box)
            for box, track_id in zip(boxes, ids)
            if crossing_detector.is_moving(int(track_id))
        ]
        if moving_tracks:
            readings = plate_ocr.recognize(tracking_frame, moving_tracks)
            for track_id, reading in readings.items():
                # Accept OCR only when character confidence is high enough.
                if reading.ocr_confidence < Config.OCR_MIN_CHAR_CONFIDENCE:
                    continue

                ocr_cache[track_id] = reading

                history = ocr_history.setdefault(track_id, Counter())
                history[reading.text] += 1

                print(
                    f"[OCR] track={track_id} "
                    f"plate={reading.text} "
                    f"conf={reading.ocr_confidence:.2f} "
                    f"count={history[reading.text]} "
                    f"history={dict(history)}"
                )

    if USE_TRACKLET_MANAGER:
        traclet_manager.update(
            byte_track_bboxes=boxes,
            byte_track_ids=ids,
            frame_id=frame_id,
            frame=tracking_frame
        )

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)

            if track_id in traclet_manager.t_storage.storage[TrackletState.STAY_UNMATCHED]:
                traclet = traclet_manager.t_storage.storage[TrackletState.STAY_UNMATCHED][track_id]
                color = (0, 0, 255)
                label = f"STAY track {traclet.track_id}"
            elif track_id in traclet_manager.t_storage.storage[TrackletState.MOVE_UNMATCHED]:
                traclet = traclet_manager.t_storage.storage[TrackletState.MOVE_UNMATCHED][track_id]
                color = (255, 0, 0)
                label = f"MOVE track {traclet.track_id}"
            elif track_id in traclet_manager.t_storage.storage[TrackletState.CONFIRMED]:
                traclet = traclet_manager.t_storage.storage[TrackletState.CONFIRMED][track_id]
                color = (0, 255, 0)
                label = f"CONFIRMED DB ID {traclet.vehicle_id}"
            else:
                continue

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                display_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
    else:
        for box, track_id, conf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0)
            label = f"ID {track_id} {conf:.2f}"

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                display_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    for box, track_id in zip(boxes, ids):
        reading = ocr_cache.get(int(track_id))
        if reading is not None:
            draw_plate_label(display_frame, box, int(track_id), reading)

    draw_transition_lines(display_frame)

    if SHOW_VIDEO:
        cv2.imshow("Tracking", display_frame)
    else:
        out.write(display_frame)

    if SHOW_VIDEO and cv2.waitKey(1) == 27:
        break

    frame_id += 1

cap.release()
transition_log.close()

if not SHOW_VIDEO:
    out.release()
    print(f"Видео сохранено в: {OUTPUT_PATH}")
    print(f"Обработано кадров: {frame_id}")

cv2.destroyAllWindows()
