import cv2
import numpy as np
from ultralytics import YOLO

from Config import Config
from TrackletModule.Tracklet import TrackletState
from TrackletModule.TrackletManager import TrackletManager

VIDEO_PATH = "../step_1/test_videos/1/side.ts"
OUTPUT_PATH = "5009.mp4"

model = YOLO(Config.YOLO_CAR_MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

SHOW_VIDEO = True          # True - показывать на экране, False - только запись в файл
USE_TRACKLET_MANAGER = False  # True - использовать TrackletManager, False - только BoT-SORT

if not SHOW_VIDEO:
    output_size = (1280, 720)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 10
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, output_size)

    print(f"Запись видео в файл: {OUTPUT_PATH}")

traclet_manager = None
if USE_TRACKLET_MANAGER:
    traclet_manager = TrackletManager(
        cam_id=0,
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

    H, W = frame.shape[:2]

    results = model.track(
        frame,
        persist=True,
        tracker=Config.YOLO_TRACK_CONFIG_PATH,
        conf=Config.YOLO_TRACK_CONF_THRESHOLD,
        iou=Config.YOLO_TRACK_IOU_THRESHOLD,
        verbose=False
    )

    r = results[0]

    if r.boxes is None or r.boxes.id is None:
        # Кадр без детекций тоже обязан дойти до менеджера и сдвинуть frame_id,
        # иначе треки не переходят в LOST, а интервалы Re-ID считаются по застывшему счётчику
        if USE_TRACKLET_MANAGER:
            traclet_manager.update(
                byte_track_bboxes=np.empty((0, 4)),
                byte_track_ids=np.empty((0,), dtype=int),
                frame_id=frame_id,
                frame=frame
            )

        if SHOW_VIDEO:
            cv2.imshow("Tracking", frame)
        else:
            out.write(frame)

        frame_id += 1
        if cv2.waitKey(1) == 27:
            break
        continue

    boxes = r.boxes.xyxy.cpu().numpy()
    ids = r.boxes.id.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()

    # --- Ветка с TrackletManager ---
    if USE_TRACKLET_MANAGER:
        traclet_manager.update(
            byte_track_bboxes=boxes,
            byte_track_ids=ids,
            frame_id=frame_id,
            frame=frame
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

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # --- Ветка только с BoT-SORT ---
    else:
        for box, track_id, conf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0)
            label = f"ID {track_id} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    if SHOW_VIDEO:
        cv2.imshow("Tracking", frame)
    else:
        out.write(frame)

    if SHOW_VIDEO and cv2.waitKey(1) == 27:
        break

    frame_id += 1

cap.release()
if not SHOW_VIDEO:
    out.release()
    print(f"Видео сохранено в: {OUTPUT_PATH}")
    print(f"Обработано кадров: {frame_id}")

cv2.destroyAllWindows()