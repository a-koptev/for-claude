import cv2
import numpy as np
from ultralytics import YOLO

from Config import Config
from TopologyModule.CameraNetwork import CameraNetwork
from TopologyModule.CrossingDetector import CrossingDetector
from TrackletModule.Tracklet import TrackletState
from TrackletModule.TrackletManager import TrackletManager

VIDEO_PATH = "../step_1/test_videos/1/side.ts"
OUTPUT_PATH = "5009.mp4"
CAMERA_ID = 1
TRANSITION_LOG_PATH = "transitions.log"

model = YOLO(Config.YOLO_CAR_MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

SHOW_VIDEO = True          # True - показывать на экране, False - только запись в файл
USE_TRACKLET_MANAGER = False  # False - текущий тест без Re-ID, только BoT-SORT + topology

# Топология нужна уже на этом этапе: пока не идентифицируем машину, а только
# фиксируем, через какую линию и в какую соседнюю камеру уходит track ID.
network = CameraNetwork.from_yaml(Config.TOPOLOGY_CONFIG_PATH)
camera = network.camera(CAMERA_ID)
crossing_detector = CrossingDetector(camera)

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

# Для текущего прогона пишем события и в консоль, и в файл.
transition_log = open(TRANSITION_LOG_PATH, "w", encoding="utf-8")

# Маршрут пока относится к локальному BoT-SORT track ID этой камеры.
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
    draw_transition_lines(frame)

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
        # Даже пустой кадр передаём детектору переходов: так он корректно
        # завершает касания линий и забывает старые track ID.
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

    # Сначала фиксируем переходы независимо от Re-ID/TrackletManager.
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
transition_log.close()

if not SHOW_VIDEO:
    out.release()
    print(f"Видео сохранено в: {OUTPUT_PATH}")
    print(f"Обработано кадров: {frame_id}")

cv2.destroyAllWindows()
