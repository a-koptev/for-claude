import cv2
from ultralytics import YOLO

from other.ClipFeaturesManager import ClipFeaturesManager
from FeaturesManager import FeaturesManager
from DB.features.feature_repository import CarFeatureRepository, DatabaseManager

VIDEO_PATH = "../videos/short_front_right.mp4"
# MODEL_PATH = "yolo_42_epoches.pt"

CONF_THRES = 0.3
IOU_THRES = 0.2

TRACKER_CFG = "./vehicle_bytetrack_conf.yaml"

model = YOLO("../Models/Tracking/YOLOv11/yolo11m.pt")
VEHICLE_CLASSES = [2]  # car
# VEHICLE_CLASSES = {2, 5, 7}  # car, bus, truck

cap = cv2.VideoCapture(VIDEO_PATH)

frame_id = 0
frame_counter_for_fps = 0
fps_in = cap.get(cv2.CAP_PROP_FPS)

images = []
feat_manager_1 = FeaturesManager(path_to_osnet_weights="../Models/Tracking/OSNet/model_both_1.pth")
feat_manager_2 = FeaturesManager(path_to_osnet_weights="../Models/Tracking/OSNet/model_veri_1.pth")
clip_feat_manager = ClipFeaturesManager()

db_manager = DatabaseManager(
    database="parking_db",
    user="postgres",
    password="postgres",
    host="127.0.0.1",
    port="5432"
)
db_manager.connect()
repo = CarFeatureRepository(db_manager=db_manager)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_counter_for_fps != 1:
        frame_counter_for_fps = 1
        continue

    frame_counter_for_fps = 0

    frame = cv2.resize(frame, (1280, 720))

    H, W = frame.shape[:2]

    results = model.track(
        frame,
        persist=True,
        tracker=TRACKER_CFG,
        classes=VEHICLE_CLASSES,
        conf=CONF_THRES,
        iou=IOU_THRES,
        verbose=False
    )

    r = results[0]

    if r.boxes is None or r.boxes.id is None:
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    boxes = r.boxes.xyxy.cpu().numpy()
    ids = r.boxes.id.cpu().numpy().astype(int)

    for box, track_id in zip(boxes, ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"track_{track_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Tracking", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key == 32:
        print("НАЖАТА space")
        track_id = 1
        x1, y1, x2, y2 = map(int, boxes[track_id - 1])
        image = frame[y1:y2, x1:x2]

        # features_1 = feat_manager_1.extract_features(image_list=images)
        features_2 = feat_manager_2.extract_features(image_list=[image])
        # features_clip = clip_feat_manager.extract_features(batch_images=[image])

        # repo.add_batch_features(car_id=9, feature_vectors=features_1)
        repo.add_batch_features(car_id=14, feature_vectors=features_2)
        # repo.add_single_feature(car_id=11, feature_vector=features_clip[0])

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
