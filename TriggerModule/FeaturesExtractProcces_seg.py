import cv2
import numpy as np
from ultralytics import YOLO
from Config import Config
from FeaturesModule.FeaturesManager import FeaturesManager
from DB.features.feature_repository import CarFeatureRepository
from DB.database_manager import DatabaseManager


class FeaturesExtractProcess:
    def __init__(self, side_cam_stream):
        self.TEMP_output_path = "output_test_extract_fr.mp4"

        # Меняем на сегментационную модель
        self.model = YOLO("../Models/Tracking/YOLO26/26s_seg_10_ep.pt")  # Нужно добавить путь в Config
        self.cap = cv2.VideoCapture(side_cam_stream)

        self.running = True

        self.roi_x1 = None
        self.roi_x2 = None
        self.roi_y1 = None
        self.roi_y2 = None

        self.features_manager = FeaturesManager("Models/Tracking/OSNet/model_both_1.pth")
        self.last_frame_id = -1
        self.car_features = []  # Для одного авто (если нужно)
        self.car_tracks = {}  # Для нескольких авто

        db_manager = DatabaseManager(
            database="parking_db",
            user="postgres",
            password="postgres",
            host="127.0.0.1",
            port="5432",
        )
        self.car_feat_repo = CarFeatureRepository(db_manager=db_manager)

    def get_roi(self):
        roi_x1 = 100
        roi_x2 = 1200
        roi_y1 = 200
        roi_y2 = 700

        self.roi_x1 = roi_x1
        self.roi_x2 = roi_x2
        self.roi_y1 = roi_y1
        self.roi_y2 = roi_y2

    def apply_roi_to_image(self, image):
        masked_image = image.copy()
        masked_image[:self.roi_y1, :] = 0
        masked_image[self.roi_y2:, :] = 0
        masked_image[:, :self.roi_x1] = 0
        masked_image[:, self.roi_x2:] = 0
        return masked_image

    @staticmethod
    def select_detection(boxes, ids, polygons=None):
        """
        Выбор лучшего обнаружения на основе площади (можно использовать площадь маски)
        """
        if len(boxes) == 1 and len(ids) == 1:
            x1, y1, x2, y2 = map(int, boxes[0])
            area = (x2 - x1) * (y2 - y1)

            if area >= Config.TRIGGER_TRACK_AREA_THRESHOLD:
                if polygons is not None:
                    return boxes[0], ids[0], polygons[0]
                return boxes[0], ids[0], None
            else:
                return None, None, None
        else:
            max_area = 0
            max_box = 0
            max_id = -1
            max_polygon = None

            for idx, (box, track_id) in enumerate(zip(boxes, ids)):
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)

                if area > max_area:
                    max_area = area
                    max_box = box
                    max_id = track_id
                    if polygons is not None:
                        max_polygon = polygons[idx]

            if max_area >= Config.TRIGGER_TRACK_AREA_THRESHOLD:
                return max_box, max_id, max_polygon
            else:
                return None, None, None

    def extract_and_compare_feature_with_polygon(self, frame, box, polygon):
        """
        Извлечение признаков с использованием полигона и размытого фона
        """
        car_image = self.features_manager.prepare_segment_frame(
            frame=frame,
            box=box,
            polygon=polygon
        )

        current_frame_feature = self.features_manager.extract_features(image_list=[car_image])[0]

        if len(self.car_features) == 0:
            self.car_features.append(current_frame_feature)

        last_feature = self.car_features[-1]
        sim = self.features_manager.calculate_similarity(
            feature_vector_1=current_frame_feature,
            feature_vector_2=last_feature
        )

        if sim <= Config.EXTRACT_SIM_THRESHOLD:
            self.car_features.append(current_frame_feature)

    @staticmethod
    def draw_polygon(frame, polygon, color, track_id=None, confidence=None):
        """
        Отрисовка полигона на кадре
        """
        contour = polygon.astype(np.int32).reshape((-1, 1, 2))

        # Полупрозрачная заливка
        overlay = frame.copy()
        cv2.fillPoly(overlay, [contour], color)
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        # Контур
        cv2.polylines(frame, [contour], True, color, 2)

        # Текст
        if track_id is not None:
            x, y = contour[0][0]
            label = f"ID:{track_id}"
            if confidence:
                label += f" | {confidence:.2f}"

            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - text_h - 8), (x + text_w, y), color, -1)
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def process_video(self):
        self.get_roi()
        SHOW_VIDEO = True

        if not SHOW_VIDEO:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 30
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.TEMP_output_path, fourcc, fps, (width, height))
            print(f"Запись видео в файл: {self.TEMP_output_path}")

        frame_id = 0
        frame_counter_for_fps = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Пропускаем каждый второй кадр для FPS
            if frame_counter_for_fps != 1:
                frame_counter_for_fps = 1
                continue
            frame_counter_for_fps = 0

            frame = cv2.resize(frame, (1280, 720))
            masked_frame = self.apply_roi_to_image(frame)
            self.last_frame_id = frame_id

            results = self.model.track(
                masked_frame,
                persist=True,
                # tracker=Config.TRIGGER_TRACK_CONFIG_PATH,
                conf=Config.TRIGGER_YOLO_TRACK_CONF_THRESHOLD,
                iou=Config.TRIGGER_YOLO_TRACK_IOU_THRESHOLD,
                verbose=False
            )

            r = results[0]

            # Проверяем наличие боксов и масок
            if r.boxes is None or r.boxes.id is None or r.masks is None:
                # Рисуем ROI
                cv2.rectangle(frame, (self.roi_x1, self.roi_y1),
                              (self.roi_x2, self.roi_y2), (0, 0, 255), 2)
                cv2.imshow("Segmentation Tracking", frame)
                if not SHOW_VIDEO:
                    out.write(frame)
                if cv2.waitKey(1) == 27:
                    break
                continue

            # Получаем данные
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            polygons = r.masks.xy  # Полигоны в абсолютных координатах!
            confidences = r.boxes.conf.cpu().numpy()

            # Выбираем лучшее обнаружение
            box, track_id, polygon = self.select_detection(boxes, ids, polygons)

            if box is None or track_id is None or polygon is None:
                cv2.rectangle(frame, (self.roi_x1, self.roi_y1),
                              (self.roi_x2, self.roi_y2), (0, 0, 255), 2)
                cv2.imshow("Segmentation Tracking", frame)
                if not SHOW_VIDEO:
                    out.write(frame)
                if cv2.waitKey(1) == 27:
                    break
                continue

            # Извлекаем признаки с полигоном
            self.extract_and_compare_feature_with_polygon(
                frame=masked_frame,
                box=box,
                polygon=polygon
            )

            # Генерируем цвет для трека
            np.random.seed(track_id)
            color = tuple(int(x) for x in np.random.randint(50, 255, 3))

            # Рисуем полигон
            confidence = confidences[list(ids).index(track_id)] if track_id in ids else None
            frame = self.draw_polygon(frame, polygon, color, track_id, confidence)

            # Рисуем ROI
            cv2.rectangle(frame, (self.roi_x1, self.roi_y1),
                          (self.roi_x2, self.roi_y2), (0, 0, 255), 2)

            if SHOW_VIDEO:
                cv2.imshow("Segmentation Tracking", frame)
            else:
                out.write(frame)

            if SHOW_VIDEO and cv2.waitKey(1) == 27:
                break

            frame_id += 1

        self.cap.release()
        if not SHOW_VIDEO:
            out.release()
            print(f"Видео сохранено в: {self.TEMP_output_path}")

        cv2.destroyAllWindows()

        # self.car_feat_repo.add_batch_features(car_id=16, feature_vectors=self.car_features)


# Использование
if __name__ == "__main__":
    fp = FeaturesExtractProcess(side_cam_stream="../videos/short_front_right.mp4")
    fp.process_video()
