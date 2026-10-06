import cv2
import numpy as np
from ultralytics import YOLO

from Config import Config
from DB.database_manager import DatabaseManager
from DB.features.feature_repository import CarFeatureRepository
from FeaturesModule.FeaturesManager import FeaturesManager


class FeaturesExtractProcess:
    def __init__(self, side_cam_stream):
        self.TEMP_output_path = "output_feat_ext.mp4"

        self.model = YOLO(Config.YOLO_CAR_MODEL_PATH)
        self.cap = cv2.VideoCapture(side_cam_stream)

        self.running = True

        # Полигональный ROI (4 точки)
        self.roi_points = None
        self.roi_mask = None
        self.roi_bbox = None  # bounding box полигона (x_min, y_min, x_max, y_max)

        self.features_manager = FeaturesManager(Config.OSNET_MODEL_PATH)
        self.last_frame_id = -1
        self.car_features = []

    def get_roi(self):
        """Определение полигона ROI по 4 точкам"""
        # Пример: точки в порядке обхода (верхний-левый, верхний-правый, нижний-правый, нижний-левый)
        roi_points = np.array([
            [10, 50],  # точка 1 (верхний-левый)
            [550, 50],  # точка 2 (верхний-правый)
            [550, 500],  # точка 3 (нижний-правый)
            [10, 500]  # точка 4 (нижний-левый)
        ], dtype=np.int32)

        self.roi_points = roi_points

        # Вычисляем bounding box полигона (минимальные/максимальные координаты)
        x_min = np.min(roi_points[:, 0])
        x_max = np.max(roi_points[:, 0])
        y_min = np.min(roi_points[:, 1])
        y_max = np.max(roi_points[:, 1])

        self.roi_bbox = (x_min, y_min, x_max, y_max)
        print(f"ROI Bounding Box: {self.roi_bbox}")

    def create_roi_mask(self, image_shape):
        """Создание маски для полигонального ROI"""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.roi_points], 255)
        return mask

    def crop_and_mask_roi(self, image):
        """
        Вырезает область по bounding box полигона и применяет маску
        Возвращает:
            - вырезанное изображение с маской (чёрное вне полигона)
            - смещение (offset_x, offset_y) для трансформации координат
        """
        if self.roi_bbox is None:
            return image, (0, 0)

        x_min, y_min, x_max, y_max = self.roi_bbox

        # Вырезаем bounding box
        cropped = image[y_min:y_max, x_min:x_max].copy()

        # Создаём маску для вырезанной области
        # Смещаем точки полигона относительно вырезанной области
        shifted_points = self.roi_points.copy()
        shifted_points[:, 0] -= x_min
        shifted_points[:, 1] -= y_min

        # Создаём маску для вырезанной области
        mask_cropped = np.zeros(cropped.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_cropped, [shifted_points], 255)

        # Применяем маску: всё вне полигона становится чёрным
        masked_cropped = cv2.bitwise_and(cropped, cropped, mask=mask_cropped)

        return masked_cropped, (x_min, y_min)

    def draw_roi(self, image):
        """Отрисовка линий полигона ROI на изображении"""
        if self.roi_points is not None:
            # Рисуем контур полигона красными линиями
            cv2.polylines(image, [self.roi_points], True, (0, 0, 255), 2)

            # Опционально: рисуем bounding box полигона (для отладки)
            if self.roi_bbox:
                x_min, y_min, x_max, y_max = self.roi_bbox
                # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)

    def transform_box_to_original(self, box, offset):
        """
        Преобразует координаты бокса из вырезанного изображения обратно в оригинальное
        """
        x1, y1, x2, y2 = box
        offset_x, offset_y = offset

        return [
            x1 + offset_x,
            y1 + offset_y,
            x2 + offset_x,
            y2 + offset_y
        ]

    def select_detection(self, boxes, ids):
        if len(boxes) == 1 and len(ids) == 1:
            x1, y1, x2, y2 = map(int, boxes[0])
            area = (x2 - x1) * (y2 - y1)

            if area >= Config.TRIGGER_TRACK_AREA_THRESHOLD and not self.bbox_is_sticking(boxes[0]):
                return boxes[0], ids[0]
            else:
                return None, None
        else:
            max_area = 0
            max_box = 0
            max_id = -1
            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)

                if area > max_area:
                    max_area = area
                    max_box = box
                    max_id = track_id

            if max_area >= Config.TRIGGER_TRACK_AREA_THRESHOLD and not self.bbox_is_sticking(max_box):
                return max_box, max_id
            else:
                return None, None

    def bbox_is_sticking(self, bbox):
        if self.roi_points is None or self.roi_bbox is None:
            print("ROI не определен")
            return True

        roi_x_min, roi_y_min, roi_x_max, roi_y_max = self.roi_bbox
        x1, y1, x2, y2 = map(int, bbox)

        dist_to_top = y1 - roi_y_min  # расстояние до верхней грани
        dist_to_bottom = roi_y_max - y2  # расстояние до нижней грани
        dist_to_left = x1 - roi_x_min  # расстояние до левой грани
        dist_to_right = roi_x_max - x2  # расстояние до правой грани

        tolerance = 20

        is_sticking_top = dist_to_top >= tolerance and dist_to_top >= -50
        is_sticking_bottom = dist_to_bottom >= tolerance and dist_to_bottom >= -50
        is_sticking_left = dist_to_left >= tolerance and dist_to_left >= -50
        is_sticking_right = dist_to_right >= tolerance and dist_to_right >= -50
        print(dist_to_top)
        print(dist_to_bottom)
        print(dist_to_left)
        print(dist_to_right)

        print(is_sticking_top or is_sticking_bottom or is_sticking_left or is_sticking_right)
        return is_sticking_top or is_sticking_bottom or is_sticking_left or is_sticking_right

    def extract_and_compare_feature(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        car_image = frame[y1:y2, x1:x2]
        current_frame_feature = self.features_manager.extract_features(image_list=[car_image])[0]

        if len(self.car_features) == 0:
            self.car_features.append(current_frame_feature)
            return

        last_feature = self.car_features[-1]
        sim = self.features_manager.calculate_similarity(feature_vector_1=current_frame_feature,
                                                         feature_vector_2=last_feature)
        print(sim)

        if sim <= Config.EXTRACT_SIM_THRESHOLD:
            self.car_features.append(current_frame_feature)

    def process_video(self):
        self.get_roi()
        SHOW_VIDEO = True  # True - показывать на экране, False - только запись в файл

        if not SHOW_VIDEO:
            # Инициализация VideoWriter для записи в файл
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.TEMP_output_path, fourcc, 30, (width, height))

            print(f"Запись видео в файл: {self.TEMP_output_path}")

        # ===============================
        # Основной цикл
        # ===============================
        frame_id = 0
        frame_counter_for_fps = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_counter_for_fps != 1:
                frame_counter_for_fps = 1
                continue
            frame_counter_for_fps = 0

            frame = cv2.resize(frame, (1280, 720))

            # КЛЮЧЕВОЙ МОМЕНТ: вырезаем область по bounding box полигона и применяем маску
            # masked_cropped - изображение, которое пойдёт в YOLO (только область полигона)
            # offset - смещение для преобразования координат обратно
            masked_cropped, offset = self.crop_and_mask_roi(frame)

            # Создаём копию для визуализации (рисуем на оригинальном кадре)
            display_frame = frame.copy()
            self.draw_roi(display_frame)

            # ---------------------------
            # YOLO + ByteTrack
            # В YOLO передаётся masked_cropped - вырезанная область с маской
            # Модель анализирует только то, что внутри полигона
            # ---------------------------
            results = self.model.track(
                masked_cropped,  # <- вырезанная область с маской
                persist=True,
                tracker=Config.TRIGGER_TRACK_CONFIG_PATH,
                conf=Config.TRIGGER_YOLO_TRACK_CONF_THRESHOLD,
                iou=Config.TRIGGER_YOLO_TRACK_IOU_THRESHOLD,
                verbose=False
            )

            r = results[0]

            if r.boxes is None or r.boxes.id is None:
                if SHOW_VIDEO:
                    cv2.imshow("Tracking", display_frame)
                else:
                    out.write(display_frame)

                if cv2.waitKey(1) == 27:
                    break
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)

            # ---------------------------
            # Обработка треков
            # ---------------------------
            box, track_id = self.select_detection(boxes, ids)
            if box is None or track_id is None:
                if SHOW_VIDEO:
                    cv2.imshow("Tracking", display_frame)
                else:
                    out.write(display_frame)

                if cv2.waitKey(1) == 27:
                    break
                print("CONTINUE")
                continue

            # Преобразуем координаты бокса из вырезанного изображения обратно в оригинальное
            original_box = self.transform_box_to_original(box, offset)

            # Извлекаем признаки из оригинального кадра с применением ROI
            x1, y1, x2, y2 = map(int, original_box)
            car_image = frame[y1:y2, x1:x2]

            self.extract_and_compare_feature(frame=car_image, box=box)

            # Рисуем bounding box на отображаемом кадре
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display_frame,
                f"ID = {track_id}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            if SHOW_VIDEO:
                cv2.imshow("Tracking", display_frame)
            else:
                out.write(display_frame)

            if SHOW_VIDEO and cv2.waitKey(1) == 27:
                break

            frame_id += 1

        self.cap.release()
        if not SHOW_VIDEO:
            out.release()
            print(f"Видео сохранено в: {self.TEMP_output_path}")
            print(f"Обработано кадров: {frame_id}")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    fp = FeaturesExtractProcess(side_cam_stream="../videos/moskva/geely/vezd.mp4")
    fp.process_video()

    print(len(fp.car_features))

# db_manager = DatabaseManager(
#     database="parking_db",
#     user="postgres",
#     password="postgres",
#     host="127.0.0.1",
#     port="5432"
# )
# db_manager.connect()
# repo = CarFeatureRepository(db_manager=db_manager)
# repo.add_batch_features(car_id=21, feature_vectors=fp.car_features)
