import cv2
from ultralytics import YOLO

from Config import Config
from DB.database_manager import DatabaseManager
from DB.features.feature_repository import CarFeatureRepository
from FeaturesModule.FeaturesManager import FeaturesManager


class FeaturesExtractProcess:
    def __init__(self, side_cam_stream):
        self.TEMP_output_path = "sboku.mp4"

        self.model = YOLO(Config.YOLO_CAR_MODEL_PATH)
        self.cap = cv2.VideoCapture(side_cam_stream)

        self.running = True

        # Прямоугольный ROI (x_min, y_min, x_max, y_max)
        self.roi = None

        self.features_manager = FeaturesManager(Config.OSNET_MODEL_PATH)
        self.car_features = []

    def get_roi(self):
        """Определение прямоугольного ROI по двум точкам"""
        self.roi = (10, 100, 1270, 670)  # (x_min, y_min, x_max, y_max)
        print(f"ROI: {self.roi}")

    def draw_roi(self, image):
        """Отрисовка прямоугольника ROI на изображении"""
        if self.roi is not None:
            x_min, y_min, x_max, y_max = self.roi
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    @staticmethod
    def transform_box_to_original(box, offset):
        """
        Преобразует координаты бокса из вырезанного ROI обратно в оригинальное изображение
        offset: (x_min, y_min) - смещение ROI
        """
        x1, y1, x2, y2 = box
        offset_x, offset_y = offset
        return [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]

    def bbox_is_inside_roi(self, bbox):
        """Проверяет, находится ли bbox полностью внутри ROI"""
        if self.roi is None:
            return True

        toler = 15

        roi_x_min, roi_y_min, roi_x_max, roi_y_max = self.roi
        x1, y1, x2, y2 = map(int, bbox)

        return (x1 >= roi_x_min + toler and y1 >= roi_y_min + toler and
                x2 <= roi_x_max - toler and y2 <= roi_y_max - toler)

    def select_detection(self, boxes, ids):
        """Выбор лучшего детекта (с максимальной площадью)"""
        if len(boxes) == 0:
            return None, None

        max_area = 0
        max_box = None
        max_id = -1

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)

            if area > max_area and area >= Config.TRIGGER_TRACK_AREA_THRESHOLD:
                max_area = area
                max_box = box
                max_id = track_id

        # if max_box and self.bbox_is_inside_roi(max_box):
        if max_box:
            return max_box, max_id
        else:
            return None, None

    def extract_and_compare_feature(self, frame, box):
        """Извлечение признаков"""
        x1, y1, x2, y2 = map(int, box)
        car_image = frame[y1:y2, x1:x2]

        current_frame_feature = self.features_manager.extract_features(image_list=[car_image])[0]

        if len(self.car_features) == 0:
            self.car_features.append(current_frame_feature)
            return

        last_feature = self.car_features[-1]
        sim = self.features_manager.calculate_similarity(feature_vector_1=current_frame_feature,
                                                         feature_vector_2=last_feature)

        if sim <= Config.EXTRACT_SIM_THRESHOLD:
            self.car_features.append(current_frame_feature)

    def process_video(self):
        self.get_roi()
        SHOW_VIDEO = True
        if not SHOW_VIDEO:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.TEMP_output_path, fourcc, 10, (1280, 720))
            print(f"Запись видео в файл: {self.TEMP_output_path}")

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
            # frame = cv2.flip(frame, 1)

            # Получаем ROI координаты
            x_min, y_min, x_max, y_max = self.roi

            # Проверяем, что ROI не выходит за границы кадра
            if y_min >= y_max or x_min >= x_max:
                print(f"Ошибка: некорректный ROI: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
                break

            # Вырезаем ROI - ВНИМАНИЕ: правильно (y, x) а не (x, y)!
            roi_frame = frame[y_min:y_max, x_min:x_max]

            # Проверяем, что ROI не пустой
            if roi_frame.size == 0:
                print(f"Пустой ROI: размер {roi_frame.shape if hasattr(roi_frame, 'shape') else 'unknown'}")
                continue

            # Для отображения используем оригинальный кадр
            display_frame = frame.copy()
            self.draw_roi(display_frame)

            # YOLO обрабатывает только вырезанный ROI
            results = self.model.track(
                roi_frame,  # <- только вырезанная область
                persist=True,
                tracker=Config.YOLO_TRACK_CONFIG_PATH,
                conf=Config.YOLO_TRACK_CONF_THRESHOLD,
                iou=Config.YOLO_TRACK_IOU_THRESHOLD,
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

            # Преобразуем координаты боксов из ROI обратно в оригинальное изображение
            original_boxes = []
            for box in boxes:
                original_box = self.transform_box_to_original(box, (x_min, y_min))
                original_boxes.append(original_box)

            # Выбираем лучший детект
            box, track_id = self.select_detection(original_boxes, ids)
            if box is None or track_id is None:
                if SHOW_VIDEO:
                    cv2.imshow("Tracking", display_frame)
                else:
                    out.write(display_frame)

                if cv2.waitKey(1) == 27:
                    break
                continue

            # Извлекаем признаки из оригинального кадра
            self.extract_and_compare_feature(frame=frame, box=box)

            # Рисуем bounding box на отображаемом кадре
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(
            #     display_frame,
            #     f"ID = {track_id}",
            #     (x1, y1 - 5),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.6,
            #     (0, 255, 0),
            #     2
            # )

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
    fp = FeaturesExtractProcess(side_cam_stream="../../step_1/test_videos/2/sboku.mp4")
    fp.process_video()

    print(f"Всего собрано признаков: {len(fp.car_features)}")

    db_manager = DatabaseManager(**Config.DB_PARAMS)
    db_manager.connect()
    repo = CarFeatureRepository(db_manager=db_manager)
    # repo.add_batch_features(car_id=22, feature_vectors=fp.car_features)
