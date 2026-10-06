import cv2
from ultralytics import YOLO

from Config import Config


class TriggerProcess:
    def __init__(self, main_cam_stream):
        self.TEMP_output_path = "output_test_trigger.mp4"

        self.model = YOLO(Config.YOLO_CAR_MODEL_PATH)
        self.cap = cv2.VideoCapture(main_cam_stream)

        self.running = True
        self.last_frame_id = 0

        self.roi_x1 = None
        self.roi_x2 = None
        self.roi_y1 = None
        self.roi_y2 = None

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
        masked_image[:self.roi_y1, :] = 0  # Верхняя часть
        masked_image[self.roi_y2:, :] = 0  # Нижняя часть
        masked_image[:, :self.roi_x1] = 0  # Левая часть
        masked_image[:, self.roi_x2:] = 0  # Правая часть
        return masked_image

    def run_trigger(self):
        self.get_roi()
        SHOW_VIDEO = True  # True - показывать на экране, False - только запись в файл
        out = None

        if not SHOW_VIDEO:
            # Инициализация VideoWriter для записи в файл
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # кодек для MP4
            fps = 30  # примерный FPS (половина от исходного, т.к. пропускаем каждый второй кадр)
            out = cv2.VideoWriter(self.TEMP_output_path, fourcc, fps, (width, height))

            print(f"Запись видео в файл: {self.TEMP_output_path}")

        # ===============================
        # Основной цикл
        # ===============================

        frame_id = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))
            masked_frame = self.apply_roi_to_image(frame)  # Применение ROI к кадру
            H, W = frame.shape[:2]

            frame_id += 1
            if frame_id - self.last_frame_id <= Config.TRIGGER_FRAMES_FOR_DETECT:
                continue
            self.last_frame_id = frame_id

            # ---------------------------
            # YOLO + ByteTrack
            # ---------------------------
            results = self.model.track(
                masked_frame,
                # classes=[2],
                persist=True,
                # tracker=Config.TRIGGER_TRACK_CONFIG_PATH,
                conf=Config.TRIGGER_YOLO_TRACK_CONF_THRESHOLD,
                iou=Config.TRIGGER_YOLO_TRACK_IOU_THRESHOLD,
                verbose=False
            )

            r = results[0]

            if r.boxes is None or r.boxes.id is None:
                if SHOW_VIDEO:
                    cv2.imshow("Tracking", frame)
                else:
                    out.write(frame)

                if SHOW_VIDEO and cv2.waitKey(1) == 27:
                    break
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)

            # ---------------------------
            # Обработка треков
            # ---------------------------
            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)

                if area >= Config.TRIGGER_TRACK_AREA_THRESHOLD:
                    self.cap.release()
                    return True

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (self.roi_x1, self.roi_y1), (self.roi_x2, self.roi_y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"ID = {track_id}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            if SHOW_VIDEO:
                cv2.imshow("Tracking", frame)
            else:
                # Записываем в файл
                out.write(frame)

            if SHOW_VIDEO and cv2.waitKey(1) == 27:
                break

            frame_id += 1

        self.cap.release()
        if not SHOW_VIDEO:
            out.release()  # Важно: закрываем файл записи
            print(f"Видео сохранено в: {self.TEMP_output_path}")
            print(f"Обработано кадров: {frame_id}")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    mt = TriggerProcess(main_cam_stream="../videos/short_front_left.mp4")
    mt.run_trigger()
