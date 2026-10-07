import os
from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).parent

    # Значения по умолчанию оставлены для локальной разработки,
    # на сервере переопределяются переменными окружения PARKING_DB_*
    DB_PARAMS = {
        'database': os.getenv("PARKING_DB_NAME", "parking_db"),
        'user': os.getenv("PARKING_DB_USER", "tracklet_manager"),
        'password': os.getenv("PARKING_DB_PASSWORD", "tracklet_manager"),
        'host': os.getenv("PARKING_DB_HOST", "127.0.0.1"),
        'port': os.getenv("PARKING_DB_PORT", "5432"),
    }

    DEVICE: str = "cpu"

    # Сеть камер и линии перехода (см. TopologyModule, проверка: python check_topology.py)
    TOPOLOGY_CONFIG_PATH = str(BASE_DIR / "topology.yaml")
    # Детектор переходов: событие возникает, когда линия находится
    # между двумя границами bbox при движении автомобиля.
    # Сколько кадров центр должен находиться дальше release_margin_px,
    # чтобы после одного перехода разрешить следующее срабатывание.
    TRANSITION_RELEASE_FRAMES = 5
    # Минимальное смещение центра bbox между соседними кадрами,
    # которое считаем реальным движением, а не дрожанием трека.
    TRANSITION_MIN_MOVE_PX = 2.0
    # Минимальное расстояние центра от линии для снятия блокировки повторного события.
    TRANSITION_RELEASE_MARGIN_PX = 30
    # Через сколько кадров без трека забываем его состояние (не меньше TRACKLET_LOST_TTL_FRAMES)
    TRANSITION_FORGET_FRAMES = 150

    YOLO_TRACK_CONFIG_PATH = str(BASE_DIR / "vehicle_botsort_conf.yaml")
    YOLO_CAR_MODEL_PATH = str(BASE_DIR / "Models/Tracking/YOLO26/1809_epoch0.pt")
    # OSNET_MODEL_PATH = str(BASE_DIR / "Models/ReID/OSNet/model_veri_1.pth")
    OSNET_MODEL_PATH = str(BASE_DIR / "Models/ReID/OSNet/model_new_6.pth")

    # OCR integration: plate detector runs inside the tracked vehicle crop, then PARSeq reads the plate crop.
    YOLO_PLATE_MODEL_PATH = os.getenv(
        "PARKING_PLATE_YOLO_MODEL_PATH",
        str(BASE_DIR / "Models/PlateRecognition/YOLOv8/best.pt"),
    )
    OCR_MODEL_PATH = os.getenv(
        "PARKING_OCR_MODEL_PATH",
        str(BASE_DIR / "ocr_plate_number_soft/best_ocr.pt"),
    )
    OCR_EVERY_N_FRAMES = int(os.getenv("PARKING_OCR_EVERY_N_FRAMES", "5"))
    OCR_PLATE_CONF_THRESHOLD = float(os.getenv("PARKING_PLATE_CONF_THRESHOLD", "0.50"))
    OCR_PLATE_IOU_THRESHOLD = float(os.getenv("PARKING_PLATE_IOU_THRESHOLD", "0.50"))
    OCR_PLATE_IMGSZ = int(os.getenv("PARKING_PLATE_IMGSZ", "640"))
    OCR_DEVICE = os.getenv("PARKING_OCR_DEVICE", "auto")
    OCR_BATCH_SIZE = int(os.getenv("PARKING_OCR_BATCH_SIZE", "32"))
    OCR_MIN_CHAR_CONFIDENCE = float(os.getenv("PARKING_OCR_MIN_CHAR_CONFIDENCE", "0.0"))

    TRACKER_MOVEMENT_ALFA = 0.2
    TRACKER_MOVEMENT_MIN_SIZE = 30
    TRACKER_MOVEMENT_THRESHOLD = 0.004
    TRACKER_MOVEMENT_SIZE_THRESHOLD = 0.005

    YOLO_TRACK_CONF_THRESHOLD = 0.25
    YOLO_TRACK_IOU_THRESHOLD = 0.7

    TRIGGER_FRAMES_FOR_DETECT = 3
    TRIGGER_TRACK_AREA_THRESHOLD = 300 * 300
    # Раньше этих констант не было, а TriggerProcess их использовал.
    # Взяты равными основным порогам трекинга, подберите свои значения.
    TRIGGER_YOLO_TRACK_CONF_THRESHOLD = YOLO_TRACK_CONF_THRESHOLD
    TRIGGER_YOLO_TRACK_IOU_THRESHOLD = YOLO_TRACK_IOU_THRESHOLD

    EXTRACT_FRAMES_FOR_DETECT = 3
    EXTRACT_SIM_THRESHOLD = 0.90

    # --- Matcher ---
    # Как сворачивать матрицу сходств tracklet x галерея в одно число:
    #   "max"       - максимум по всем парам (поведение до правок, самый оптимистичный)
    #   "topk_mean" - среднее по MATCH_SIM_TOPK лучшим парам (консервативнее)
    MATCH_SIM_AGGREGATION = "max"
    MATCH_SIM_TOPK = 5
    # Сколько очков теряет кандидат на машину, если в этом цикле совпадения не было.
    # 1 - мягкое затухание, большое число (например 99) - требуем совпадения строго подряд.
    CANDIDATE_MISS_PENALTY = 1

    # --- Жизненный цикл tracklet'ов ---
    # Через сколько кадров без трека LOST-tracklet закрывается и удаляется
    # (должно быть больше track_buffer трекера, сейчас 30)
    TRACKLET_LOST_TTL_FRAMES = 150
    # Минимальная сторона кропа (px) для извлечения Re-ID признаков
    FEATURE_MIN_CROP_SIZE = 8
