# FeatureZoneModule

Модуль собирает Re-ID gallery для локального трека в зоне съёма признаков. Он использует
публичный интерфейс FeaturesManager: extract_features(image_list) и calculate_similarity(...);
сам FeaturesManager не меняется.

## Поведение

- Каждые N кадров рассматриваются bbox активных треков.
- Зона проверяется по нижней центральной точке bbox и полигону из feature_zones.yaml.
- Кропы всех подходящих треков текущего кадра и их cv2.flip(crop, 1) передаются одним batch-вызовом.
- Фильтрация сравнивает только новый ORIGINAL embedding с последним принятым ORIGINAL embedding.
  При cosine similarity >= 0.90 обе версии пропускаются; при меньшем сходстве оригинал и flip
  добавляются в разные массивы. Первый образец принимается без сравнения.
- Вектора L2-нормализуются. Состояние разделено по (camera_id, local_track_id).

## Зоны и шлагбаумы

feature_zones.yaml содержит пустые полигоны-заглушки: реальные координаты нельзя корректно
угадать по линиям топологии. Пока полигон пустой, профиль выключен.

Для камер 2 и 3 нужен entry_gate="LEFT" или entry_gate="RIGHT", сохранённый при регистрации
на камере 1 и переданный как контекст автомобиля. Модуль не угадывает сторону по локальному
треку камер 2/3. Камеры 1 и 4–9 используют DEFAULT.

## Жизненный цикл

Когда CrossingDetector выдаёт событие EXIT для текущей камеры, вызывающий код передаёт
события в end_tracks_on_exit(camera_id, events). Галерея соответствующего локального трека
закрывается и возвращается вызывающему коду для привязки к регистрации/сохранения. Это
завершает сбор на исходной камере, но не подтверждает идентичность трека на целевой камере.
При остановке потока можно вызвать close_all().

Пример:

    completed = feature_zones.end_tracks_on_exit(CAMERA_ID, events)
    for local_track_id, gallery in completed.items():
        persist_gallery(gallery)

## Использование

    from FeatureZoneModule import FeatureZoneManager
    from FeaturesModule.FeaturesManager import FeaturesManager
    from Config import Config

    features = FeaturesManager(Config.OSNET_MODEL_PATH)
    zones = FeatureZoneManager(features, "feature_zones.yaml")
    accepted = zones.process_frame(
        camera_id=2, frame_id=frame_id, frame=frame,
        tracks=list(zip(track_ids, boxes)), entry_gate=registration_entry_gate,
    )

registration_entry_gate — persistent LEFT/RIGHT атрибут регистрации, а не то, что нужно
заново вычислять на камерах 2 и 3.

## Проверки

Запустить python -m unittest discover -s tests -p test_feature_zone.py. Тесты используют fake extractor и не требуют
весов OSNet. Нужны зависимости проекта OpenCV, NumPy и PyYAML.


## Диагностический запуск без БД

Из корня репозитория после заполнения полигонов в feature_zones.yaml:

    python test_feature_zones.py --video ../step_1/test_videos/1/side.ts --camera-id 1 --show

Для камеры 2 или 3 укажи сторону, которая сохранена в контексте регистрации:

    python test_feature_zones.py --video path/to/video.mp4 --camera-id 2 --entry-gate LEFT --show

Без --show скрипт работает без окна. Можно ограничить прогон, например, первыми 500 кадрами:

    python test_feature_zones.py --video path/to/video.mp4 --camera-id 1 --max-frames 500

Скрипт не подключается к БД и ничего туда не записывает. Он создаёт feature_zone_test.log
и feature_zone_test_report.json. В логе есть FEATURE_ACCEPT, FEATURE_REJECT_SIMILAR,
TRANSITION и GALLERY_CLOSED. Если полигон пустой, он явно предупредит, что съём отключён.
Камера и трек визуализируются для отладки, но маски/полигоны зон пока не рисуются поверх кадра.
