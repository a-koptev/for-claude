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


## Запуск из PyCharm без аргументов

Открой в PyCharm и запусти зелёной кнопкой Run файл
[`run_feature_zone_debug.py`](https://github.com/a-koptev/for-claude/blob/feature-zone/run_feature_zone_debug.py).
Аргументы командной строки не нужны. В начале файла есть блок `SETTINGS`:

- `CAMERA_ID` — номер камеры;
- `VIDEO_SOURCE` — путь к видео, индекс камеры вроде `"0"` или RTSP URL;
- `ENTRY_GATE` — `"LEFT"` / `"RIGHT"` для камер 2 и 3;
- `RUN_OSNET_FEATURES` — включить/выключить извлечение признаков.

Если путь к видео не найден, откроется диалог выбора файла. Скрипт не подключается к БД и ничего туда не записывает.

### Управление в окне

- **E** — редактирование полигона зоны признаков: кликай вершины по порядку;
- **B** — режим разметки двух линий шлагбаума на камере 1;
- **1 / 2** — выбрать левую / правую линию шлагбаума (в режиме B) либо профиль LEFT / RIGHT для камеры 2/3;
- **D** — профиль DEFAULT;
- **S** — сохранить разметку;
- **C** — очистить редактируемый полигон или выбранную линию;
- **Space** — пауза, **Q** или **Esc** — выход.

Линии левого и правого шлагбаума сохраняются в отдельный `barrier_lines.yaml` и пока служат для визуальной проверки/разметки: сами по себе они ещё не присваивают регистрации `entry_gate`. Полигоны признаков сохраняются в `feature_zones.yaml`. При пустом полигоне извлечение признаков для этой зоны отключено.

В окне отображаются bbox/ID треков, точка проверки зоны (нижняя середина bbox), полигоны, линии перехода из `topology.yaml` и счётчики принятых/отброшенных признаков. Подробности пишутся в `feature_zone_debug.log`, сводка — в `feature_zone_debug_report.json`.

Старый скрипт `test_feature_zones.py` с аргументами командной строки также оставлен для пакетных диагностических запусков.
