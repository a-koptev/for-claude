# Feature capture preproduction: sessions, queue, and concurrency

Branch: `feature-zone`  
Status: implementation building blocks and documentation; integration with the main multi-camera runtime is still pending.

## Decisions

### 1. Zones activate tracks; they do not bound feature sampling

For cameras 1–3, the camera-specific anchor point is tested against configured polygons:

- Camera 1: bottom-center of bbox; `LEFT` / `RIGHT` profiles.
- Camera 2: center of the right bbox edge; `LEFT` / `RIGHT` profiles.
- Camera 3: center of the top bbox edge; `DEFAULT` profile.
- Cameras 4–9: no feature-zone activation and no feature extraction in this module.

When a track's anchor enters a polygon, its local `(camera_id, track_id)` gets an independent gallery. Sampling continues when the same activated track moves outside the polygon, as long as it remains visible and the caller has not closed the gallery. Trigger evaluation is not skipped when feature sampling is frame-throttled.

A new local ID that enters a zone gets a separate gallery. Galleries are not merged based on spatial overlap alone.

### 2. Multiple cars in one frame

Every visible activated track on a sampling frame is cropped. All originals and horizontal flips are sent through one extractor batch, then mapped back to their original local track IDs. A slow extraction job must not disable trigger evaluation for later cars.

The manager itself is synchronous. The eventual runtime should use a bounded worker queue/thread or process pool around feature extraction, selected after measuring OSNet's CPU/GPU execution behavior. Do not start an unbounded OS process per trigger.

### 3. Trigger handling and entry order

The trigger stays enabled while other cars are being sampled. Each confirmed camera-1 entry gets a monotonically increasing `entry_order` and a unique `session_id`. Repeated zone hits for the same known camera-1 track must be idempotent.

Example:

| Event | Entry queue |
|---|---|
| Car A activates at camera 1 | A = order 1 |
| Car B activates while A is still visible | A = 1, B = 2 |
| Non-entering/interloper car C passes between them | Queue remains A = 1, B = 2 |
| Confirmed entrant A reaches camera 2 | Bind camera-2 track to A |
| Confirmed entrant B reaches camera 2 | Bind camera-2 track to B |

FIFO is valid because the site guarantees that registered entrants preserve their order between cameras. However, the code must only advance FIFO for a **confirmed registered entrant**, not every detected vehicle. An interloper must not consume an entry slot. If the caller cannot establish that a camera-2 track is one of the registered entrants, it must not bind that track by FIFO alone.

The coordinator exposes both explicit association (`bind_track`) and guarded FIFO (`bind_next_in_order`). The current local track ID is not itself the cross-camera identity.

### 4. Session completion and OCR

For the first preproduction iteration, the bound camera-2 track leaving the frame is the global session-completion signal. The video runtime must decide that the track has truly left after a suitable lost-track grace period; one missed detector frame must not end registration.

Each camera's local gallery can be finalized when that local track leaves its frame, at a different time from other cameras. The overall entry session remains pending until camera 2 signals completion, then the accumulated session result can be handed to persistence.

License-plate OCR runs independently. OCR success/failure or latency must not terminate feature capture. If the database schema requires a recognized plate to find/create a car row, hold the finished feature result in a pending-persistence state until the association is resolved; do not discard features or stop capture while waiting.

### 5. Persistence and safety boundaries

The coordinator stores session metadata and in-memory gallery payloads only. It does not open database connections, write features, run OCR, or spawn workers. Before DB writes, the runtime must resolve which registered vehicle/session the gallery belongs to and use the existing repository interfaces.

The coordinator and feature manager accept only cameras 1–3 for feature capture. Existing topology transitions involving cameras 4–9 may still be consumed by the wider tracking pipeline, but those cameras must never be passed into feature-zone extraction.

## Current code

- `FeatureZoneManager.py`: zone-triggered activation, ongoing sampling outside the zone, per-track galleries, batched originals/flips.
- `EntrySessionManager.py`: ordered entry sessions, duplicate-trigger guard, guarded FIFO association, camera-2 completion signal.
- `tests/test_feature_zone.py`: includes outside-zone continuation and multi-track batch cases.
- `tests/test_entry_session_manager.py`: queue order, duplicate trigger, interloper, completion, and camera restrictions.
- `README_FEATURE_ZONE.md`: quick module usage and keyboard controls for the polygon debugger.

## Not yet complete

This is not yet wired into the production/preproduction camera loop. The integration still needs to connect:
1. camera-1 zone activation to `register_entry`;
2. each camera's activated local tracks to `FeatureZoneManager.process_frame`;
3. confirmed cross-camera entrants to explicit/FIFO session binding;
4. per-camera track-loss detection with a grace period to gallery finalization;
5. camera-2 track loss to `finish_on_camera2_track_lost`;
6. OCR completion and DB persistence as independent asynchronous work;
7. bounded feature-extraction workers and orderly shutdown.

Tests have been added but have not been executed in the project's runtime environment as part of this change.


## Trigger-driven runner

Added `run_feature_capture.py` as an executable first-stage runner. It displays camera 1 while waiting for a zone activation, then switches to the three-camera capture view. A camera-2 topology EXIT event closes that camera's local gallery and prints a mock persistence record with a generated `MOCK-CAR-####` ID and original/flipped/total embedding counts. It never opens a real database connection.

This is intentionally a mock boundary, not full production persistence. The mock record currently describes the camera-2 local gallery only; cross-camera session/gallery association and confirmation that a camera-2 entrant belongs to the registered-entry queue still need to be wired before treating counts as one physical car's total.


## Стабилизация номера по наблюдениям OCR

В \`run_feature_capture.py\` добавлено накопление OCR-уверенностей отдельно для каждого \`(camera_id, track_id)\`. Номер с максимальной средней уверенностью становится итоговым; число повторений используется как дополнительный критерий при равенстве средних. Нормализация визуально одинаковых кириллических и латинских символов выполняется внутри `PlateOCRModule`; runner использует готовый текст распознавания.

Файл \`feature_capture_run.log\` оставлен для коротких итоговых строк по завершённым трекам: выбранный номер, средняя OCR-уверенность, число original/flipped признаков и EXIT-линия с номером следующей камеры. Подробный поток кадров и промежуточные OCR-ответы в этот лог не записываются.
