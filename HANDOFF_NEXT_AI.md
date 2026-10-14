# Handoff for the next AI session

## Project goal

This repository implements vehicle tracking and cross-camera re-identification for a parking system with nine cameras. Keep three concepts separate:

- **Local track**: BoT-SORT track ID within one camera only. A numeric ID is not a global vehicle identity.
- **Gallery**: a set of Re-ID embeddings collected for one local track.
- **Registration/session**: the logical vehicle entering the parking facility. Eventually it must be linked to local tracks across cameras using evidence, not by assuming equal track IDs.

Read this document first, then read `README_FEATURE_CAPTURE_RUN.md`, `README_FEATURE_ZONE.md`, `FEATURE_CAPTURE_PREPRODUCTION.md`, `TRACKING.md`, and `TOPOLOGY.md`. Those files have different scopes: the README files describe current feature-capture behavior; TRACKING.md mainly describes the intended future architecture.

## Current branch and merge status

- Working branch: `feature-zone`.
- Repository default branch: `reid_and_tracking`.
- **There is currently no branch named `main` on GitHub.** Do not assume that `main` exists. Ask whether to merge into `reid_and_tracking` or create a `main` branch before performing the merge.
- The feature branch is 70 commits ahead of `reid_and_tracking` and 0 commits behind it (at the last comparison). Review the whole branch diff before merging, not just the latest runner change.
- No real database writes are performed by the feature-capture runner.

## Current feature-capture implementation

Run from the repository root:

```bash
python run_feature_capture.py
```

Main files:
- `run_feature_capture.py`: trigger-driven local-video runner, OCR voting, topology EXIT handling, concise log and JSON report.
- `FeatureZoneModule/FeatureZoneManager.py`: zone-triggered gallery activation and embedding sampling.
- `FeatureZoneModule/EntrySessionManager.py`: preproduction session/queue coordinator; not wired into the runner yet.
- `feature_zones.yaml`: camera 1–3 activation polygons, sample interval, similarity threshold.
- `topology.yaml`: nine-camera transition topology.
- `Config.py`: model paths, video paths, OCR and tracker configuration.

### Runtime behavior

1. Before the trigger, only camera 1's video source is opened/read and its vehicle YOLO tracker runs.
2. Camera 1's bbox anchor entering LEFT or RIGHT zone activates a local gallery. Anchors: camera 1 bottom-center; camera 2 right-edge center; camera 3 top-center.
3. Feature sampling is every 3 frames by default. The original crop and horizontal flip are sent in one extractor batch; both outputs are L2-normalized and stored separately.
4. The new original embedding is compared only with the last accepted original embedding for the same camera-local track. If cosine similarity is at least 0.90, that original/flip pair is rejected as a near-duplicate.
5. Once camera 1 triggers, camera 2 and camera 3 videos and their vehicle YOLO models are opened lazily. They seek to the next frame index that camera 1 will read, then each camera is advanced in the sequential display loop.
6. This is **approximate frame-index alignment**, not timestamp synchronization. It assumes clips have compatible FPS and aligned starts; seeking accuracy depends on the codec/keyframes. For exact synchronization, use timestamps or precomputed frame alignment.
7. A topology EXIT event closes that camera-local gallery. The runner logs a short line such as `track=2:4 plate=... features=...`. Here `2:4` means camera 2, local track ID 4.
8. A camera-2 EXIT produces a mock record with `MOCK-CAR-####`. It is not a database record and is not guaranteed to represent all features for one physical vehicle.
9. OCR votes are grouped by `(camera_id, track_id)`; the selected plate is the variant with the highest mean confidence, with vote count as tie-breaker. OCR confidence is a heuristic, not proof that the plate is correct.
10. Outputs: `feature_capture_run.log` and `feature_capture_run_report.json`.

### Important startup limitation

Although camera 2/3 video sources and their YOLO models are lazy-loaded, `FeaturesManager` (OSNet/Re-ID) and `PlateOCRTracker` (plate detector + PARSeq) are still initialized **before** the trigger wait loop. The log therefore prints model-loading messages before “Waiting for camera 1 zone trigger.” This is a known reason the process can take a while to reach the waiting state. To reduce that startup time, separate lightweight zone-trigger evaluation from feature extraction, then initialize Re-ID/OCR after the trigger. Do not simply set the managers' extractor to null while still calling `process_frame`: that method performs extraction when a track activates.

## What the preproduction building blocks do not yet do

`EntrySessionManager` is not integrated into `run_feature_capture.py`. The runner does not yet:
- create a session per confirmed camera-1 entrant and associate camera 1/2/3 galleries to that session;
- distinguish a registered entrant from an interloper before FIFO association;
- use lost-track grace periods to finalize each local gallery safely;
- wait for independent OCR and persistence tasks;
- persist features to PostgreSQL or another database;
- run bounded asynchronous feature-extraction workers.

The current runner is a functional capture/debug stage, not the finished production registration pipeline.

## Documentation audit / known discrepancies

- `README_FEATURE_CAPTURE_RUN.md` previously said camera inputs were read in parallel while waiting for the trigger. This was incorrect and has been corrected: only camera 1 is read before the trigger.
- `TRACKING.md` describes the target architecture, including registration by a line event. The current feature runner instead uses feature-zone polygons as a preproduction trigger. A scope note has been added to avoid treating the target design as implemented behavior.
- Secondary-camera frame alignment had an off-by-one error: the trigger frame from camera 1 had already been consumed, so the secondary streams must seek to `frame_ids[1]`, not `frame_ids[1] - 1`. The code and docs have been corrected. Verify this against actual clips; no video-runtime test was run here.
- `topology.yaml` maps EXIT events to peers in the full nine-camera network. The example logs showing `next_camera=9` are consistent with the current topology, even though feature extraction itself is limited to cameras 1–3.
- The five Markdown files are not one specification: `TRACKING.md` is primarily future intent, `TOPOLOGY.md` explains transition detection, and the feature-capture README/preproduction notes describe the current test stage. Keep that distinction explicit when making changes.

## Verification

Documented unit-test commands include:

```bash
python -m unittest discover -s tests -p test_feature_zone.py
python -m unittest discover -s tests -p test_entry_session_manager.py
python check_topology.py
```

The current change has **not** been executed against the user's local video files or model weights in this environment. Before calling it verified, run the tests locally and check:
- the trigger activates on the intended camera-1 zone;
- cameras 2/3 start at the intended timeline position;
- no startup/runtime exceptions occur with the actual models;
- OCR plate voting and EXIT log output behave as expected;
- the JSON report correctly lists unfinished galleries on manual stop/end-of-video.

## Next engineering steps

1. Run tests and validate trigger/frame alignment on the real three clips.
2. Decide whether to move Re-ID/OCR model loading after the trigger to improve time-to-ready.
3. Wire `EntrySessionManager` into the runner without blocking camera-1 trigger evaluation.
4. Add lost-track grace periods and explicit evidence-based camera association; never bind every detected car to FIFO.
5. Implement global vehicle/session identity and database persistence only after cross-camera association rules are clear.
6. Add gallery-wide cosine matching. With L2-normalized embeddings, matrix multiplication gives cosine similarity: for gallery `G` of shape `(N, 512)` and query `q` of shape `(512,)`, compute `G @ q` to get all `N` similarities in one operation. For roughly 400 vectors, this is only about 204,800 multiply-accumulate terms; compute cost is tiny. Decide how to aggregate many embeddings per vehicle (e.g. top-k mean or a representative/prototype vector) so vehicles with more samples do not dominate. Keep similarity (higher is better) distinct from cosine distance `1 - similarity` (lower is better).

## Constraints to preserve

- Feature capture only on cameras 1–3; cameras 4–9 remain part of the wider topology/tracking pipeline.
- Never interpret a local `track_id` as a global car ID.
- Do not merge galleries solely because boxes overlap in space/time.
- OCR failure or latency must not stop feature capture.
- Do not claim a mock record was persisted to the database.
- Keep camera 2/3 startup lazy unless profiling demonstrates a better design.
