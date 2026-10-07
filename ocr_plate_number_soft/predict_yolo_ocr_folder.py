from __future__ import annotations

"""
YOLO -> crop -> PARSeq-GOST OCR -> annotated images.

Place this file in the same folder as:
    checkpoint.py
    data.py
    ocr_pipeline.py
    parseq_gost_ocr.py

Install:
    pip install ultralytics opencv-python pillow tqdm

The OCR checkpoint must be produced by the current 9-class trainer.
"""

import csv
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from ocr_pipeline import OCRPipeline


# =====================================================================
# SETTINGS — EDIT THESE PATHS
# =====================================================================

YOLO_MODEL = Path(
    r"C:\Users\asmikhalev\Desktop\ocr_plate_number_soft\best_yolo26.pt"
)

OCR_CHECKPOINT = Path(
    r"C:\Users\asmikhalev\Desktop\ocr_plate_number_soft\best_ocr.pt"
)

INPUT_DIR = Path(
    r"C:\Users\asmikhalev\Desktop\vehicle_dataset_builder\dataset_output\6\crop"
)

OUTPUT_DIR = Path(
    r"C:\Users\asmikhalev\Desktop\ocr_plate_number_soft\rez"
)


# =====================================================================
# YOLO SETTINGS
# =====================================================================

YOLO_CONF = 0.75
YOLO_IOU = 0.50
YOLO_IMGSZ = 640

# Number of full images sent to YOLO at once.
YOLO_BATCH_SIZE = 16

# 0 = first CUDA GPU, "cpu" = CPU.
YOLO_DEVICE = 0


# =====================================================================
# OCR SETTINGS
# =====================================================================

OCR_DEVICE = "auto"
OCR_BATCH_SIZE = 128

# Optional: do not draw low-confidence OCR results.
# 0.0 = draw everything.
MIN_OCR_CONFIDENCE = 0.0


# =====================================================================
# OUTPUT SETTINGS
# =====================================================================

RECURSIVE = True
SAVE_CROPS = False
LINE_THICKNESS = 3

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


# =====================================================================
# UNICODE-SAFE IMAGE IO FOR WINDOWS
# =====================================================================

def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        ext = ".jpg"
        path = path.with_suffix(ext)

    encode_ext = ".tif" if ext == ".tiff" else ext

    ok, encoded = cv2.imencode(encode_ext, image)
    if not ok:
        path = path.with_suffix(".jpg")
        ok, encoded = cv2.imencode(".jpg", image)

    if not ok:
        raise ValueError(f"Cannot encode image: {path}")

    encoded.tofile(str(path))
    return path


# =====================================================================
# HELPERS
# =====================================================================

def collect_images(root: Path) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Input folder not found: {root}")

    iterator = root.rglob("*") if RECURSIVE else root.iterdir()

    paths = [
        p for p in iterator
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths.sort()
    return paths


def clip_box(
    xyxy: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy

    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width, int(round(x2))))
    y2 = max(0, min(height, int(round(y2))))

    return x1, y1, x2, y2


def crop_to_pil(image_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = box
    crop = image_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        raise ValueError(f"Empty crop for box {box}")

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def draw_prediction(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    text: str,
) -> None:
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]

    font_scale = max(0.55, min(1.15, w / 1400.0))
    thickness = max(1, int(round(font_scale * 2)))

    (tw, th), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )

    pad = 6

    text_y = y1 - 8
    if text_y - th - baseline - pad < 0:
        text_y = min(h - baseline - pad, y1 + th + baseline + 12)

    bg_x1 = max(0, x1)
    bg_y1 = max(0, text_y - th - baseline - pad)
    bg_x2 = min(w - 1, x1 + tw + 2 * pad)
    bg_y2 = min(h - 1, text_y + baseline + pad)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        LINE_THICKNESS,
    )

    cv2.rectangle(
        image,
        (bg_x1, bg_y1),
        (bg_x2, bg_y2),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        image,
        text,
        (x1 + pad, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    if not YOLO_MODEL.is_file():
        raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL}")

    if not OCR_CHECKPOINT.is_file():
        raise FileNotFoundError(f"OCR checkpoint not found: {OCR_CHECKPOINT}")

    image_paths = collect_images(INPUT_DIR)
    if not image_paths:
        raise RuntimeError(f"No images found in: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annotated_dir = OUTPUT_DIR / "annotated"
    crops_dir = OUTPUT_DIR / "crops"

    print("=" * 80)
    print("YOLO model :", YOLO_MODEL)
    print("OCR model  :", OCR_CHECKPOINT)
    print("Input      :", INPUT_DIR)
    print("Output     :", OUTPUT_DIR)
    print("Images     :", len(image_paths))
    print("=" * 80)

    yolo = YOLO(str(YOLO_MODEL))

    ocr = OCRPipeline(
        OCR_CHECKPOINT,
        device=OCR_DEVICE,
        batch_size=OCR_BATCH_SIZE,
    )

    print("OCR device :", ocr.device)
    if ocr.device.type == "cuda":
        print("GPU        :", torch.cuda.get_device_name(ocr.device))
    print()

    csv_rows = []
    global_detection_id = 0

    for batch_start in tqdm(
        range(0, len(image_paths), YOLO_BATCH_SIZE),
        desc="images",
    ):
        batch_paths = image_paths[
            batch_start:batch_start + YOLO_BATCH_SIZE
        ]

        batch_images = [imread_unicode(path) for path in batch_paths]

        yolo_results = yolo.predict(
            source=batch_images,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            device=YOLO_DEVICE,
            verbose=False,
        )

        ocr_crops: List[Image.Image] = []
        crop_meta = []

        for local_image_idx, (path, image, result) in enumerate(
            zip(batch_paths, batch_images, yolo_results)
        ):
            h, w = image.shape[:2]

            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
            yolo_confs = result.boxes.conf.detach().cpu().numpy()

            for detection_idx, (xyxy, yolo_conf) in enumerate(
                zip(boxes_xyxy, yolo_confs)
            ):
                box = clip_box(tuple(xyxy.tolist()), w, h)
                x1, y1, x2, y2 = box

                if x2 <= x1 or y2 <= y1:
                    continue

                try:
                    pil_crop = crop_to_pil(image, box)
                except ValueError:
                    continue

                ocr_crops.append(pil_crop)
                crop_meta.append({
                    "local_image_idx": local_image_idx,
                    "source_path": path,
                    "box": box,
                    "yolo_confidence": float(yolo_conf),
                    "detection_idx": detection_idx,
                })

        if ocr_crops:
            ocr_results = ocr.recognize_images(ocr_crops)
        else:
            ocr_results = []

        if len(ocr_results) != len(crop_meta):
            raise RuntimeError(
                f"OCR result count mismatch: "
                f"{len(ocr_results)} predictions for {len(crop_meta)} crops"
            )

        for meta, pred, pil_crop in zip(
            crop_meta,
            ocr_results,
            ocr_crops,
        ):
            global_detection_id += 1

            image = batch_images[meta["local_image_idx"]]
            source_path: Path = meta["source_path"]
            box = meta["box"]

            subtype = str(pred["subtype"])
            number = str(pred["text"])

            subtype_conf = float(pred.get("subtype_confidence", 0.0))

            # Mean probability of the autoregressively selected characters.
            # This is the OCR confidence shown in the image label.
            ocr_conf = float(pred.get("char_confidence", 0.0))

            label = (
                f"{subtype}: {number} "
                f"({ocr_conf * 100:.1f}%)"
            )

            if ocr_conf >= MIN_OCR_CONFIDENCE:
                draw_prediction(image, box, label)

            relative = source_path.relative_to(INPUT_DIR)

            if SAVE_CROPS:
                crop_name = (
                    f"{source_path.stem}"
                    f"_plate_{meta['detection_idx'] + 1:02d}.jpg"
                )
                crop_path = crops_dir / relative.parent / crop_name
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                pil_crop.save(crop_path, quality=95)
            else:
                crop_path = None

            x1, y1, x2, y2 = box

            csv_rows.append({
                "image": relative.as_posix(),
                "detection_id": global_detection_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "yolo_confidence": meta["yolo_confidence"],
                "subtype": subtype,
                "number": number,
                "subtype_confidence": subtype_conf,
                "ocr_confidence": ocr_conf,
                "bucket": pred.get("bucket", ""),
                "crop": (
                    crop_path.relative_to(OUTPUT_DIR).as_posix()
                    if crop_path is not None
                    else ""
                ),
            })

        for path, image in zip(batch_paths, batch_images):
            relative = path.relative_to(INPUT_DIR)
            output_path = annotated_dir / relative
            imwrite_unicode(output_path, image)

    csv_path = OUTPUT_DIR / "predictions.csv"

    fieldnames = [
        "image",
        "detection_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "yolo_confidence",
        "subtype",
        "number",
        "subtype_confidence",
        "ocr_confidence",
        "bucket",
        "crop",
    ]

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print()
    print("=" * 80)
    print("DONE")
    print("Images       :", len(image_paths))
    print("Detections   :", len(csv_rows))
    print("Annotated    :", annotated_dir)
    print("CSV          :", csv_path)
    if SAVE_CROPS:
        print("Crops        :", crops_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
