from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from ocr_plate_number_soft.ocr_pipeline import OCRPipeline


@dataclass
class PlateReading:
    text: str
    ocr_confidence: float
    plate_confidence: float
    subtype: str


class PlateOCRTracker:
    """Plate detector + PARSeq OCR for vehicle track crops."""

    def __init__(
        self,
        plate_model_path: str | Path,
        ocr_checkpoint_path: str | Path,
        plate_conf: float = 0.50,
        plate_iou: float = 0.50,
        plate_imgsz: int = 640,
        ocr_device: str = "auto",
        ocr_batch_size: int = 32,
    ) -> None:
        plate_model_path = Path(plate_model_path)
        ocr_checkpoint_path = Path(ocr_checkpoint_path)

        if not plate_model_path.is_file():
            raise FileNotFoundError(
                f"Plate YOLO model not found: {plate_model_path}. "
                "Set PARKING_PLATE_YOLO_MODEL_PATH."
            )
        if not ocr_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"OCR checkpoint not found: {ocr_checkpoint_path}. "
                "Set PARKING_OCR_MODEL_PATH."
            )

        self.plate_model = YOLO(str(plate_model_path))
        self.ocr = OCRPipeline(
            ocr_checkpoint_path,
            device=ocr_device,
            batch_size=ocr_batch_size,
        )
        self.plate_conf = float(plate_conf)
        self.plate_iou = float(plate_iou)
        self.plate_imgsz = int(plate_imgsz)

    @staticmethod
    def _crop_vehicle(frame: np.ndarray, box: Sequence[float]) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size else None

    def recognize(
        self,
        frame: np.ndarray,
        tracks: Sequence[Tuple[int, Sequence[float]]],
    ) -> Dict[int, PlateReading]:
        """Detect a plate inside each vehicle crop, then run OCR on the best plate."""
        if not tracks:
            return {}

        vehicle_crops: List[np.ndarray] = []
        track_ids: List[int] = []

        for track_id, box in tracks:
            crop = self._crop_vehicle(frame, box)
            if crop is not None:
                vehicle_crops.append(crop)
                track_ids.append(int(track_id))

        if not vehicle_crops:
            return {}

        yolo_results = self.plate_model.predict(
            source=vehicle_crops,
            imgsz=self.plate_imgsz,
            conf=self.plate_conf,
            iou=self.plate_iou,
            verbose=False,
        )

        plate_images: List[Image.Image] = []
        plate_meta: List[Tuple[int, float]] = []

        for track_id, vehicle_crop, result in zip(
            track_ids, vehicle_crops, yolo_results
        ):
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            best_idx = int(np.argmax(confs))
            x1, y1, x2, y2 = boxes[best_idx]
            x1, y1, x2, y2 = (
                max(0, int(round(x1))),
                max(0, int(round(y1))),
                min(vehicle_crop.shape[1], int(round(x2))),
                min(vehicle_crop.shape[0], int(round(y2))),
            )

            if x2 <= x1 or y2 <= y1:
                continue

            plate = vehicle_crop[y1:y2, x1:x2]
            if plate.size == 0:
                continue

            rgb = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
            plate_images.append(Image.fromarray(rgb))
            plate_meta.append((track_id, float(confs[best_idx])))

        if not plate_images:
            return {}

        predictions = self.ocr.recognize_images(plate_images)
        readings: Dict[int, PlateReading] = {}

        for (track_id, plate_conf), pred in zip(plate_meta, predictions):
            text = str(pred.get("text", "")).strip()
            if not text:
                continue

            readings[track_id] = PlateReading(
                text=text,
                ocr_confidence=float(pred.get("char_confidence", 0.0)),
                plate_confidence=plate_conf,
                subtype=str(pred.get("subtype", "")),
            )

        return readings
