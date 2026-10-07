from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from PIL import Image

try:
    from .checkpoint import load_checkpoint
except ImportError:
    from checkpoint import load_checkpoint
from data import AspectBucketizer, letterbox_to_bucket


class OCRPipeline:
    """
    Complete crop -> aspect bucket -> model -> subtype -> GOST-constrained OCR pipeline.
    Input images must already be cropped license plates.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "auto",
        batch_size: int = 32,
        invert: bool = True,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, self.ckpt = load_checkpoint(checkpoint, self.device)
        self.model.eval()
        self.batch_size = int(batch_size)
        self.bucketizer = AspectBucketizer()
        self.invert = invert

    @torch.inference_mode()
    def recognize_images(
        self,
        images: Sequence[Image.Image],
        force_subtypes: Optional[Sequence[str]] = None,
    ) -> List[dict]:
        groups: Dict[int, List[int]] = defaultdict(list)
        for i, img in enumerate(images):
            groups[self.bucketizer.choose_id(img.width, img.height)].append(i)

        results: List[Optional[dict]] = [None] * len(images)
        for bid, indices in groups.items():
            bucket = self.bucketizer.buckets[bid]
            for start in range(0, len(indices), self.batch_size):
                ids = indices[start:start + self.batch_size]
                batch = torch.stack([
                    letterbox_to_bucket(images[i], bucket, invert=self.invert)
                    for i in ids
                ]).to(self.device)
                forced = None
                if force_subtypes is not None:
                    forced = [force_subtypes[i] for i in ids]
                pred = self.model.recognize(batch, force_subtypes=forced)
                for i, p in zip(ids, pred):
                    p = dict(p)
                    p["bucket"] = bucket.name
                    results[i] = p
        return [x for x in results if x is not None]

    @torch.inference_mode()
    def recognize_paths(
        self,
        paths: Sequence[str | Path],
        force_subtypes: Optional[Sequence[str]] = None,
    ) -> List[dict]:
        pil_images: List[Image.Image] = []
        names: List[str] = []
        for p in paths:
            p = Path(p)
            with Image.open(p) as im:
                pil_images.append(im.convert("RGB"))
            names.append(str(p))
        out = self.recognize_images(pil_images, force_subtypes=force_subtypes)
        for path, item in zip(names, out):
            item["path"] = path
        return out
