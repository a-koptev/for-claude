from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from data import (
    SUPPORTED_SUBTYPES,
    AspectBucketBatchSampler,
    PlateDataset,
    collate_plate_batch,
    load_dataset_split,
)
from parseq_gost_ocr import ModelConfig, PARSeqGostOCR


SAMPLES = {
    "type1": "A123BC77",
    "type1a": "T706HX799",
    "type1b": "AB12377",
    "type9": "001CD177",
    "type10": "001D01377",
    "AM": "01UA070",
    "BY": "0001AA6",
    "KG": "S7881AI",
    "KZ": "001ABD02",
}

FOLDERS = {
    "type1": "type1",
    "type1a": "type1a",
    "type1b": "type1b",
    "type9": "type9",
    "type10": "type10",
    "AM": "Am",
    "BY": "By",
    "KG": "Kg",
    "KZ": "Kz",
}

MIXED = {"type1a", "type1b", "type9", "type10"}


def write_csv(path: Path, image_rel: str, label: str, subtype: str) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["image", "plate_num", "subtype"],
            delimiter=";",
        )
        w.writeheader()
        w.writerow({
            "image": image_rel,
            "plate_num": label,
            "subtype": subtype,
        })


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        for subtype, label in SAMPLES.items():
            folder = FOLDERS[subtype]

            if subtype in MIXED:
                for source in ("real", "synthetic"):
                    d = root / folder / "train" / source
                    (d / "images").mkdir(parents=True)
                    name = f"{subtype}_{source}.jpg"
                    width = 180 if subtype == "type1a" else 400
                    Image.new("RGB", (width, 100), "white").save(d / "images" / name)
                    write_csv(d / "train.csv", f"images/{name}", label, subtype)

                d = root / folder / "val"
                (d / "images").mkdir(parents=True)
                name = f"{subtype}_val.jpg"
                width = 180 if subtype == "type1a" else 400
                Image.new("RGB", (width, 100), "white").save(d / "images" / name)
                write_csv(d / "val.csv", f"images/{name}", label, subtype)

            else:
                for split in ("train", "val"):
                    d = root / folder / split
                    (d / "img").mkdir(parents=True)
                    name = f"{subtype}_{split}.jpg"
                    Image.new("RGB", (400, 100), "white").save(d / "img" / name)
                    # User's flat val folders may also contain train.csv.
                    write_csv(d / "train.csv", f"img/{name}", label, subtype)

        train_records, train_report = load_dataset_split(root, "train")
        val_records, val_report = load_dataset_split(root, "val")

        expected_train = 5 + 4 * 2
        expected_val = 9
        assert len(train_records) == expected_train, (len(train_records), train_report)
        assert len(val_records) == expected_val, (len(val_records), val_report)

        ds = PlateDataset(train_records, augment=False)
        sampler = AspectBucketBatchSampler(ds, batch_size=3, shuffle=False)
        loader = DataLoader(
            ds,
            batch_sampler=sampler,
            collate_fn=collate_plate_batch,
        )

        cfg = ModelConfig(
            dim=128,
            encoder_depth=1,
            decoder_depth=1,
            heads=4,
            subtypes=tuple(SUPPORTED_SUBTYPES),
            max_chars=9,
        )
        model = PARSeqGostOCR(cfg)

        for batch in loader:
            losses = model.compute_loss(
                batch["images"],
                batch["texts"],
                batch["subtypes"],
                num_permutations=1,
            )
            assert torch.isfinite(losses["loss"]), losses

        print("SUPPORTED:", SUPPORTED_SUBTYPES)
        print("train:", [(r.subtype, r.text) for r in train_records])
        print("val:", [(r.subtype, r.text) for r in val_records])
        print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
