from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
import csv
import json
import math
import random
import re

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, Sampler


# Canonical OCR alphabet:
# keep all labels as uppercase LATIN A-Z + digits.
# Russian Cyrillic look-alikes are converted to their visual Latin equivalents.
_CYR_TO_LAT = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
    "Ё": "E",
})

_VALID_CHARS = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

SUPPORTED_SUBTYPES = (
    "type1",
    "type1a",
    "type1b",
    "type9",
    "type10",
    "AM",
    "BY",
    "KG",
    "KZ",
)

# Folder names on disk. CSV subtype values remain AM/BY/KG/KZ.
SUBTYPE_FOLDERS = {
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

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PlateRecord:
    image_path: Path
    text: str
    subtype: str


@dataclass(frozen=True)
class Bucket:
    height: int
    width: int

    @property
    def ratio(self) -> float:
        return self.width / self.height

    @property
    def name(self) -> str:
        return f"{self.height}x{self.width}"


DEFAULT_BUCKETS: Tuple[Bucket, ...] = (
    # Near-square / two-line plates
    Bucket(64, 80),
    Bucket(64, 96),
    Bucket(64, 112),
    Bucket(56, 112),
    # Intermediate / perspective-distorted crops
    Bucket(48, 128),
    Bucket(40, 144),
    # Standard long Russian plates
    Bucket(36, 152),
    Bucket(32, 160),
)


class AspectBucketizer:
    """Chooses the closest aspect-ratio bucket without stretching the crop."""

    def __init__(self, buckets: Sequence[Bucket] = DEFAULT_BUCKETS):
        if not buckets:
            raise ValueError("At least one bucket is required")
        for b in buckets:
            if b.height % 4 or b.width % 4:
                raise ValueError(f"Bucket {b.name} must be divisible by 4")
        self.buckets = tuple(buckets)

    def choose_id(self, width: int, height: int) -> int:
        ratio = max(width, 1) / max(height, 1)
        # Log distance treats r=2->4 similarly to r=.5->1.
        scores = [abs(math.log(max(ratio, 1e-6) / b.ratio)) for b in self.buckets]
        return int(min(range(len(scores)), key=scores.__getitem__))

    def choose(self, width: int, height: int) -> Bucket:
        return self.buckets[self.choose_id(width, height)]


def normalize_plate_text(text: str) -> str:
    """Normalize every label to uppercase ASCII A-Z + digits.

    This intentionally unifies visually equivalent Russian and Latin glyphs.
    Examples:
        А123ВС77 -> A123BC77
        001CD177 -> 001CD177
        001D01377 -> 001D01377
    """
    raw = str(text).strip().upper().translate(_CYR_TO_LAT)
    # '#' means an unreadable / unknown character in the source annotation.
    # Keep it in the label: it preserves the true plate length and position.
    compact = "".join(ch for ch in raw if ch.isalnum() or ch == "#")

    bad = [c for c in compact if c != "#" and c not in _VALID_CHARS]
    if bad:
        raise ValueError(
            f"Unsupported plate characters {sorted(set(bad))!r} in {text!r}"
        )
    if not compact:
        raise ValueError("Empty plate label")
    return compact


def normalize_subtype(value: object) -> str:
    if value is None:
        return ""

    raw = str(value).strip()
    compact = raw.lower().replace("-", "").replace("_", "").replace(" ", "")

    aliases = {
        "1": "type1", "type1": "type1", "t1": "type1",
        "1a": "type1a", "type1a": "type1a", "t1a": "type1a",
        "1b": "type1b", "type1b": "type1b", "t1b": "type1b",
        "9": "type9", "type9": "type9", "t9": "type9",
        "10": "type10", "type10": "type10", "t10": "type10",
        "am": "AM", "armenia": "AM",
        "by": "BY", "belarus": "BY",
        "kg": "KG", "kyrgyzstan": "KG", "kyrgyz": "KG",
        "kz": "KZ", "kazakhstan": "KZ", "kazakh": "KZ",
    }
    return aliases.get(compact, raw)


# Strict formats for the Russian GOST subtypes.
# Template syntax here matches the OCR grammar:
#   # = digit, R = one of ABEKMHOPCTYX, X = alphanumeric,
#   other symbols are fixed literals.
# In a DATA LABEL, '#' is also accepted as an unknown/unreadable wildcard.
_STRICT_SUBTYPE_TEMPLATES = {
    "type1":  ("R###RR##", "R###RR###"),
    "type1a": ("R###RR##", "R###RR###"),
    "type1b": ("RR#####",),
    "type9":  ("###CD###",),
    "type10": ("###D#####", "###T#####"),
}

_FOREIGN_LENGTHS = {
    "AM": {6, 7},
    "BY": {7},
    "KG": {5, 6, 7, 8},
    "KZ": {6, 7, 8},
}


def _label_matches_template(text: str, template: str) -> bool:
    if len(text) != len(template):
        return False

    gost_letters = set("ABEKMHOPCTYX")
    for value, expected in zip(text, template):
        # Unknown source symbol: accept it in any position.
        if value == "#":
            continue

        if expected == "#":
            if not value.isdigit():
                return False
        elif expected == "R":
            if value not in gost_letters:
                return False
        elif expected == "X":
            if value not in _VALID_CHARS:
                return False
        elif value != expected:
            return False

    return True


def validate_plate_format(text: str, subtype: str) -> None:
    templates = _STRICT_SUBTYPE_TEMPLATES.get(subtype)
    if templates is not None:
        if not any(_label_matches_template(text, template) for template in templates):
            raise ValueError(f"Label {text!r} does not match {subtype} format")
        return

    lengths = _FOREIGN_LENGTHS.get(subtype)
    if lengths is not None:
        if len(text) not in lengths:
            raise ValueError(
                f"Label {text!r} has length {len(text)}; "
                f"{subtype} expects one of {sorted(lengths)}"
            )
        bad = [ch for ch in text if ch != "#" and ch not in _VALID_CHARS]
        if bad:
            raise ValueError(
                f"Label {text!r} contains unsupported symbols: {sorted(set(bad))}"
            )

def _recursive_find(obj: object, keys: Sequence[str]) -> Optional[object]:
    keys_l = {k.lower() for k in keys}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in keys_l and v not in (None, ""):
                return v
        for v in obj.values():
            found = _recursive_find(v, keys)
            if found not in (None, ""):
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _recursive_find(v, keys)
            if found not in (None, ""):
                return found
    return None


def _resolve_image_path(root: Path, raw: object, basename_index: Optional[Dict[str, Path]] = None) -> Optional[Path]:
    if isinstance(raw, dict):
        raw = _recursive_find(raw, ["path", "image_path", "file_name", "filename", "file"])
    if raw is None:
        return None
    p = Path(str(raw).replace("\\", "/"))
    candidates = [
        p if p.is_absolute() else root / p,
        root / "images" / p,
        root / "image" / p,
        root / "imgs" / p,
        root / "renders" / p,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()
    if basename_index is not None:
        return basename_index.get(p.name)
    return None


def _build_basename_index(root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            index.setdefault(p.name, p.resolve())
    return index


def _infer_subtype_from_aspect(path: Path) -> str:
    with Image.open(path) as im:
        ratio = im.width / max(im.height, 1)
    # Type 1A is approximately 1.6-1.8, Type 1 approximately 4.5-4.7.
    return "type1a" if ratio < 2.6 else "type1"


def load_generator_folder(
    root: Path,
    allowed_subtypes: Optional[Sequence[str]] = SUPPORTED_SUBTYPES,
    infer_missing_subtype: bool = True,
) -> List[PlateRecord]:
    """
    Reads russian-license-plate-anpr generator output.

    The official generator writes annotations.jsonl.  The parser intentionally
    accepts several field aliases so that it remains usable if the generator's
    metadata schema changes slightly.
    """
    root = Path(root)
    ann = root / "annotations.jsonl"
    if not ann.exists():
        raise FileNotFoundError(f"annotations.jsonl not found in {root}")

    allowed = {normalize_subtype(x) for x in allowed_subtypes} if allowed_subtypes else None
    basename_index: Optional[Dict[str, Path]] = None
    records: List[PlateRecord] = []

    image_keys = ["image_path", "path", "image", "file_name", "filename", "file", "render"]
    text_keys = ["plate_text", "text", "number", "plate_number", "registration", "reg_number", "label", "value"]
    type_keys = ["subtype", "plate_type", "gost_type", "type", "template", "kind"]

    with ann.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {ann}:{line_no}: {e}") from e

            raw_path = _recursive_find(obj, image_keys)
            path = _resolve_image_path(root, raw_path, basename_index)
            if path is None:
                if basename_index is None:
                    basename_index = _build_basename_index(root)
                    path = _resolve_image_path(root, raw_path, basename_index)
                if path is None:
                    raise FileNotFoundError(
                        f"Cannot resolve image for annotation {line_no}: {raw_path!r}"
                    )

            raw_text = _recursive_find(obj, text_keys)
            if raw_text is None:
                raise ValueError(f"No plate text in annotation {line_no}")
            text = normalize_plate_text(str(raw_text))

            raw_type = _recursive_find(obj, type_keys)
            subtype = normalize_subtype(raw_type)
            if not subtype and infer_missing_subtype:
                subtype = _infer_subtype_from_aspect(path)
            if not subtype:
                subtype = "unknown"

            if allowed is not None and subtype not in allowed:
                continue
            records.append(PlateRecord(path, text, subtype))

    if not records:
        raise ValueError(f"No usable records loaded from {ann}")
    return records


def load_tsv(
    path: Path,
    allowed_subtypes: Optional[Sequence[str]] = SUPPORTED_SUBTYPES,
    infer_missing_subtype: bool = True,
) -> List[PlateRecord]:
    """
    Supports both headered and simple TSV files.

    Recommended format:
        image_path<TAB>text<TAB>subtype
    """
    path = Path(path)
    root = path.parent
    allowed = {normalize_subtype(x) for x in allowed_subtypes} if allowed_subtypes else None
    records: List[PlateRecord] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    rows = [r for r in rows if r and any(c.strip() for c in r)]
    if not rows:
        return []

    header_l = [c.strip().lower() for c in rows[0]]
    has_header = any(x in header_l for x in ("image", "image_path", "path", "filename", "text", "plate_text", "subtype"))

    if has_header:
        header = header_l
        data_rows = rows[1:]
        def col(row: List[str], names: Sequence[str]) -> str:
            for name in names:
                if name in header:
                    idx = header.index(name)
                    if idx < len(row):
                        return row[idx]
            return ""
        for row in data_rows:
            raw_path = col(row, ["image_path", "path", "image", "filename", "file"])
            raw_text = col(row, ["plate_text", "text", "number", "label"])
            raw_type = col(row, ["subtype", "plate_type", "type", "gost_type"])
            if not raw_path or not raw_text:
                continue
            p = Path(raw_path)
            if not p.is_absolute():
                p = (root / p).resolve()
            subtype = normalize_subtype(raw_type)
            if not subtype and infer_missing_subtype:
                subtype = _infer_subtype_from_aspect(p)
            subtype = subtype or "unknown"
            if allowed is not None and subtype not in allowed:
                continue
            records.append(PlateRecord(p, normalize_plate_text(raw_text), subtype))
    else:
        for row in rows:
            if len(row) < 2:
                continue
            p = Path(row[0])
            if not p.is_absolute():
                p = (root / p).resolve()
            text = normalize_plate_text(row[1])
            subtype = normalize_subtype(row[2]) if len(row) > 2 else ""
            if not subtype and infer_missing_subtype:
                subtype = _infer_subtype_from_aspect(p)
            subtype = subtype or "unknown"
            if allowed is not None and subtype not in allowed:
                continue
            records.append(PlateRecord(p, text, subtype))

    if not records:
        raise ValueError(f"No usable records loaded from {path}")
    return records


def _read_text_auto(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise UnicodeDecodeError("unknown", data, 0, min(1, len(data)), f"Cannot decode {path}")


def _detect_delimiter(path: Path, text: str) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    first = next((line for line in text.splitlines() if line.strip()), "")
    counts = {";": first.count(";"), "\t": first.count("\t"), ",": first.count(",")}
    return max(counts, key=counts.get) if max(counts.values(), default=0) else ";"


def _find_csv_file(folder: Path, split: str) -> Path:
    priority = [folder / f"{split}.csv", folder / "train.csv", folder / "val.csv"]
    for p in priority:
        if p.is_file():
            return p
    csv_files = sorted(folder.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {folder}")
    raise RuntimeError(f"Several CSV files found in {folder}: {[p.name for p in csv_files]}")


def _resolve_csv_image(source_root: Path, csv_path: Path, raw_path: str, basename_index: Dict[str, Path]) -> Optional[Path]:
    raw = str(raw_path).strip().replace("\\", "/")
    if not raw:
        return None
    p = Path(raw)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            csv_path.parent / p,
            source_root / p,
            csv_path.parent / "img" / p,
            csv_path.parent / "images" / p,
            source_root / "img" / p,
            source_root / "images" / p,
            csv_path.parent / "img" / p.name,
            csv_path.parent / "images" / p.name,
            source_root / "img" / p.name,
            source_root / "images" / p.name,
        ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return basename_index.get(p.name)


def load_csv_records(
    csv_path: str | Path,
    subtype: str,
    source_root: str | Path | None = None,
    strict: bool = False,
) -> Tuple[List[PlateRecord], List[str]]:
    """Load one of the project's compact CSV files: image;plate_num;subtype."""
    csv_path = Path(csv_path)
    source_root = Path(source_root) if source_root is not None else csv_path.parent
    expected_subtype = normalize_subtype(subtype)
    if expected_subtype not in SUPPORTED_SUBTYPES:
        raise ValueError(f"Unsupported requested subtype: {expected_subtype}")

    text = _read_text_auto(csv_path)
    delimiter = _detect_delimiter(csv_path, text)
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    if not reader.fieldnames:
        raise ValueError(f"CSV has no header: {csv_path}")
    fields = {str(x).strip().lower(): x for x in reader.fieldnames if x is not None}

    image_key = next((fields[k] for k in ("image", "image_path", "path", "filename", "file") if k in fields), None)
    text_key = next((fields[k] for k in ("plate_num", "text", "plate_text", "number", "label") if k in fields), None)
    subtype_key = next((fields[k] for k in ("subtype", "plate_type", "type") if k in fields), None)
    if image_key is None or text_key is None:
        raise ValueError(
            f"CSV {csv_path} must contain image and plate_num columns. Header={reader.fieldnames}"
        )

    basename_index = _build_basename_index(source_root)
    records: List[PlateRecord] = []
    rejected: List[str] = []

    for line_no, row in enumerate(reader, 2):
        try:
            row_subtype = normalize_subtype(row.get(subtype_key, "") if subtype_key else "") or expected_subtype
            if row_subtype != expected_subtype:
                raise ValueError(f"CSV subtype {row_subtype!r} != folder subtype {expected_subtype!r}")
            label = normalize_plate_text(row.get(text_key, ""))
            validate_plate_format(label, row_subtype)
            image_path = _resolve_csv_image(
                source_root, csv_path, str(row.get(image_key, "")), basename_index
            )
            if image_path is None:
                raise FileNotFoundError(f"image not found: {row.get(image_key, '')!r}")
            records.append(PlateRecord(image_path, label, row_subtype))
        except Exception as exc:
            rejected.append(f"{csv_path.name}:{line_no}: {type(exc).__name__}: {exc}")

    if strict and rejected:
        preview = "\n".join(rejected[:20])
        raise RuntimeError(f"Rejected {len(rejected)} rows from {csv_path}:\n{preview}")
    if not records:
        preview = "\n".join(rejected[:20])
        raise ValueError(f"No usable records in {csv_path}. Rejected examples:\n{preview}")
    return records, rejected


def load_dataset_split(
    dataset_root: str | Path,
    split: str,
    allowed_subtypes: Sequence[str] = SUPPORTED_SUBTYPES,
    strict: bool = False,
) -> Tuple[List[PlateRecord], Dict[str, object]]:
    """Load the complete nine-class dataset tree.

    Flat layout (AM/BY/KG/KZ/type1):
        <folder>/train/img + train.csv
        <folder>/val/img   + train.csv (or val.csv)

    Mixed real/synthetic layout (type1a/type1b/type9/type10):
        <folder>/train/real/images      + train.csv
        <folder>/train/synthetic/images + train.csv
        <folder>/val/images             + val.csv

    Folder aliases:
        AM -> Am, BY -> By, KG -> Kg, KZ -> Kz.
    """
    dataset_root = Path(dataset_root)
    split = split.lower().strip()
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")

    normalized_subtypes = tuple(normalize_subtype(s) for s in allowed_subtypes)
    unknown = [s for s in normalized_subtypes if s not in SUPPORTED_SUBTYPES]
    if unknown:
        raise ValueError(f"Unsupported subtypes for current model: {unknown}")

    all_records: List[PlateRecord] = []
    report: Dict[str, object] = {"split": split, "subtypes": {}, "total": 0, "rejected": 0}

    for subtype in normalized_subtypes:
        folder_name = SUBTYPE_FOLDERS.get(subtype, subtype)
        split_root = dataset_root / folder_name / split
        if not split_root.is_dir():
            raise FileNotFoundError(
                f"Missing dataset folder for subtype {subtype}: {split_root}"
            )

        if split == "train" and (split_root / "real").is_dir() and (split_root / "synthetic").is_dir():
            source_dirs = [split_root / "real", split_root / "synthetic"]
        else:
            source_dirs = [split_root]

        subtype_records: List[PlateRecord] = []
        subtype_rejected: List[str] = []
        source_report: Dict[str, object] = {}

        for source_dir in source_dirs:
            csv_path = _find_csv_file(source_dir, split)
            records, rejected = load_csv_records(
                csv_path, subtype=subtype, source_root=source_dir, strict=strict
            )
            subtype_records.extend(records)
            subtype_rejected.extend(rejected)
            source_report[source_dir.name] = {
                "csv": str(csv_path),
                "records": len(records),
                "rejected": len(rejected),
                "rejected_examples": rejected[:10],
            }

        all_records.extend(subtype_records)
        report["subtypes"][subtype] = {
            "records": len(subtype_records),
            "rejected": len(subtype_rejected),
            "sources": source_report,
        }
        report["rejected"] += len(subtype_rejected)

    report["total"] = len(all_records)
    return all_records, report


def load_records(
    sources: Sequence[str | Path],
    allowed_subtypes: Optional[Sequence[str]] = SUPPORTED_SUBTYPES,
) -> List[PlateRecord]:
    records: List[PlateRecord] = []
    for src in sources:
        p = Path(src)
        if p.is_dir():
            records.extend(load_generator_folder(p, allowed_subtypes=allowed_subtypes))
        elif p.is_file() and p.suffix.lower() in {".tsv", ".txt", ".csv"}:
            records.extend(load_tsv(p, allowed_subtypes=allowed_subtypes))
        else:
            raise FileNotFoundError(f"Unsupported data source: {p}")
    return records


def split_records(
    records: Sequence[PlateRecord],
    val_fraction: float,
    seed: int = 42,
) -> Tuple[List[PlateRecord], List[PlateRecord]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(round(len(idx) * val_fraction)))
    val_idx = set(idx[:n_val])
    train = [r for i, r in enumerate(records) if i not in val_idx]
    val = [r for i, r in enumerate(records) if i in val_idx]
    return train, val


def _augment_pil(img: Image.Image, rng: random.Random) -> Image.Image:
    # Mild augmentations only; the generator already models strong 3D/lighting effects.
    if rng.random() < 0.55:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.72, 1.28))
    if rng.random() < 0.55:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.72, 1.35))
    if rng.random() < 0.25:
        img = ImageEnhance.Sharpness(img).enhance(rng.uniform(0.6, 1.5))
    if rng.random() < 0.22:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.25, 1.15)))
    if rng.random() < 0.20:
        # Simulate low-resolution CCTV crop without changing final geometry.
        scale = rng.uniform(0.45, 0.8)
        w = max(8, int(img.width * scale))
        h = max(8, int(img.height * scale))
        img = img.resize((w, h), Image.Resampling.BILINEAR).resize(
            (max(1, int(w / scale)), max(1, int(h / scale))), Image.Resampling.BILINEAR
        )
    if rng.random() < 0.18:
        angle = rng.uniform(-3.0, 3.0)
        img = img.rotate(
            angle, Image.Resampling.BICUBIC, expand=False,
            fillcolor=_border_fill_color(img),
        )
    return img


def _border_fill_color(img: Image.Image) -> Tuple[int, int, int]:
    """Estimate padding color from the crop border (works for white/yellow/red plates)."""
    arr = np.asarray(img.convert("RGB"))
    if arr.size == 0:
        return (245, 245, 245)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]], axis=0)
    med = np.median(border, axis=0)
    return tuple(int(np.clip(round(v), 0, 255)) for v in med)


def letterbox_to_bucket(
    img: Image.Image,
    bucket: Bucket,
    invert: bool = True,
    fill: int | Tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Aspect-ratio preserving resize + centered padding. No geometric stretching."""
    img = img.convert("RGB")
    src_w, src_h = img.size
    scale = min(bucket.width / max(src_w, 1), bucket.height / max(src_h, 1))
    new_w = max(1, min(bucket.width, int(round(src_w * scale))))
    new_h = max(1, min(bucket.height, int(round(src_h * scale))))
    resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    if fill is None:
        fill_color = _border_fill_color(img)
    elif isinstance(fill, tuple):
        fill_color = fill
    else:
        fill_color = (int(fill), int(fill), int(fill))
    canvas = Image.new("RGB", (bucket.width, bucket.height), fill_color)
    x = (bucket.width - new_w) // 2
    y = (bucket.height - new_h) // 2
    canvas.paste(resized, (x, y))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    if invert:
        arr = 1.0 - arr
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


class PlateDataset(Dataset):
    def __init__(
        self,
        records: Sequence[PlateRecord],
        bucketizer: Optional[AspectBucketizer] = None,
        augment: bool = False,
        invert: bool = True,
        seed: int = 42,
    ):
        self.records = list(records)
        self.bucketizer = bucketizer or AspectBucketizer()
        self.augment = augment
        self.invert = invert
        self.seed = seed
        self._size_cache: Dict[int, Tuple[int, int]] = {}
        self._bucket_cache: Dict[int, int] = {}

    def __len__(self) -> int:
        return len(self.records)

    def image_size(self, idx: int) -> Tuple[int, int]:
        if idx not in self._size_cache:
            with Image.open(self.records[idx].image_path) as im:
                self._size_cache[idx] = (im.width, im.height)
        return self._size_cache[idx]

    def precompute_buckets(
        self,
        workers: int = 16,
        label: str = "dataset",
        progress_every: int = 5000,
    ) -> None:
        """
        Pre-read image dimensions in parallel and fill the bucket cache.

        The previous implementation calculated bucket_id() synchronously inside
        AspectBucketBatchSampler.__init__, which opens EVERY image one by one.
        On a dataset with 100k-300k images this can look like the program has
        frozen immediately after the AMP message.

        This method performs the same work using an I/O thread pool and prints
        progress so startup is visible.
        """
        missing = [
            idx for idx in range(len(self.records))
            if idx not in self._bucket_cache
        ]

        if not missing:
            return

        workers = max(1, int(workers))
        progress_every = max(1, int(progress_every))

        print(
            f"[buckets] {label}: reading sizes for {len(missing)} images "
            f"with {workers} threads...",
            flush=True,
        )

        def read_one(idx: int):
            with Image.open(self.records[idx].image_path) as im:
                w, h = int(im.width), int(im.height)
            return idx, w, h

        def store(result):
            idx, w, h = result
            self._size_cache[idx] = (w, h)
            self._bucket_cache[idx] = self.bucketizer.choose_id(w, h)

        if workers == 1:
            for done, idx in enumerate(missing, start=1):
                store(read_one(idx))
                if done % progress_every == 0 or done == len(missing):
                    print(
                        f"[buckets] {label}: {done}/{len(missing)}",
                        flush=True,
                    )
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for done, result in enumerate(
                    executor.map(read_one, missing),
                    start=1,
                ):
                    store(result)
                    if done % progress_every == 0 or done == len(missing):
                        print(
                            f"[buckets] {label}: {done}/{len(missing)}",
                            flush=True,
                        )

    def bucket_id(self, idx: int) -> int:
        if idx not in self._bucket_cache:
            w, h = self.image_size(idx)
            self._bucket_cache[idx] = self.bucketizer.choose_id(w, h)
        return self._bucket_cache[idx]

    def __getitem__(self, idx: int) -> Dict[str, object]:
        record = self.records[idx]
        bid = self.bucket_id(idx)
        bucket = self.bucketizer.buckets[bid]
        with Image.open(record.image_path) as im:
            img = im.convert("RGB")
        if self.augment:
            # Different workers/epochs still receive stochastic augmentation because
            # Python's worker RNG state changes; idx is mixed in for reproducibility.
            rng = random.Random(random.getrandbits(64) ^ (self.seed + idx * 1000003))
            img = _augment_pil(img, rng)
        tensor = letterbox_to_bucket(img, bucket, invert=self.invert)
        return {
            "image": tensor,
            "text": record.text,
            "subtype": record.subtype,
            "path": str(record.image_path),
            "bucket_id": bid,
            "bucket": bucket.name,
        }


class AspectBucketBatchSampler(Sampler[List[int]]):
    """Yields batches containing only images assigned to the same HxW bucket."""

    def __init__(
        self,
        dataset: PlateDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.groups: Dict[int, List[int]] = {}
        for idx in range(len(dataset)):
            self.groups.setdefault(dataset.bucket_id(idx), []).append(idx)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches: List[List[int]] = []
        for _, indices in sorted(self.groups.items()):
            ids = list(indices)
            if self.shuffle:
                rng.shuffle(ids)
            for i in range(0, len(ids), self.batch_size):
                batch = ids[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        total = 0
        for ids in self.groups.values():
            if self.drop_last:
                total += len(ids) // self.batch_size
            else:
                total += math.ceil(len(ids) / self.batch_size)
        return total


def collate_plate_batch(items: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not items:
        raise ValueError("Empty batch")
    shapes = {tuple(x["image"].shape) for x in items}
    if len(shapes) != 1:
        raise RuntimeError(
            f"Mixed bucket shapes in one batch: {shapes}. Use AspectBucketBatchSampler."
        )
    return {
        "images": torch.stack([x["image"] for x in items], dim=0),
        "texts": [str(x["text"]) for x in items],
        "subtypes": [str(x["subtype"]) for x in items],
        "paths": [str(x["path"]) for x in items],
        "bucket": str(items[0]["bucket"]),
    }
