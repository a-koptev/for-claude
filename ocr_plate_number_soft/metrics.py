from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                cur[-1] + 1,
                prev[j] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = cur
    return prev[-1]




def wildcard_text_equal(gt: str, pred: str) -> bool:
    """Exact-length match where '#' in GT accepts any predicted character."""
    if len(gt) != len(pred):
        return False
    return all(g == "#" or g == p for g, p in zip(gt, pred))


def wildcard_distance_target(gt: str, pred: str) -> str:
    """Replace aligned GT wildcards by prediction chars before edit distance."""
    chars = list(gt)
    for i, ch in enumerate(chars):
        if ch == "#" and i < len(pred):
            chars[i] = pred[i]
    return "".join(chars)

def _empty() -> Dict[str, float]:
    return {"n": 0, "text_correct": 0, "type_correct": 0, "joint_correct": 0, "ned_sum": 0.0}


def update_metrics(
    store: Dict[str, Dict[str, float]],
    gt_texts: Sequence[str],
    gt_types: Sequence[str],
    predictions: Sequence[dict],
) -> None:
    for gt_text, gt_type, pred in zip(gt_texts, gt_types, predictions):
        for key in ("overall", gt_type):
            if key not in store:
                store[key] = _empty()
            m = store[key]
            pred_text = str(pred["text"])
            pred_type = str(pred["subtype"])
            text_ok = wildcard_text_equal(gt_text, pred_text)
            type_ok = pred_type == gt_type
            dist_gt = wildcard_distance_target(gt_text, pred_text)
            dist = levenshtein(dist_gt, pred_text)
            denom = max(len(gt_text), len(pred_text), 1)
            m["n"] += 1
            m["text_correct"] += int(text_ok)
            m["type_correct"] += int(type_ok)
            m["joint_correct"] += int(text_ok and type_ok)
            m["ned_sum"] += 1.0 - dist / denom


def finalize_metrics(store: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key, m in store.items():
        n = max(int(m["n"]), 1)
        out[key] = {
            "n": int(m["n"]),
            "text_accuracy": m["text_correct"] / n,
            "subtype_accuracy": m["type_correct"] / n,
            "joint_accuracy": m["joint_correct"] / n,
            "normalized_edit_similarity": m["ned_sum"] / n,
        }
    return out
