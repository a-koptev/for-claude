from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

try:
    from .parseq_gost_ocr import ModelConfig, PARSeqGostOCR, PlateAlphabet
except ImportError:
    from parseq_gost_ocr import ModelConfig, PARSeqGostOCR, PlateAlphabet


def save_checkpoint(
    path: str | Path,
    model: PARSeqGostOCR,
    epoch: int,
    best_metric: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "format": "parseq_gost_ocr_v1",
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "model_config": asdict(model.cfg),
        "alphabet_letters": "".join(
            model.alphabet.itos[i] for i in model.alphabet.letter_ids
        ),
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if extra:
        payload["extra"] = extra
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    load_optimizer: Optional[torch.optim.Optimizer] = None,
    load_scheduler: Optional[Any] = None,
) -> Tuple[PARSeqGostOCR, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(f"Unsupported checkpoint format: {path}")

    cfg_data = dict(ckpt["model_config"])
    if isinstance(cfg_data.get("subtypes"), list):
        cfg_data["subtypes"] = tuple(cfg_data["subtypes"])
    cfg = ModelConfig(**cfg_data)
    alphabet = PlateAlphabet(letters=ckpt.get("alphabet_letters", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    model = PARSeqGostOCR(cfg, alphabet=alphabet)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)

    if load_optimizer is not None and "optimizer_state" in ckpt:
        load_optimizer.load_state_dict(ckpt["optimizer_state"])
    if load_scheduler is not None and "scheduler_state" in ckpt:
        load_scheduler.load_state_dict(ckpt["scheduler_state"])
    return model, ckpt
