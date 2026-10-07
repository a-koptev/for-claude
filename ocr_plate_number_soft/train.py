from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from checkpoint import load_checkpoint, save_checkpoint
from data import (
    SUPPORTED_SUBTYPES,
    AspectBucketBatchSampler,
    AspectBucketizer,
    PlateDataset,
    PlateRecord,
    collate_plate_batch,
    load_dataset_split,
)
from metrics import finalize_metrics, update_metrics
from parseq_gost_ocr import ModelConfig, PARSeqGostOCR, make_ltr_masks


DEFAULT_DATASET_ROOT = Path(r"E:\Проекты\OCR_dataset")
DEFAULT_OUTPUT = Path(r"outputs\parseq_gost_9classes")

DEFAULT_SUBTYPES = ("type1", "type1a", "type1b", "type9", "type10", "AM", "BY", "KG", "KZ")


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train PARSeq-GOST OCR on the project dataset tree: "
            "type1/type1a/type1b/type9/type10/AM/BY/KG/KZ"
        )
    )
    p.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    p.add_argument("--subtypes", nargs="+", default=list(DEFAULT_SUBTYPES))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument(
        "--bucket-scan-workers",
        type=int,
        default=16,
        help="Threads used once at startup to read image sizes for aspect-ratio buckets",
    )

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=2.0)
    p.add_argument("--num-permutations", type=int, default=3)
    p.add_argument("--grad-clip", type=float, default=5.0)

    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--encoder-depth", type=int, default=6)
    p.add_argument("--decoder-depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--type-loss-weight", type=float, default=0.25)
    p.add_argument("--max-chars", type=int, default=9)

    p.add_argument("--device", default="auto")
    p.add_argument("--amp", choices=["auto", "none", "fp16", "bf16"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--strict-data", action="store_true")
    p.add_argument("--fail-on-leakage", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


# =====================================================================
# Runtime helpers
# =====================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    return device


def resolve_amp(device: torch.device, mode: str) -> Tuple[bool, torch.dtype]:
    if mode == "none" or device.type != "cuda":
        return False, torch.float32
    if mode == "fp16":
        return True, torch.float16
    if mode == "bf16":
        return True, torch.bfloat16
    bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    return True, torch.bfloat16 if bf16 else torch.float16


def make_loader(
    dataset: PlateDataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    prefetch_factor: int,
) -> Tuple[DataLoader, AspectBucketBatchSampler]:
    sampler = AspectBucketBatchSampler(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        seed=seed,
    )
    kwargs = dict(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=collate_plate_batch,
    )
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(**kwargs)
    return loader, sampler


def cosine_lr(
    base_lr: float,
    min_lr: float,
    step: int,
    total_steps: int,
    warmup_steps: int,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def summarize_records(records: List[PlateRecord]) -> Dict[str, object]:
    by_type = Counter(r.subtype for r in records)
    lengths = Counter(len(r.text) for r in records)

    pattern_by_type: Dict[str, Counter] = {}
    for record in records:
        signature = "".join(
            "#" if ch.isdigit() else "L"
            for ch in record.text
        )
        pattern_by_type.setdefault(record.subtype, Counter())[signature] += 1

    return {
        "total": len(records),
        "by_subtype": dict(sorted(by_type.items())),
        "text_lengths": dict(sorted(lengths.items())),
        "patterns_by_subtype": {
            subtype: dict(counter.most_common(20))
            for subtype, counter in sorted(pattern_by_type.items())
        },
    }


def leakage_pairs(train_records: List[PlateRecord], val_records: List[PlateRecord]) -> List[Tuple[str, str]]:
    train_keys = {(r.subtype, r.text) for r in train_records}
    val_keys = {(r.subtype, r.text) for r in val_records}
    return sorted(train_keys & val_keys)


# =====================================================================
# FAST batch autoregressive validation
# =====================================================================

@torch.no_grad()
def recognize_batch_fast(
    model: PARSeqGostOCR,
    images: torch.Tensor,
) -> List[Dict[str, object]]:
    """Batched greedy decoding: one decoder forward per character position."""
    memory, subtype_logits = model.encode_image(images)
    subtype_probs = subtype_logits.softmax(dim=-1)

    batch_size = images.shape[0]
    device = images.device
    max_chars = model.cfg.max_chars

    subtype_ids = subtype_probs.argmax(dim=-1)
    subtypes = [model.id_to_subtype[int(i.item())] for i in subtype_ids]
    subtype_conf = [
        float(subtype_probs[i, subtype_ids[i]].item())
        for i in range(batch_size)
    ]

    chars = torch.full(
        (batch_size, max_chars),
        model.alphabet.pad_id,
        dtype=torch.long,
        device=device,
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    output_ids: List[List[int]] = [[] for _ in range(batch_size)]
    output_probs: List[List[float]] = [[] for _ in range(batch_size)]

    cm, qm = make_ltr_masks(batch_size, max_chars + 1, device)

    for pos in range(max_chars + 1):
        content_ids = model._make_content_ids(chars)
        logits = model.decoder(content_ids, memory, cm, qm)[:, pos, :]

        constrained = logits.clone()
        for i in range(batch_size):
            if finished[i]:
                constrained[i].fill_(float("-inf"))
                constrained[i, model.alphabet.eos_id] = 0.0
                continue
            allowed = model.grammar.allowed_token_ids(subtypes[i], pos)
            if allowed:
                mask = torch.full_like(constrained[i], float("-inf"))
                idx = torch.tensor(allowed, dtype=torch.long, device=device)
                mask[idx] = 0.0
                constrained[i] += mask

        probs = constrained.softmax(dim=-1)
        next_ids = probs.argmax(dim=-1)
        next_probs = probs.gather(1, next_ids[:, None]).squeeze(1)

        for i in range(batch_size):
            if finished[i]:
                continue
            token_id = int(next_ids[i].item())
            prob = float(next_probs[i].item())

            if token_id == model.alphabet.eos_id:
                finished[i] = True
                continue
            if token_id in (model.alphabet.pad_id, model.alphabet.bos_id):
                finished[i] = True
                continue

            output_ids[i].append(token_id)
            output_probs[i].append(prob)
            if pos < max_chars:
                chars[i, pos] = token_id

        if bool(finished.all()):
            break

    results: List[Dict[str, object]] = []
    for i in range(batch_size):
        results.append({
            "text": model.alphabet.decode_chars(output_ids[i]),
            "subtype": subtypes[i],
            "subtype_confidence": subtype_conf[i],
            "char_confidence": (
                float(sum(output_probs[i]) / len(output_probs[i]))
                if output_probs[i] else 0.0
            ),
        })
    return results


@torch.inference_mode()
def evaluate(
    model: PARSeqGostOCR,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_batches: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    store: Dict[str, Dict[str, float]] = {}
    for batch_idx, batch in enumerate(tqdm(loader, desc="validation", leave=False)):
        images = batch["images"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            pred = recognize_batch_fast(model, images)
        update_metrics(store, batch["texts"], batch["subtypes"], pred)
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break
    return finalize_metrics(store)


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    allowed_subtypes = tuple(args.subtypes)
    unsupported = sorted(set(allowed_subtypes) - set(SUPPORTED_SUBTYPES))
    if unsupported:
        raise ValueError(
            f"Current model supports only {list(SUPPORTED_SUBTYPES)}; got {unsupported}"
        )
    if args.max_chars < 9:
        raise ValueError("--max-chars must be at least 9 for the current nine-class dataset")

    dataset_root = Path(args.dataset_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    train_records, train_report = load_dataset_split(
        dataset_root,
        "train",
        allowed_subtypes=allowed_subtypes,
        strict=args.strict_data,
    )
    val_records, val_report = load_dataset_split(
        dataset_root,
        "val",
        allowed_subtypes=allowed_subtypes,
        strict=args.strict_data,
    )

    dataset_report = {
        "dataset_root": str(dataset_root),
        "train": train_report,
        "val": val_report,
        "train_summary": summarize_records(train_records),
        "val_summary": summarize_records(val_records),
    }
    (output / "dataset_report.json").write_text(
        json.dumps(dataset_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("DATASET")
    print(json.dumps(dataset_report["train_summary"], ensure_ascii=False, indent=2))
    print("VALIDATION")
    print(json.dumps(dataset_report["val_summary"], ensure_ascii=False, indent=2))

    overlaps = leakage_pairs(train_records, val_records)
    if overlaps:
        msg = (
            f"WARNING: {len(overlaps)} (subtype, plate_num) labels occur in both train and val. "
            f"Examples: {overlaps[:20]}"
        )
        print(msg)
        if args.fail_on_leakage:
            raise RuntimeError(msg)
    else:
        print("Leakage check: OK (no identical subtype+plate_num across train/val)")

    device = resolve_device(args.device)
    use_amp, amp_dtype = resolve_amp(device, args.amp)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"AMP: {use_amp} ({amp_dtype})")

    bucketizer = AspectBucketizer()
    train_ds = PlateDataset(
        train_records,
        bucketizer=bucketizer,
        augment=not args.no_augment,
        seed=args.seed,
    )
    val_ds = PlateDataset(
        val_records,
        bucketizer=bucketizer,
        augment=False,
        seed=args.seed,
    )

    # AspectBucketBatchSampler needs the aspect-ratio bucket for every image.
    # Do this explicitly and in parallel; otherwise it opens all images
    # sequentially here and the program appears to hang after the AMP line.
    bucket_t0 = time.perf_counter()
    print("Preparing aspect-ratio buckets...", flush=True)
    train_ds.precompute_buckets(
        workers=args.bucket_scan_workers,
        label="train",
    )
    val_ds.precompute_buckets(
        workers=args.bucket_scan_workers,
        label="val",
    )
    print(
        f"Aspect-ratio buckets ready in {time.perf_counter() - bucket_t0:.1f} s",
        flush=True,
    )

    train_loader, train_sampler = make_loader(
        train_ds,
        args.batch_size,
        args.workers,
        True,
        args.seed,
        args.prefetch_factor,
    )
    val_loader, _ = make_loader(
        val_ds,
        args.batch_size,
        args.workers,
        False,
        args.seed,
        args.prefetch_factor,
    )

    cfg = ModelConfig(
        dim=args.dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        heads=args.heads,
        dropout=args.dropout,
        max_chars=args.max_chars,
        subtypes=allowed_subtypes,
        type_loss_weight=args.type_loss_weight,
    )

    start_epoch = 0
    best_joint = -1.0
    if args.resume:
        model, ckpt = load_checkpoint(args.resume, device)
        if tuple(model.cfg.subtypes) != allowed_subtypes:
            raise RuntimeError(
                "Checkpoint subtype classes do not match this run. "
                f"checkpoint={model.cfg.subtypes}, requested={allowed_subtypes}. "
                "Old type1/type1a checkpoints cannot be resumed directly because the subtype head size changed."
            )
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_joint = float(ckpt.get("best_metric", -1.0))
        print(f"Resumed: {args.resume}, next epoch={start_epoch + 1}")
    else:
        model = PARSeqGostOCR(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )
    if args.resume:
        raw = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "optimizer_state" in raw:
            optimizer.load_state_dict(raw["optimizer_state"])

    history_path = output / "history.jsonl"
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    global_step = start_epoch * steps_per_epoch

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(use_amp and amp_dtype == torch.float16),
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        running_loss = 0.0
        running_text = 0.0
        running_type = 0.0
        seen_batches = 0
        epoch_started = time.perf_counter()

        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            images = batch["images"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            lr = cosine_lr(args.lr, args.min_lr, global_step, total_steps, warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                losses = model.compute_loss(
                    images,
                    texts=batch["texts"],
                    subtype_labels=batch["subtypes"],
                    num_permutations=args.num_permutations,
                )
                loss = losses["loss"]

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            global_step += 1
            seen_batches += 1
            running_loss += float(loss.detach())
            running_text += float(losses["text_loss"])
            running_type += float(losses["type_loss"])

            pbar.set_postfix(
                loss=f"{running_loss / seen_batches:.4f}",
                text=f"{running_text / seen_batches:.4f}",
                type=f"{running_type / seen_batches:.4f}",
                lr=f"{lr:.2e}",
                bucket=batch["bucket"],
            )

            if args.dry_run and batch_idx >= 2:
                break

        metrics = evaluate(
            model,
            val_loader,
            device,
            use_amp,
            amp_dtype,
            max_batches=3 if args.dry_run else None,
        )
        overall = metrics.get("overall", {})
        joint = float(overall.get("joint_accuracy", 0.0))

        summary = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(seen_batches, 1),
            "train_text_loss": running_text / max(seen_batches, 1),
            "train_type_loss": running_type / max(seen_batches, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.perf_counter() - epoch_started, 2),
            "metrics": metrics,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

        save_checkpoint(
            output / "last.pt",
            model,
            epoch,
            max(best_joint, joint),
            optimizer=optimizer,
            extra={"metrics": metrics, "args": vars(args)},
        )

        if joint > best_joint:
            best_joint = joint
            save_checkpoint(
                output / "best.pt",
                model,
                epoch,
                best_joint,
                optimizer=optimizer,
                extra={"metrics": metrics, "args": vars(args)},
            )
            print(f"Saved new best.pt, joint_accuracy={best_joint:.6f}")

        if args.dry_run:
            print("Dry run completed successfully.")
            break


if __name__ == "__main__":
    main()
