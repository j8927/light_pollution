"""Faster R-CNN ResNet50 FPN V2 학습 (torchvision).

PowerShell 사용 예:
    python training/train_faster_rcnn.py
    python training/train_faster_rcnn.py --epochs 50 --batch 2
    python training/train_faster_rcnn.py --overwrite

YOLO 계열과 동일한 split(splits/*.txt), 동일한 image size, seed, epoch 수를 사용한다.
YOLO txt 라벨은 training/yolo_dataset.py 가 실행 시점에 xyxy pixel 로 변환해 읽는다
(원본 라벨 파일은 수정하지 않으며 COCO JSON 도 만들지 않는다).

YOLO 계열과 구조적으로 다를 수밖에 없는 부분(기록만 하고 억지로 통일하지 않는다)
    - optimizer : SGD (Ultralytics 는 auto/AdamW)
    - augmentation : 좌우 반전만 적용 (mosaic/HSV 등 YOLO 고유 증강 없음)
    - loss : RPN + ROI head loss 4종

학습 결과: runs/faster_rcnn/
    weights/best.pt, weights/last.pt, results.csv, training_meta.json
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lpcompare import dataset as ds  # noqa: E402
from lpcompare.config import MODEL_DISPLAY, load_config, set_global_seed  # noqa: E402
from lpcompare.logging_utils import get_logger, log_banner, log_kv  # noqa: E402
from lpcompare.metrics import Detections, GroundTruth, evaluate_ap, evaluate_pr  # noqa: E402
from lpcompare.model_info import gpu_name_and_vram  # noqa: E402
from lpcompare.runtime import build_faster_rcnn, load_training_meta, resolve_weights, save_training_meta  # noqa: E402
from training.yolo_dataset import build_dataloader  # noqa: E402

MODEL_KEY = "faster_rcnn"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Faster R-CNN ResNet50 FPN V2 학습 (빛공해 모델 비교 실험)")
    p.add_argument("--config", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda | cpu | 0")
    p.add_argument("--workers", type=int, default=None, help="DataLoader worker 수 (Windows 기본 0)")
    p.add_argument("--overwrite", action="store_true", help="기존 학습 결과가 있어도 다시 학습")
    p.add_argument("--allow-cpu", action="store_true",
                   help="CUDA 를 쓸 수 없어도 CPU 로 학습을 강행한다 "
                        "(기본값: common.require_cuda=true 이면 중단)")
    p.add_argument("--val-map-interval", type=int, default=1,
                   help="몇 epoch 마다 val mAP 를 계산할지 (기본 1, 0 이면 계산 안 함)")
    p.add_argument("--amp", dest="amp", action="store_true", default=None, help="AMP 사용 (CUDA 전용)")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="AMP 사용 안 함")
    return p


def _is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in text or "cuda oom" in text or "cublas_status_alloc_failed" in text


def _set_bn_eval(model) -> None:
    """val loss 계산 시 BatchNorm 통계가 갱신되지 않도록 고정한다."""
    import torch.nn as nn

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def _lr_lambda(step: int, warmup_iters: int, total_iters: int) -> float:
    if warmup_iters > 0 and step < warmup_iters:
        return (step + 1) / float(warmup_iters)
    progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def evaluate_epoch(model, loader, device: str, num_classes: int,
                   compute_map: bool) -> Dict[str, Optional[float]]:
    """val 손실과 (선택적으로) val mAP / Precision / Recall 을 계산한다.

    letterbox 된 좌표계에서 계산하며, 학습 모니터링 용도다.
    최종 비교 지표는 evaluation/evaluate_models.py 가 원본 좌표계에서 다시 계산한다.
    """
    import torch

    out: Dict[str, Optional[float]] = {
        "val_loss": None, "precision": None, "recall": None, "map50": None, "map50_95": None,
    }

    # --- val loss (train 모드이되 BN 통계는 고정) ---
    model.train()
    _set_bn_eval(model)
    total, n_batches = 0.0, 0
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            tgts = [{k: (v.to(device) if hasattr(v, "to") else v) for k, v in t.items() if k != "path"}
                    for t in targets]
            loss_dict = model(images, tgts)
            total += float(sum(l.item() for l in loss_dict.values()))
            n_batches += 1
    if n_batches:
        out["val_loss"] = round(total / n_batches, 5)

    if not compute_map:
        return out

    # --- val mAP / P / R ---
    model.eval()
    preds: List[Detections] = []
    gts: List[GroundTruth] = []
    with torch.inference_mode():
        for images, targets in loader:
            images_d = [img.to(device) for img in images]
            outputs = model(images_d)
            for o, t, img in zip(outputs, targets, images):
                boxes = o["boxes"].detach().cpu().numpy().astype(np.float32)
                scores = o["scores"].detach().cpu().numpy().astype(np.float32)
                labels = o["labels"].detach().cpu().numpy().astype(np.int32) - 1
                keep = labels >= 0
                preds.append(Detections(boxes[keep], scores[keep], labels[keep]))
                gt_boxes = t["boxes"].numpy().astype(np.float32)
                gt_labels = (t["labels"].numpy() - 1).astype(np.int32)
                h, w = int(img.shape[1]), int(img.shape[2])
                gts.append(GroundTruth(gt_boxes, gt_labels, w, h))

    ap = evaluate_ap(preds, gts, num_classes)
    pr = evaluate_pr(preds, gts, num_classes, conf=0.25, iou_thr=0.5)
    out["map50"] = None if math.isnan(ap["map50"]) else round(ap["map50"], 5)
    out["map50_95"] = None if math.isnan(ap["map50_95"]) else round(ap["map50_95"], 5)
    out["precision"] = None if math.isnan(pr["precision_mean"]) else round(pr["precision_mean"], 5)
    out["recall"] = None if math.isnan(pr["recall_mean"]) else round(pr["recall_mean"], 5)
    return out


def train_once(cfg, args, logger, meta: Dict[str, Any], batch_size: int) -> Dict[str, Any]:
    """batch_size 로 1회 학습을 시도한다. CUDA OOM 은 호출자가 처리한다."""
    import torch

    device = meta["device_detail"]
    run_dir = Path(meta["run_dir"])
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    layout = ds.detect_layout(cfg.dataset_root, cfg.data_yaml, cfg.fallback_names)
    names = layout.names
    num_classes = len(names)
    splits = ds.load_splits(cfg.splits_dir)

    model_cfg = cfg.model(MODEL_KEY)
    epochs = meta["epochs"]
    imgsz = meta["image_size"]
    workers = meta["workers"]

    train_loader = build_dataloader(
        splits["train"], num_classes, batch_size, imgsz, train=True,
        workers=workers, seed=cfg.seed,
        hflip_prob=float(model_cfg.get("horizontal_flip", 0.5)), class_names=names,
    )
    val_loader = build_dataloader(
        splits["val"], num_classes, max(1, batch_size), imgsz, train=False,
        workers=workers, seed=cfg.seed, class_names=names,
    )

    model = build_faster_rcnn(num_classes=num_classes + 1, pretrained=cfg.pretrained, image_size=imgsz)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=float(model_cfg.get("lr", 0.005)),
        momentum=float(model_cfg.get("momentum", 0.9)),
        weight_decay=float(model_cfg.get("weight_decay", 0.0005)),
    )
    iters_per_epoch = max(1, len(train_loader))
    total_iters = iters_per_epoch * epochs
    warmup_iters = min(int(model_cfg.get("warmup_iters", 500)), max(1, total_iters // 10))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_lambda(step, warmup_iters, total_iters)
    )

    use_amp = bool(meta["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    results_path = run_dir / "results.csv"
    fieldnames = ["epoch", "lr", "train_loss", "loss_classifier", "loss_box_reg",
                  "loss_objectness", "loss_rpn_box_reg", "val_loss",
                  "precision", "recall", "mAP50", "mAP50-95", "epoch_time_sec"]
    with open(results_path, "w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    best_score = -1.0
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    display = meta["display_name"]
    val_interval = max(0, int(args.val_map_interval))

    for epoch in range(1, epochs + 1):
        logger.info("=" * 40)
        logger.info("Model: %s", display)
        logger.info("Epoch: %d / %d", epoch, epochs)
        logger.info("=" * 40)

        model.train()
        t_epoch = time.perf_counter()
        sums = {"loss": 0.0, "loss_classifier": 0.0, "loss_box_reg": 0.0,
                "loss_objectness": 0.0, "loss_rpn_box_reg": 0.0}
        n_batches = 0

        for bi, (images, targets) in enumerate(train_loader, start=1):
            images = [img.to(device, non_blocking=True) for img in images]
            tgts = [{k: (v.to(device) if hasattr(v, "to") else v) for k, v in t.items() if k != "path"}
                    for t in targets]

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss_dict = model(images, tgts)
                    loss = sum(loss_dict.values())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(images, tgts)
                loss = sum(loss_dict.values())
                loss.backward()
                optimizer.step()
            scheduler.step()

            if not torch.isfinite(loss):
                logger.warning("  [경고] epoch %d batch %d 에서 loss 가 발산했습니다 (%s).", epoch, bi, loss)
            sums["loss"] += float(loss.item())
            for k in ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"):
                if k in loss_dict:
                    sums[k] += float(loss_dict[k].item())
            n_batches += 1
            if bi % 10 == 0 or bi == len(train_loader):
                logger.info("  [%s] epoch %d/%d  batch %d/%d  loss %.4f  lr %.6f",
                            display, epoch, epochs, bi, len(train_loader),
                            sums["loss"] / max(1, n_batches), optimizer.param_groups[0]["lr"])

        compute_map = val_interval > 0 and (epoch % val_interval == 0 or epoch == epochs)
        val_stats = evaluate_epoch(model, val_loader, device, num_classes, compute_map)

        row = {
            "epoch": epoch,
            "lr": round(optimizer.param_groups[0]["lr"], 8),
            "train_loss": round(sums["loss"] / max(1, n_batches), 5),
            "loss_classifier": round(sums["loss_classifier"] / max(1, n_batches), 5),
            "loss_box_reg": round(sums["loss_box_reg"] / max(1, n_batches), 5),
            "loss_objectness": round(sums["loss_objectness"] / max(1, n_batches), 5),
            "loss_rpn_box_reg": round(sums["loss_rpn_box_reg"] / max(1, n_batches), 5),
            "val_loss": val_stats["val_loss"],
            "precision": val_stats["precision"],
            "recall": val_stats["recall"],
            "mAP50": val_stats["map50"],
            "mAP50-95": val_stats["map50_95"],
            "epoch_time_sec": round(time.perf_counter() - t_epoch, 2),
        }
        history.append(row)
        with open(results_path, "a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        logger.info("  -> train_loss %.4f | val_loss %s | mAP50 %s | mAP50-95 %s | %.1f초",
                    row["train_loss"], row["val_loss"], row["mAP50"], row["mAP50-95"],
                    row["epoch_time_sec"])

        # best 선택: val mAP50-95 우선, 없으면 val_loss 기준
        if row["mAP50-95"] is not None:
            score = float(row["mAP50-95"])
        elif row["val_loss"] is not None:
            score = -float(row["val_loss"])
        else:
            score = -float(row["train_loss"])

        ckpt = {
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes + 1,
            "class_names": names,
            "image_size": imgsz,
            "epoch": epoch,
            "metrics": row,
            "architecture": "fasterrcnn_resnet50_fpn_v2",
        }
        torch.save(ckpt, weights_dir / "last.pt")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(ckpt, weights_dir / "best.pt")
            logger.info("  -> best 갱신 (epoch %d, score %.5f)", epoch, score)

    meta["best_epoch"] = best_epoch
    meta["best_score"] = round(best_score, 5) if best_score > -1 else None
    meta["history"] = history
    meta["results_csv"] = str(results_path)
    meta["best_weights"] = str(weights_dir / "best.pt")
    meta["num_classes"] = num_classes
    meta["class_names"] = names
    return meta


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config)
    display = MODEL_DISPLAY[MODEL_KEY]
    logger = get_logger(MODEL_KEY, cfg.logs_dir)
    log_banner(logger, f"{display} 학습 시작")

    model_cfg = cfg.model(MODEL_KEY)
    run_dir = cfg.run_dir(MODEL_KEY)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        cfg.raw["common"]["device"] = args.device
    device_info = cfg.resolve_device()
    for w in device_info["warnings"]:
        logger.warning(w)

    if device_info.get("blocked") and not getattr(args, "allow_cpu", False):
        msg = ("GPU 를 사용할 수 없어 학습을 중단했습니다 (config 의 common.require_cuda=true). "
               + device_info.get("block_reason", ""))
        logger.error("[중단] %s", msg)
        logger.error("        해결: python tools/check_environment.py 로 필요한 조치를 확인하세요.")
        logger.error("        CPU 로 강행하려면 --allow-cpu 옵션을 사용하거나 "
                     "config/experiment.yaml 의 common.require_cuda 를 false 로 바꾸세요.")
        return {"model_key": MODEL_KEY, "display_name": display, "success": False,
                "error": msg, "device": "cpu",
                "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    batch = args.batch if args.batch else int(model_cfg.get("batch", 4))
    use_amp = args.amp if args.amp is not None else (device_info["device"] == "cuda")

    meta: Dict[str, Any] = {
        "model_key": MODEL_KEY,
        "display_name": display,
        "framework": "torchvision",
        "requested_weights": str(model_cfg.get("weights", "fasterrcnn_resnet50_fpn_v2")),
        "actual_weights": "fasterrcnn_resnet50_fpn_v2 (COCO pretrained)" if cfg.pretrained
                          else "fasterrcnn_resnet50_fpn_v2 (scratch)",
        "weights_changed_reason": "",
        "dataset_root": str(cfg.dataset_root),
        "data_yaml": str(cfg.compare_data_yaml),
        "epochs": args.epochs if args.epochs else cfg.epochs,
        "image_size": args.imgsz if args.imgsz else cfg.image_size,
        "requested_batch": batch,
        "actual_batch": batch,
        "seed": cfg.seed,
        "device": device_info["device"],
        "device_detail": device_info["torch"],
        "pretrained": cfg.pretrained,
        "workers": args.workers if args.workers is not None else cfg.workers,
        "amp": bool(use_amp),
        "optimizer": model_cfg.get("optimizer", "SGD"),
        "lr": model_cfg.get("lr", 0.005),
        "lr_scheduler": model_cfg.get("lr_scheduler", "cosine"),
        "augmentation": f"horizontal_flip={model_cfg.get('horizontal_flip', 0.5)} (YOLO 고유 증강 미적용)",
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": None,
        "training_time_sec": None,
        "train_gpu_memory_mb": None,
        "success": False,
        "error": None,
        "oom_retries": [],
        "notes": [],
        "run_dir": str(run_dir),
    }
    meta.update(gpu_name_and_vram())

    if not ds.splits_exist(cfg.splits_dir):
        msg = f"split 파일이 없습니다: {cfg.splits_dir}. 먼저 `python tools/prepare_splits.py` 를 실행하세요."
        logger.error("[실패] %s", msg)
        meta["error"] = msg
        save_training_meta(run_dir, meta)
        return meta

    existing = resolve_weights(run_dir)
    if existing and not args.overwrite:
        prev = load_training_meta(run_dir)
        logger.info("이미 학습된 weight 가 있어 학습을 건너뜁니다: %s", existing)
        logger.info("다시 학습하려면 --overwrite 옵션을 사용하세요.")
        if prev:
            return prev
        meta["success"] = True
        meta["best_weights"] = str(existing)
        meta["notes"].append("기존 weight 재사용 (학습 생략)")
        return meta

    counts = {k: len(v) for k, v in ds.load_splits(cfg.splits_dir).items()}
    log_kv(logger, f"{display} 학습 조건", {
        "architecture": "torchvision fasterrcnn_resnet50_fpn_v2",
        "pretrained": "COCO (FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)" if cfg.pretrained else "없음",
        "train / val / test": f"{counts['train']} / {counts['val']} / {counts['test']} 장",
        "epochs": meta["epochs"],
        "image size": meta["image_size"],
        "batch": batch,
        "optimizer": f"{meta['optimizer']} (lr={meta['lr']}, cosine + warmup)",
        "augmentation": meta["augmentation"],
        "seed": cfg.seed,
        "device": device_info["torch"],
        "AMP": meta["amp"],
        "workers": meta["workers"],
        "결과 폴더": run_dir,
    })
    if device_info["device"] != "cuda":
        logger.warning("[경고] CPU 로 Faster R-CNN 을 학습하면 매우 오래 걸립니다. "
                       "`python tools/check_environment.py` 로 CUDA 상태를 확인하세요.")

    set_global_seed(cfg.seed, cfg.deterministic)

    import torch

    min_batch = int(cfg.oom_retry.get("min_batch", 1))
    oom_enabled = bool(cfg.oom_retry.get("enabled", True))

    # 같은 프로세스에서 여러 모델을 연달아 학습할 때 앞 모델의 최대치가 섞이지 않도록 초기화
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    start = time.perf_counter()

    while True:
        try:
            meta = train_once(cfg, args, logger, meta, meta["actual_batch"])
            meta["success"] = Path(meta.get("best_weights", "")).is_file()
            break
        except Exception as exc:
            if oom_enabled and _is_oom(exc) and meta["actual_batch"] > min_batch:
                new_batch = max(min_batch, meta["actual_batch"] // 2)
                logger.warning("[OOM] CUDA 메모리 부족 — batch %d -> %d 로 낮춰 재시도합니다.",
                               meta["actual_batch"], new_batch)
                meta["oom_retries"].append({"from_batch": meta["actual_batch"], "to_batch": new_batch,
                                            "error": f"{type(exc).__name__}: {exc}"[:400]})
                meta["actual_batch"] = new_batch
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
                continue
            meta["error"] = f"{type(exc).__name__}: {exc}"
            meta["traceback"] = traceback.format_exc()
            logger.error("[FAILED] %s 학습 실패: %s", display, meta["error"])
            logger.error(meta["traceback"])
            break

    meta["training_time_sec"] = round(time.perf_counter() - start, 2)
    meta["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        if device_info["device"] == "cuda":
            meta["train_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    except Exception:
        pass
    if meta["actual_batch"] != meta["requested_batch"]:
        meta["notes"].append(
            f"CUDA OOM 으로 batch 를 {meta['requested_batch']} -> {meta['actual_batch']} 로 낮춰 학습했습니다."
        )

    logger.info("")
    logger.info("[%s] %s 학습 종료 — 소요 %.1f초 (%.2f분)",
                "SUCCESS" if meta["success"] else "FAILED", display,
                meta["training_time_sec"], meta["training_time_sec"] / 60)
    if meta["success"]:
        logger.info("best weight: %s (epoch %s)", meta.get("best_weights"), meta.get("best_epoch"))
    save_training_meta(run_dir, meta)
    return meta


def main() -> int:
    args = build_parser().parse_args()
    meta = run_training(args)
    return 0 if meta.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
