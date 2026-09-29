"""동일한 기준으로 모델 지표를 계산하고 클래스별 CSV 를 쓰는 모듈.

evaluate_models.py 가 이 모듈을 사용하며, 단독 실행으로 한 모델만 평가할 수도 있다.

PowerShell 사용 예:
    python evaluation/calculate_metrics.py --model yolov8
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lpcompare import dataset as ds  # noqa: E402
from lpcompare.config import MODEL_DISPLAY, load_config  # noqa: E402
from lpcompare.logging_utils import get_logger, log_banner  # noqa: E402
from lpcompare.metrics import (Detections, GroundTruth, evaluate_all,  # noqa: E402
                               torchmetrics_cross_check)
from lpcompare.runtime import build_detector, load_training_meta, resolve_weights  # noqa: E402


def nan_to_none(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else round(f, 5)


def load_test_set(cfg, split: str = "test") -> Tuple[List[Path], List[GroundTruth], List[str]]:
    """split 목록과 정답(GT)을 원본 이미지 좌표계로 읽는다. 모든 모델이 이 목록을 공유한다."""
    layout = ds.detect_layout(cfg.dataset_root, cfg.data_yaml, cfg.fallback_names)
    names = layout.names
    nc = len(names)
    paths = ds.read_split_file(Path(cfg.splits_dir) / f"{split}.txt")

    images: List[Path] = []
    gts: List[GroundTruth] = []
    for p in paths:
        if not p.is_file():
            continue
        try:
            w, h = ds.image_size(p)
        except Exception:
            continue
        boxes, labels = ds.load_ground_truth(p, w, h, nc)
        images.append(p)
        gts.append(GroundTruth(
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(labels, dtype=np.int32).reshape(-1), w, h,
        ))
    return images, gts, names


def run_predictions(detector, image_paths: Sequence[Path], logger=None,
                    progress_every: int = 20) -> Tuple[List[Detections], Dict[str, float]]:
    """모든 test 이미지에 대해 예측을 수행한다 (평가용 — 속도 측정은 benchmark_speed.py)."""
    preds: List[Detections] = []
    acc = {"load": 0.0, "preprocess": 0.0, "inference": 0.0, "postprocess": 0.0, "total": 0.0}
    for i, p in enumerate(image_paths, start=1):
        det, timing = detector.predict_path(p)
        preds.append(det)
        for k in acc:
            acc[k] += float(timing.get(k, 0.0))
        if logger and progress_every and (i % progress_every == 0 or i == len(image_paths)):
            logger.info("    예측 진행: %d / %d", i, len(image_paths))
    n = max(1, len(image_paths))
    return preds, {k: round(v / n, 3) for k, v in acc.items()}


def compute_metrics(preds: Sequence[Detections], gts: Sequence[GroundTruth], names: Sequence[str],
                    conf: float, iou: float, area_criterion: str) -> Dict[str, Any]:
    result = evaluate_all(preds, gts, len(names), conf=conf, iou_thr=iou, area_criterion=area_criterion)
    cross = torchmetrics_cross_check(preds, gts)
    if cross:
        result["torchmetrics"] = cross
    return result


def write_class_metrics_csv(path: Path, names: Sequence[str], result: Dict[str, Any],
                            gts: Sequence[GroundTruth]) -> Path:
    """클래스별 Precision / Recall / F1 / AP50 / AP50-95 CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pr = result["pr"]
    overall = result["overall"]

    images_with_class = np.zeros(len(names), dtype=np.int64)
    for gt in gts:
        for c in set(int(x) for x in gt.labels.tolist()):
            if 0 <= c < len(names):
                images_with_class[c] += 1

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Class", "Images", "GT_Instances", "TP", "FP", "FN",
                     "Precision", "Recall", "F1", "AP50", "AP50_95"])
        for i, name in enumerate(names):
            wr.writerow([
                name,
                int(images_with_class[i]),
                int(overall["gt_counts"][i]),
                int(pr["tp"][i]), int(pr["fp"][i]), int(pr["fn"][i]),
                nan_to_none(pr["precision"][i]),
                nan_to_none(pr["recall"][i]),
                nan_to_none(pr["f1"][i]),
                nan_to_none(overall["ap50"][i]),
                nan_to_none(overall["ap50_95"][i]),
            ])
        wr.writerow([
            "all",
            len(gts),
            int(overall["gt_counts"].sum()),
            int(pr["tp"].sum()), int(pr["fp"].sum()), int(pr["fn"].sum()),
            nan_to_none(pr["precision_mean"]),
            nan_to_none(pr["recall_mean"]),
            nan_to_none(pr["f1_mean"]),
            nan_to_none(overall["map50"]),
            nan_to_none(overall["map50_95"]),
        ])
    return path


def resolve_model_weights(cfg, model_key: str) -> Tuple[Optional[Path], Dict[str, Any]]:
    run_dir = cfg.run_dir(model_key)
    meta = load_training_meta(run_dir)
    weights = None
    if meta.get("best_weights") and Path(meta["best_weights"]).is_file():
        weights = Path(meta["best_weights"])
    else:
        weights = resolve_weights(run_dir)
    return weights, meta


def evaluate_single_model(cfg, model_key: str, logger, images=None, gts=None, names=None) -> Dict[str, Any]:
    """한 모델을 평가해 지표 dict 를 돌려준다. 실패 시 error 키를 채운다."""
    display = MODEL_DISPLAY.get(model_key, model_key)
    out: Dict[str, Any] = {"model_key": model_key, "display_name": display, "error": None}

    if images is None or gts is None or names is None:
        images, gts, names = load_test_set(cfg)
    out["test_images"] = len(images)

    weights, meta = resolve_model_weights(cfg, model_key)
    out["training_meta"] = meta
    if not weights:
        out["error"] = (f"학습된 weight 를 찾을 수 없습니다: {cfg.run_dir(model_key)}. "
                        f"먼저 학습을 실행하세요.")
        return out
    out["weights"] = str(weights)

    device_info = cfg.resolve_device()
    ev = cfg.evaluation
    detector = build_detector(
        model_key, weights, device_info["torch"], cfg.image_size,
        conf=float(ev["map_conf_threshold"]), nms_iou=float(ev["nms_iou"]),
        max_det=int(ev["max_detections"]), class_names=names,
    )
    logger.info("  weight : %s", weights)
    logger.info("  device : %s", device_info["torch"])

    preds, avg_time = run_predictions(detector, images, logger)
    out["predict_avg_ms"] = avg_time

    result = compute_metrics(preds, gts, names,
                             conf=float(ev["conf_threshold"]), iou=float(ev["iou_threshold"]),
                             area_criterion=str(ev.get("area_criterion", "coco")))
    out["result"] = result
    out["info"] = detector.info()
    out["preds"] = preds
    out["detector"] = detector
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="단일 모델 지표 계산")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", required=True, choices=list(MODEL_DISPLAY.keys()))
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("evaluation", cfg.logs_dir)
    log_banner(logger, f"{MODEL_DISPLAY[args.model]} 지표 계산")

    out = evaluate_single_model(cfg, args.model, logger)
    if out.get("error"):
        logger.error("[FAILED] %s", out["error"])
        return 1

    r = out["result"]
    names = ds.detect_layout(cfg.dataset_root, cfg.data_yaml, cfg.fallback_names).names
    csv_path = cfg.results_dir / f"{args.model}_class_metrics.csv"
    _, gts, _ = load_test_set(cfg)
    write_class_metrics_csv(csv_path, names, r, gts)

    logger.info("mAP50 %.4f | mAP50-95 %.4f | P %.4f | R %.4f | F1 %.4f",
                r["overall"]["map50"], r["overall"]["map50_95"],
                r["pr"]["precision_mean"], r["pr"]["recall_mean"], r["pr"]["f1_mean"])
    logger.info("클래스별 지표: %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
