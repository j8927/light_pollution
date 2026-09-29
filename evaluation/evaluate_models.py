"""학습된 모델들을 **동일한 test dataset · 동일한 기준**으로 평가하고 결과를 통합한다.

PowerShell 사용 예:
    python evaluation/evaluate_models.py
    python evaluation/evaluate_models.py --model yolov8
    python evaluation/evaluate_models.py --no-speed        # 속도 측정 생략

생성되는 결과
    results/model_comparison.csv          4개 모델 통합 비교표
    results/<model>_class_metrics.csv     클래스별 지표
    results/object_size_metrics.csv       객체 크기별(Small/Medium/Large) AP
    results/confusion_matrix/*.png        모델별 혼동행렬
    results/evaluation_summary.json       원본 수치(JSON)

측정하지 못한 값은 빈 칸으로 두고 그 이유를 로그와 JSON 에 남긴다. 값을 임의로 만들지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.benchmark_speed import benchmark_detector  # noqa: E402
from evaluation.calculate_metrics import (evaluate_single_model, load_test_set,  # noqa: E402
                                          nan_to_none, write_class_metrics_csv)
from lpcompare.config import MODEL_DISPLAY, load_config  # noqa: E402
from lpcompare.logging_utils import get_logger, log_banner  # noqa: E402
from lpcompare.viz import plot_confusion_matrix, setup_matplotlib  # noqa: E402

COMPARISON_COLUMNS = [
    "Model", "Model_Key", "Framework", "Status",
    "Precision", "Recall", "F1", "mAP50", "mAP50_95",
    "FPS", "Inference_ms", "Preprocess_ms", "Postprocess_ms", "Total_ms",
    "Parameters", "GFLOPs", "Model_Size_MB",
    "GPU_Memory_MB", "Train_GPU_Memory_MB",
    "Training_Time", "Training_Time_sec",
    "Batch_Size", "Requested_Batch", "Image_Size", "Epochs", "Seed", "Device",
    "TP", "FP", "FN", "Test_Images",
    "Small_AP50_95", "Medium_AP50_95", "Large_AP50_95",
    "Conf_Threshold", "IoU_Threshold", "Weights", "Note",
]


def _fmt_seconds(sec: Optional[float]) -> str:
    if sec is None:
        return ""
    try:
        return str(timedelta(seconds=int(round(float(sec)))))
    except Exception:
        return ""


def evaluate_model(cfg, model_key: str, logger, images, gts, names,
                   run_speed: bool = True) -> Dict[str, Any]:
    display = MODEL_DISPLAY[model_key]
    logger.info("")
    logger.info("-" * 78)
    logger.info("[평가] %s", display)
    logger.info("-" * 78)

    out = evaluate_single_model(cfg, model_key, logger, images, gts, names)
    if out.get("error"):
        logger.error("[FAILED] %s — %s", display, out["error"])
        return out

    result = out["result"]
    overall, pr = result["overall"], result["pr"]
    logger.info("  mAP50 %.4f | mAP50-95 %.4f | P %.4f | R %.4f | F1 %.4f (conf=%.2f, IoU=%.2f)",
                overall["map50"], overall["map50_95"], pr["precision_mean"],
                pr["recall_mean"], pr["f1_mean"], pr["conf"], pr["iou"])
    logger.info("  TP %d / FP %d / FN %d  (FP=오탐, FN=미탐)",
                pr["tp_total"], pr["fp_total"], pr["fn_total"])
    tm = result.get("torchmetrics") or {}
    if tm.get("unavailable"):
        logger.info("  [교차검증] torchmetrics mAP 생략 — %s", tm["unavailable"])
    elif tm:
        logger.info("  [교차검증] torchmetrics mAP50 %.4f / mAP50-95 %.4f "
                    "(위 값과 차이가 크면 평가 코드를 점검할 것)",
                    tm["map50"], tm["map50_95"])

    # 클래스별 CSV
    class_csv = cfg.results_dir / f"{model_key}_class_metrics.csv"
    write_class_metrics_csv(class_csv, names, result, gts)
    out["class_metrics_csv"] = str(class_csv)
    logger.info("  클래스별 지표: %s", class_csv)

    # 혼동행렬
    try:
        cm_path = cfg.results_dir / "confusion_matrix" / f"{model_key}_confusion_matrix.png"
        plot_confusion_matrix(cm_path, result["confusion_matrix"], names,
                              f"{display} Confusion Matrix (conf={pr['conf']}, IoU={pr['iou']})")
        out["confusion_matrix_png"] = str(cm_path)
        logger.info("  혼동행렬: %s", cm_path)
    except Exception as exc:
        logger.warning("  혼동행렬 생성 실패: %s: %s", type(exc).__name__, exc)

    # 속도 측정
    if run_speed:
        bench = cfg.benchmark
        logger.info("  속도 측정 중 (warmup %s, repeat %s, 최대 %s장)...",
                    bench["warmup"], bench["repeat"], bench["max_images"])
        try:
            out["speed"] = benchmark_detector(
                out["detector"], images, int(bench["warmup"]), int(bench["repeat"]),
                int(bench["max_images"]), logger,
            )
            s = out["speed"]
            logger.info("  %.2f FPS | total %.2f ms (pre %.2f / inf %.2f / post %.2f)",
                        s["fps"] or 0.0, s["total_ms"], s["preprocess_ms"],
                        s["inference_ms"], s["postprocess_ms"])
            logger.info("  계측 방식: %s", s["framework_note"])
        except Exception as exc:
            logger.warning("  속도 측정 실패: %s: %s", type(exc).__name__, exc)
            out["speed"] = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        out["speed"] = {"error": "--no-speed 옵션으로 측정하지 않음"}

    info = out.get("info", {})
    logger.info("  파라미터 %s | 모델 크기 %s MB | GFLOPs %s",
                f"{info.get('parameters'):,}" if info.get("parameters") else "N/A",
                info.get("model_size_mb") or "N/A",
                info.get("gflops") if info.get("gflops") is not None else "N/A")
    if info.get("gflops") is None and info.get("gflops_source"):
        logger.info("  GFLOPs 미기록 사유: %s", info["gflops_source"])
    return out


def build_row(cfg, out: Dict[str, Any]) -> Dict[str, Any]:
    key = out["model_key"]
    display = out["display_name"]
    meta = out.get("training_meta", {}) or {}
    ev = cfg.evaluation

    row = {c: "" for c in COMPARISON_COLUMNS}
    row["Model"] = display
    row["Model_Key"] = key
    row["Framework"] = meta.get("framework", "torchvision" if key == "faster_rcnn" else "ultralytics")
    row["Image_Size"] = meta.get("image_size", cfg.image_size)
    row["Epochs"] = meta.get("epochs", "")
    row["Seed"] = meta.get("seed", cfg.seed)
    row["Device"] = meta.get("device_detail", "")
    row["Batch_Size"] = meta.get("actual_batch", "")
    row["Requested_Batch"] = meta.get("requested_batch", "")
    row["Training_Time_sec"] = meta.get("training_time_sec", "")
    row["Training_Time"] = _fmt_seconds(meta.get("training_time_sec"))
    row["Train_GPU_Memory_MB"] = meta.get("train_gpu_memory_mb") or ""
    row["Conf_Threshold"] = ev["conf_threshold"]
    row["IoU_Threshold"] = ev["iou_threshold"]
    row["Weights"] = out.get("weights", "")
    notes: List[str] = list(meta.get("notes") or [])

    if out.get("error"):
        row["Status"] = "FAILED"
        notes.append(out["error"])
        row["Note"] = " / ".join(n for n in notes if n)
        return row

    row["Status"] = "OK"
    result = out["result"]
    overall, pr = result["overall"], result["pr"]
    row["Precision"] = nan_to_none(pr["precision_mean"])
    row["Recall"] = nan_to_none(pr["recall_mean"])
    row["F1"] = nan_to_none(pr["f1_mean"])
    row["mAP50"] = nan_to_none(overall["map50"])
    row["mAP50_95"] = nan_to_none(overall["map50_95"])
    row["TP"] = int(pr["tp_total"])
    row["FP"] = int(pr["fp_total"])
    row["FN"] = int(pr["fn_total"])
    row["Test_Images"] = out.get("test_images", "")

    size_ap = result["size_ap"]
    row["Small_AP50_95"] = nan_to_none(size_ap["small"]["map50_95"])
    row["Medium_AP50_95"] = nan_to_none(size_ap["medium"]["map50_95"])
    row["Large_AP50_95"] = nan_to_none(size_ap["large"]["map50_95"])

    speed = out.get("speed") or {}
    if not speed.get("error"):
        row["FPS"] = speed.get("fps", "")
        row["Inference_ms"] = speed.get("inference_ms", "")
        row["Preprocess_ms"] = speed.get("preprocess_ms", "")
        row["Postprocess_ms"] = speed.get("postprocess_ms", "")
        row["Total_ms"] = speed.get("total_ms", "")
        row["GPU_Memory_MB"] = speed.get("inference_gpu_memory_mb") or ""
    else:
        notes.append(f"속도 미측정: {speed.get('error')}")

    info = out.get("info", {})
    row["Parameters"] = info.get("parameters") or ""
    row["GFLOPs"] = info.get("gflops") if info.get("gflops") is not None else ""
    row["Model_Size_MB"] = info.get("model_size_mb") or ""
    if info.get("gflops") is None and info.get("gflops_source"):
        notes.append(f"GFLOPs {info['gflops_source']}")
    if not row["GPU_Memory_MB"]:
        notes.append("GPU 메모리 미측정 (CPU 실행 또는 측정 실패)")

    row["Note"] = " / ".join(n for n in notes if n)
    return row


def write_comparison_csv(path: Path, rows: List[Dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COMPARISON_COLUMNS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: ("" if r.get(k) is None else r.get(k, "")) for k in COMPARISON_COLUMNS})
    return path


def write_object_size_csv(path: Path, outs: List[Dict[str, Any]], criterion: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    desc = ("COCO 기준 (small < 32^2 px, medium < 96^2 px, large >= 96^2 px)"
            if criterion == "coco" else
            "상대 면적 기준 (small < 0.1%, medium < 1%, large >= 1% of image area)")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["# 기준", desc])
        wr.writerow(["Model", "Criterion", "Small_AP50", "Small_AP50_95", "Small_GT",
                     "Medium_AP50", "Medium_AP50_95", "Medium_GT",
                     "Large_AP50", "Large_AP50_95", "Large_GT"])
        for out in outs:
            if out.get("error"):
                continue
            s = out["result"]["size_ap"]
            wr.writerow([
                out["display_name"], criterion,
                nan_to_none(s["small"]["map50"]), nan_to_none(s["small"]["map50_95"]), s["small"]["gt_count"],
                nan_to_none(s["medium"]["map50"]), nan_to_none(s["medium"]["map50_95"]), s["medium"]["gt_count"],
                nan_to_none(s["large"]["map50"]), nan_to_none(s["large"]["map50_95"]), s["large"]["gt_count"],
            ])
    return path


def print_summary(logger, rows: List[Dict[str, Any]]) -> None:
    logger.info("")
    logger.info("=" * 100)
    logger.info("%-14s%-11s%-9s%-8s%-9s%-11s%-9s%-12s%s",
                "Model", "Precision", "Recall", "F1", "mAP50", "mAP50-95", "FPS", "Size(MB)", "Status")
    logger.info("=" * 100)

    def fmt(v, nd=4):
        if v in ("", None):
            return "N/A"
        try:
            return f"{float(v):.{nd}f}"
        except (TypeError, ValueError):
            return str(v)

    for r in rows:
        logger.info("%-14s%-11s%-9s%-8s%-9s%-11s%-9s%-12s%s",
                    r["Model"], fmt(r["Precision"]), fmt(r["Recall"]), fmt(r["F1"]),
                    fmt(r["mAP50"]), fmt(r["mAP50_95"]), fmt(r["FPS"], 2),
                    fmt(r["Model_Size_MB"], 2), r["Status"])
    logger.info("=" * 100)
    logger.info("※ 모델 크기와 GPU 요구량이 서로 다르므로 정확도만으로 우열을 판단하지 마세요.")
    logger.info("※ 각 모델의 batch size / 학습 시간 / GFLOPs 차이는 results/model_comparison.csv 에 그대로 기록되어 있습니다.")


def main() -> int:
    parser = argparse.ArgumentParser(description="4개 모델 통합 평가")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", action="append", choices=list(MODEL_DISPLAY.keys()),
                        help="평가할 모델 (여러 번 지정 가능, 지정하지 않으면 설정에서 enabled 된 모든 모델)")
    parser.add_argument("--no-speed", action="store_true", help="추론 속도 측정을 생략한다")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="평가에 사용할 split (기본 test)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("evaluation", cfg.logs_dir)
    log_banner(logger, "모델 평가 (동일 Test Dataset · 동일 기준)")
    setup_matplotlib()

    device_info = cfg.resolve_device()
    for w in device_info["warnings"]:
        logger.warning(w)

    images, gts, names = load_test_set(cfg, args.split)
    if not images:
        logger.error("[실패] %s split 이미지가 없습니다. `python tools/prepare_splits.py` 를 먼저 실행하세요.",
                     args.split)
        return 2
    ev = cfg.evaluation
    logger.info("평가 대상: %s split %d장 / 클래스 %d개 (%s)", args.split, len(images), len(names), ", ".join(names))
    logger.info("공통 평가 조건: conf=%s (P/R/F1), mAP conf=%s, IoU=%s, NMS IoU=%s, imgsz=%s",
                ev["conf_threshold"], ev["map_conf_threshold"], ev["iou_threshold"],
                ev["nms_iou"], cfg.image_size)

    keys = args.model if args.model else cfg.enabled_models
    outs: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for key in keys:
        try:
            out = evaluate_model(cfg, key, logger, images, gts, names, run_speed=not args.no_speed)
        except Exception as exc:
            import traceback

            logger.error("[FAILED] %s 평가 중 예외: %s: %s", MODEL_DISPLAY[key], type(exc).__name__, exc)
            logger.error(traceback.format_exc())
            out = {"model_key": key, "display_name": MODEL_DISPLAY[key],
                   "error": f"{type(exc).__name__}: {exc}"}
        outs.append(out)
        rows.append(build_row(cfg, out))

    # --- 기존 결과와 병합 (한 모델만 평가해도 다른 모델 결과를 지우지 않는다) ---
    comparison_path = cfg.results_dir / "model_comparison.csv"
    merged: Dict[str, Dict[str, Any]] = {}
    if comparison_path.is_file():
        try:
            with open(comparison_path, "r", encoding="utf-8-sig", newline="") as f:
                for old in csv.DictReader(f):
                    if old.get("Model_Key"):
                        merged[old["Model_Key"]] = {k: old.get(k, "") for k in COMPARISON_COLUMNS}
        except Exception as exc:
            logger.warning("기존 model_comparison.csv 를 읽지 못했습니다: %s", exc)
    for r in rows:
        merged[r["Model_Key"]] = r
    ordered = [merged[k] for k in MODEL_DISPLAY if k in merged]

    write_comparison_csv(comparison_path, ordered)
    logger.info("")
    logger.info("통합 비교표: %s", comparison_path)

    size_csv = write_object_size_csv(cfg.results_dir / "object_size_metrics.csv", outs,
                                     str(ev.get("area_criterion", "coco")))
    logger.info("객체 크기별 성능: %s", size_csv)

    # 원본 수치 JSON (그래프/문서에서 재사용)
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "split": args.split,
        "num_images": len(images),
        "class_names": names,
        "evaluation_settings": dict(ev),
        "image_size": cfg.image_size,
        "device": device_info["torch"],
        "models": {},
    }
    for out, row in zip(outs, rows):
        entry: Dict[str, Any] = {"row": row, "error": out.get("error")}
        if not out.get("error"):
            r = out["result"]
            entry["metrics"] = {
                "map50": nan_to_none(r["overall"]["map50"]),
                "map50_95": nan_to_none(r["overall"]["map50_95"]),
                "per_class_ap50": [nan_to_none(v) for v in r["overall"]["ap50"]],
                "per_class_ap50_95": [nan_to_none(v) for v in r["overall"]["ap50_95"]],
                "precision": nan_to_none(r["pr"]["precision_mean"]),
                "recall": nan_to_none(r["pr"]["recall_mean"]),
                "f1": nan_to_none(r["pr"]["f1_mean"]),
                "confusion_matrix": r["confusion_matrix"].tolist(),
                "size_ap": r["size_ap"],
                "torchmetrics_cross_check": r.get("torchmetrics"),
            }
            entry["speed"] = out.get("speed")
            entry["model_info"] = out.get("info")
            entry["training_meta"] = out.get("training_meta")
        summary["models"][out["model_key"]] = entry

    json_path = cfg.results_dir / "evaluation_summary.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    logger.info("원본 수치 JSON: %s", json_path)

    print_summary(logger, ordered)
    failed = [r["Model"] for r in rows if r["Status"] != "OK"]
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
