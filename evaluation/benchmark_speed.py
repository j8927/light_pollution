"""추론 속도(FPS / Inference Time) 측정.

PowerShell 사용 예:
    python evaluation/benchmark_speed.py
    python evaluation/benchmark_speed.py --model yolov8

공정한 비교를 위해 모든 모델에 동일하게 적용하는 조건
    - 같은 device, 같은 이미지 크기(640), 같은 test 이미지 목록
    - 이미지를 미리 메모리에 올려 **디스크 로딩 시간을 추론 시간과 분리**
    - warm-up 을 수행하고 첫 실행 시간은 평균에서 제외
    - 여러 이미지를 여러 번 반복 추론해 평균 계산
    - preprocess / inference / postprocess / total 을 나눠 기록

구조적 차이(로그에 함께 기록)
    - YOLO 계열/RT-DETR : Ultralytics 가 보고하는 preprocess / inference / postprocess (ms)
    - Faster R-CNN      : letterbox+텐서 변환(preprocess), forward(inference),
                          좌표 역변환(postprocess) 을 직접 계측.
                          NMS 와 box decoding 이 forward 내부(roi_heads)에서 일어나므로
                          같은 이름의 구간이라도 포함하는 연산 범위가 완전히 같지는 않다.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lpcompare.config import MODEL_DISPLAY, load_config  # noqa: E402
from lpcompare.logging_utils import get_logger, log_banner  # noqa: E402
from lpcompare.runtime import read_image  # noqa: E402

FRAMEWORK_NOTE = {
    "ultralytics": "Ultralytics 내부 speed 계측 (preprocess/inference/postprocess)",
    "torchvision": "직접 계측 — NMS/box decoding 은 forward(inference) 구간에 포함됨",
}


def benchmark_detector(detector, image_paths: Sequence[Path], warmup: int = 10, repeat: int = 3,
                       max_images: int = 100, logger=None) -> Dict[str, Any]:
    """detector 를 반복 추론하며 평균 시간과 FPS 를 측정한다."""
    paths = [Path(p) for p in image_paths][:max(1, int(max_images))]

    # --- 이미지 로딩 (측정에서 분리) ---
    t0 = time.perf_counter()
    images = []
    for p in paths:
        try:
            images.append(read_image(p))
        except Exception:
            continue
    load_total_ms = (time.perf_counter() - t0) * 1000.0
    if not images:
        return {"error": "속도 측정에 사용할 이미지를 읽지 못했습니다."}

    # --- warm-up (측정 제외) ---
    for i in range(max(0, int(warmup))):
        detector.predict(images[i % len(images)])

    device = detector.cfg.device
    gpu_peak = None
    if device.startswith("cuda"):
        try:
            import torch

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    per_image_total: List[float] = []
    sums = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0, "total": 0.0}
    count = 0
    for r in range(max(1, int(repeat))):
        for img in images:
            t = time.perf_counter()
            _det, timing = detector.predict(img)
            wall_ms = (time.perf_counter() - t) * 1000.0
            for k in ("preprocess", "inference", "postprocess"):
                sums[k] += float(timing.get(k, 0.0))
            total = float(timing.get("total") or wall_ms)
            sums["total"] += total
            per_image_total.append(total)
            count += 1
        if logger:
            logger.info("    반복 %d/%d 완료 (%d장)", r + 1, int(repeat), len(images))

    if device.startswith("cuda"):
        try:
            import torch

            torch.cuda.synchronize()
            gpu_peak = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
        except Exception:
            gpu_peak = None

    avg = {k: round(v / max(1, count), 3) for k, v in sums.items()}
    out: Dict[str, Any] = {
        "images_used": len(images),
        "repeat": int(repeat),
        "warmup": int(warmup),
        "total_inferences": count,
        "preprocess_ms": avg["preprocess"],
        "inference_ms": avg["inference"],
        "postprocess_ms": avg["postprocess"],
        "total_ms": avg["total"],
        "median_total_ms": round(statistics.median(per_image_total), 3) if per_image_total else None,
        "fps": round(1000.0 / avg["total"], 2) if avg["total"] > 0 else None,
        "image_load_ms_per_image": round(load_total_ms / max(1, len(images)), 3),
        "device": device,
        "framework": detector.cfg.framework,
        "framework_note": FRAMEWORK_NOTE.get(detector.cfg.framework, ""),
        "inference_gpu_memory_mb": gpu_peak,
        "error": None,
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="모델 추론 속도 측정")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None, choices=list(MODEL_DISPLAY.keys()),
                        help="지정하지 않으면 학습된 모든 모델을 측정")
    args = parser.parse_args()

    from evaluation.calculate_metrics import load_test_set, resolve_model_weights
    from lpcompare.runtime import build_detector

    cfg = load_config(args.config)
    logger = get_logger("benchmark", cfg.logs_dir)
    log_banner(logger, "추론 속도 측정 (Benchmark)")

    images, _gts, names = load_test_set(cfg)
    if not images:
        logger.error("[실패] test 이미지가 없습니다. `python tools/prepare_splits.py` 를 먼저 실행하세요.")
        return 2

    device_info = cfg.resolve_device()
    for w in device_info["warnings"]:
        logger.warning(w)

    keys = [args.model] if args.model else cfg.enabled_models
    bench = cfg.benchmark
    ev = cfg.evaluation
    ok = 0
    for key in keys:
        display = MODEL_DISPLAY[key]
        weights, _meta = resolve_model_weights(cfg, key)
        if not weights:
            logger.warning("[SKIP] %s — 학습된 weight 없음 (%s)", display, cfg.run_dir(key))
            continue
        logger.info("")
        logger.info("[%s] 속도 측정 시작 — %s", display, weights)
        try:
            detector = build_detector(
                key, weights, device_info["torch"], cfg.image_size,
                conf=float(ev["conf_threshold"]), nms_iou=float(ev["nms_iou"]),
                max_det=int(ev["max_detections"]), class_names=names,
            )
            res = benchmark_detector(detector, images, int(bench["warmup"]), int(bench["repeat"]),
                                     int(bench["max_images"]), logger)
            logger.info("  preprocess %.2f ms | inference %.2f ms | postprocess %.2f ms | total %.2f ms | %.2f FPS",
                        res["preprocess_ms"], res["inference_ms"], res["postprocess_ms"],
                        res["total_ms"], res["fps"] or 0.0)
            logger.info("  계측 방식: %s", res["framework_note"])
            ok += 1
        except Exception as exc:
            logger.error("[FAILED] %s 속도 측정 실패: %s: %s", display, type(exc).__name__, exc)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
