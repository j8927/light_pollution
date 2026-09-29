"""학습된 4개 모델 중 하나로 추론을 수행하는 공통 진입점.

PowerShell 사용 예:
    python inference.py --model yolov8 --image test.jpg
    python inference.py --model yolo11 --image data/images/test/images/sample.jpg
    python inference.py --model rtdetr --image test.jpg --conf 0.4
    python inference.py --model faster_rcnn --image test.jpg
    python inference.py --model yolov8 --video sample.mp4          # 동영상
    python inference.py --model yolov8 --webcam 0                  # 웹캠 / 외부 카메라

출력
    - 검출된 클래스, confidence, bbox(xmin, ymin, xmax, ymax), 추론 시간
    - bbox 가 그려진 결과 이미지/영상 (기본 저장 위치: results/inference/)

학습 코드(training/)와 완전히 분리되어 있어, 이후 영상·카메라 파이프라인에 그대로 재사용할 수 있다.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lpcompare import dataset as ds  # noqa: E402
from lpcompare.config import MODEL_DISPLAY, load_config  # noqa: E402
from lpcompare.logging_utils import get_logger, log_banner  # noqa: E402
from lpcompare.runtime import build_detector, load_training_meta, read_image, resolve_weights  # noqa: E402
from lpcompare.viz import draw_detections, save_image  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="빛공해 객체탐지 — 단일 이미지 / 영상 / 카메라 추론")
    p.add_argument("--model", required=True, choices=list(MODEL_DISPLAY.keys()))
    p.add_argument("--config", default=None)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="이미지 파일 경로")
    src.add_argument("--video", help="동영상 파일 경로")
    src.add_argument("--webcam", type=int, help="웹캠/외부 카메라 인덱스 (예: 0)")
    p.add_argument("--weights", default=None, help="사용할 weight 경로 (기본: runs/<model>/weights/best.pt)")
    p.add_argument("--conf", type=float, default=None, help="confidence 임계값 (기본: 설정값)")
    p.add_argument("--device", default=None, help="cuda | cpu | 0")
    p.add_argument("--out", default=None, help="결과 저장 폴더 (기본: results/inference)")
    p.add_argument("--no-save", action="store_true", help="결과 이미지를 저장하지 않는다")
    p.add_argument("--max-frames", type=int, default=0, help="영상/카메라에서 처리할 최대 프레임 수 (0 = 제한 없음)")
    return p


def load_detector(cfg, model_key: str, weights: Optional[str], conf: Optional[float],
                  device_override: Optional[str], logger):
    layout = ds.detect_layout(cfg.dataset_root, cfg.data_yaml, cfg.fallback_names)
    names = layout.names

    if weights:
        weight_path = Path(weights)
    else:
        meta = load_training_meta(cfg.run_dir(model_key))
        weight_path = None
        if meta.get("best_weights") and Path(meta["best_weights"]).is_file():
            weight_path = Path(meta["best_weights"])
        else:
            weight_path = resolve_weights(cfg.run_dir(model_key))
    if not weight_path or not Path(weight_path).is_file():
        raise FileNotFoundError(
            f"{MODEL_DISPLAY[model_key]} 의 학습된 weight 를 찾을 수 없습니다 ({cfg.run_dir(model_key)}). "
            f"먼저 학습하거나 --weights 로 경로를 지정하세요."
        )

    if device_override:
        cfg.raw["common"]["device"] = device_override
    device_info = cfg.resolve_device()
    for w in device_info["warnings"]:
        logger.warning(w)

    ev = cfg.evaluation
    detector = build_detector(
        model_key, weight_path, device_info["torch"], cfg.image_size,
        conf=float(conf) if conf is not None else float(ev["conf_threshold"]),
        nms_iou=float(ev["nms_iou"]), max_det=int(ev["max_detections"]),
        class_names=names,
    )
    logger.info("모델   : %s", MODEL_DISPLAY[model_key])
    logger.info("weight : %s", weight_path)
    logger.info("device : %s", device_info["torch"])
    logger.info("conf   : %s / NMS IoU: %s / imgsz: %s", detector.cfg.conf, ev["nms_iou"], cfg.image_size)
    return detector


def print_detections(logger, det, names: List[str], timing) -> None:
    logger.info("")
    logger.info("검출 결과: %d개", len(det))
    if len(det):
        logger.info("  %-4s%-12s%-12s%s", "No", "Class", "Confidence", "BBox (xmin, ymin, xmax, ymax)")
        order = np.argsort(-det.scores)
        for n, i in enumerate(order, start=1):
            cls = int(det.labels[i])
            name = names[cls] if 0 <= cls < len(names) else f"class{cls}"
            x1, y1, x2, y2 = det.boxes[i]
            logger.info("  %-4d%-12s%-12.4f(%.1f, %.1f, %.1f, %.1f)",
                        n, name, float(det.scores[i]), x1, y1, x2, y2)
    else:
        logger.info("  (검출된 객체 없음)")
    logger.info("")
    logger.info("추론 시간: preprocess %.2f ms + inference %.2f ms + postprocess %.2f ms = %.2f ms (%.1f FPS)",
                timing.get("preprocess", 0.0), timing.get("inference", 0.0),
                timing.get("postprocess", 0.0), timing.get("total", 0.0),
                1000.0 / timing["total"] if timing.get("total") else 0.0)
    if "load" in timing:
        logger.info("이미지 로딩 시간(추론 시간과 별도): %.2f ms", timing["load"])


def run_image(cfg, args, detector, logger) -> int:
    path = Path(args.image)
    if not path.is_file():
        logger.error("[실패] 이미지를 찾을 수 없습니다: %s", path)
        return 2

    det, timing = detector.predict_path(path)
    names = detector.class_names
    print_detections(logger, det, names, timing)

    if not args.no_save:
        out_dir = Path(args.out) if args.out else (cfg.results_dir / "inference")
        drawn = draw_detections(read_image(path), det.boxes, det.scores, det.labels, names,
                                title=f"{detector.cfg.display_name}  ({len(det)}개, {timing['total']:.1f}ms)")
        out_path = save_image(out_dir / f"{path.stem}_{detector.cfg.key}.jpg", drawn)
        logger.info("결과 이미지 저장: %s", out_path)
    return 0


def run_stream(cfg, args, detector, logger) -> int:
    """동영상 파일 또는 카메라 입력 처리 (이후 실시간 파이프라인 확장을 위한 기본 구현)."""
    import cv2

    if args.video:
        source: object = str(args.video)
        label = Path(args.video).stem
        if not Path(args.video).is_file():
            logger.error("[실패] 동영상을 찾을 수 없습니다: %s", args.video)
            return 2
    else:
        source = int(args.webcam)
        label = f"webcam{args.webcam}"

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("[실패] 입력을 열 수 없습니다: %s", source)
        return 2

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 640
    writer = None
    out_path = None
    if not args.no_save:
        out_dir = Path(args.out) if args.out else (cfg.results_dir / "inference")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{label}_{detector.cfg.key}.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_in if fps_in > 0 else 20.0, (width, height))

    names = detector.class_names
    frames, total_ms, total_det = 0, 0.0, 0
    t_start = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            det, timing = detector.predict(frame)
            drawn = draw_detections(frame, det.boxes, det.scores, det.labels, names,
                                    title=f"{detector.cfg.display_name}  {timing['total']:.1f}ms")
            if writer is not None:
                writer.write(cv2.resize(drawn, (width, height)))
            frames += 1
            total_ms += float(timing.get("total", 0.0))
            total_det += len(det)
            if frames % 30 == 0:
                logger.info("  %d 프레임 처리 — 평균 %.1f ms (%.1f FPS)",
                            frames, total_ms / frames, 1000.0 * frames / max(total_ms, 1e-6))
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    wall = time.perf_counter() - t_start
    logger.info("")
    logger.info("처리 프레임 : %d", frames)
    logger.info("총 검출 수  : %d", total_det)
    if frames:
        logger.info("평균 추론   : %.2f ms (%.1f FPS, 모델 기준)", total_ms / frames,
                    1000.0 * frames / max(total_ms, 1e-6))
        logger.info("전체 처리   : %.1f초 (%.1f FPS, 디코딩·그리기 포함)", wall, frames / max(wall, 1e-6))
    if out_path:
        logger.info("결과 영상 저장: %s", out_path)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    logger = get_logger("inference", cfg.logs_dir)
    log_banner(logger, f"{MODEL_DISPLAY[args.model]} 추론")

    try:
        detector = load_detector(cfg, args.model, args.weights, args.conf, args.device, logger)
    except Exception as exc:
        logger.error("[실패] %s", exc)
        return 2

    if args.image:
        return run_image(cfg, args, detector, logger)
    return run_stream(cfg, args, detector, logger)


if __name__ == "__main__":
    raise SystemExit(main())
