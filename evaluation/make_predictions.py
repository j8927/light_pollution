"""동일한 test 이미지에 대해 4개 모델의 예측 이미지와 비교 이미지를 생성한다.

PowerShell 사용 예:
    python evaluation/make_predictions.py
    python evaluation/make_predictions.py --num 20

출력
    results/predictions/original/       원본 이미지 (주석 없음)
    results/predictions/ground_truth/   정답 bbox 를 그린 이미지
    results/predictions/<model>/        모델별 예측 결과 (클래스명 + confidence 표시)
    results/comparison_images/          같은 이미지에 대한 정답 + 4개 모델 결과를 한 장에 배치

모든 모델에 **정확히 같은 test 이미지**를 입력한다 (splits/test.txt 기준, seed 42 로 선택).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.calculate_metrics import load_test_set, resolve_model_weights  # noqa: E402
from lpcompare import dataset as ds  # noqa: E402
from lpcompare.config import MODEL_DISPLAY, load_config  # noqa: E402
from lpcompare.logging_utils import get_logger, log_banner  # noqa: E402
from lpcompare.runtime import build_detector, read_image  # noqa: E402
from lpcompare.viz import draw_detections, make_grid, save_image  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="모델별 예측 이미지 / 비교 이미지 생성")
    parser.add_argument("--config", default=None)
    parser.add_argument("--num", type=int, default=12, help="사용할 test 이미지 수 (기본 12)")
    parser.add_argument("--all", action="store_true", help="test 이미지 전체에 대해 생성")
    parser.add_argument("--conf", type=float, default=None, help="표시할 confidence 임계값 (기본: 설정값)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("predictions", cfg.logs_dir)
    log_banner(logger, "예측 이미지 / 비교 이미지 생성")

    images, gts, names = load_test_set(cfg)
    if not images:
        logger.error("[실패] test 이미지가 없습니다. `python tools/prepare_splits.py` 를 먼저 실행하세요.")
        return 2

    if args.all or args.num >= len(images):
        selected_idx = list(range(len(images)))
    else:
        # seed 42 로 고정 선택 — 실행할 때마다 같은 이미지를 쓴다
        order = ds.deterministic_shuffle(images, cfg.seed)
        chosen = {str(p) for p in order[:max(1, args.num)]}
        selected_idx = [i for i, p in enumerate(images) if str(p) in chosen]

    sel_images = [images[i] for i in selected_idx]
    sel_gts = [gts[i] for i in selected_idx]
    logger.info("선택된 test 이미지 %d장 (전체 %d장 중)", len(sel_images), len(images))

    ev = cfg.evaluation
    conf = float(args.conf) if args.conf is not None else float(ev["conf_threshold"])
    device_info = cfg.resolve_device()
    for w in device_info["warnings"]:
        logger.warning(w)

    pred_root = cfg.results_dir / "predictions"
    comp_root = cfg.results_dir / "comparison_images"

    # --- 원본 / 정답 이미지 ---
    raw_images: Dict[str, np.ndarray] = {}
    for p, gt in zip(sel_images, sel_gts):
        img = read_image(p)
        raw_images[str(p)] = img
        save_image(pred_root / "original" / f"{p.stem}.jpg", img)
        gt_img = draw_detections(img, gt.boxes, np.ones(len(gt), dtype=np.float32), gt.labels,
                                 names, title="Ground Truth", show_conf=False)
        save_image(pred_root / "ground_truth" / f"{p.stem}.jpg", gt_img)
    logger.info("원본 / 정답 이미지 저장: %s", pred_root)

    # --- 모델별 예측 ---
    panels_per_image: Dict[str, List] = {str(p): [] for p in sel_images}
    used_models: List[str] = []

    for key in cfg.enabled_models:
        display = MODEL_DISPLAY[key]
        weights, _meta = resolve_model_weights(cfg, key)
        if not weights:
            logger.warning("[SKIP] %s — 학습된 weight 없음 (%s)", display, cfg.run_dir(key))
            continue
        logger.info("")
        logger.info("[%s] 예측 이미지 생성 — %s", display, weights)
        try:
            detector = build_detector(
                key, weights, device_info["torch"], cfg.image_size,
                conf=conf, nms_iou=float(ev["nms_iou"]),
                max_det=int(ev["max_detections"]), class_names=names,
            )
        except Exception as exc:
            logger.error("[FAILED] %s 모델 로딩 실패: %s: %s", display, type(exc).__name__, exc)
            continue

        used_models.append(key)
        for p in sel_images:
            img = raw_images[str(p)]
            try:
                det, timing = detector.predict(img)
                # 개별 저장본에는 이미지 안에 모델명을 표시하고, 비교 이미지에는 패널 제목만 사용한다
                save_image(pred_root / key / f"{p.stem}.jpg",
                           draw_detections(img, det.boxes, det.scores, det.labels, names,
                                           title=f"{display}  ({len(det)}개, {timing['total']:.1f}ms)"))
                panel = draw_detections(img, det.boxes, det.scores, det.labels, names)
                panels_per_image[str(p)].append(
                    (f"{display}  |  검출 {len(det)}개  |  {timing['total']:.1f}ms", panel))
            except Exception as exc:
                logger.error("  [FAILED] %s: %s: %s", p.name, type(exc).__name__, exc)
        logger.info("  저장: %s", pred_root / key)

    if not used_models:
        logger.error("[실패] 사용할 수 있는 학습된 모델이 없습니다.")
        return 1

    # --- 비교 이미지 ---
    logger.info("")
    made = 0
    for p, gt in zip(sel_images, sel_gts):
        panels = list(panels_per_image[str(p)])
        if not panels:
            continue
        gt_img = draw_detections(raw_images[str(p)], gt.boxes,
                                 np.ones(len(gt), dtype=np.float32), gt.labels, names,
                                 show_conf=False)
        panels = [(f"Ground Truth  |  정답 {len(gt)}개", gt_img)] + panels
        try:
            grid = make_grid(panels, cols=2, cell_width=640)
            save_image(comp_root / f"{p.stem}_comparison.jpg", grid)
            made += 1
        except Exception as exc:
            logger.error("  [FAILED] 비교 이미지 생성 실패 (%s): %s: %s", p.name, type(exc).__name__, exc)
    logger.info("비교 이미지 %d장 저장: %s", made, comp_root)
    logger.info("표시 조건: conf >= %.2f, NMS IoU = %s, imgsz = %s (모든 모델 동일)",
                conf, ev["nms_iou"], cfg.image_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
