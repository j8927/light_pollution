"""빛공해 객체탐지 모델 비교 실험 전체 실행 스크립트.

PowerShell 사용 예:
    python run_all.py                        # 전체 (환경검사 -> 데이터검사 -> split -> 학습 4종 -> 평가 -> 결과)
    python run_all.py --evaluate-only        # 학습 없이 기존 weight 로 평가만
    python run_all.py --model yolov8         # 특정 모델만 (여러 번 지정 가능)
    python run_all.py --model rtdetr --model faster_rcnn
    python run_all.py --overwrite            # 기존 학습 결과가 있어도 다시 학습
    python run_all.py --epochs 10            # 빠른 동작 확인용

실행 순서
    1  환경 검사              2  Dataset 검사           3  Dataset split 확인
    4  YOLOv8 학습            5  YOLO11 학습            6  RT-DETR 학습
    7  Faster R-CNN 학습      8  동일 Test Dataset 평가  9~10 성능 통합 / 비교 CSV
    11 그래프 생성            12~13 예측 / 비교 이미지   14 최종 요약

기본 동작은 **기존 결과 보존**이다. 이미 학습된 weight 가 있으면 학습을 건너뛰고 평가만 수행한다.
한 모델이 실패해도 나머지 모델은 계속 진행하며, 마지막에 성공/실패를 모두 표시한다.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lpcompare.config import MODEL_DISPLAY, load_config  # noqa: E402
from lpcompare.logging_utils import get_logger, log_banner  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="빛공해 객체탐지 모델 비교 실험 전체 실행")
    p.add_argument("--config", default=None, help="실험 설정 파일 (기본: config/experiment.yaml)")
    p.add_argument("--model", action="append", choices=list(MODEL_DISPLAY.keys()),
                   help="처리할 모델 (여러 번 지정 가능, 기본: 설정에서 enabled 된 모든 모델)")
    p.add_argument("--evaluate-only", action="store_true", help="학습을 건너뛰고 평가만 수행")
    p.add_argument("--overwrite", action="store_true", help="기존 학습 결과가 있어도 다시 학습")
    p.add_argument("--epochs", type=int, default=None, help="설정값을 덮어쓸 epoch 수")
    p.add_argument("--device", default=None, help="cuda | cpu | 0")
    p.add_argument("--allow-cpu", action="store_true",
                   help="CUDA 를 쓸 수 없어도 CPU 로 학습을 강행한다 "
                        "(기본값: common.require_cuda=true 이면 학습을 시작하지 않는다)")
    p.add_argument("--no-speed", action="store_true", help="추론 속도 측정 생략")
    p.add_argument("--skip-graphs", action="store_true", help="그래프 생성 생략")
    p.add_argument("--skip-predictions", action="store_true", help="예측/비교 이미지 생성 생략")
    p.add_argument("--num-predictions", type=int, default=12, help="예측 이미지로 만들 test 이미지 수")
    p.add_argument("--val-map-interval", type=int, default=1,
                   help="Faster R-CNN 이 몇 epoch 마다 val mAP 를 계산할지 (기본 1, 0 이면 계산 안 함). "
                        "val 이미지가 많으면 값을 키우면 학습이 빨라진다")
    p.add_argument("--force-splits", action="store_true",
                   help="기존 split 파일을 무시하고 다시 생성 (비교 일관성이 깨질 수 있음)")
    return p


def _step(logger, idx: int, total: int, title: str) -> None:
    logger.info("")
    logger.info("#" * 78)
    logger.info("# [%d/%d] %s", idx, total, title)
    logger.info("#" * 78)


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    logger = get_logger("experiment", cfg.logs_dir)

    started = datetime.now()
    log_banner(logger, "빛공해 객체탐지 모델 비교 실험 (YOLOv8 / YOLO11 / RT-DETR / Faster R-CNN)")
    logger.info("시작 시각 : %s", started.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("설정 파일 : %s", cfg.path)
    logger.info("데이터셋  : %s", cfg.dataset_root)

    models: List[str] = args.model if args.model else cfg.enabled_models
    logger.info("대상 모델 : %s", ", ".join(MODEL_DISPLAY[m] for m in models))
    logger.info("모드      : %s", "평가만 (--evaluate-only)" if args.evaluate_only else "학습 + 평가")

    status: Dict[str, Dict[str, Any]] = {m: {"train": "-", "eval": "-", "reason": ""} for m in models}
    total_steps = 14

    # ---------------- 1. 환경 검사 ----------------
    _step(logger, 1, total_steps, "환경 검사")
    from tools.check_environment import collect, print_report

    env = collect()
    print_report(env, cfg.logs_dir)

    if args.device:
        cfg.raw["common"]["device"] = args.device
    device_info = cfg.resolve_device()
    if device_info["device"] == "cuda":
        logger.info("[OK] 학습/평가를 GPU(%s)에서 실행합니다.", device_info["torch"])
    elif device_info.get("blocked") and not args.evaluate_only and not args.allow_cpu:
        logger.error("")
        logger.error("=" * 78)
        logger.error("[중단] GPU 를 사용할 수 없어 학습을 시작하지 않았습니다.")
        logger.error("       %s", device_info.get("block_reason", ""))
        logger.error("=" * 78)
        logger.error("해결 방법을 확인하려면: python tools/check_environment.py")
        logger.error("기존 weight 로 평가만 하려면 : python run_all.py --evaluate-only")
        logger.error("CPU 로 강행하려면            : python run_all.py --allow-cpu")
        logger.error("설정을 바꾸려면             : config/experiment.yaml 의 common.require_cuda 를 false 로")
        return 2
    else:
        logger.warning("[경고] CUDA 를 사용할 수 없어 CPU 로 실행됩니다. 학습에 매우 오랜 시간이 걸립니다.")

    # ---------------- 2. Dataset 검사 ----------------
    _step(logger, 2, total_steps, "Dataset 검사")
    from tools.validate_dataset import main as validate_main

    sys_argv = sys.argv
    try:
        sys.argv = ["validate_dataset.py"] + (["--config", args.config] if args.config else [])
        ds_code = validate_main()
    except Exception as exc:
        logger.error("[FAILED] 데이터셋 검사 중 예외: %s: %s", type(exc).__name__, exc)
        logger.error(traceback.format_exc())
        ds_code = 2
    finally:
        sys.argv = sys_argv

    if ds_code == 2:
        logger.error("")
        logger.error("데이터셋을 찾을 수 없어 실험을 계속할 수 없습니다.")
        logger.error("config/experiment.yaml 의 dataset.root 가 실제 YOLO 데이터셋 폴더를 가리키는지 확인하세요.")
        logger.error("(현재 설정: %s)", cfg.dataset_root)
        return 2

    # ---------------- 3. Dataset split ----------------
    _step(logger, 3, total_steps, "Dataset split 확인")
    from tools.prepare_splits import main as splits_main

    try:
        sys.argv = (["prepare_splits.py"] + (["--config", args.config] if args.config else [])
                    + (["--force"] if args.force_splits else []))
        sp_code = splits_main()
    except Exception as exc:
        logger.error("[FAILED] split 생성 중 예외: %s: %s", type(exc).__name__, exc)
        logger.error(traceback.format_exc())
        sp_code = 2
    finally:
        sys.argv = sys_argv

    if sp_code != 0:
        logger.error("split 생성에 실패하여 실험을 계속할 수 없습니다.")
        return 2

    # ---------------- 4~7. 학습 ----------------
    train_order = [m for m in MODEL_DISPLAY if m in models]
    if args.evaluate_only:
        for i, key in enumerate(train_order, start=4):
            _step(logger, i, total_steps, f"{MODEL_DISPLAY[key]} 학습 (--evaluate-only 이므로 건너뜀)")
            status[key]["train"] = "SKIPPED"
            status[key]["reason"] = "--evaluate-only"
    else:
        from training.train_faster_rcnn import run_training as run_frcnn
        from training.train_ultralytics import run_training as run_ultra

        for i, key in enumerate(train_order, start=4):
            display = MODEL_DISPLAY[key]
            _step(logger, i, total_steps, f"[{train_order.index(key) + 1}/{len(train_order)}] Training {display}")
            t0 = time.perf_counter()
            try:
                if key == "faster_rcnn":
                    ns = Namespace(config=args.config, epochs=args.epochs, batch=None, imgsz=None,
                                   device=args.device, workers=None, overwrite=args.overwrite,
                                   val_map_interval=args.val_map_interval, amp=None,
                                   allow_cpu=args.allow_cpu)
                    meta = run_frcnn(ns)
                else:
                    ns = Namespace(config=args.config, epochs=args.epochs, batch=None, imgsz=None,
                                   weights=None, device=args.device, overwrite=args.overwrite,
                                   allow_cpu=args.allow_cpu)
                    meta = run_ultra(key, ns)
                if meta.get("success"):
                    status[key]["train"] = "SUCCESS"
                    if meta.get("notes"):
                        status[key]["reason"] = " / ".join(meta["notes"])
                else:
                    status[key]["train"] = "FAILED"
                    status[key]["reason"] = str(meta.get("error") or "원인 불명")
                    logger.error("[FAILED] %s — %s", display, status[key]["reason"])
            except Exception as exc:
                status[key]["train"] = "FAILED"
                status[key]["reason"] = f"{type(exc).__name__}: {exc}"
                logger.error("[FAILED] %s 학습 중 예외: %s", display, status[key]["reason"])
                logger.error(traceback.format_exc())
            logger.info("소요 시간: %.1f초", time.perf_counter() - t0)

    # ---------------- 8~10. 평가 + 통합 CSV ----------------
    _step(logger, 8, total_steps, "동일 Test Dataset 평가 / 성능 통합 / 비교 CSV 생성")
    from evaluation.evaluate_models import main as evaluate_main

    eval_code = 1
    try:
        argv = ["evaluate_models.py"]
        if args.config:
            argv += ["--config", args.config]
        if args.no_speed:
            argv += ["--no-speed"]
        if args.model:
            # 사용자가 고른 모델만 평가한다 (학습하지 않은 모델의 기존 결과를 덮어쓰지 않도록)
            for m in models:
                argv += ["--model", m]
        sys.argv = argv
        eval_code = evaluate_main()
    except Exception as exc:
        logger.error("[FAILED] 평가 중 예외: %s: %s", type(exc).__name__, exc)
        logger.error(traceback.format_exc())
    finally:
        sys.argv = sys_argv

    # 평가 결과를 status 에 반영
    import csv as _csv

    comparison_path = cfg.results_dir / "model_comparison.csv"
    if comparison_path.is_file():
        try:
            with open(comparison_path, "r", encoding="utf-8-sig", newline="") as f:
                for row in _csv.DictReader(f):
                    k = row.get("Model_Key")
                    if k in status:
                        status[k]["eval"] = row.get("Status", "-")
                        if row.get("Status") != "OK" and row.get("Note"):
                            status[k]["reason"] = row["Note"]
        except Exception as exc:
            logger.warning("model_comparison.csv 읽기 실패: %s", exc)

    # ---------------- 11. 그래프 ----------------
    _step(logger, 11, total_steps, "그래프 생성")
    if args.skip_graphs:
        logger.info("--skip-graphs 옵션으로 생략했습니다.")
    else:
        from evaluation.make_graphs import main as graphs_main

        try:
            sys.argv = ["make_graphs.py"] + (["--config", args.config] if args.config else [])
            graphs_main()
        except Exception as exc:
            logger.error("[FAILED] 그래프 생성 실패: %s: %s", type(exc).__name__, exc)
            logger.error(traceback.format_exc())
        finally:
            sys.argv = sys_argv

    # ---------------- 12~13. 예측 / 비교 이미지 ----------------
    _step(logger, 12, total_steps, "예측 이미지 / 모델 비교 이미지 생성")
    if args.skip_predictions:
        logger.info("--skip-predictions 옵션으로 생략했습니다.")
    else:
        from evaluation.make_predictions import main as predictions_main

        try:
            sys.argv = (["make_predictions.py"] + (["--config", args.config] if args.config else [])
                        + ["--num", str(args.num_predictions)])
            predictions_main()
        except Exception as exc:
            logger.error("[FAILED] 예측 이미지 생성 실패: %s: %s", type(exc).__name__, exc)
            logger.error(traceback.format_exc())
        finally:
            sys.argv = sys_argv

    # ---------------- 14. 최종 요약 ----------------
    _step(logger, 14, total_steps, "최종 요약")
    elapsed = (datetime.now() - started).total_seconds()
    logger.info("")
    logger.info("모델별 진행 결과")
    logger.info("-" * 78)
    logger.info("%-16s%-12s%-12s%s", "Model", "Training", "Evaluation", "비고")
    logger.info("-" * 78)
    for key in train_order:
        s = status[key]
        tag_train = f"[{s['train']}]" if s["train"] != "-" else "-"
        tag_eval = f"[{s['eval']}]" if s["eval"] != "-" else "-"
        logger.info("%-16s%-12s%-12s%s", MODEL_DISPLAY[key], tag_train, tag_eval,
                    (s["reason"] or "")[:60])
    logger.info("-" * 78)

    logger.info("")
    logger.info("결과 위치")
    logger.info("  통합 비교표      : %s", cfg.results_dir / "model_comparison.csv")
    logger.info("  클래스별 지표    : %s", cfg.results_dir / "<model>_class_metrics.csv")
    logger.info("  객체 크기별 성능 : %s", cfg.results_dir / "object_size_metrics.csv")
    logger.info("  데이터셋 리포트  : %s", cfg.results_dir / "dataset_report.txt")
    logger.info("  그래프           : %s", cfg.results_dir / "graphs")
    logger.info("  학습 곡선        : %s", cfg.results_dir / "training_curves")
    logger.info("  혼동행렬         : %s", cfg.results_dir / "confusion_matrix")
    logger.info("  예측 이미지      : %s", cfg.results_dir / "predictions")
    logger.info("  비교 이미지      : %s", cfg.results_dir / "comparison_images")
    logger.info("  로그             : %s", cfg.logs_dir)
    logger.info("")
    logger.info("총 소요 시간: %.1f초 (%.2f분)", elapsed, elapsed / 60)
    logger.info("종료 시각   : %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    failed = [MODEL_DISPLAY[k] for k in train_order
              if status[k]["train"] == "FAILED" or status[k]["eval"] not in ("OK", "-")]
    if failed:
        logger.warning("")
        logger.warning("실패한 모델: %s — 자세한 원인은 logs/ 의 모델별 로그를 확인하세요.", ", ".join(failed))
        return 1
    return 0 if eval_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
