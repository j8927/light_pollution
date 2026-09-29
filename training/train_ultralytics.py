"""Ultralytics 계열(YOLOv8 / YOLO11 / RT-DETR) 공통 학습 루틴.

train_yolov8.py / train_yolo11.py / train_rtdetr.py 가 이 모듈을 호출한다.
공통 조건(epochs / imgsz / seed / pretrained / device / split)은 config/experiment.yaml 에서만 읽으며,
모델별로 다른 값(weights, batch)은 같은 파일의 models 섹션에서 읽는다.

기존 train_model.py (웹 서비스용 YOLOv8 학습)는 이 파일과 무관하게 그대로 동작한다.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lpcompare import dataset as ds  # noqa: E402
from lpcompare.config import MODEL_DISPLAY, load_config, set_global_seed  # noqa: E402
from lpcompare.logging_utils import get_logger, log_banner, log_kv  # noqa: E402
from lpcompare.model_info import gpu_name_and_vram  # noqa: E402
from lpcompare.runtime import load_training_meta, resolve_weights, save_training_meta  # noqa: E402


def build_parser(model_key: str) -> argparse.ArgumentParser:
    display = MODEL_DISPLAY.get(model_key, model_key)
    p = argparse.ArgumentParser(description=f"{display} 학습 (빛공해 모델 비교 실험)")
    p.add_argument("--config", default=None, help="실험 설정 파일 (기본: config/experiment.yaml)")
    p.add_argument("--epochs", type=int, default=None, help="설정값을 덮어쓸 epoch 수")
    p.add_argument("--batch", type=int, default=None, help="설정값을 덮어쓸 batch size")
    p.add_argument("--imgsz", type=int, default=None, help="설정값을 덮어쓸 이미지 크기")
    p.add_argument("--weights", default=None, help="설정값을 덮어쓸 사전학습 weight")
    p.add_argument("--device", default=None, help="cuda | cpu | 0 (설정값 덮어쓰기)")
    p.add_argument("--overwrite", action="store_true",
                   help="이미 학습된 결과가 있어도 다시 학습한다 (기본은 기존 결과 보존)")
    p.add_argument("--allow-cpu", action="store_true",
                   help="CUDA 를 쓸 수 없어도 CPU 로 학습을 강행한다 "
                        "(기본값: common.require_cuda=true 이면 중단)")
    return p


def _is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in text or "cuda oom" in text or "cublas_status_alloc_failed" in text


def _free_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def run_training(model_key: str, args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config)
    display = MODEL_DISPLAY.get(model_key, model_key)
    logger = get_logger(model_key, cfg.logs_dir)
    log_banner(logger, f"{display} 학습 시작")

    model_cfg = cfg.model(model_key)
    run_dir = cfg.run_dir(model_key)
    epochs = args.epochs if args.epochs else cfg.epochs
    imgsz = args.imgsz if args.imgsz else cfg.image_size
    requested_weights = args.weights or model_cfg["weights"]
    batch = args.batch if args.batch else int(model_cfg.get("batch", 16))

    device_info = cfg.resolve_device()
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
        meta = {"model_key": model_key, "display_name": display, "success": False,
                "error": msg, "device": "cpu", "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        return meta

    meta: Dict[str, Any] = {
        "model_key": model_key,
        "display_name": display,
        "framework": "ultralytics",
        "requested_weights": str(requested_weights),
        "actual_weights": str(requested_weights),
        "weights_changed_reason": "",
        "dataset_root": str(cfg.dataset_root),
        "data_yaml": str(cfg.compare_data_yaml),
        "epochs": epochs,
        "image_size": imgsz,
        "requested_batch": batch,
        "actual_batch": batch,
        "seed": cfg.seed,
        "device": device_info["device"],
        "device_detail": device_info["torch"],
        "pretrained": cfg.pretrained,
        "workers": cfg.workers,
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

    # --- 사전 조건: split 파일 ---
    if not ds.splits_exist(cfg.splits_dir):
        msg = (f"split 파일이 없습니다: {cfg.splits_dir}. "
               "먼저 `python tools/prepare_splits.py` 를 실행하세요.")
        logger.error("[실패] %s", msg)
        meta["error"] = msg
        save_training_meta(run_dir, meta)
        return meta
    if not cfg.compare_data_yaml.is_file():
        layout = ds.detect_layout(cfg.dataset_root, cfg.data_yaml, cfg.fallback_names)
        ds.build_compare_data_yaml(cfg.compare_data_yaml, cfg.root, cfg.splits_dir, layout.names)

    counts = {k: len(v) for k, v in ds.load_splits(cfg.splits_dir).items()}

    # --- 이미 학습된 결과가 있으면 기본적으로 보존 ---
    existing = resolve_weights(run_dir)
    if existing and not args.overwrite:
        prev = load_training_meta(run_dir)
        logger.info("이미 학습된 weight 가 있어 학습을 건너뜁니다: %s", existing)
        logger.info("다시 학습하려면 --overwrite 옵션을 사용하세요.")
        if prev:
            return prev
        meta["success"] = True
        meta["notes"].append("기존 weight 재사용 (학습 생략)")
        meta["best_weights"] = str(existing)
        return meta

    log_kv(logger, f"{display} 학습 조건", {
        "model": requested_weights,
        "data.yaml": cfg.compare_data_yaml,
        "train / val / test": f"{counts['train']} / {counts['val']} / {counts['test']} 장",
        "epochs": epochs,
        "image size": imgsz,
        "batch": batch,
        "seed": cfg.seed,
        "device": device_info["torch"],
        "pretrained": cfg.pretrained,
        "workers": cfg.workers,
        "결과 폴더": run_dir,
    })

    set_global_seed(cfg.seed, cfg.deterministic)

    from ultralytics import RTDETR, YOLO

    loader = RTDETR if model_key == "rtdetr" else YOLO
    min_batch = int(cfg.oom_retry.get("min_batch", 1))
    oom_enabled = bool(cfg.oom_retry.get("enabled", True))

    # 같은 프로세스에서 여러 모델을 연달아 학습할 때 앞 모델의 최대치가 섞이지 않도록 초기화
    _free_cuda()

    start = time.perf_counter()
    result = None
    while True:
        try:
            model = loader(str(meta["actual_weights"]))

            def _on_epoch_end(trainer, _display=display, _epochs=epochs):
                try:
                    cur = int(getattr(trainer, "epoch", 0)) + 1
                    logger.info("=" * 40)
                    logger.info("Model: %s", _display)
                    logger.info("Epoch: %d / %d", cur, _epochs)
                    logger.info("=" * 40)
                except Exception:
                    pass

            model.add_callback("on_train_epoch_end", _on_epoch_end)

            result = model.train(
                data=str(cfg.compare_data_yaml),
                epochs=epochs,
                imgsz=imgsz,
                batch=meta["actual_batch"],
                seed=cfg.seed,
                deterministic=cfg.deterministic,
                device=device_info["ultralytics"],
                pretrained=cfg.pretrained,
                workers=cfg.workers,
                project=str(cfg.runs_dir),
                name=model_key,
                exist_ok=True,
                plots=True,
                val=True,
                verbose=True,
            )
            break
        except Exception as exc:
            if oom_enabled and _is_oom(exc) and meta["actual_batch"] > min_batch:
                new_batch = max(min_batch, meta["actual_batch"] // 2)
                logger.warning("[OOM] CUDA 메모리 부족 — batch %d -> %d 로 낮춰 재시도합니다.",
                               meta["actual_batch"], new_batch)
                meta["oom_retries"].append({"from_batch": meta["actual_batch"], "to_batch": new_batch,
                                            "error": f"{type(exc).__name__}: {exc}"[:400]})
                meta["actual_batch"] = new_batch
                _free_cuda()
                continue
            meta["error"] = f"{type(exc).__name__}: {exc}"
            meta["traceback"] = traceback.format_exc()
            logger.error("[FAILED] %s 학습 실패: %s", display, meta["error"])
            logger.error(meta["traceback"])
            meta["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            meta["training_time_sec"] = round(time.perf_counter() - start, 2)
            save_training_meta(run_dir, meta)
            return meta

    meta["training_time_sec"] = round(time.perf_counter() - start, 2)
    meta["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        import torch

        if device_info["device"] == "cuda":
            meta["train_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    except Exception:
        pass

    save_dir = Path(getattr(result, "save_dir", run_dir)) if result is not None else run_dir
    meta["run_dir"] = str(save_dir)
    best = resolve_weights(save_dir)
    meta["best_weights"] = str(best) if best else None
    meta["success"] = best is not None
    if meta["actual_batch"] != meta["requested_batch"]:
        meta["notes"].append(
            f"CUDA OOM 으로 batch 를 {meta['requested_batch']} -> {meta['actual_batch']} 로 낮춰 학습했습니다."
        )

    logger.info("")
    logger.info("[%s] %s 학습 종료 — 소요 %.1f초 (%.2f분)",
                "SUCCESS" if meta["success"] else "FAILED", display,
                meta["training_time_sec"], meta["training_time_sec"] / 60)
    logger.info("결과 폴더 : %s", save_dir)
    logger.info("best weight: %s", meta["best_weights"])
    save_training_meta(save_dir, meta)
    if str(save_dir) != str(run_dir):
        save_training_meta(run_dir, meta)
    return meta


def main(model_key: str) -> int:
    args = build_parser(model_key).parse_args()
    meta = run_training(model_key, args)
    return 0 if meta.get("success") else 1
