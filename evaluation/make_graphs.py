"""비교 그래프와 학습 곡선을 생성한다 (실제 측정값만 사용).

PowerShell 사용 예:
    python evaluation/make_graphs.py

입력
    results/model_comparison.csv      (evaluation/evaluate_models.py 가 생성)
    runs/<model>/results.csv          (각 모델 학습이 생성한 epoch 별 기록)

출력
    results/graphs/*.png              지표별 모델 비교 막대그래프
    results/training_curves/*.png     모델별 학습 곡선 + 모델 간 비교 곡선
    results/training_curves/<model>_ultralytics_results.png
                                      Ultralytics 가 이미 만든 그래프 복사본 (원본은 runs/ 에 그대로 둔다)
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lpcompare.config import MODEL_DISPLAY, load_config  # noqa: E402
from lpcompare.logging_utils import get_logger, log_banner  # noqa: E402
from lpcompare.viz import PALETTE, bar_chart, setup_matplotlib  # noqa: E402

# (파일명, CSV 컬럼, 제목, y축 라벨, 표시 형식)
GRAPH_SPECS = [
    ("map50_comparison.png", "mAP50", "모델별 mAP@0.5 비교", "mAP@0.5", "{:.4f}"),
    ("map50_95_comparison.png", "mAP50_95", "모델별 mAP@0.5:0.95 비교", "mAP@0.5:0.95", "{:.4f}"),
    ("precision_comparison.png", "Precision", "모델별 Precision 비교", "Precision", "{:.4f}"),
    ("recall_comparison.png", "Recall", "모델별 Recall 비교", "Recall", "{:.4f}"),
    ("f1_comparison.png", "F1", "모델별 F1-score 비교", "F1-score", "{:.4f}"),
    ("fps_comparison.png", "FPS", "모델별 추론 속도(FPS) 비교", "FPS (frames/sec)", "{:.1f}"),
    ("inference_time_comparison.png", "Inference_ms", "모델별 추론 시간 비교", "Inference Time (ms)", "{:.2f}"),
    ("model_size_comparison.png", "Model_Size_MB", "모델별 파일 크기 비교", "Model Size (MB)", "{:.1f}"),
    ("parameters_comparison.png", "Parameters", "모델별 파라미터 수 비교", "Parameters (개)", "{:,.0f}"),
    ("gflops_comparison.png", "GFLOPs", "모델별 연산량 비교", "GFLOPs", "{:.1f}"),
    ("gpu_memory_comparison.png", "GPU_Memory_MB", "모델별 추론 GPU 메모리 사용량 비교", "GPU Memory (MB)", "{:.0f}"),
    ("training_time_comparison.png", "Training_Time_sec", "모델별 학습 시간 비교", "Training Time (초)", "{:.0f}"),
    ("small_object_ap_comparison.png", "Small_AP50_95", "작은 광원(Small) AP@0.5:0.95 비교", "Small AP@0.5:0.95", "{:.4f}"),
]

NOTE = "※ 모델 규모(파라미터/연산량)와 batch size 가 서로 다릅니다. results/model_comparison.csv 의 조건을 함께 확인하세요."


def read_comparison(path: Path) -> List[Dict[str, str]]:
    if not Path(path).is_file():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f)]


def _to_float(v) -> Optional[float]:
    if v in ("", None, "N/A"):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def make_bar_graphs(cfg, rows: List[Dict[str, str]], logger) -> int:
    out_dir = cfg.results_dir / "graphs"
    labels = [r["Model"] for r in rows]
    made = 0
    for filename, column, title, ylabel, fmt in GRAPH_SPECS:
        values = [_to_float(r.get(column)) for r in rows]
        path = bar_chart(out_dir / filename, labels, values, title, ylabel, NOTE, fmt)
        if path:
            made += 1
            logger.info("  생성: %s", path)
        else:
            logger.warning("  건너뜀: %s — %s 값이 없습니다 (측정되지 않음).", filename, column)
    return made


# ---------------------------------------------------------------------------
# 학습 곡선
# ---------------------------------------------------------------------------
def parse_results_csv(path: Path) -> Dict[str, List[float]]:
    """Ultralytics / Faster R-CNN results.csv 를 공통 시계열로 정규화한다."""
    series: Dict[str, List[float]] = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "precision": [], "recall": [], "map50": [], "map50_95": [],
    }
    if not Path(path).is_file():
        return series

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in (reader.fieldnames or [])]

        def pick(*candidates: str) -> Optional[str]:
            for c in candidates:
                for h in headers:
                    if h.lower() == c.lower():
                        return h
            return None

        col_epoch = pick("epoch")
        col_p = pick("metrics/precision(B)", "precision")
        col_r = pick("metrics/recall(B)", "recall")
        col_m50 = pick("metrics/mAP50(B)", "mAP50")
        col_m5095 = pick("metrics/mAP50-95(B)", "mAP50-95")
        col_vloss = pick("val_loss")
        train_loss_cols = [h for h in headers if h.lower().startswith("train/") and h.lower().endswith("loss")]
        val_loss_cols = [h for h in headers if h.lower().startswith("val/") and h.lower().endswith("loss")]
        single_train_loss = pick("train_loss")

        for i, row in enumerate(reader, start=1):
            def g(col):
                return _to_float(row.get(col)) if col else None

            series["epoch"].append(g(col_epoch) if col_epoch else float(i))
            if train_loss_cols:
                vals = [g(c) for c in train_loss_cols]
                vals = [v for v in vals if v is not None]
                series["train_loss"].append(sum(vals) if vals else float("nan"))
            elif single_train_loss:
                v = g(single_train_loss)
                series["train_loss"].append(v if v is not None else float("nan"))
            else:
                series["train_loss"].append(float("nan"))

            if val_loss_cols:
                vals = [g(c) for c in val_loss_cols]
                vals = [v for v in vals if v is not None]
                series["val_loss"].append(sum(vals) if vals else float("nan"))
            elif col_vloss:
                v = g(col_vloss)
                series["val_loss"].append(v if v is not None else float("nan"))
            else:
                series["val_loss"].append(float("nan"))

            for key, col in (("precision", col_p), ("recall", col_r),
                             ("map50", col_m50), ("map50_95", col_m5095)):
                v = g(col)
                series[key].append(v if v is not None else float("nan"))
    return series


def _has_data(values: List[float]) -> bool:
    return any(v is not None and np.isfinite(v) for v in values)


def plot_training_curve(out_path: Path, display: str, series: Dict[str, List[float]]) -> Optional[Path]:
    import matplotlib.pyplot as plt

    epochs = series["epoch"]
    if not epochs:
        return None

    panels = [
        ("Loss", [("train loss", series["train_loss"]), ("val loss", series["val_loss"])], "Loss"),
        ("Precision / Recall", [("precision", series["precision"]), ("recall", series["recall"])], "값"),
        ("mAP@0.5", [("mAP50", series["map50"])], "mAP@0.5"),
        ("mAP@0.5:0.95", [("mAP50-95", series["map50_95"])], "mAP@0.5:0.95"),
    ]
    panels = [p for p in panels if any(_has_data(v) for _, v in p[1])]
    if not panels:
        return None

    cols = 2
    rows_n = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(11, 4.2 * rows_n), squeeze=False)
    for idx, (title, lines, ylabel) in enumerate(panels):
        ax = axes[idx // cols][idx % cols]
        for j, (label, values) in enumerate(lines):
            if not _has_data(values):
                continue
            color = tuple(c / 255 for c in PALETTE[j % len(PALETTE)])
            ax.plot(epochs, values, label=label, color=color, linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(fontsize=9)
    for idx in range(len(panels), rows_n * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle(f"{display} 학습 곡선", fontsize=14)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_combined_curves(out_path: Path, curves: Dict[str, Dict[str, List[float]]],
                         key: str, title: str, ylabel: str) -> Optional[Path]:
    import matplotlib.pyplot as plt

    usable = {d: s for d, s in curves.items() if s["epoch"] and _has_data(s[key])}
    if not usable:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i, (display, s) in enumerate(usable.items()):
        color = tuple(c / 255 for c in PALETTE[i % len(PALETTE)])
        ax.plot(s["epoch"], s[key], label=display, color=color, linewidth=1.9)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def collect_training_curves(cfg, logger) -> int:
    out_dir = cfg.results_dir / "training_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    curves: Dict[str, Dict[str, List[float]]] = {}
    made = 0

    for key in MODEL_DISPLAY:
        run_dir = cfg.run_dir(key)
        display = MODEL_DISPLAY[key]
        results_csv = run_dir / "results.csv"
        if not results_csv.is_file():
            hits = sorted(run_dir.glob("**/results.csv"))
            results_csv = hits[0] if hits else results_csv
        if not results_csv.is_file():
            logger.warning("  건너뜀: %s — results.csv 없음 (%s)", display, run_dir)
            continue

        series = parse_results_csv(results_csv)
        curves[display] = series
        path = plot_training_curve(out_dir / f"{key}_training_curve.png", display, series)
        if path:
            made += 1
            logger.info("  생성: %s", path)

        # Ultralytics 가 이미 만든 그래프는 복사만 한다 (원본은 runs/ 에 그대로 유지)
        for src_name, dst_name in (("results.png", f"{key}_ultralytics_results.png"),
                                   ("BoxPR_curve.png", f"{key}_pr_curve.png"),
                                   ("PR_curve.png", f"{key}_pr_curve.png")):
            src = results_csv.parent / src_name
            if src.is_file():
                shutil.copyfile(src, out_dir / dst_name)
                logger.info("  복사: %s -> %s", src, out_dir / dst_name)

    for key, title, ylabel in (("map50", "모델별 mAP@0.5 학습 곡선", "mAP@0.5"),
                               ("map50_95", "모델별 mAP@0.5:0.95 학습 곡선", "mAP@0.5:0.95"),
                               ("train_loss", "모델별 train loss 곡선 (loss 정의가 모델마다 다름)", "Train Loss")):
        p = plot_combined_curves(out_dir / f"all_models_{key}.png", curves, key, title, ylabel)
        if p:
            made += 1
            logger.info("  생성: %s", p)
    return made


def main() -> int:
    parser = argparse.ArgumentParser(description="모델 비교 그래프 / 학습 곡선 생성")
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-curves", action="store_true", help="학습 곡선 생성을 건너뛴다")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("graphs", cfg.logs_dir)
    log_banner(logger, "그래프 생성")
    font = setup_matplotlib()
    logger.info("matplotlib 한글 폰트: %s", font)

    rows = read_comparison(cfg.results_dir / "model_comparison.csv")
    rows = [r for r in rows if r.get("Status") == "OK"]
    if not rows:
        logger.error("[실패] results/model_comparison.csv 에 평가 결과가 없습니다. "
                     "먼저 `python evaluation/evaluate_models.py` 를 실행하세요.")
        return 2

    logger.info("비교 막대그래프 생성 (%d개 모델)", len(rows))
    made = make_bar_graphs(cfg, rows, logger)

    if not args.skip_curves:
        logger.info("")
        logger.info("학습 곡선 생성")
        made += collect_training_curves(cfg, logger)

    logger.info("")
    logger.info("그래프 %d개 생성 완료 — %s, %s", made,
                cfg.results_dir / "graphs", cfg.results_dir / "training_curves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
