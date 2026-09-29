"""YOLO 데이터셋 검사 스크립트 (원본은 읽기만 하고 절대 수정하지 않는다).

PowerShell 사용 예:
    python tools/validate_dataset.py
    python tools/validate_dataset.py --no-image-size     # 이미지 크기 분석 생략(빠름)

검사 항목
    전체/ split 별 이미지 수, 전체/클래스별 bbox 수, 이미지-라벨 1:1 대응,
    잘못된 class id, bbox 좌표 오류(w<=0, h<=0, 0~1 범위 초과, 경계 초과),
    라벨 없는 이미지, 이미지 없는 라벨, 클래스 불균형, 객체 크기 분포(COCO 기준)

결과
    results/dataset_report.txt
    results/dataset_report.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lpcompare import dataset as ds  # noqa: E402
from lpcompare.config import load_config  # noqa: E402
from lpcompare.logging_utils import get_logger, log_banner  # noqa: E402
from lpcompare.metrics import COCO_AREA_RANGES  # noqa: E402


def analyze(cfg, with_image_size: bool = True) -> Dict[str, Any]:
    layout = ds.detect_layout(cfg.dataset_root, cfg.data_yaml, cfg.fallback_names)
    names = layout.names
    nc = len(names)

    report: Dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root": str(cfg.dataset_root),
        "layout_style": layout.style,
        "data_yaml": str(layout.data_yaml) if layout.data_yaml else None,
        "class_names": names,
        "num_classes": nc,
        "notes": list(layout.notes),
        "splits": {},
        "totals": {"images": 0, "labeled_images": 0, "boxes": 0, "missing_label_files": 0,
                   "empty_label_files": 0, "orphan_label_files": 0},
        "class_counts": {n: 0 for n in names},
        "issues": [],
        "issue_counts": Counter(),
        "size_distribution": {"small": 0, "medium": 0, "large": 0, "unknown": 0},
        "exists": layout.exists,
    }

    if not layout.exists:
        return report

    for split in layout.available_splits:
        img_dir = layout.images_dirs[split]
        lbl_dir = layout.labels_dirs.get(split)
        images = ds.list_images(img_dir)

        split_info = {
            "images_dir": str(img_dir),
            "labels_dir": str(lbl_dir) if lbl_dir else None,
            "image_count": len(images),
            "labeled_images": 0,
            "background_images": 0,     # 라벨 파일은 있지만 bbox 가 0개
            "missing_label_files": 0,   # 라벨 파일 자체가 없음
            "box_count": 0,
            "class_counts": {n: 0 for n in names},
            "orphan_label_files": 0,
        }

        image_stems = set()
        for img in images:
            image_stems.add(img.stem)
            lp = ds.label_path_for(img)
            if not lp.is_file():
                split_info["missing_label_files"] += 1
                report["issues"].append({
                    "split": split, "kind": "missing_label_file",
                    "image": str(img), "label": str(lp), "line": "", "detail": "라벨 파일 없음",
                })
                report["issue_counts"]["missing_label_file"] += 1
                continue

            boxes, issues = ds.parse_label_file(lp, nc)
            for line_no, kind, detail in issues:
                report["issues"].append({
                    "split": split, "kind": kind, "image": str(img),
                    "label": str(lp), "line": line_no, "detail": detail,
                })
                report["issue_counts"][kind] += 1

            if boxes:
                split_info["labeled_images"] += 1
            else:
                split_info["background_images"] += 1

            split_info["box_count"] += len(boxes)
            for cls, xc, yc, w, h in boxes:
                if 0 <= cls < nc:
                    split_info["class_counts"][names[cls]] += 1

            if with_image_size and boxes:
                try:
                    iw, ih = ds.image_size(img)
                except Exception as exc:
                    report["issues"].append({
                        "split": split, "kind": "image_read_error", "image": str(img),
                        "label": "", "line": "", "detail": f"{type(exc).__name__}: {exc}",
                    })
                    report["issue_counts"]["image_read_error"] += 1
                    report["size_distribution"]["unknown"] += len(boxes)
                    continue
                for _cls, _xc, _yc, w, h in boxes:
                    area = (w * iw) * (h * ih)
                    if area < COCO_AREA_RANGES["small"][1]:
                        report["size_distribution"]["small"] += 1
                    elif area < COCO_AREA_RANGES["medium"][1]:
                        report["size_distribution"]["medium"] += 1
                    else:
                        report["size_distribution"]["large"] += 1
            elif boxes:
                report["size_distribution"]["unknown"] += len(boxes)

        # 이미지가 없는 라벨 파일
        if lbl_dir and Path(lbl_dir).is_dir():
            for lp in sorted(Path(lbl_dir).rglob("*.txt")):
                if lp.stem not in image_stems:
                    split_info["orphan_label_files"] += 1
                    report["issues"].append({
                        "split": split, "kind": "orphan_label_file", "image": "",
                        "label": str(lp), "line": "", "detail": "대응하는 이미지 없음",
                    })
                    report["issue_counts"]["orphan_label_file"] += 1

        report["splits"][split] = split_info
        report["totals"]["images"] += split_info["image_count"]
        report["totals"]["labeled_images"] += split_info["labeled_images"]
        report["totals"]["boxes"] += split_info["box_count"]
        report["totals"]["missing_label_files"] += split_info["missing_label_files"]
        report["totals"]["empty_label_files"] += split_info["background_images"]
        report["totals"]["orphan_label_files"] += split_info["orphan_label_files"]
        for n in names:
            report["class_counts"][n] += split_info["class_counts"][n]

    # 클래스 불균형
    counts = [report["class_counts"][n] for n in names]
    total_boxes = sum(counts)
    imbalance = []
    if total_boxes > 0:
        mx, mn = max(counts), min(counts)
        for n in names:
            c = report["class_counts"][n]
            imbalance.append({
                "class": n, "count": c,
                "ratio": round(c / total_boxes, 4),
                "vs_max": round(c / mx, 4) if mx else 0.0,
            })
        report["imbalance_ratio"] = round(mx / mn, 2) if mn > 0 else None
    report["class_balance"] = imbalance
    return report


def write_txt(path: Path, report: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    w = lines.append
    w("=" * 78)
    w("빛공해 YOLO 데이터셋 검사 리포트")
    w("=" * 78)
    w(f"생성 시각      : {report['generated_at']}")
    w(f"데이터셋 경로  : {report['dataset_root']}")
    w(f"디렉터리 구조  : {report['layout_style']}")
    w(f"data.yaml      : {report['data_yaml'] or '없음'}")
    w(f"클래스 수      : {report['num_classes']}")
    w(f"클래스 이름    : {', '.join(report['class_names']) if report['class_names'] else '없음'}")
    for n in report["notes"]:
        w(f"[참고] {n}")
    w("")

    if not report["exists"]:
        w("!" * 78)
        w("데이터셋을 찾을 수 없어 검사를 수행하지 못했습니다.")
        w("config/experiment.yaml 의 dataset.root 경로를 확인하세요.")
        w("!" * 78)
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    w("-" * 78)
    w("Split 별 요약")
    w("-" * 78)
    w(f"{'split':<8}{'images':>9}{'labeled':>9}{'배경':>8}{'라벨없음':>10}{'boxes':>9}{'고아라벨':>10}")
    for split, s in report["splits"].items():
        w(f"{split:<8}{s['image_count']:>9}{s['labeled_images']:>9}{s['background_images']:>8}"
          f"{s['missing_label_files']:>10}{s['box_count']:>9}{s['orphan_label_files']:>10}")
    t = report["totals"]
    w(f"{'합계':<8}{t['images']:>9}{t['labeled_images']:>9}{t['empty_label_files']:>8}"
      f"{t['missing_label_files']:>10}{t['boxes']:>9}{t['orphan_label_files']:>10}")
    w("")

    w("-" * 78)
    w("클래스별 bbox 개수")
    w("-" * 78)
    for row in report.get("class_balance", []):
        w(f"  {row['class']:<12} {row['count']:>8} 개   (전체의 {row['ratio'] * 100:5.2f}%, 최다 클래스 대비 {row['vs_max'] * 100:5.1f}%)")
    if report.get("imbalance_ratio") is not None:
        w(f"  최다/최소 클래스 비율 : {report['imbalance_ratio']} : 1")
        if report["imbalance_ratio"] and report["imbalance_ratio"] >= 3:
            w("  [경고] 클래스 불균형이 큽니다. 소수 클래스의 Recall 이 낮게 나올 수 있습니다.")
    w("")

    w("-" * 78)
    w("split 별 클래스 분포")
    w("-" * 78)
    header = f"{'split':<8}" + "".join(f"{n:>12}" for n in report["class_names"])
    w(header)
    for split, s in report["splits"].items():
        w(f"{split:<8}" + "".join(f"{s['class_counts'][n]:>12}" for n in report["class_names"]))
    w("")

    sd = report["size_distribution"]
    if sum(sd.values()) > 0:
        w("-" * 78)
        w("객체 크기 분포 (COCO 기준: small < 32^2 px, medium < 96^2 px)")
        w("-" * 78)
        tot = sum(sd.values())
        for k in ("small", "medium", "large", "unknown"):
            w(f"  {k:<8} {sd[k]:>8} 개  ({sd[k] / tot * 100:5.2f}%)")
        w("  * 작은 광원 탐지 성능 비교(results/object_size_metrics.csv)의 기준과 동일합니다.")
        w("")

    w("-" * 78)
    w("오류 / 경고 요약")
    w("-" * 78)
    if not report["issue_counts"]:
        w("  발견된 문제 없음")
    else:
        for kind, cnt in sorted(report["issue_counts"].items(), key=lambda x: -x[1]):
            w(f"  {kind:<24} {cnt:>6} 건")
        w("")
        w("  상세 목록 (최대 50건, 전체는 dataset_report.csv 참고)")
        for issue in report["issues"][:50]:
            loc = f"{issue['label'] or issue['image']}"
            line = f":{issue['line']}" if issue["line"] else ""
            w(f"   - [{issue['kind']}] {loc}{line} — {issue['detail']}")
    w("")
    w("원본 데이터셋은 검사만 수행했으며 어떤 파일도 수정하지 않았습니다.")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_csv(path: Path, report: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["category", "split", "key", "value", "detail"])
        wr.writerow(["meta", "", "generated_at", report["generated_at"], ""])
        wr.writerow(["meta", "", "dataset_root", report["dataset_root"], ""])
        wr.writerow(["meta", "", "layout_style", report["layout_style"], ""])
        wr.writerow(["meta", "", "data_yaml", report["data_yaml"] or "", ""])
        wr.writerow(["meta", "", "num_classes", report["num_classes"], ""])
        wr.writerow(["meta", "", "class_names", "|".join(report["class_names"]), ""])
        wr.writerow(["meta", "", "dataset_found", report["exists"], ""])

        for split, s in report["splits"].items():
            for key in ("image_count", "labeled_images", "background_images",
                        "missing_label_files", "box_count", "orphan_label_files"):
                wr.writerow(["split", split, key, s[key], ""])
            for n, c in s["class_counts"].items():
                wr.writerow(["split_class", split, n, c, "bbox 개수"])

        for key, val in report["totals"].items():
            wr.writerow(["total", "", key, val, ""])
        for n, c in report["class_counts"].items():
            wr.writerow(["class_total", "", n, c, "bbox 개수"])
        for row in report.get("class_balance", []):
            wr.writerow(["class_balance", "", row["class"], row["ratio"], f"최다 대비 {row['vs_max']}"])
        for k, v in report["size_distribution"].items():
            wr.writerow(["size_distribution", "", k, v, "COCO 면적 기준"])
        for kind, cnt in report["issue_counts"].items():
            wr.writerow(["issue_count", "", kind, cnt, ""])
        for issue in report["issues"]:
            wr.writerow(["issue", issue["split"], issue["kind"],
                         issue["label"] or issue["image"], f"line {issue['line']}: {issue['detail']}"])
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="빛공해 YOLO 데이터셋 검사 (원본 수정 없음)")
    parser.add_argument("--config", default=None, help="실험 설정 파일 (기본: config/experiment.yaml)")
    parser.add_argument("--no-image-size", action="store_true",
                        help="이미지 크기 기반 객체 크기 분포 분석을 생략한다")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("dataset", cfg.logs_dir)
    log_banner(logger, "데이터셋 검사 (Dataset Validation)")
    logger.info("데이터셋 루트: %s", cfg.dataset_root)

    report = analyze(cfg, with_image_size=not args.no_image_size)

    txt_path = write_txt(cfg.results_dir / "dataset_report.txt", report)
    csv_path = write_csv(cfg.results_dir / "dataset_report.csv", report)

    if not report["exists"]:
        logger.error("[실패] 데이터셋을 찾을 수 없습니다: %s", cfg.dataset_root)
        for n in report["notes"]:
            logger.error("        %s", n)
        logger.error("        config/experiment.yaml 의 dataset.root 를 실제 데이터셋 경로로 지정하세요.")
        logger.info("리포트 저장: %s", txt_path)
        logger.info("리포트 저장: %s", csv_path)
        return 2

    t = report["totals"]
    logger.info("이미지 %d장 / bbox %d개 / 클래스 %d개", t["images"], t["boxes"], report["num_classes"])
    for split, s in report["splits"].items():
        logger.info("  %-5s : 이미지 %4d장, bbox %5d개, 라벨없음 %d개",
                    split, s["image_count"], s["box_count"], s["missing_label_files"])
    if report["issue_counts"]:
        logger.warning("문제 %d건 발견 — 상세 내용은 리포트를 확인하세요.", sum(report["issue_counts"].values()))
        for kind, cnt in sorted(report["issue_counts"].items(), key=lambda x: -x[1]):
            logger.warning("   - %s : %d건", kind, cnt)
    else:
        logger.info("발견된 문제 없음")
    logger.info("리포트 저장: %s", txt_path)
    logger.info("리포트 저장: %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
