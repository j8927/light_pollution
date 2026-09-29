"""모든 모델이 공유할 train / val / test split 목록을 만든다.

PowerShell 사용 예:
    python tools/prepare_splits.py
    python tools/prepare_splits.py --force      # 기존 split 을 다시 생성 (주의: 비교 일관성 깨짐)

원본 이미지를 이동하거나 복사하지 않고, **이미지 경로 목록 파일**만 만든다.
    splits/train.txt
    splits/val.txt
    splits/test.txt
    splits/split_info.json      (어떤 방식으로 나눴는지 기록)
    derived_data/data_compare.yaml  (4개 모델 공용 data.yaml — 원본 data.yaml 은 그대로 둔다)

test split 전략 (config/experiment.yaml 의 splits.test_strategy)
    existing        : 데이터셋에 test 폴더가 있으면 그대로 사용
    split_val       : test 가 없으면 기존 val 을 seed 42 로 val/test 로 나눔 (train 은 손대지 않음)
    use_val_as_test : val 을 test 로 재사용 (val == test 이므로 성능이 과대평가됨 — 기록에 남긴다)
    resplit_all     : 전체를 70/15/15 로 재분할 (기존 split 이 깨지므로 기본값 아님)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lpcompare import dataset as ds  # noqa: E402
from lpcompare.config import load_config  # noqa: E402
from lpcompare.logging_utils import get_logger, log_banner  # noqa: E402


def build_splits(cfg, logger, force: bool = False) -> Dict[str, object]:
    layout = ds.detect_layout(cfg.dataset_root, cfg.data_yaml, cfg.fallback_names)
    info: Dict[str, object] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root": str(cfg.dataset_root),
        "layout_style": layout.style,
        "seed": cfg.splits["seed"],
        "strategy": cfg.splits["test_strategy"],
        "class_names": layout.names,
        "counts": {},
        "notes": list(layout.notes),
        "ok": False,
    }

    if not layout.exists:
        logger.error("[실패] 데이터셋을 찾을 수 없습니다: %s", cfg.dataset_root)
        for n in layout.notes:
            logger.error("        %s", n)
        return info

    seed = int(cfg.splits["seed"])
    strategy = str(cfg.splits["test_strategy"])
    available = layout.available_splits
    logger.info("데이터셋 구조: %s / 발견된 split: %s", layout.style, ", ".join(available))

    # --- 라벨 포맷 점검: 폴리곤(세그멘테이션) 라벨이 섞여 있으면 파생 데이터셋으로 정규화 ---
    mode = str(cfg.splits.get("normalize_polygon_labels", "auto"))
    scan = ds.scan_label_formats(layout)
    poly_lines = sum(c["polygon_lines"] for c in scan.values())
    mixed_bbox = sum(c["bbox_lines_in_mixed"] for c in scan.values())
    info["label_scan"] = scan
    if poly_lines:
        logger.warning("[주의] 폴리곤(세그멘테이션) 라벨 %d줄이 섞여 있습니다. "
                       "혼재 파일 안의 bbox 라인 %d줄은 Ultralytics 가 잘못 해석합니다.",
                       poly_lines, mixed_bbox)
    if mode == "always" or (mode == "auto" and poly_lines > 0):
        norm_root = cfg.derived_dir / "dataset_normalized"
        logger.info("라벨 정규화 파생 데이터셋 생성: %s (원본은 수정하지 않음)", norm_root)
        norm = ds.build_normalized_dataset(layout, norm_root, logger)
        layout = norm["layout"]
        available = layout.available_splits
        info["label_normalization"] = {k: v for k, v in norm.items() if k != "layout"}
        info["notes"].append(
            f"폴리곤 라벨 {norm['converted_polygon_lines']}줄을 외접 bbox 로 변환하고, "
            f"혼재 파일의 bbox {norm['recovered_bbox_lines']}줄을 원래 좌표로 보존한 "
            f"파생 데이터셋({norm_root})을 만들어 4개 모델이 모두 이 데이터를 사용합니다."
        )
        info["dataset_used"] = str(norm_root)
    else:
        info["dataset_used"] = str(cfg.dataset_root)

    pools = {s: ds.list_images(layout.images_dirs[s]) for s in available}
    for s, imgs in pools.items():
        logger.info("  원본 %-5s : %d장 (%s)", s, len(imgs), layout.images_dirs[s])

    final: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}

    if strategy == "resplit_all":
        allimgs: List[Path] = []
        for s in available:
            allimgs.extend(pools[s])
        shuffled = ds.deterministic_shuffle(allimgs, seed)
        r = cfg.splits["ratios"]
        n = len(shuffled)
        n_train = int(round(n * float(r["train"])))
        n_val = int(round(n * float(r["val"])))
        final["train"] = shuffled[:n_train]
        final["val"] = shuffled[n_train:n_train + n_val]
        final["test"] = shuffled[n_train + n_val:]
        info["notes"].append(f"전체 {n}장을 seed {seed} 로 {r['train']}/{r['val']}/{r['test']} 재분할했습니다.")

    elif "test" in pools and pools["test"] and strategy in ("existing", "split_val", "use_val_as_test"):
        final["train"] = pools.get("train", [])
        final["val"] = pools.get("val", [])
        final["test"] = pools["test"]
        info["strategy"] = "existing"
        info["notes"].append("데이터셋에 test 폴더가 있어 기존 train/val/test 를 그대로 사용했습니다.")

    elif strategy == "use_val_as_test":
        final["train"] = pools.get("train", [])
        final["val"] = pools.get("val", [])
        final["test"] = list(final["val"])
        info["notes"].append(
            "test 폴더가 없어 val 을 test 로 그대로 재사용했습니다. "
            "val 과 test 가 동일하므로 test 성능은 과대평가된 값입니다."
        )

    elif strategy == "split_val":
        if not pools.get("val"):
            logger.error("[실패] val split 이 없어 split_val 전략을 사용할 수 없습니다.")
            info["notes"].append("val 폴더가 없어 split_val 전략을 적용하지 못했습니다.")
            return info
        final["train"] = pools.get("train", [])
        shuffled = ds.deterministic_shuffle(pools["val"], seed)
        ratio = float(cfg.splits.get("test_ratio_of_val", 0.5))
        n_test = int(round(len(shuffled) * ratio))
        n_test = max(1, min(len(shuffled) - 1, n_test)) if len(shuffled) > 1 else len(shuffled)
        final["test"] = sorted(shuffled[:n_test], key=lambda p: str(p))
        final["val"] = sorted(shuffled[n_test:], key=lambda p: str(p))
        info["notes"].append(
            f"test 폴더가 없어 기존 val {len(pools['val'])}장을 seed {seed} 로 "
            f"val {len(final['val'])}장 / test {len(final['test'])}장 으로 나눴습니다. "
            "train 은 원본 그대로 유지했습니다."
        )
    else:
        logger.error("[실패] 알 수 없는 test_strategy: %s", strategy)
        return info

    # 중복 검사 — 모든 모델이 같은 데이터를 쓰되 split 간에는 겹치지 않아야 한다
    sets = {k: {str(p).lower() for p in v} for k, v in final.items()}
    overlaps = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        common = sets[a] & sets[b]
        if common:
            overlaps[f"{a}&{b}"] = len(common)
    if overlaps:
        info["overlaps"] = overlaps
        if not (info["strategy"] == "use_val_as_test" and set(overlaps) == {"val&test"}):
            logger.warning("[경고] split 간 중복 이미지가 있습니다: %s", overlaps)
        else:
            logger.warning("[경고] val 과 test 가 동일합니다 (use_val_as_test 전략).")

    if not final["test"]:
        logger.error("[실패] test split 이 비어 있습니다. 평가를 진행할 수 없습니다.")
        return info

    # 파일 기록
    paths = ds.split_files(cfg.splits_dir)
    for s in ("train", "val", "test"):
        ds.write_split_file(paths[s], final[s])
        info["counts"][s] = len(final[s])
        logger.info("  %-5s -> %s (%d장)", s, paths[s], len(final[s]))

    data_yaml = ds.build_compare_data_yaml(cfg.compare_data_yaml, cfg.root, cfg.splits_dir, layout.names)
    info["data_yaml"] = str(data_yaml)
    info["ok"] = True
    logger.info("공용 data.yaml 생성: %s", data_yaml)

    info_path = Path(cfg.splits_dir) / "split_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    logger.info("split 정보 기록: %s", info_path)
    for n in info["notes"]:
        logger.info("[참고] %s", n)
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="4개 모델 공용 train/val/test split 생성")
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true",
                        help="기존 split 파일이 있어도 다시 생성한다 (모델 간 비교 일관성이 깨질 수 있음)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("splits", cfg.logs_dir)
    log_banner(logger, "데이터 split 준비 (Train / Val / Test)")

    if ds.splits_exist(cfg.splits_dir) and not args.force:
        counts = {k: len(v) for k, v in ds.load_splits(cfg.splits_dir).items()}
        logger.info("기존 split 파일을 그대로 사용합니다: %s", cfg.splits_dir)
        logger.info("  train %d장 / val %d장 / test %d장", counts["train"], counts["val"], counts["test"])
        if not cfg.compare_data_yaml.is_file():
            layout = ds.detect_layout(cfg.dataset_root, cfg.data_yaml, cfg.fallback_names)
            ds.build_compare_data_yaml(cfg.compare_data_yaml, cfg.root, cfg.splits_dir, layout.names)
            logger.info("공용 data.yaml 생성: %s", cfg.compare_data_yaml)
        logger.info("다시 만들려면 --force 옵션을 사용하세요.")
        return 0

    info = build_splits(cfg, logger, force=args.force)
    return 0 if info.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
