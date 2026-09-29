"""YOLO 데이터셋 탐색 / 검사 / split 관리.

원본 데이터셋은 **읽기 전용**으로만 다룬다. 이 모듈은 원본 이미지·라벨·data.yaml 을
절대 수정하거나 이동하지 않으며, 필요한 파생 파일은 splits/ 와 derived_data/ 에만 만든다.

지원하는 디렉터리 구조 (자동 판별)
  1) split_first : <root>/train/images, <root>/train/labels, <root>/val/images ...
                   (현재 프로젝트의 data/images 구조, Roboflow YOLOv8 export 기본형)
  2) type_first  : <root>/images/train, <root>/labels/train, ...
  3) flat        : <root>/images, <root>/labels  (단일 split)
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
SPLITS = ("train", "val", "test")
# val 폴더 이름 변형 (Roboflow 는 valid 를 쓰는 경우가 있다)
SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("val", "valid", "validation"),
    "test": ("test",),
}


# ---------------------------------------------------------------------------
# 경로 유틸
# ---------------------------------------------------------------------------
def label_path_for(image_path: Path) -> Path:
    """이미지 경로 -> 라벨(.txt) 경로.

    Ultralytics 와 동일하게 경로에서 **마지막** `images` 디렉터리를 `labels` 로 바꾼다.
    두 레이아웃 모두에서 올바르게 동작한다.
      data/images/train/images/a.jpg -> data/images/train/labels/a.txt
      dataset/images/train/a.jpg     -> dataset/labels/train/a.txt
    """
    parts = list(Path(image_path).parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def list_images(directory: Path) -> List[Path]:
    if not directory or not Path(directory).is_dir():
        return []
    files = [p for p in sorted(Path(directory).rglob("*")) if p.suffix.lower() in IMG_EXTS]
    return files


# ---------------------------------------------------------------------------
# 데이터셋 레이아웃
# ---------------------------------------------------------------------------
@dataclass
class DatasetLayout:
    root: Path
    style: str                                   # split_first | type_first | flat | missing
    images_dirs: Dict[str, Path] = field(default_factory=dict)
    labels_dirs: Dict[str, Path] = field(default_factory=dict)
    data_yaml: Optional[Path] = None
    names: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def available_splits(self) -> List[str]:
        return [s for s in SPLITS if s in self.images_dirs]

    @property
    def exists(self) -> bool:
        return self.style != "missing"


def _first_existing(root: Path, candidates: Sequence[Path]) -> Optional[Path]:
    for c in candidates:
        if c.is_dir():
            return c
    return None


def detect_layout(root: Path, data_yaml: Optional[Path] = None,
                  fallback_names: Optional[List[str]] = None) -> DatasetLayout:
    root = Path(root)
    layout = DatasetLayout(root=root, style="missing")

    if not root.is_dir():
        layout.notes.append(f"데이터셋 루트를 찾을 수 없습니다: {root}")
        layout.names = list(fallback_names or [])
        return layout

    images_dirs: Dict[str, Path] = {}
    labels_dirs: Dict[str, Path] = {}

    # 1) split_first : <root>/<split>/images
    for split, aliases in SPLIT_ALIASES.items():
        img = _first_existing(root, [root / a / "images" for a in aliases])
        if img:
            images_dirs[split] = img
            lbl = img.parent / "labels"
            labels_dirs[split] = lbl
    if images_dirs:
        layout.style = "split_first"

    # 2) type_first : <root>/images/<split>
    if not images_dirs:
        for split, aliases in SPLIT_ALIASES.items():
            img = _first_existing(root, [root / "images" / a for a in aliases])
            if img:
                images_dirs[split] = img
                labels_dirs[split] = root / "labels" / img.name
        if images_dirs:
            layout.style = "type_first"

    # 3) flat : <root>/images
    if not images_dirs and (root / "images").is_dir():
        images_dirs["train"] = root / "images"
        labels_dirs["train"] = root / "labels"
        layout.style = "flat"
        layout.notes.append("단일 images/labels 구조입니다. train 으로 간주하고 split 을 새로 만들어야 합니다.")

    layout.images_dirs = images_dirs
    layout.labels_dirs = labels_dirs

    if not images_dirs:
        layout.style = "missing"
        layout.notes.append(
            f"{root} 아래에서 YOLO 데이터셋 구조(train/images 또는 images/train)를 찾지 못했습니다."
        )

    # data.yaml 탐색 (원본은 읽기만 한다)
    yml = Path(data_yaml) if data_yaml else find_data_yaml(root)
    layout.data_yaml = yml
    layout.names = load_class_names(yml, fallback_names)
    if yml is None:
        layout.notes.append("data.yaml 을 찾지 못해 config 의 fallback_names 를 사용합니다.")
    return layout


def find_data_yaml(root: Path) -> Optional[Path]:
    root = Path(root)
    for name in ("data.yaml", "data.yml", "light_pollution.yaml", "dataset.yaml"):
        p = root / name
        if p.is_file():
            return p
    if root.is_dir():
        for p in sorted(root.glob("*.yaml")):
            return p
    return None


def load_class_names(data_yaml: Optional[Path], fallback: Optional[List[str]] = None) -> List[str]:
    if data_yaml and Path(data_yaml).is_file():
        try:
            with open(data_yaml, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            names = cfg.get("names")
            if isinstance(names, dict):
                return [names[k] for k in sorted(names, key=lambda x: int(x))]
            if isinstance(names, list) and names:
                return list(names)
        except Exception:
            pass
    return list(fallback or [])


# ---------------------------------------------------------------------------
# 라벨 파싱 / 검사
# ---------------------------------------------------------------------------
@dataclass
class LabelIssue:
    image: str
    label: str
    line_no: int
    kind: str
    detail: str


def parse_label_file(label_path: Path, num_classes: int) -> Tuple[List[Tuple[int, float, float, float, float]], List[Tuple[int, str, str]]]:
    """YOLO 라벨 파일을 읽어 (boxes, issues) 를 돌려준다.

    boxes  : [(cls, xc, yc, w, h), ...]  (normalized 원본 값 그대로)
    issues : [(line_no, kind, detail), ...]
    """
    boxes: List[Tuple[int, float, float, float, float]] = []
    issues: List[Tuple[int, str, str]] = []
    if not Path(label_path).is_file():
        return boxes, issues

    with open(label_path, "r", encoding="utf-8", errors="replace") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                issues.append((i, "format_error", f"필드 개수 {len(parts)}개 (5개 필요): '{line}'"))
                continue

            if len(parts) > 5:
                # YOLO 세그멘테이션(폴리곤) 라벨: class x1 y1 x2 y2 ...
                # Ultralytics 의 segments2boxes 와 동일하게 폴리곤의 외접 사각형으로 변환한다.
                coords = parts[1:]
                if len(coords) >= 6 and len(coords) % 2 == 0:
                    try:
                        cls = int(float(parts[0]))
                        xs = [float(v) for v in coords[0::2]]
                        ys = [float(v) for v in coords[1::2]]
                    except ValueError:
                        issues.append((i, "format_error", f"숫자로 변환할 수 없습니다: '{line[:80]}'"))
                        continue
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                    xc, yc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    w, h = x2 - x1, y2 - y1
                    issues.append((i, "polygon_label",
                                   f"폴리곤 라벨({len(xs)}개 점)을 외접 bbox 로 변환했습니다."))
                else:
                    issues.append((i, "format_error",
                                   f"필드 개수 {len(parts)}개 — bbox(5개)도 폴리곤(홀수 개, 점 3개 이상)도 아닙니다."))
                    continue
            else:
                try:
                    cls = int(float(parts[0]))
                    xc, yc, w, h = (float(v) for v in parts[1:5])
                except ValueError:
                    issues.append((i, "format_error", f"숫자로 변환할 수 없습니다: '{line}'"))
                    continue

            if num_classes and not (0 <= cls < num_classes):
                issues.append((i, "invalid_class_id", f"class_id={cls} (유효 범위 0~{num_classes - 1})"))
            if w <= 0:
                issues.append((i, "nonpositive_width", f"width={w}"))
            if h <= 0:
                issues.append((i, "nonpositive_height", f"height={h}"))
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                issues.append((i, "out_of_range", f"정규화 범위(0~1) 밖: xc={xc}, yc={yc}, w={w}, h={h}"))
            elif (xc - w / 2 < -1e-6) or (yc - h / 2 < -1e-6) or (xc + w / 2 > 1 + 1e-6) or (yc + h / 2 > 1 + 1e-6):
                issues.append((i, "bbox_overflow", f"bbox 가 이미지 경계를 벗어납니다: xc={xc}, yc={yc}, w={w}, h={h}"))

            boxes.append((cls, xc, yc, w, h))
    return boxes, issues


def yolo_to_xyxy(box: Sequence[float], width: int, height: int) -> Tuple[float, float, float, float]:
    """(xc, yc, w, h) normalized -> (xmin, ymin, xmax, ymax) pixel."""
    xc, yc, w, h = box
    xmin = (xc - w / 2.0) * width
    ymin = (yc - h / 2.0) * height
    xmax = (xc + w / 2.0) * width
    ymax = (yc + h / 2.0) * height
    return xmin, ymin, xmax, ymax


def load_ground_truth(image_path: Path, width: int, height: int,
                      num_classes: int = 0) -> Tuple[List[List[float]], List[int]]:
    """원본 라벨을 읽어 pixel 좌표 xyxy 와 class id 리스트로 돌려준다 (원본 파일은 수정하지 않음)."""
    lp = label_path_for(image_path)
    boxes_norm, _ = parse_label_file(lp, num_classes)
    boxes, labels = [], []
    for cls, xc, yc, w, h in boxes_norm:
        if w <= 0 or h <= 0:
            continue
        x1, y1, x2, y2 = yolo_to_xyxy((xc, yc, w, h), width, height)
        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(int(cls))
    return boxes, labels


def image_size(image_path: Path) -> Tuple[int, int]:
    """(width, height). Pillow 로 헤더만 읽는다."""
    from PIL import Image

    with Image.open(image_path) as im:
        return im.size


# ---------------------------------------------------------------------------
# split 파일 (splits/train.txt, val.txt, test.txt)
# ---------------------------------------------------------------------------
def write_split_file(path: Path, images: Sequence[Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in images:
            f.write(str(Path(p).resolve()) + "\n")
    return path


def read_split_file(path: Path) -> List[Path]:
    path = Path(path)
    if not path.is_file():
        return []
    out: List[Path] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Path(line))
    return out


def split_files(splits_dir: Path) -> Dict[str, Path]:
    return {s: Path(splits_dir) / f"{s}.txt" for s in SPLITS}


def splits_exist(splits_dir: Path) -> bool:
    return all(p.is_file() for p in split_files(splits_dir).values())


def load_splits(splits_dir: Path) -> Dict[str, List[Path]]:
    return {s: read_split_file(p) for s, p in split_files(splits_dir).items()}


def deterministic_shuffle(items: List[Path], seed: int) -> List[Path]:
    """경로 문자열 기준으로 정렬한 뒤 고정 seed 로 섞는다 (OS/파일시스템에 무관하게 동일 결과)."""
    ordered = sorted(items, key=lambda p: str(p).replace("\\", "/").lower())
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered


# ---------------------------------------------------------------------------
# 라벨 정규화 (폴리곤/세그멘테이션 라벨이 섞인 데이터셋 대응)
# ---------------------------------------------------------------------------
def scan_label_formats(layout: DatasetLayout) -> Dict[str, Dict[str, int]]:
    """split 별로 bbox 전용 / 폴리곤 포함 파일 수를 집계한다 (원본은 읽기만 한다)."""
    out: Dict[str, Dict[str, int]] = {}
    for split in layout.available_splits:
        counts = {"files": 0, "polygon_files": 0, "mixed_files": 0,
                  "polygon_lines": 0, "bbox_lines": 0, "bbox_lines_in_mixed": 0}
        for img in list_images(layout.images_dirs[split]):
            lp = label_path_for(img)
            if not lp.is_file():
                continue
            counts["files"] += 1
            try:
                raw = lp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            poly = bbox = 0
            for line in raw.strip().splitlines():
                parts = line.split()
                if len(parts) > 6:
                    poly += 1
                elif len(parts) == 5:
                    bbox += 1
            counts["polygon_lines"] += poly
            counts["bbox_lines"] += bbox
            if poly:
                counts["polygon_files"] += 1
                if bbox:
                    counts["mixed_files"] += 1
                    counts["bbox_lines_in_mixed"] += bbox
        out[split] = counts
    return out


def needs_label_normalization(scan: Dict[str, Dict[str, int]]) -> bool:
    return any(c["polygon_lines"] > 0 for c in scan.values())


def _link_or_copy_images(src: Path, dst: Path) -> str:
    """이미지 폴더를 파생 위치에 연결한다. Windows 디렉터리 junction 우선, 실패 시 복사."""
    import shutil
    import subprocess

    src, dst = Path(src), Path(dst)
    if dst.exists():
        return "existing"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        # mklink /J 는 관리자 권한 없이 동작하며 디스크를 추가로 쓰지 않는다
        res = subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                             capture_output=True, text=True)
        if res.returncode == 0 and dst.exists():
            return "junction"
    try:
        dst.symlink_to(src, target_is_directory=True)
        return "symlink"
    except Exception:
        pass
    shutil.copytree(src, dst)
    return "copy"


def build_normalized_dataset(layout: DatasetLayout, out_root: Path, logger=None) -> Dict[str, object]:
    """폴리곤 라벨을 외접 bbox 로 바꾼 **파생 데이터셋**을 만든다.

    원본 이미지/라벨/data.yaml 은 전혀 건드리지 않는다.
        <out_root>/<split>/labels/*.txt   정규화된 라벨 (새로 생성)
        <out_root>/<split>/images         원본 이미지 폴더로의 junction (복사 아님)

    폴리곤 라벨이 한 줄이라도 섞인 파일은 Ultralytics 가 파일 전체를 폴리곤으로 간주해
    같은 파일의 정상 bbox 라인까지 잘못 해석한다. 정규화 후에는 모든 줄이 5필드 bbox 이므로
    YOLO 계열과 Faster R-CNN 이 완전히 동일한 정답을 사용하게 된다.
    """
    out_root = Path(out_root)
    result: Dict[str, object] = {
        "out_root": str(out_root), "splits": {}, "images_link": {},
        "converted_polygon_lines": 0, "recovered_bbox_lines": 0,
        "dropped_lines": 0, "total_files": 0,
    }

    for split in layout.available_splits:
        img_dir = layout.images_dirs[split]
        dst_split = out_root / split
        link_mode = _link_or_copy_images(img_dir, dst_split / "images")
        result["images_link"][split] = link_mode

        dst_labels = dst_split / "labels"
        dst_labels.mkdir(parents=True, exist_ok=True)

        n_files = n_poly = n_mixed_bbox = n_dropped = 0
        for img in list_images(img_dir):
            src_lp = label_path_for(img)
            dst_lp = dst_labels / (img.stem + ".txt")
            if not src_lp.is_file():
                continue
            boxes, issues = parse_label_file(src_lp, len(layout.names))
            poly_here = sum(1 for _, kind, _ in issues if kind == "polygon_label")
            dropped = sum(1 for _, kind, _ in issues if kind == "format_error")
            bbox_here = len(boxes) - poly_here
            if poly_here and bbox_here > 0:
                n_mixed_bbox += bbox_here
            n_poly += poly_here
            n_dropped += dropped

            lines = [f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for cls, xc, yc, w, h in boxes]
            dst_lp.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
            n_files += 1

        result["splits"][split] = {
            "files": n_files, "polygon_lines": n_poly,
            "bbox_lines_in_mixed_files": n_mixed_bbox, "dropped_lines": n_dropped,
            "images": link_mode,
        }
        result["total_files"] += n_files
        result["converted_polygon_lines"] += n_poly
        result["recovered_bbox_lines"] += n_mixed_bbox
        result["dropped_lines"] += n_dropped
        if logger:
            logger.info("  %-5s : 라벨 %d개 정규화 (폴리곤 %d줄 변환, 혼재 파일의 bbox %d줄 보정) / images: %s",
                        split, n_files, n_poly, n_mixed_bbox, link_mode)

    # 정규화된 트리를 가리키는 layout 을 새로 만든다
    normalized = DatasetLayout(
        root=out_root, style="split_first",
        images_dirs={s: out_root / s / "images" for s in layout.available_splits},
        labels_dirs={s: out_root / s / "labels" for s in layout.available_splits},
        data_yaml=layout.data_yaml, names=list(layout.names),
        notes=["폴리곤 라벨을 bbox 로 정규화한 파생 데이터셋 (원본은 수정하지 않음)"],
    )
    result["layout"] = normalized
    return result


def build_compare_data_yaml(out_path: Path, project_root: Path, splits_dir: Path,
                            names: List[str]) -> Path:
    """4개 모델이 공유하는 파생 data.yaml 을 만든다 (원본 data.yaml 은 그대로 둔다).

    Ultralytics 는 train/val/test 값으로 이미지 경로 목록이 담긴 .txt 파일을 지원한다.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    splits_dir = Path(splits_dir).resolve()
    # split 폴더가 프로젝트와 다른 드라이브(C:/D:/E:)에 있어도 동작하도록 절대경로로 기록한다.
    cfg = {
        "path": str(Path(project_root).resolve()),
        "train": str(splits_dir / "train.txt"),
        "val": str(splits_dir / "val.txt"),
        "test": str(splits_dir / "test.txt"),
        "nc": len(names),
        "names": list(names),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
    return out_path
