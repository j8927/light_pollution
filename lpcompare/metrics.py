"""모든 모델에 동일하게 적용되는 탐지 성능 지표 계산.

4개 모델(YOLOv8 / YOLO11 / RT-DETR / Faster R-CNN)의 예측 결과를 같은 포맷
(pixel xyxy + score + class id)으로 모은 뒤, 여기서 **하나의 동일한 코드**로
지표를 계산한다. 모델별 프레임워크가 제공하는 서로 다른 평가기를 쓰지 않기 때문에
평가 기준 차이로 인한 왜곡이 없다.

계산 방식
  - AP : COCO 방식 (IoU 0.50:0.05:0.95, 101-point interpolation)
  - Precision / Recall / F1 : conf_threshold 이상 예측을 IoU 0.5 로 greedy 매칭
  - 객체 크기별 AP : COCO 면적 기준(small < 32^2, medium < 96^2 px) 또는 상대 면적 기준
  - Confusion Matrix : Ultralytics 와 동일한 (nc+1) x (nc+1), 마지막 행/열은 background
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# COCO 기본 IoU 구간
IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)

# 객체 크기 구간
COCO_AREA_RANGES = {
    "all": (0.0, float("inf")),
    "small": (0.0, 32.0 ** 2),
    "medium": (32.0 ** 2, 96.0 ** 2),
    "large": (96.0 ** 2, float("inf")),
}
# 상대 기준 (이미지 면적 대비 비율) — COCO 절대 기준이 맞지 않는 데이터셋용
RELATIVE_AREA_RANGES = {
    "all": (0.0, float("inf")),
    "small": (0.0, 0.001),      # 이미지 면적의 0.1% 미만
    "medium": (0.001, 0.01),    # 0.1% ~ 1%
    "large": (0.01, float("inf")),  # 1% 이상
}


@dataclass
class Detections:
    """한 이미지에 대한 예측 결과 (pixel 좌표계, 원본 이미지 기준)."""

    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    scores: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    labels: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    def filter(self, conf: float) -> "Detections":
        if len(self) == 0:
            return self
        keep = self.scores >= conf
        return Detections(self.boxes[keep], self.scores[keep], self.labels[keep])


@dataclass
class GroundTruth:
    """한 이미지에 대한 정답 (pixel 좌표계)."""

    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    labels: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    width: int = 0
    height: int = 0

    def __len__(self) -> int:
        return int(self.boxes.shape[0])


def to_arrays(boxes, labels, scores=None):
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    l = np.asarray(labels, dtype=np.int32).reshape(-1)
    if scores is None:
        return b, l
    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    return b, l, s


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (N,4), b (M,4) xyxy -> IoU (N,M)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)


def box_areas(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return (np.clip(boxes[:, 2] - boxes[:, 0], 0, None) *
            np.clip(boxes[:, 3] - boxes[:, 1], 0, None)).astype(np.float32)


def _area_values(gt: GroundTruth, criterion: str) -> np.ndarray:
    """면적 기준값. coco 는 절대 px^2, relative 는 이미지 면적 대비 비율."""
    areas = box_areas(gt.boxes)
    if criterion == "relative":
        denom = float(max(1, gt.width * gt.height))
        return areas / denom
    return areas


# ---------------------------------------------------------------------------
# COCO 방식 AP
# ---------------------------------------------------------------------------
def _ap_from_pr(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> Tuple[float, np.ndarray, np.ndarray]:
    """누적 TP/FP 배열에서 101-point interpolated AP 를 계산한다."""
    if n_gt == 0:
        return float("nan"), np.zeros(0), np.zeros(0)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)

    # precision 을 단조 감소하도록 보정 (COCO/VOC 공통)
    mpre = precision.copy()
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.searchsorted(recall, RECALL_POINTS, side="left")
    q = np.zeros_like(RECALL_POINTS)
    valid = idx < len(mpre)
    q[valid] = mpre[idx[valid]]
    return float(q.mean()), recall, precision


def evaluate_ap(preds: Sequence[Detections], gts: Sequence[GroundTruth], num_classes: int,
                area_range: Tuple[float, float] = (0.0, float("inf")),
                area_criterion: str = "coco",
                iou_thresholds: np.ndarray = IOU_THRESHOLDS) -> Dict[str, object]:
    """클래스별 / IoU별 AP 를 계산한다.

    COCO 와 동일하게, 면적 범위를 벗어난 GT 는 ignore 로 처리하고
    그 GT 에 매칭된 예측은 FP 로 세지 않는다.
    """
    n_iou = len(iou_thresholds)
    ap = np.full((num_classes, n_iou), np.nan, dtype=np.float64)
    gt_counts = np.zeros(num_classes, dtype=np.int64)

    for c in range(num_classes):
        # --- 이미지별 GT / 예측 수집 ---
        img_gt_boxes, img_gt_ignore = [], []
        det_records = []  # (score, img_idx, box)
        n_pos = 0
        for i, gt in enumerate(gts):
            if len(gt):
                sel = gt.labels == c
                gboxes = gt.boxes[sel]
                gareas = _area_values(gt, area_criterion)[sel]
                ignore = ~((gareas >= area_range[0]) & (gareas < area_range[1]))
            else:
                gboxes = np.zeros((0, 4), dtype=np.float32)
                ignore = np.zeros((0,), dtype=bool)
            img_gt_boxes.append(gboxes)
            img_gt_ignore.append(ignore)
            n_pos += int((~ignore).sum())

            pred = preds[i]
            if len(pred):
                psel = pred.labels == c
                for box, score in zip(pred.boxes[psel], pred.scores[psel]):
                    det_records.append((float(score), i, box))

        gt_counts[c] = n_pos
        if n_pos == 0:
            continue  # GT 가 없는 클래스는 AP 를 정의하지 않는다 (nan)
        if not det_records:
            ap[c, :] = 0.0
            continue

        det_records.sort(key=lambda r: -r[0])
        n_det = len(det_records)

        for t_idx, thr in enumerate(iou_thresholds):
            matched = [np.zeros(len(b), dtype=bool) for b in img_gt_boxes]
            tp = np.zeros(n_det, dtype=np.float64)
            fp = np.zeros(n_det, dtype=np.float64)
            ignored = np.zeros(n_det, dtype=bool)

            for d_idx, (_score, img_idx, box) in enumerate(det_records):
                gboxes = img_gt_boxes[img_idx]
                if len(gboxes) == 0:
                    fp[d_idx] = 1.0
                    continue
                ious = iou_matrix(box.reshape(1, 4), gboxes)[0]
                ignore_flags = img_gt_ignore[img_idx]
                used = matched[img_idx]

                # 1) ignore 가 아닌 GT 중 최고 IoU
                cand = (~used) & (~ignore_flags) & (ious >= thr)
                if cand.any():
                    j = int(np.argmax(np.where(cand, ious, -1.0)))
                    used[j] = True
                    tp[d_idx] = 1.0
                    continue
                # 2) ignore GT 에 매칭되면 이 예측은 평가에서 제외
                cand_ig = (~used) & ignore_flags & (ious >= thr)
                if cand_ig.any():
                    j = int(np.argmax(np.where(cand_ig, ious, -1.0)))
                    used[j] = True
                    ignored[d_idx] = True
                    continue
                fp[d_idx] = 1.0

            keep = ~ignored
            ap_val, _, _ = _ap_from_pr(tp[keep], fp[keep], n_pos)
            ap[c, t_idx] = ap_val

    with warnings.catch_warnings():
        # GT 가 전혀 없는 클래스는 AP 가 nan 이므로 "Mean of empty slice" 경고가 나온다 (정상 동작)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ap50_95 = np.nanmean(ap, axis=1)
        map50 = float(np.nanmean(ap[:, 0])) if np.isfinite(ap[:, 0]).any() else float("nan")
        map50_95 = float(np.nanmean(ap)) if np.isfinite(ap).any() else float("nan")

    return {
        "ap": ap,                       # (nc, n_iou)
        "iou_thresholds": iou_thresholds,
        "gt_counts": gt_counts,
        "ap50": ap[:, 0],
        "ap50_95": ap50_95,
        "map50": map50,
        "map50_95": map50_95,
    }


# ---------------------------------------------------------------------------
# Precision / Recall / F1 (고정 conf, 고정 IoU)
# ---------------------------------------------------------------------------
def evaluate_pr(preds: Sequence[Detections], gts: Sequence[GroundTruth], num_classes: int,
                conf: float = 0.25, iou_thr: float = 0.5) -> Dict[str, object]:
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    for pred, gt in zip(preds, gts):
        p = pred.filter(conf)
        order = np.argsort(-p.scores) if len(p) else np.zeros(0, dtype=int)
        used = np.zeros(len(gt), dtype=bool)
        for d in order:
            cls = int(p.labels[d])
            if len(gt):
                same = (gt.labels == cls) & (~used)
                ious = iou_matrix(p.boxes[d].reshape(1, 4), gt.boxes)[0]
                cand = same & (ious >= iou_thr)
            else:
                cand = np.zeros(0, dtype=bool)
            if cand.any():
                j = int(np.argmax(np.where(cand, ious, -1.0)))
                used[j] = True
                tp[cls] += 1
            else:
                fp[cls] += 1
        for j in range(len(gt)):
            if not used[j]:
                fn[int(gt.labels[j])] += 1

    precision = np.divide(tp, np.maximum(tp + fp, 1), dtype=np.float64)
    recall = np.divide(tp, np.maximum(tp + fn, 1), dtype=np.float64)
    precision = np.where((tp + fp) == 0, np.nan, precision)
    recall = np.where((tp + fn) == 0, np.nan, recall)
    with np.errstate(invalid="ignore"):
        f1 = 2 * precision * recall / np.maximum(precision + recall, np.finfo(np.float64).eps)

    def _mean(x):
        return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "precision_mean": _mean(precision),
        "recall_mean": _mean(recall),
        "f1_mean": _mean(f1),
        "conf": conf, "iou": iou_thr,
        "tp_total": int(tp.sum()), "fp_total": int(fp.sum()), "fn_total": int(fn.sum()),
    }


# ---------------------------------------------------------------------------
# Confusion Matrix (Ultralytics 와 동일한 형태)
# ---------------------------------------------------------------------------
def confusion_matrix(preds: Sequence[Detections], gts: Sequence[GroundTruth], num_classes: int,
                     conf: float = 0.25, iou_thr: float = 0.5) -> np.ndarray:
    """(nc+1, nc+1) 행렬. matrix[pred, gt], 마지막 인덱스는 background."""
    nc = num_classes
    matrix = np.zeros((nc + 1, nc + 1), dtype=np.int64)

    for pred, gt in zip(preds, gts):
        p = pred.filter(conf)
        gt_used = np.zeros(len(gt), dtype=bool)
        pred_used = np.zeros(len(p), dtype=bool)

        if len(p) and len(gt):
            ious = iou_matrix(p.boxes, gt.boxes)
            # IoU 가 큰 쌍부터 greedy 매칭 (클래스 무관 — 오분류를 보기 위해)
            pairs = np.argwhere(ious >= iou_thr)
            if pairs.size:
                order = np.argsort(-ious[pairs[:, 0], pairs[:, 1]])
                for k in order:
                    di, gi = pairs[k]
                    if pred_used[di] or gt_used[gi]:
                        continue
                    pred_used[di] = True
                    gt_used[gi] = True
                    matrix[int(p.labels[di]), int(gt.labels[gi])] += 1

        for gi in range(len(gt)):
            if not gt_used[gi]:
                matrix[nc, int(gt.labels[gi])] += 1  # 미탐 (background 로 예측됨)
        for di in range(len(p)):
            if not pred_used[di]:
                matrix[int(p.labels[di]), nc] += 1   # 오탐 (background 를 객체로 예측)
    return matrix


# ---------------------------------------------------------------------------
# 통합 실행
# ---------------------------------------------------------------------------
def evaluate_all(preds: Sequence[Detections], gts: Sequence[GroundTruth], num_classes: int,
                 conf: float = 0.25, iou_thr: float = 0.5,
                 area_criterion: str = "coco") -> Dict[str, object]:
    ranges = COCO_AREA_RANGES if area_criterion == "coco" else RELATIVE_AREA_RANGES
    overall = evaluate_ap(preds, gts, num_classes, ranges["all"], area_criterion)
    size_ap = {}
    for key in ("small", "medium", "large"):
        r = evaluate_ap(preds, gts, num_classes, ranges[key], area_criterion)
        size_ap[key] = {
            "map50": r["map50"],
            "map50_95": r["map50_95"],
            "gt_count": int(r["gt_counts"].sum()),
        }
    pr = evaluate_pr(preds, gts, num_classes, conf, iou_thr)
    cm = confusion_matrix(preds, gts, num_classes, conf, iou_thr)
    return {
        "overall": overall,
        "size_ap": size_ap,
        "pr": pr,
        "confusion_matrix": cm,
        "area_criterion": area_criterion,
        "area_ranges": {k: list(v) for k, v in ranges.items()},
    }


def torchmetrics_cross_check(preds: Sequence[Detections], gts: Sequence[GroundTruth]) -> Dict[str, Any]:
    """torchmetrics 로 mAP 를 한 번 더 계산해 교차 검증 값을 돌려준다 (선택 사항).

    성공 시 {"map50": float, "map50_95": float},
    사용할 수 없으면 {"unavailable": 사유} — 조용히 건너뛰지 않고 이유를 남긴다.
    """
    try:
        import torch
        from torchmetrics.detection import MeanAveragePrecision
    except Exception as exc:
        return {"unavailable": f"torchmetrics 를 불러올 수 없습니다 ({type(exc).__name__}: {exc})"}
    try:
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    except Exception as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}
    try:
        for p, g in zip(preds, gts):
            metric.update(
                [{"boxes": torch.as_tensor(p.boxes, dtype=torch.float32).reshape(-1, 4),
                  "scores": torch.as_tensor(p.scores, dtype=torch.float32).reshape(-1),
                  "labels": torch.as_tensor(p.labels, dtype=torch.int64).reshape(-1)}],
                [{"boxes": torch.as_tensor(g.boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.as_tensor(g.labels, dtype=torch.int64).reshape(-1)}],
            )
        res = metric.compute()
        return {"map50": float(res["map_50"]), "map50_95": float(res["map"])}
    except Exception as exc:
        return {"unavailable": f"계산 실패 — {type(exc).__name__}: {exc}"}
