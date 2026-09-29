"""4개 모델을 동일한 인터페이스로 다루는 추론 래퍼.

평가(evaluation/), 속도 측정(benchmark), 예측 이미지 생성, 단일 이미지 inference 가
모두 이 모듈을 사용한다. 덕분에 모델별로 다른 코드 경로 때문에 결과가 달라지는 일을 막는다.

공통 규약
  - 입력  : BGR numpy 배열(OpenCV) 또는 이미지 경로
  - 출력  : Detections(boxes=pixel xyxy(원본 좌표계), scores, labels=0-based class id)
  - 시간  : preprocess / inference / postprocess (ms) 분리 기록
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .metrics import Detections

TRAIN_META_NAME = "training_meta.json"


# ---------------------------------------------------------------------------
# 전처리 (YOLO letterbox 와 동일한 방식 — Faster R-CNN 에도 같은 기하 변환을 적용)
# ---------------------------------------------------------------------------
def letterbox(image: np.ndarray, new_size: int = 640,
              color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """가로세로 비율을 유지한 채 new_size x new_size 로 패딩한다.

    반환: (이미지, scale, (pad_left, pad_top))
    """
    import cv2

    h, w = image.shape[:2]
    scale = min(new_size / max(1, h), new_size / max(1, w))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    if (nh, nw) != (h, w):
        interp = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
        resized = cv2.resize(image, (nw, nh), interpolation=interp)
    else:
        resized = image
    pad_w, pad_h = new_size - nw, new_size - nh
    left, top = pad_w // 2, pad_h // 2
    right, bottom = pad_w - left, pad_h - top
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, scale, (float(left), float(top))


def unletterbox_boxes(boxes: np.ndarray, scale: float, pad: Tuple[float, float],
                      width: int, height: int) -> np.ndarray:
    """letterbox 좌표계의 xyxy 를 원본 이미지 좌표계로 되돌린다."""
    if boxes.size == 0:
        return boxes
    out = boxes.copy().astype(np.float32)
    out[:, [0, 2]] -= pad[0]
    out[:, [1, 3]] -= pad[1]
    out /= max(scale, 1e-9)
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, width)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, height)
    return out


def read_image(path) -> np.ndarray:
    """경로를 BGR numpy 로 읽는다. 한글/유니코드 경로(Windows)도 처리한다."""
    import cv2

    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        # Windows 에서 비ASCII 경로일 때 imread 가 실패하는 경우가 있다
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return img


# ---------------------------------------------------------------------------
# 학습 메타 정보 (batch size / 학습 시간 등 실제 사용된 값 기록)
# ---------------------------------------------------------------------------
def save_training_meta(run_dir: Path, meta: Dict[str, Any]) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / TRAIN_META_NAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return path


def load_training_meta(run_dir: Path) -> Dict[str, Any]:
    path = Path(run_dir) / TRAIN_META_NAME
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def resolve_weights(run_dir: Path, prefer: str = "best") -> Optional[Path]:
    """학습 결과 폴더에서 weight 파일을 찾는다 (best -> last 순)."""
    run_dir = Path(run_dir)
    candidates = [run_dir / "weights" / f"{prefer}.pt", run_dir / "weights" / "last.pt"]
    for c in candidates:
        if c.is_file():
            return c
    # ultralytics 가 runs/<key>/<name>/weights 형태로 저장한 경우 대비
    hits = sorted(run_dir.glob(f"**/weights/{prefer}.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if hits:
        return hits[0]
    hits = sorted(run_dir.glob("**/weights/last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# Detector 공통 인터페이스
# ---------------------------------------------------------------------------
@dataclass
class DetectorConfig:
    key: str
    display_name: str
    framework: str
    weights: Path
    image_size: int = 640
    conf: float = 0.001
    nms_iou: float = 0.7
    max_det: int = 300
    device: str = "cpu"
    class_names: List[str] = field(default_factory=list)


class BaseDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.model = None
        self.notes: List[str] = []

    # --- 공개 API ---
    def load(self) -> "BaseDetector":
        raise NotImplementedError

    def predict(self, image: np.ndarray) -> Tuple[Detections, Dict[str, float]]:
        raise NotImplementedError

    def predict_path(self, path) -> Tuple[Detections, Dict[str, float]]:
        t0 = time.perf_counter()
        img = read_image(path)
        load_ms = (time.perf_counter() - t0) * 1000.0
        det, timing = self.predict(img)
        timing = dict(timing)
        timing["load"] = load_ms
        return det, timing

    def info(self) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def class_names(self) -> List[str]:
        return self.cfg.class_names


class UltralyticsDetector(BaseDetector):
    """YOLOv8 / YOLO11 / RT-DETR 공통 (ultralytics 패키지)."""

    def load(self) -> "UltralyticsDetector":
        from ultralytics import RTDETR, YOLO

        weights = str(self.cfg.weights)
        loader = RTDETR if self.cfg.key == "rtdetr" else YOLO
        self.model = loader(weights)
        try:
            self.model.to(self.cfg.device)
        except Exception:
            pass
        names = self.model.names
        if isinstance(names, dict):
            self.cfg.class_names = [names[k] for k in sorted(names)]
        elif names:
            self.cfg.class_names = list(names)
        return self

    def predict(self, image: np.ndarray) -> Tuple[Detections, Dict[str, float]]:
        results = self.model.predict(
            source=image,
            imgsz=self.cfg.image_size,
            conf=self.cfg.conf,
            iou=self.cfg.nms_iou,
            max_det=self.cfg.max_det,
            device=self.cfg.device,
            verbose=False,
        )
        r = results[0]
        if r.boxes is not None and len(r.boxes):
            boxes = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            scores = r.boxes.conf.detach().cpu().numpy().astype(np.float32)
            labels = r.boxes.cls.detach().cpu().numpy().astype(np.int32)
        else:
            boxes = np.zeros((0, 4), np.float32)
            scores = np.zeros((0,), np.float32)
            labels = np.zeros((0,), np.int32)
        speed = getattr(r, "speed", {}) or {}
        timing = {
            "preprocess": float(speed.get("preprocess", 0.0)),
            "inference": float(speed.get("inference", 0.0)),
            "postprocess": float(speed.get("postprocess", 0.0)),
        }
        timing["total"] = timing["preprocess"] + timing["inference"] + timing["postprocess"]
        return Detections(boxes, scores, labels), timing

    def info(self) -> Dict[str, Any]:
        from .model_info import compute_gflops, count_parameters, file_size_mb

        module = getattr(self.model, "model", None)
        out: Dict[str, Any] = {
            "parameters": count_parameters(module) if module is not None else None,
            "model_size_mb": file_size_mb(self.cfg.weights),
            "gflops": None,
            "gflops_source": "",
        }
        try:
            info = module.info(detailed=False, verbose=False)  # (layers, params, grads, flops)
            if info and len(info) >= 4 and info[3]:
                out["gflops"] = round(float(info[3]), 3)
                out["gflops_source"] = "ultralytics model.info()"
            if info and len(info) >= 2 and info[1]:
                out["parameters"] = int(info[1])
        except Exception:
            pass
        if out["gflops"] is None and module is not None:
            g, src = compute_gflops(module, self.cfg.image_size, self.cfg.device)
            out["gflops"], out["gflops_source"] = g, src
        return out


class FasterRCNNDetector(BaseDetector):
    """torchvision Faster R-CNN ResNet50 FPN V2."""

    def load(self) -> "FasterRCNNDetector":
        import torch

        ckpt = torch.load(str(self.cfg.weights), map_location="cpu", weights_only=False)
        num_classes = int(ckpt.get("num_classes", len(self.cfg.class_names) + 1))
        names = ckpt.get("class_names")
        if names:
            self.cfg.class_names = list(names)
        model = build_faster_rcnn(num_classes=num_classes, pretrained=False,
                                  image_size=self.cfg.image_size)
        model.load_state_dict(ckpt["model_state_dict"])
        model.roi_heads.score_thresh = self.cfg.conf
        model.roi_heads.nms_thresh = self.cfg.nms_iou
        model.roi_heads.detections_per_img = self.cfg.max_det
        model.eval().to(self.cfg.device)
        self.model = model
        return self

    def predict(self, image: np.ndarray) -> Tuple[Detections, Dict[str, float]]:
        import torch

        h, w = image.shape[:2]
        device = self.cfg.device

        # --- preprocess (YOLO 와 동일한 letterbox 기하 변환) ---
        t0 = time.perf_counter()
        lb, scale, pad = letterbox(image, self.cfg.image_size)
        rgb = lb[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).to(device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # --- inference ---
        with torch.inference_mode():
            outputs = self.model([tensor])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        # --- postprocess (좌표 역변환 + 0-based class id) ---
        o = outputs[0]
        boxes = o["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = o["scores"].detach().cpu().numpy().astype(np.float32)
        labels = o["labels"].detach().cpu().numpy().astype(np.int32) - 1  # 0 = background
        boxes = unletterbox_boxes(boxes, scale, pad, w, h)
        keep = labels >= 0
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        t3 = time.perf_counter()

        timing = {
            "preprocess": (t1 - t0) * 1000.0,
            "inference": (t2 - t1) * 1000.0,
            "postprocess": (t3 - t2) * 1000.0,
        }
        timing["total"] = timing["preprocess"] + timing["inference"] + timing["postprocess"]
        return Detections(boxes, scores, labels), timing

    def info(self) -> Dict[str, Any]:
        from .model_info import count_parameters, file_size_mb

        out: Dict[str, Any] = {
            "parameters": count_parameters(self.model),
            "model_size_mb": file_size_mb(self.cfg.weights),
            "gflops": None,
            "gflops_source": "",
        }
        # torchvision 의 Faster R-CNN 은 입력이 list[Tensor] 이고 내부 transform 이 있어
        # thop 의 표준 프로파일링이 동작하지 않는다. 측정 불가 사유를 기록만 한다.
        out["gflops_source"] = (
            "측정 안 함: torchvision Faster R-CNN 은 list[Tensor] 입력과 내부 GeneralizedRCNNTransform "
            "때문에 thop 표준 프로파일링이 적용되지 않음"
        )
        return out


def build_faster_rcnn(num_classes: int, pretrained: bool = True, image_size: int = 640,
                      trainable_backbone_layers: Optional[int] = None):
    """Faster R-CNN ResNet50 FPN V2 생성 (COCO pretrained 가중치 사용 가능).

    num_classes 는 background 를 포함한 값이다 (예: 클래스 3개 -> 4).
    입력은 이미 640x640 으로 letterbox 된 이미지이므로 내부 transform 의 리사이즈가
    추가로 일어나지 않도록 min_size = max_size = image_size 로 고정한다.
    """
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    weights = None
    weights_backbone = None
    if pretrained:
        try:
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

            weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        except Exception:
            weights = "DEFAULT"

    kwargs: Dict[str, Any] = {
        "weights": weights,
        "weights_backbone": weights_backbone,
        "min_size": image_size,
        "max_size": image_size,
    }
    if trainable_backbone_layers is not None:
        kwargs["trainable_backbone_layers"] = trainable_backbone_layers

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(**kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_detector(model_key: str, weights: Path, device: str, image_size: int,
                   conf: float, nms_iou: float, max_det: int,
                   class_names: List[str], display_name: str = "",
                   framework: str = "") -> BaseDetector:
    from .config import MODEL_DISPLAY

    cfg = DetectorConfig(
        key=model_key,
        display_name=display_name or MODEL_DISPLAY.get(model_key, model_key),
        framework=framework or ("torchvision" if model_key == "faster_rcnn" else "ultralytics"),
        weights=Path(weights),
        image_size=image_size,
        conf=conf,
        nms_iou=nms_iou,
        max_det=max_det,
        device=device,
        class_names=list(class_names),
    )
    det = FasterRCNNDetector(cfg) if model_key == "faster_rcnn" else UltralyticsDetector(cfg)
    return det.load()
