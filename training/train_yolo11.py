"""YOLO11 학습 (빛공해 모델 비교 실험용).

PowerShell 사용 예:
    python training/train_yolo11.py
    python training/train_yolo11.py --epochs 50 --batch 8
    python training/train_yolo11.py --overwrite

YOLOv8 과 동일한 data.yaml / split / epoch / imgsz / seed / pretrained 조건을 사용한다.
기본 모델은 yolo11n.pt (config/experiment.yaml 의 models.yolo11.weights).
학습 결과는 runs/yolo11/ 에 저장된다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.train_ultralytics import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main("yolo11"))
