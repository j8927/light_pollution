"""RT-DETR 학습 (빛공해 모델 비교 실험용).

PowerShell 사용 예:
    python training/train_rtdetr.py
    python training/train_rtdetr.py --batch 2
    python training/train_rtdetr.py --overwrite

Ultralytics 의 RTDETR 클래스를 사용하며, 기본 모델은 rtdetr-l.pt
(config/experiment.yaml 의 models.rtdetr.weights).

RT-DETR 은 YOLO nano 계열보다 모델이 크고 GPU 메모리를 많이 사용한다.
CUDA Out Of Memory 가 발생하면 batch 를 절반씩 낮춰 재시도하며,
실제 사용된 batch 와 재시도 이력은 runs/rtdetr/training_meta.json 및
results/model_comparison.csv 에 그대로 기록된다 (차이를 숨기지 않는다).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.train_ultralytics import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main("rtdetr"))
