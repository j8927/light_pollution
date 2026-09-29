"""YOLOv8 학습 (빛공해 모델 비교 실험용).

PowerShell 사용 예:
    python training/train_yolov8.py
    python training/train_yolov8.py --epochs 50 --batch 8
    python training/train_yolov8.py --overwrite          # 기존 학습 결과가 있어도 다시 학습

공통 설정은 config/experiment.yaml 에서 읽는다 (기본: yolov8n.pt, 100 epoch, 640px, seed 42).
학습 결과는 runs/yolov8/ 에 저장된다.

※ 웹 서비스용 기존 학습 스크립트인 train_model.py 는 그대로 유지되며 이 파일과 무관하게 동작한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.train_ultralytics import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main("yolov8"))
