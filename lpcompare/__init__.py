"""빛공해 객체탐지 모델 비교 실험 공용 패키지.

YOLOv8 / YOLO11 / RT-DETR / Faster R-CNN 을 동일한 데이터셋·동일한 평가 기준으로
비교하기 위한 설정 로딩, 데이터셋 스캔, 지표 계산, 모델 래퍼, 시각화 유틸을 제공한다.

기존 웹 서비스 코드(backend.py)와 기존 학습 코드(train_model.py)는 이 패키지를
import 하지 않으며, 이 패키지도 기존 코드나 원본 데이터셋을 수정하지 않는다.
"""

__all__ = [
    "config",
    "dataset",
    "logging_utils",
    "metrics",
    "model_info",
    "runtime",
    "viz",
]
