"""YOLO 형식 라벨을 그대로 읽어 torchvision Faster R-CNN 용 데이터로 변환하는 Dataset.

원본 라벨 파일(.txt)은 읽기만 하며 수정하지 않는다. COCO JSON 등 별도 포맷으로
변환해 저장하지 않고, 학습/평가 시점에 메모리에서 바로 변환한다.

좌표 변환
    YOLO   : class_id x_center y_center width height   (0~1 정규화)
    변환 후 : xmin, ymin, xmax, ymax                   (letterbox 된 640x640 pixel)

    xmin = (x_center - width  / 2) * 이미지너비
    ymin = (y_center - height / 2) * 이미지높이
    xmax = (x_center + width  / 2) * 이미지너비
    ymax = (y_center + height / 2) * 이미지높이
    -> letterbox scale/padding 을 적용해 640x640 좌표로 옮긴다.

클래스 id
    YOLO 0-based  ->  Faster R-CNN 1-based (0 은 background 로 예약)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lpcompare import dataset as ds  # noqa: E402
from lpcompare.runtime import letterbox, read_image  # noqa: E402


class YoloDetectionDataset(Dataset):
    """YOLO txt 라벨을 읽는 torchvision detection 용 Dataset.

    Args:
        image_paths   : 사용할 이미지 경로 목록 (splits/*.txt 에서 읽은 값)
        num_classes   : 배경을 제외한 클래스 수
        image_size    : letterbox 목표 크기 (기본 640)
        train         : True 면 좌우 반전 augmentation 적용
        hflip_prob    : 좌우 반전 확률 (YOLO 기본 fliplr=0.5 와 맞춤)
    """

    def __init__(self, image_paths: Sequence[Path], num_classes: int, image_size: int = 640,
                 train: bool = False, hflip_prob: float = 0.5,
                 class_names: Optional[List[str]] = None):
        self.image_paths = [Path(p) for p in image_paths]
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.hflip_prob = float(hflip_prob)
        self.class_names = list(class_names or [])

    def __len__(self) -> int:
        return len(self.image_paths)

    # -- 라벨 로딩 -------------------------------------------------------
    def load_annotation(self, index: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """원본 이미지 좌표계의 (boxes xyxy, labels 0-based, width, height)."""
        img_path = self.image_paths[index]
        width, height = ds.image_size(img_path)
        boxes, labels = ds.load_ground_truth(img_path, width, height, self.num_classes)
        b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        l = np.asarray(labels, dtype=np.int64).reshape(-1)
        keep = (l >= 0) & (l < self.num_classes)
        return b[keep], l[keep], width, height

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_path = self.image_paths[index]
        image = read_image(img_path)                    # BGR
        height, width = image.shape[:2]

        boxes, labels = ds.load_ground_truth(img_path, width, height, self.num_classes)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        keep = (labels >= 0) & (labels < self.num_classes)
        boxes, labels = boxes[keep], labels[keep]

        # letterbox (YOLO 계열과 동일한 기하 변환)
        lb, scale, pad = letterbox(image, self.image_size)
        if len(boxes):
            boxes = boxes * scale
            boxes[:, [0, 2]] += pad[0]
            boxes[:, [1, 3]] += pad[1]

        if self.train and self.hflip_prob > 0 and float(torch.rand(1)) < self.hflip_prob:
            lb = lb[:, ::-1].copy()
            if len(boxes):
                x1 = boxes[:, 0].copy()
                x2 = boxes[:, 2].copy()
                boxes[:, 0] = self.image_size - x2
                boxes[:, 2] = self.image_size - x1

        # 유효하지 않은 박스 제거 (letterbox 후 면적이 0이 되는 경우 방지)
        if len(boxes):
            valid = (boxes[:, 2] - boxes[:, 0] > 1e-3) & (boxes[:, 3] - boxes[:, 1] > 1e-3)
            boxes, labels = boxes[valid], labels[valid]

        rgb = lb[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)

        target: Dict[str, Any] = {
            "boxes": torch.from_numpy(boxes).float().reshape(-1, 4),
            # torchvision 규약: 0 = background 이므로 +1
            "labels": torch.from_numpy(labels + 1).long().reshape(-1),
            "image_id": torch.tensor([index]),
            "area": torch.from_numpy(
                ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).astype(np.float32)
            ).reshape(-1) if len(boxes) else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            # 예측을 원본 좌표계로 되돌릴 때 사용
            "orig_size": torch.tensor([height, width]),
            "scale": torch.tensor([scale], dtype=torch.float32),
            "pad": torch.tensor(list(pad), dtype=torch.float32),
            "path": str(img_path),
        }
        return tensor, target


def collate_fn(batch):
    """detection 모델은 이미지마다 객체 수가 다르므로 기본 collate 를 쓸 수 없다."""
    return tuple(zip(*batch))


def build_dataloader(image_paths: Sequence[Path], num_classes: int, batch_size: int,
                     image_size: int = 640, train: bool = False, workers: int = 0,
                     seed: int = 42, hflip_prob: float = 0.5,
                     class_names: Optional[List[str]] = None):
    """Windows 에서도 안전하게 동작하도록 기본 num_workers 는 0 이다."""
    from torch.utils.data import DataLoader

    dataset = YoloDetectionDataset(
        image_paths, num_classes=num_classes, image_size=image_size,
        train=train, hflip_prob=hflip_prob if train else 0.0, class_names=class_names,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=max(0, int(workers)),
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator if train else None,
        persistent_workers=bool(workers) and train,
    )
