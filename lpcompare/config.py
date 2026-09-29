"""config/experiment.yaml 로딩 및 경로 / 디바이스 해석."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# 프로젝트 루트 = lpcompare/config.py -> lpcompare -> 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "experiment.yaml"

MODEL_KEYS = ["yolov8", "yolo11", "rtdetr", "faster_rcnn"]
MODEL_DISPLAY = {
    "yolov8": "YOLOv8",
    "yolo11": "YOLO11",
    "rtdetr": "RT-DETR",
    "faster_rcnn": "Faster R-CNN",
}


def ensure_project_on_path() -> None:
    """`python training/train_yolov8.py` 처럼 직접 실행해도 lpcompare import 가 되도록 한다."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


@dataclass
class ExperimentConfig:
    raw: Dict[str, Any]
    path: Path

    # ---------------- 경로 ----------------
    @property
    def root(self) -> Path:
        return PROJECT_ROOT

    def abs_path(self, value) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    @property
    def dataset_root(self) -> Path:
        return self.abs_path(self.raw["dataset"]["root"])

    @property
    def data_yaml(self) -> Optional[Path]:
        v = self.raw["dataset"].get("data_yaml")
        return self.abs_path(v) if v else None

    @property
    def fallback_names(self) -> List[str]:
        return list(self.raw["dataset"].get("fallback_names") or [])

    @property
    def splits_dir(self) -> Path:
        return self.abs_path(self.raw["splits"]["dir"])

    @property
    def results_dir(self) -> Path:
        return self.abs_path(self.raw["output"]["results_dir"])

    @property
    def logs_dir(self) -> Path:
        return self.abs_path(self.raw["output"]["logs_dir"])

    @property
    def derived_dir(self) -> Path:
        return self.abs_path(self.raw["output"]["derived_dir"])

    @property
    def runs_dir(self) -> Path:
        return self.abs_path(self.raw["output"].get("runs_dir", "runs"))

    def run_dir(self, model_key: str) -> Path:
        return self.abs_path(self.model(model_key)["run_dir"])

    @property
    def compare_data_yaml(self) -> Path:
        """4개 모델이 공통으로 사용하는 파생 data.yaml (원본 data.yaml 은 건드리지 않는다)."""
        return self.derived_dir / "data_compare.yaml"

    # ---------------- 공통 실험 조건 ----------------
    @property
    def epochs(self) -> int:
        return int(self.raw["common"]["epochs"])

    @property
    def image_size(self) -> int:
        return int(self.raw["common"]["image_size"])

    @property
    def seed(self) -> int:
        return int(self.raw["common"]["seed"])

    @property
    def pretrained(self) -> bool:
        return bool(self.raw["common"]["pretrained"])

    @property
    def workers(self) -> int:
        return int(self.raw["common"].get("workers", 0))

    @property
    def deterministic(self) -> bool:
        return bool(self.raw["common"].get("deterministic", True))

    @property
    def require_cuda(self) -> bool:
        """true 면 CUDA 를 쓸 수 없을 때 CPU 로 대체하지 않고 학습을 중단한다."""
        return bool(self.raw["common"].get("require_cuda", False))

    # ---------------- 섹션 ----------------
    @property
    def splits(self) -> Dict[str, Any]:
        return self.raw["splits"]

    @property
    def evaluation(self) -> Dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def benchmark(self) -> Dict[str, Any]:
        return self.raw["benchmark"]

    @property
    def oom_retry(self) -> Dict[str, Any]:
        return self.raw.get("oom_retry", {"enabled": True, "min_batch": 1})

    def model(self, key: str) -> Dict[str, Any]:
        try:
            return self.raw["models"][key]
        except KeyError as exc:
            raise KeyError(f"config/experiment.yaml 의 models 에 '{key}' 항목이 없습니다.") from exc

    @property
    def enabled_models(self) -> List[str]:
        return [k for k in MODEL_KEYS if self.raw["models"].get(k, {}).get("enabled", False)]

    # ---------------- 디바이스 ----------------
    def resolve_device(self) -> Dict[str, Any]:
        """설정값과 실제 환경을 함께 보고 사용할 디바이스를 결정한다.

        반환: {"device": "cuda"|"cpu", "index": int|None, "torch": str,
               "ultralytics": str, "requested": str, "warnings": [str, ...],
               "blocked": bool, "block_reason": str}

        blocked 가 True 면 common.require_cuda 설정에 따라 학습을 중단해야 한다는 뜻이다
        (CPU 로 조용히 대체 실행되는 것을 막는다).
        """
        requested = str(self.raw["common"].get("device", "auto"))
        warnings: List[str] = []
        try:
            import torch
        except Exception as exc:  # torch 가 아예 없는 환경
            return {
                "device": "cpu", "index": None, "torch": "cpu", "ultralytics": "cpu",
                "requested": requested, "warnings": [f"torch import 실패: {exc}"],
                "blocked": self.require_cuda, "block_reason": f"torch import 실패: {exc}",
            }

        cuda_ok = torch.cuda.is_available()

        if requested == "cpu":
            if cuda_ok:
                warnings.append("CUDA 를 쓸 수 있지만 설정(common.device=cpu)에 따라 CPU 로 실행합니다.")
            return {"device": "cpu", "index": None, "torch": "cpu", "ultralytics": "cpu",
                    "requested": requested, "warnings": warnings,
                    "blocked": False, "block_reason": ""}

        if not cuda_ok:
            build = getattr(torch.version, "cuda", None)
            if not build:
                reason = (f"설치된 PyTorch 가 CPU 전용 빌드입니다 (torch {torch.__version__}, "
                          "torch.version.cuda = None). 이 빌드에는 CUDA 커널이 없어 "
                          "설정이나 코드 수정만으로는 GPU 를 사용할 수 없습니다.")
                warnings.append(
                    f"[경고] CUDA 사용 불가 — {reason} "
                    "해결 방법은 `python tools/check_environment.py` 를 실행해 확인하세요."
                )
            else:
                reason = (f"CUDA 빌드 PyTorch({torch.__version__}, CUDA {build})는 설치돼 있지만 "
                          "torch.cuda.is_available() 이 False 입니다. NVIDIA 드라이버 / GPU 상태를 확인하세요.")
                warnings.append(f"[경고] CUDA 사용 불가 — {reason}")
            if requested != "auto":
                warnings.append(f"설정에서 device={requested} 를 요청했지만 GPU 를 쓸 수 없습니다.")
            return {"device": "cpu", "index": None, "torch": "cpu", "ultralytics": "cpu",
                    "requested": requested, "warnings": warnings,
                    "blocked": self.require_cuda, "block_reason": reason}

        index = int(requested) if requested.isdigit() else 0
        return {
            "device": "cuda", "index": index,
            "torch": f"cuda:{index}", "ultralytics": str(index),
            "requested": requested, "warnings": warnings,
            "blocked": False, "block_reason": "",
        }


def load_config(path=None) -> ExperimentConfig:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"실험 설정 파일을 찾을 수 없습니다: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig(raw=raw, path=cfg_path)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """random / numpy / torch 시드를 한 번에 고정한다."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
