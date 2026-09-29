"""모델 규모 정보(파라미터 수 / 파일 크기 / GFLOPs)와 GPU 메모리 측정 유틸.

측정이 불가능한 항목은 값을 만들어내지 않고 None 을 돌려주며, 이유를 문자열로 함께 반환한다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def file_size_mb(path) -> Optional[float]:
    p = Path(path) if path else None
    if p and p.is_file():
        return round(p.stat().st_size / (1024 * 1024), 3)
    return None


def count_parameters(module) -> Optional[int]:
    try:
        return int(sum(p.numel() for p in module.parameters()))
    except Exception:
        return None


def compute_gflops(module, image_size: int = 640, device: str = "cpu") -> Tuple[Optional[float], str]:
    """thop 으로 GFLOPs 를 계산한다. 실패하면 (None, 이유)."""
    try:
        import torch
    except Exception as exc:
        return None, f"torch 없음: {exc}"
    try:
        try:
            from thop import profile  # ultralytics-thop 가 제공
        except Exception:
            from ultralytics.utils.torch_utils import get_flops  # 폴백 (Ultralytics 모델 전용)

            flops = get_flops(module, image_size)
            return (float(flops), "ultralytics.get_flops") if flops else (None, "ultralytics.get_flops 가 0 을 반환")

        module = module.eval()
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        macs, _params = profile(module, inputs=(dummy,), verbose=False)
        return round(float(macs) * 2 / 1e9, 3), "thop.profile (MACs x 2)"
    except Exception as exc:
        return None, f"계산 실패: {type(exc).__name__}: {exc}"


class GpuMemoryTracker:
    """with 블록 동안의 GPU 최대 할당량(MB)을 기록한다. CPU 환경에서는 None."""

    def __init__(self, device: str = "cpu"):
        self.device = str(device)
        self.enabled = self.device.startswith("cuda")
        self.peak_mb: Optional[float] = None
        self.reason: str = ""

    def __enter__(self) -> "GpuMemoryTracker":
        if not self.enabled:
            self.reason = "CUDA 를 사용하지 않아 GPU 메모리를 측정할 수 없습니다."
            return self
        import torch

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc) -> None:
        if not self.enabled:
            return
        import torch

        torch.cuda.synchronize()
        self.peak_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)


def gpu_name_and_vram() -> Dict[str, Any]:
    out: Dict[str, Any] = {"gpu_name": None, "vram_gb": None}
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            out["gpu_name"] = props.name
            out["vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
    except Exception:
        pass
    return out
