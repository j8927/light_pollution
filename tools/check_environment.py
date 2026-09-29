"""실행 전 환경 점검 스크립트.

PowerShell 사용 예:
    python tools/check_environment.py
    python tools/check_environment.py --json

Python / PyTorch / torchvision / ultralytics 버전과 CUDA·GPU 상태를 출력한다.
NVIDIA GPU 가 있는데 PyTorch 가 CPU 전용 빌드면 경고와 함께 필요한 설치 명령을 안내한다.
이 스크립트는 어떤 패키지도 설치하거나 제거하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lpcompare.config import load_config  # noqa: E402
from lpcompare.logging_utils import get_logger, log_banner, log_kv  # noqa: E402

OPTIONAL_PACKAGES = ["torchmetrics", "pandas", "matplotlib", "numpy", "cv2", "PIL", "yaml", "tqdm", "thop"]


def _version(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "설치됨(버전 정보 없음)")
    except Exception:
        return "미설치"


def nvidia_smi() -> Dict[str, Any]:
    """torch 가 CPU 빌드여도 실제 GPU 존재 여부를 확인하기 위해 nvidia-smi 를 호출한다."""
    out: Dict[str, Any] = {"available": False, "gpus": [], "driver": None, "error": None}
    exe = shutil.which("nvidia-smi")
    if not exe:
        out["error"] = "nvidia-smi 를 찾을 수 없습니다 (NVIDIA 드라이버 미설치이거나 PATH 에 없음)"
        return out
    try:
        res = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        if res.returncode != 0:
            out["error"] = (res.stderr or res.stdout).strip()
            return out
        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                out["gpus"].append({"name": parts[0], "memory": parts[1]})
                out["driver"] = parts[2]
        out["available"] = bool(out["gpus"])
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def collect() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "torch_version": "미설치",
        "torchvision_version": _version("torchvision"),
        "ultralytics_version": _version("ultralytics"),
        "cuda_available": False,
        "torch_cuda_version": None,
        "cudnn_version": None,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "current_device": "cpu",
        "device_count": 0,
        "warnings": [],
        "hints": [],
        "optional_packages": {},
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        info["cuda_available"] = bool(torch.cuda.is_available())
        try:
            info["cudnn_version"] = torch.backends.cudnn.version()
        except Exception:
            info["cudnn_version"] = None
        if info["cuda_available"]:
            info["device_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
            info["current_device"] = f"cuda:{torch.cuda.current_device()}"
    except Exception as exc:
        info["warnings"].append(f"PyTorch import 실패: {type(exc).__name__}: {exc}")

    smi = nvidia_smi()
    info["nvidia_smi"] = smi

    for pkg in OPTIONAL_PACKAGES:
        info["optional_packages"][pkg] = _version(pkg)

    # ---- 경고 / 안내 ----
    if not info["cuda_available"]:
        if smi.get("available"):
            names = ", ".join(g["name"] for g in smi["gpus"])
            info["warnings"].append(
                f"NVIDIA GPU({names})가 감지되었지만 PyTorch 에서 CUDA 를 사용할 수 없습니다."
            )
            if not info["torch_cuda_version"]:
                info["warnings"].append(
                    f"설치된 PyTorch 가 CPU 전용 빌드입니다 (torch {info['torch_version']}). "
                    "이 상태에서는 학습과 추론이 모두 CPU 로 실행되어 매우 느립니다."
                )
                tv = info["torchvision_version"]
                info["hints"].append(
                    "CUDA 빌드 PyTorch 설치가 필요합니다. 현재 가상환경을 활성화한 뒤 "
                    "GPU/드라이버에 맞는 CUDA 빌드를 직접 설치하세요 (기존 환경이 바뀌므로 반드시 확인 후 실행)."
                )
                base_t = info["torch_version"].split("+")[0]
                base_tv = str(tv).split("+")[0]
                info["hints"].append(
                    "먼저 해당 CUDA 채널에 현재 버전이 있는지 확인하세요 (설치 없이 조회만 함):\n"
                    "         pip index versions torch --index-url https://download.pytorch.org/whl/cu130\n"
                    "       버전이 바뀌면 다른 패키지와 충돌할 수 있으므로, 가능하면 지금과 같은 버전"
                    f"(torch {base_t}, torchvision {base_tv}) 의 CUDA 빌드를 고르세요."
                )
                info["hints"].append(
                    "[주의] 설치할 때 반드시 '+cuXXX' 로컬 버전까지 붙이세요. "
                    f"`torch=={base_t}` 처럼 쓰면 pip 이 이미 설치된 {info['torch_version']} 를 "
                    "조건 충족으로 보고 아무것도 설치하지 않습니다 (PEP 440 은 로컬 버전 라벨을 무시함).\n"
                    f"         예: pip install --index-url https://download.pytorch.org/whl/cu130 "
                    f"torch=={base_t}+cu130 torchvision=={base_tv}+cu130\n"
                    "       설치 후 이 스크립트를 다시 실행해 CUDA Available 이 True 인지 확인하세요."
                )
                info["hints"].append(
                    "CUDA 채널은 GPU 세대와 드라이버에 맞춰 고릅니다 (cu126 / cu128 / cu129 / cu130 등). "
                    "위 nvidia-smi 의 CUDA 버전 이하 채널이어야 하며, 최신 GPU 는 낮은 채널을 지원하지 않을 수 있습니다. "
                    "채널별 지원 목록: https://pytorch.org/get-started/locally/"
                )
            else:
                info["hints"].append("NVIDIA 드라이버 버전과 PyTorch CUDA 버전의 호환 여부를 확인하세요.")
        else:
            info["warnings"].append(
                "CUDA 를 사용할 수 없고 NVIDIA GPU 도 감지되지 않았습니다. CPU 로만 실행됩니다."
            )
    if info["optional_packages"].get("torchmetrics") == "미설치":
        info["hints"].append(
            "torchmetrics 는 선택 사항입니다. 설치하면 평가 시 mAP 교차 검증 값이 함께 기록됩니다: "
            "pip install torchmetrics"
        )
    return info


def print_report(info: Dict[str, Any], logs_dir: Path) -> None:
    logger = get_logger("environment", logs_dir)
    log_banner(logger, "환경 검사 (Environment Check)")
    log_kv(logger, "기본 환경", {
        "Python Version": info["python_version"],
        "Python Executable": info["python_executable"],
        "Platform": info["platform"],
        "PyTorch Version": info["torch_version"],
        "Torchvision Version": info["torchvision_version"],
        "Ultralytics Version": info["ultralytics_version"],
    })
    log_kv(logger, "GPU / CUDA", {
        "CUDA Available": info["cuda_available"],
        "PyTorch CUDA Version": info["torch_cuda_version"] or "N/A (CPU 전용 빌드)",
        "cuDNN Version": info["cudnn_version"] or "N/A",
        "GPU Name": info["gpu_name"] or "N/A",
        "GPU VRAM": f"{info['gpu_vram_gb']} GB" if info["gpu_vram_gb"] else "N/A",
        "Device Count": info["device_count"],
        "Current Device": info["current_device"],
        "nvidia-smi GPU": ", ".join(f"{g['name']} ({g['memory']})" for g in info["nvidia_smi"]["gpus"]) or "감지 안 됨",
        "nvidia-smi Driver": info["nvidia_smi"].get("driver") or "N/A",
    })
    log_kv(logger, "선택 패키지", info["optional_packages"])

    if info["warnings"]:
        logger.info("")
        for w in info["warnings"]:
            logger.warning("[경고] %s", w)
    if info["hints"]:
        logger.info("")
        for h in info["hints"]:
            logger.info("[안내] %s", h)
    if info["cuda_available"]:
        logger.info("")
        logger.info("[OK] CUDA 사용 가능 — 학습/평가가 GPU 에서 실행됩니다.")
    logger.info("")


def main() -> int:
    parser = argparse.ArgumentParser(description="빛공해 모델 비교 실험 환경 검사")
    parser.add_argument("--config", default=None, help="실험 설정 파일 경로 (기본: config/experiment.yaml)")
    parser.add_argument("--json", action="store_true", help="JSON 으로만 출력")
    args = parser.parse_args()

    info = collect()
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0

    try:
        cfg = load_config(args.config)
        logs_dir = cfg.logs_dir
    except Exception:
        logs_dir = Path(__file__).resolve().parents[1] / "logs"
    print_report(info, logs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
