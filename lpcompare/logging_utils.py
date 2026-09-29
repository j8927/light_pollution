"""모델별 로그 파일 + 콘솔 진행 상태 출력 유틸."""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_CONFIGURED: dict = {}


def get_logger(name: str, logs_dir: Path, filename: Optional[str] = None) -> logging.Logger:
    """`logs/<name>.log` 와 콘솔에 동시에 기록하는 로거를 돌려준다."""
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / (filename or f"{name}.log")

    logger = logging.getLogger(f"lpcompare.{name}")
    if _CONFIGURED.get(name):
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

    _CONFIGURED[name] = True
    logger.info("=" * 78)
    logger.info("로그 시작: %s (%s)", name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("로그 파일: %s", log_path)
    return logger


def banner(text: str, char: str = "=", width: int = 78) -> str:
    return f"\n{char * width}\n{text}\n{char * width}"


def log_banner(logger: logging.Logger, text: str, char: str = "=") -> None:
    for line in banner(text, char).splitlines():
        logger.info(line)


def log_kv(logger: logging.Logger, title: str, data: dict) -> None:
    """설정/환경 정보를 key : value 형태로 기록한다."""
    logger.info("-" * 78)
    logger.info(title)
    logger.info("-" * 78)
    width = max((len(str(k)) for k in data), default=0)
    for k, v in data.items():
        logger.info("  %-*s : %s", width, k, v)
    logger.info("-" * 78)
