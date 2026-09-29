"""시각화 유틸 — bbox 그리기, 모델 비교 패널, 그래프 공통 설정.

클래스 이름이 한글(간판/조명/가로등)이므로 OpenCV 대신 Pillow 로 텍스트를 그리고,
matplotlib 에도 Windows 기본 한글 폰트(맑은 고딕)를 설정한다.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# 클래스별 고정 색상 (BGR 이 아니라 RGB)
PALETTE = [
    (255, 96, 86), (86, 180, 255), (120, 220, 130), (255, 196, 64),
    (198, 120, 255), (64, 220, 220), (255, 140, 200), (170, 170, 170),
]

_KOREAN_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\malgunsl.ttf",
    r"C:\Windows\Fonts\NanumGothic.ttf",
    r"C:\Windows\Fonts\gulim.ttc",
]


def find_korean_font() -> Optional[str]:
    for p in _KOREAN_FONT_CANDIDATES:
        if Path(p).is_file():
            return p
    return None


def setup_matplotlib() -> str:
    """matplotlib 한글 폰트를 설정하고 사용된 폰트 이름을 돌려준다."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = find_korean_font()
    used = "default"
    if font_path:
        try:
            font_manager.fontManager.addfont(font_path)
            name = font_manager.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.family"] = name
            used = name
        except Exception:
            pass
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.bbox"] = "tight"
    return used


def _pil_font(size: int):
    from PIL import ImageFont

    path = find_korean_font()
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_detections(image_bgr: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
                    labels: np.ndarray, class_names: Sequence[str],
                    title: Optional[str] = None, show_conf: bool = True) -> np.ndarray:
    """BGR 이미지에 bbox + 클래스명 + confidence 를 그려서 BGR 로 돌려준다."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(image_bgr[:, :, ::-1].copy())
    draw = ImageDraw.Draw(img)
    h, w = image_bgr.shape[:2]
    line_w = max(2, int(round(min(h, w) / 400)))
    font = _pil_font(max(14, int(round(min(h, w) / 45))))

    for box, score, cls in zip(np.asarray(boxes).reshape(-1, 4),
                               np.asarray(scores).reshape(-1),
                               np.asarray(labels).reshape(-1)):
        color = PALETTE[int(cls) % len(PALETTE)]
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
        name = class_names[int(cls)] if 0 <= int(cls) < len(class_names) else f"class{int(cls)}"
        text = f"{name} {score:.2f}" if show_conf else name
        try:
            tb = draw.textbbox((0, 0), text, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = len(text) * 8, 14
        ty = max(0.0, y1 - th - 4)
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
        draw.text((x1 + 3, ty + 2), text, fill=(0, 0, 0), font=font)

    if title:
        tfont = _pil_font(max(18, int(round(min(h, w) / 30))))
        try:
            tb = draw.textbbox((0, 0), title, font=tfont)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = len(title) * 10, 18
        draw.rectangle([0, 0, tw + 12, th + 10], fill=(0, 0, 0))
        draw.text((6, 4), title, fill=(255, 255, 255), font=tfont)

    return np.asarray(img)[:, :, ::-1].copy()


def make_grid(panels: List[Tuple[str, np.ndarray]], cols: int = 2,
              cell_width: int = 640, pad: int = 8,
              bg: Tuple[int, int, int] = (30, 30, 30)) -> np.ndarray:
    """(제목, BGR 이미지) 목록을 격자로 합친다. 각 패널 상단에 제목 띠를 그린다."""
    import cv2

    if not panels:
        raise ValueError("패널이 비어 있습니다.")

    header = 34
    cells = []
    for title, img in panels:
        h, w = img.shape[:2]
        scale = cell_width / max(1, w)
        resized = cv2.resize(img, (cell_width, max(1, int(round(h * scale)))))
        cells.append((title, resized))

    cell_h = max(c.shape[0] for _, c in cells) + header
    rows = (len(cells) + cols - 1) // cols
    canvas = np.full((rows * cell_h + pad * (rows + 1),
                      cols * cell_width + pad * (cols + 1), 3), bg, dtype=np.uint8)

    from PIL import Image, ImageDraw

    for i, (title, img) in enumerate(cells):
        r, c = divmod(i, cols)
        y0 = pad + r * (cell_h + pad)
        x0 = pad + c * (cell_width + pad)
        pil = Image.fromarray(canvas[:, :, ::-1].copy())
        d = ImageDraw.Draw(pil)
        font = _pil_font(22)
        d.rectangle([x0, y0, x0 + cell_width, y0 + header], fill=(45, 45, 45))
        d.text((x0 + 8, y0 + 5), title, fill=(255, 255, 255), font=font)
        canvas = np.asarray(pil)[:, :, ::-1].copy()
        canvas[y0 + header:y0 + header + img.shape[0], x0:x0 + cell_width] = img
    return canvas


def save_image(path: Path, image_bgr: np.ndarray) -> Path:
    """한글 경로에서도 안전하게 저장한다."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".jpg"
    ok, buf = cv2.imencode(ext, image_bgr)
    if not ok:
        raise RuntimeError(f"이미지 인코딩 실패: {path}")
    buf.tofile(str(path))
    return path


def bar_chart(path: Path, labels: Sequence[str], values: Sequence[Optional[float]],
              title: str, ylabel: str, note: str = "", value_fmt: str = "{:.4g}") -> Optional[Path]:
    """모델 비교용 막대그래프. 값이 하나도 없으면 생성하지 않고 None 을 돌려준다."""
    import matplotlib.pyplot as plt

    pairs = [(l, v) for l, v in zip(labels, values)
             if v is not None and isinstance(v, (int, float)) and np.isfinite(v)]
    if not pairs:
        return None

    names = [p[0] for p in pairs]
    vals = [float(p[1]) for p in pairs]
    colors = [tuple(c / 255 for c in PALETTE[i % len(PALETTE)]) for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(names, vals, color=colors, edgecolor="#333333", linewidth=0.8)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("모델", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    top = max(vals) if max(vals) > 0 else 1.0
    ax.set_ylim(0, top * 1.18)
    for b, v in zip(bars, vals):
        ax.annotate(value_fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=10)
    if note:
        fig.text(0.01, -0.02, note, fontsize=8, color="#555555")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_confusion_matrix(path: Path, matrix: np.ndarray, class_names: Sequence[str],
                          title: str, normalize: bool = False) -> Path:
    import matplotlib.pyplot as plt

    labels = list(class_names) + ["background"]
    m = matrix.astype(np.float64)
    if normalize:
        col = m.sum(axis=0, keepdims=True)
        m = np.divide(m, np.maximum(col, 1e-9))

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("정답 (Ground Truth)")
    ax.set_ylabel("예측 (Prediction)")
    ax.set_title(title, pad=12)
    thresh = m.max() / 2 if m.max() > 0 else 0.5
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            txt = f"{m[i, j]:.2f}" if normalize else f"{int(matrix[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if m[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
