"""Korean signboard OCR and store-name candidate extraction."""

import os
import re
import threading

import cv2
import numpy as np


_OCR_ENGINE = None
_OCR_ERROR = None
_OCR_LOCK = threading.Lock()
_MIN_SCORE = 0.35
_GENERIC_WORDS = {
    "간판", "상가", "매장", "영업", "문의", "예약", "주차", "입구", "출구",
    "전화", "배달", "포장", "전문", "본점", "지점", "오픈", "open", "tel",
}
_GENERIC_MODIFIERS = ("전문", "본점", "지점", "영업", "문의", "예약", "주차", "입구", "출구")


def _enabled():
    return os.getenv("SIGNBOARD_OCR_ENABLED", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }


def _get_engine():
    """Create one shared EasyOCR instance."""
    global _OCR_ENGINE, _OCR_ERROR
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE
    if _OCR_ERROR is not None or not _enabled():
        return None

    with _OCR_LOCK:
        if _OCR_ENGINE is not None:
            return _OCR_ENGINE
        try:
            import easyocr
            model_dir = os.path.abspath(
                os.getenv("EASYOCR_MODEL_DIR", os.path.join("models", "easyocr"))
            )
            os.makedirs(model_dir, exist_ok=True)
            _OCR_ENGINE = easyocr.Reader(
                ["ko", "en"],
                gpu=False,
                verbose=False,
                model_storage_directory=model_dir,
                user_network_directory=model_dir,
            )
        except Exception as exc:
            _OCR_ERROR = f"OCR initialization failed: {exc}"
            return None
    return _OCR_ENGINE


def _prepare_crop(rgb_crop):
    """Enlarge a dark sign and improve local contrast without discarding color."""
    height, width = rgb_crop.shape[:2]
    scale = max(1.0, min(3.0, 960.0 / max(width, height)))
    resized = cv2.resize(
        rgb_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )
    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(
        cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR
    )


def _parse_easyocr_results(results):
    lines = []
    for result in results or []:
        try:
            _, text, score = result
            score = float(score)
        except (TypeError, ValueError):
            continue
        if str(text).strip() and score >= _MIN_SCORE:
            lines.append((str(text).strip(), score))
    return lines


def extract_store_name(lines):
    """Select the most store-name-like OCR line using transparent heuristics."""
    candidates = []
    for order, (raw_text, confidence) in enumerate(lines):
        text = re.sub(r"\s+", " ", raw_text).strip(" |·•_-.,")
        compact = re.sub(r"\s", "", text)
        if not compact or re.search(r"(?:\d[ -]?){7,}", compact):
            continue
        if re.fullmatch(r"(?:https?://)?(?:www\.)?[^ ]+\.(?:com|net|kr)", compact, re.I):
            continue
        letters = re.findall(r"[가-힣A-Za-z]", text)
        if len(letters) < 2:
            continue
        normalized = re.sub(r"[^가-힣A-Za-z]", "", text).lower()
        generic_only = normalized in _GENERIC_WORDS
        generic_modifier_count = sum(word in normalized for word in _GENERIC_MODIFIERS)
        length_score = min(len(letters), 12) / 12
        korean_bonus = 0.18 if re.search(r"[가-힣]", text) else 0.0
        generic_penalty = (0.65 if generic_only else 0.0) + generic_modifier_count * 0.3
        long_penalty = max(0, len(compact) - 24) * 0.025
        score = float(confidence) + length_score + korean_bonus - generic_penalty - long_penalty
        candidates.append((score, -order, text))
    return max(candidates, default=(0, 0, None))[2]


def recognize_signboard(rgb_crop):
    """Return OCR text, confidence and a best-effort store-name candidate."""
    empty = {
        "storeName": None,
        "ocrText": "",
        "ocrConfidence": 0.0,
        "ocrStatus": "disabled" if not _enabled() else "unavailable",
    }
    if not isinstance(rgb_crop, np.ndarray) or rgb_crop.size == 0:
        empty["ocrStatus"] = "empty crop"
        return empty
    engine = _get_engine()
    if engine is None:
        empty["ocrStatus"] = _OCR_ERROR or empty["ocrStatus"]
        return empty

    image = _prepare_crop(rgb_crop)
    try:
        with _OCR_LOCK:
            results = engine.readtext(
                image,
                detail=1,
                paragraph=False,
                decoder="beamsearch",
                rotation_info=[90, 180, 270],
                text_threshold=0.45,
                low_text=0.25,
                link_threshold=0.3,
                contrast_ths=0.05,
                adjust_contrast=0.7,
            )
            lines = _parse_easyocr_results(results)
    except Exception as exc:
        empty["ocrStatus"] = f"OCR inference failed: {exc}"
        return empty
    if not lines:
        empty["ocrStatus"] = "no text"
        return empty
    return {
        "storeName": extract_store_name(lines),
        "ocrText": " ".join(text for text, _ in lines),
        "ocrConfidence": round(sum(score for _, score in lines) / len(lines), 3),
        "ocrStatus": "success",
    }
