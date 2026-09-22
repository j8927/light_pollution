"""Korean signboard OCR and store-name candidate extraction."""

import difflib
import os
import re
import threading

import cv2
import numpy as np


_OCR_ENGINE = None
_OCR_ERROR = None
_OCR_LOCK = threading.Lock()

# EasyOCR confidence is frequently low for glowing or stylised Korean lettering.
# Keep plausible low-confidence lines for candidate ranking, then use geometry and
# repeated readings to reject noise.
_MIN_SCORE = 0.15
_DISPLAY_MIN_SCORE = 0.25
_GENERIC_WORDS = {
    "간판", "상가", "매장", "영업", "문의", "예약", "주차", "입구", "출구",
    "전화", "배달", "포장", "전문", "전문점", "본점", "지점", "오픈", "open",
    "tel", "메뉴", "할인", "행사",
}
_GENERIC_MODIFIERS = (
    "전문점", "전문", "본점", "지점", "영업", "문의", "예약", "주차", "입구",
    "출구", "전화", "배달", "포장", "오픈", "open", "tel", "메뉴", "할인",
)
_DESCRIPTOR_SUFFIXES = (
    "전문점", "전문", "판매점", "대리점", "직영점", "할인점", "영업점",
)


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


def crop_signboard_region(rgb_image, box, padding_x=0.08, padding_y=0.25):
    """Crop a detection with room for brand text placed around its border."""
    if not isinstance(rgb_image, np.ndarray) or rgb_image.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)

    image_height, image_width = rgb_image.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in box)
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    pad_x = max(4, int(round(box_width * padding_x)))
    pad_y = max(6, int(round(box_height * padding_y)))
    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(image_width, x2 + pad_x)
    bottom = min(image_height, y2 + pad_y)
    return rgb_image[top:bottom, left:right]


def _resize_crop(rgb_crop):
    """Upscale small text while keeping CPU/memory usage bounded."""
    height, width = rgb_crop.shape[:2]
    longest_side = max(width, height)
    scale = max(1.0, min(3.5, 1200.0 / max(longest_side, 1)))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized_rgb = cv2.resize(
        rgb_crop, None, fx=scale, fy=scale, interpolation=interpolation
    )
    return cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)


def _prepare_variants(rgb_crop):
    """Build complementary inputs for ordinary and overexposed night signs."""
    color = _resize_crop(rgb_crop)
    lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_lightness = clahe.apply(lightness)
    enhanced = cv2.cvtColor(
        cv2.merge((enhanced_lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR
    )

    # Bright, blooming strokes often become more legible as dark glyphs on a
    # clean background. This variant is only used when the primary pass is weak.
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, inverted_binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return enhanced, inverted_binary


def _prepare_crop(rgb_crop):
    """Backward-compatible primary preprocessing helper."""
    return _prepare_variants(rgb_crop)[0]


def _box_metrics(box):
    try:
        points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        width = max(1.0, float(max_x - min_x))
        height = max(1.0, float(max_y - min_y))
        return {
            "x": float(min_x),
            "y": float(min_y),
            "width": width,
            "height": height,
            "area": width * height,
        }
    except (TypeError, ValueError):
        return {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0, "area": 1.0}


def _clean_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip(" |·•_-.,'\"")


def _normalize_text(value):
    return re.sub(r"[^가-힣A-Za-z0-9]", "", str(value or "")).lower()


def _parse_easyocr_results(results, variant=0):
    lines = []
    for order, result in enumerate(results or []):
        try:
            box, raw_text, score = result
            score = float(score)
        except (TypeError, ValueError):
            continue
        text = _clean_text(raw_text)
        letters = re.findall(r"[가-힣A-Za-z]", text)
        if not text or len(letters) < 2 or score < _MIN_SCORE:
            continue
        metrics = _box_metrics(box)
        lines.append({
            "text": text,
            "confidence": score,
            "order": order,
            "variant": variant,
            "seen": 1,
            **metrics,
        })
    return lines


def _text_similarity(left, right):
    left_normalized = _normalize_text(left)
    right_normalized = _normalize_text(right)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0
    return difflib.SequenceMatcher(None, left_normalized, right_normalized).ratio()


def _merge_variant_lines(lines):
    """Deduplicate near-identical readings and keep the most reliable spelling."""
    merged = []
    for line in sorted(lines, key=lambda item: item["confidence"], reverse=True):
        match = next(
            (
                existing
                for existing in merged
                if _text_similarity(line["text"], existing["text"]) >= 0.82
            ),
            None,
        )
        if match is None:
            merged.append(dict(line))
            continue
        match["seen"] += 1
        if line["confidence"] > match["confidence"]:
            seen = match["seen"]
            match.update(line)
            match["seen"] = seen
    return sorted(merged, key=lambda item: (item["y"], item["x"]))


def _candidate_score(line, max_height, max_area):
    text = line["text"]
    compact = re.sub(r"\s", "", text)
    normalized = _normalize_text(text)
    letters = re.findall(r"[가-힣A-Za-z]", text)
    if not normalized or len(letters) < 2:
        return None
    if re.search(r"(?:\d[ -]?){7,}", compact):
        return None
    if re.fullmatch(r"(?:https?://)?(?:www\.)?[^ ]+\.(?:com|net|kr)", compact, re.I):
        return None

    generic_only = normalized in _GENERIC_WORDS
    modifier_count = sum(word in normalized for word in _GENERIC_MODIFIERS)
    descriptor_suffix = any(normalized.endswith(word) for word in _DESCRIPTOR_SUFFIXES)
    sentence_like = len(compact) > 18 or len(re.findall(r"[+|,:]", text)) > 1
    height_score = min(1.0, line.get("height", 1.0) / max(max_height, 1.0))
    area_score = min(1.0, line.get("area", 1.0) / max(max_area, 1.0))
    length_score = min(len(letters), 10) / 10.0
    repeat_bonus = min(2, max(0, line.get("seen", 1) - 1)) * 0.16

    score = (
        float(line["confidence"]) * 1.25
        + height_score * 0.85
        + area_score * 0.25
        + length_score * 0.45
        + (0.12 if re.search(r"[가-힣]", text) else 0.0)
        + repeat_bonus
    )
    score -= 0.9 if generic_only else 0.0
    score -= modifier_count * 0.18
    score -= 1.25 if descriptor_suffix else 0.0
    score -= 0.7 if sentence_like else 0.0
    score -= max(0, len(compact) - 24) * 0.04
    return score


def extract_store_name(lines):
    """Select the most store-name-like OCR line using text and box prominence."""
    normalized_lines = []
    for order, line in enumerate(lines or []):
        if isinstance(line, dict):
            item = dict(line)
            item["text"] = _clean_text(item.get("text"))
            item["confidence"] = float(item.get("confidence", 0.0))
            item.setdefault("height", 1.0)
            item.setdefault("area", item.get("width", 1.0) * item["height"])
            item.setdefault("seen", 1)
        else:
            try:
                raw_text, confidence = line
            except (TypeError, ValueError):
                continue
            item = {
                "text": _clean_text(raw_text),
                "confidence": float(confidence),
                "height": 1.0,
                "area": 1.0,
                "seen": 1,
            }
        item["order"] = order
        normalized_lines.append(item)

    if not normalized_lines:
        return None
    max_height = max(item.get("height", 1.0) for item in normalized_lines)
    max_area = max(item.get("area", 1.0) for item in normalized_lines)
    candidates = []
    for item in normalized_lines:
        score = _candidate_score(item, max_height, max_area)
        if score is not None:
            candidates.append((score, -item["order"], item["text"]))
    return max(candidates, default=(0, 0, None))[2]


def _read_variant(engine, image, variant):
    height, width = image.shape[:2]
    rotation_info = [90, 270] if height > width * 1.2 else None
    results = engine.readtext(
        image,
        detail=1,
        paragraph=False,
        decoder="beamsearch",
        rotation_info=rotation_info,
        text_threshold=0.38 if variant == 0 else 0.30,
        low_text=0.20 if variant == 0 else 0.15,
        link_threshold=0.25,
        contrast_ths=0.04,
        adjust_contrast=0.7,
        canvas_size=1920,
        mag_ratio=1.2,
    )
    return _parse_easyocr_results(results, variant=variant)


def _primary_is_weak(lines):
    if not lines:
        return True
    best_name = extract_store_name(lines)
    best_confidence = max(
        (line["confidence"] for line in lines if line["text"] == best_name),
        default=0.0,
    )
    return not best_name or best_confidence < 0.55


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

    primary, glare_reduced = _prepare_variants(rgb_crop)
    try:
        with _OCR_LOCK:
            lines = _read_variant(engine, primary, variant=0)
            if _primary_is_weak(lines):
                lines.extend(_read_variant(engine, glare_reduced, variant=1))
        lines = _merge_variant_lines(lines)
    except Exception as exc:
        empty["ocrStatus"] = f"OCR inference failed: {exc}"
        return empty
    if not lines:
        empty["ocrStatus"] = "no text"
        return empty

    display_lines = [
        line for line in lines
        if line["confidence"] >= _DISPLAY_MIN_SCORE or line.get("seen", 1) > 1
    ]
    if not display_lines:
        display_lines = lines[:3]
    return {
        "storeName": extract_store_name(lines),
        "ocrText": " ".join(line["text"] for line in display_lines),
        "ocrConfidence": round(
            sum(line["confidence"] for line in display_lines) / len(display_lines), 3
        ),
        "ocrStatus": "success",
    }
