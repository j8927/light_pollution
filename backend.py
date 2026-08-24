from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import base64
import hashlib
import io
import json
import math
import re
import time
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageOps
from signboard_ocr import recognize_signboard
import requests as req_lib  # Nominatim 역지오코딩용

app = Flask(__name__)
CORS(app)

BUSAN_COMMERCIAL_ENDPOINT = (
    'https://apis.data.go.kr/6260000/BusanCommercialHistoryService/'
    'getCommercialHistoryList'
)
BUSAN_COMMERCIAL_MAX_PAGES = 0  # 0 means scan every page from totalCount.
BUSAN_COMMERCIAL_NUM_OF_ROWS = 1000
BUSAN_COMMERCIAL_CACHE_PATH = os.path.join('data', 'busan_commercial_cache.json')
BUSAN_COMMERCIAL_CACHE = None


def load_local_env(path='.env'):
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding='utf-8') as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except OSError:
        return


load_local_env()

MODEL = None
MODEL_STATUS = "모델 로드 중..."

try:
    from ultralytics import YOLO
    custom_path = 'models/light_pollution_best.pt'
    if os.path.exists(custom_path):
        MODEL = YOLO(custom_path)
        MODEL_STATUS = f"커스텀 모델 로드 완료({custom_path})"
    else:
        MODEL = YOLO('yolov8n.pt')
        MODEL_STATUS = "YOLOv8 기본 모델 로드 완료"
except Exception as e:
    MODEL = None
    MODEL_STATUS = f"모델 로드 실패: {e}"

# ---- YOLO 탐지 클래스 매핑 ----
COCO_TO_KR = {
    'traffic light': '가로등',
    'street sign': '간판',
    'stop sign': '간판',
    'light_signboard': '간판',
    'lighting': '조명',
    'streetlight': '가로등',
    'parking meter': '조명',
    'bench': '구조물',
    'person': '사람',
    'car': '차량'
}

DIRECT_LIGHT_CLASSES = {'간판', '가로등', '조명', '구조물', '사람', '차량'}

# 촬영 조건별 보정 프로필
CAMERA_PROFILES = {
    'smartphone': {
        'label': '스마트폰 기본',
        'iso_ref': 100.0,
        'exposure_ref_ms': 12.0,
        'brightness_bias': 1.00,
        'glare_bias': 1.02,
        'angle_power': 0.82,
    },
    'iphone': {
        'label': '아이폰 계열',
        'iso_ref': 80.0,
        'exposure_ref_ms': 10.0,
        'brightness_bias': 0.98,
        'glare_bias': 1.00,
        'angle_power': 0.80,
    },
    'galaxy': {
        'label': '갤럭시 계열',
        'iso_ref': 90.0,
        'exposure_ref_ms': 11.0,
        'brightness_bias': 1.00,
        'glare_bias': 1.03,
        'angle_power': 0.83,
    },
    'dslr': {
        'label': '디지털카메라/DSLR',
        'iso_ref': 200.0,
        'exposure_ref_ms': 20.0,
        'brightness_bias': 0.95,
        'glare_bias': 0.97,
        'angle_power': 0.76,
    },
    'action': {
        'label': '액션캠/드론',
        'iso_ref': 160.0,
        'exposure_ref_ms': 16.0,
        'brightness_bias': 1.06,
        'glare_bias': 1.08,
        'angle_power': 0.90,
    },
    'cctv': {
        'label': 'CCTV/고정형',
        'iso_ref': 140.0,
        'exposure_ref_ms': 18.0,
        'brightness_bias': 1.08,
        'glare_bias': 1.10,
        'angle_power': 0.88,
    },
    'default': {
        'label': '기본값',
        'iso_ref': 100.0,
        'exposure_ref_ms': 12.0,
        'brightness_bias': 1.00,
        'glare_bias': 1.00,
        'angle_power': 0.82,
    },
}

# YOLO 탐지 카테고리 → 조명 법규 유형 매핑
LIGHT_TYPE_MAP = {
    '가로등': '공간조명',   # 조도(lux) 최대값 기준
    '간판':  '광고물',      # 휘도(cd/m²) 최대값 기준
    '조명':  '장식조명',    # 휘도(cd/m²) 평균/최대값 기준
}

# 빛 공해 4대 분류
POLLUTION_TYPES = {
    '침입광': '원치 않는 공간(창가/주거 방향)으로 유입되는 조명',
    '눈부심': '고휘도 광원이 시야 불편을 유발하는 상태',
    '산란광': '하늘 방향으로 퍼지는 확산광/배경 밝아짐',
    '군집된빛': '간판·장식 조명이 과도하게 밀집된 상태',
}

# 대 분류 튜닝 파라미터, 필요 시 조정
POLLUTION_THRESHOLDS = {
    'edge_margin': 0.14,
    'intrusion_avg': 52.0,
    'glare_p95': 230.0,
    'glare_gamma': 2.45,
    'glare_ratio_upper': 14.0,
    'scatter_area': 0.22,
    'scatter_ratio': 4.2,
    'scatter_sat_max': 0.40,
    'cluster_sat': 0.30,
    'cluster_ratio': 2.0,
    'cluster_count_bonus': 2,
    # 탐지 유지 임계값(어두운 장면 누락 완화)
    'min_confidence': 0.24,
    'min_brightness': 60.0,
    'min_p95': 175.0,
    'min_bright_ratio': 1.2,
}


def should_keep_detection(obj):
    """어두운 환경 누락을 줄이기 위한 적응형 탐지 필터."""
    th = POLLUTION_THRESHOLDS
    if obj['type'] not in ('간판', '가로등', '조명'):
        return False

    conf_ok = obj.get('confidence', 0.0) >= th['min_confidence']
    if not conf_ok:
        return False

    # 기존 평균 밝기 기준 + 상위 밝기(p95) + 밝은 픽셀 비율 중 하나라도 만족하면 유지
    return (
        obj.get('brightness', 0) >= th['min_brightness'] or
        obj.get('brightnessP95', 0) >= th['min_p95'] or
        obj.get('brightPixelRatio', 0.0) >= th['min_bright_ratio']
    )


def _clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def _parse_float(value, default):
    try:
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def build_capture_context(payload):
    """카메라 기종/ISO/노출/각도에 따른 밝기 보정 계수를 계산한다."""
    model_key = str(payload.get('cameraModel') or 'default').strip().lower()
    profile = CAMERA_PROFILES.get(model_key, CAMERA_PROFILES['default'])
    iso = _clamp(_parse_float(payload.get('iso'), profile['iso_ref']), 50.0, 12800.0)
    exposure_ms = _clamp(_parse_float(payload.get('exposureMs'), profile['exposure_ref_ms']), 1.0, 2000.0)
    angle_deg = _clamp(_parse_float(payload.get('angleDeg'), 0.0), 0.0, 80.0)

    iso_scale = profile['iso_ref'] / iso
    exposure_scale = math.sqrt(profile['exposure_ref_ms'] / exposure_ms)
    angle_radians = math.radians(angle_deg)
    angle_scale = 1.0 / max(0.55, math.cos(angle_radians) ** profile['angle_power'])

    brightness_scale = _clamp(
        profile['brightness_bias'] * iso_scale * exposure_scale * angle_scale,
        0.55,
        2.40,
    )
    glare_scale = _clamp(profile['glare_bias'] * (0.92 + 0.08 * angle_scale), 0.85, 1.25)
    pixel_ratio_scale = _clamp(1.0 / max(0.82, brightness_scale ** 0.35), 0.74, 1.12)

    return {
        'cameraModel': model_key if model_key in CAMERA_PROFILES else 'default',
        'cameraLabel': profile['label'],
        'iso': round(iso, 1),
        'exposureMs': round(exposure_ms, 1),
        'angleDeg': round(angle_deg, 1),
        'brightnessScale': round(brightness_scale, 3),
        'glareScale': round(glare_scale, 3),
        'pixelRatioScale': round(pixel_ratio_scale, 3),
        'locationMode': str(payload.get('locationMode', 'auto')).lower(),
        'locationZone': str(payload.get('locationZone', '제3종')),
    }


def apply_capture_adjustment(metrics, capture_context):
    """촬영 조건에 맞게 ROI 밝기와 법규 판정용 값을 정규화한다."""
    brightness_scale = capture_context.get('brightnessScale', 1.0)
    glare_scale = capture_context.get('glareScale', 1.0)
    pixel_ratio_scale = capture_context.get('pixelRatioScale', 1.0)

    adjusted_brightness = _clamp(metrics['brightness'] * brightness_scale, 0.0, 255.0)
    adjusted_brightness_p95 = _clamp(metrics['brightness_p95'] * brightness_scale * glare_scale, 0.0, 255.0)
    adjusted_brightness_max = _clamp(metrics['brightness_max'] * brightness_scale * glare_scale, 0.0, 255.0)
    adjusted_bright_pixel_ratio = _clamp(metrics['bright_pixel_ratio'] * pixel_ratio_scale, 0.0, 100.0)
    adjusted_saturation = _clamp(metrics['saturation'] * (1.0 / max(0.92, brightness_scale ** 0.18)), 0.0, 1.0)
    adjusted_gamma = _clamp(metrics['gamma'] * (0.98 + 0.04 * min(1.0, capture_context.get('angleDeg', 0.0) / 45.0)), 1.0, 3.5)

    adjusted_luminance_avg = estimate_luminance_cd_m2(adjusted_brightness)
    adjusted_luminance_max = estimate_luminance_cd_m2(adjusted_brightness_max)
    adjusted_illuminance_avg = estimate_illuminance_lux(adjusted_brightness)
    adjusted_illuminance_max = estimate_illuminance_lux(adjusted_brightness_max)

    return {
        'raw': metrics,
        'brightness': adjusted_brightness,
        'brightness_p95': adjusted_brightness_p95,
        'brightness_max': adjusted_brightness_max,
        'bright_pixel_ratio': adjusted_bright_pixel_ratio,
        'saturation': adjusted_saturation,
        'gamma': adjusted_gamma,
        'luminance_cd_m2_avg': adjusted_luminance_avg,
        'luminance_cd_m2_max': adjusted_luminance_max,
        'illuminance_lux_avg': adjusted_illuminance_avg,
        'illuminance_lux_max': adjusted_illuminance_max,
    }

# ---- 조명환경관리구역 구분 (인공조명에 의한 빛공해 방지법 시행규칙) ----
ZONE_LABELS = {
    '제1종': '자연환경 보존지역',
    '제2종': '농림지역',
    '제3종': '주거지역',
    '제4종': '상업·공업지역',
}

# ---- 빛공해 법규 기준값 (별표) ----
ZONE_STANDARDS = {
    '공간조명': {
        # 가로등 — 주거지 인접면 조도(lux) 최대값
        '제1종': 10, '제2종': 10, '제3종': 10, '제4종': 25,
    },
    '광고물': {
        # 간판 — 발광표면휘도(cd/m²) 최대값
        '제1종': 50, '제2종': 400, '제3종': 800, '제4종': 1000,
    },
    '장식조명': {
        # 조명 — 발광표면휘도(cd/m²) 평균값/최대값
        '제1종': {'avg': 5,  'max': 20},
        '제2종': {'avg': 5,  'max': 60},
        '제3종': {'avg': 15, 'max': 180},
        '제4종': {'avg': 25, 'max': 300},
    },
}

# ---- 과태료 부과 기준 (인공조명에 의한 빛공해 방지법 시행령 제8조) ----
# 1차 위반 기준 적용 (100만원 이하)
# 기준값 초과 배율: ~1.5배=1단계, 1.5~2배=2단계, 2배 초과=3단계
FINE_AMOUNTS = {
    '1단계': {'label': '1차위반 수준',      'amount': 50},   # 50만원
    '2단계': {'label': '1차위반 수준',      'amount': 75},   # 75만원
    '3단계': {'label': '1차위반 최고 수준', 'amount': 100},  # 100만원 (1차 위반 상한)
}


# ---- 지역 코드 정규화 ----
def normalize_zone_code(zone_value):
    """요청 파라미터를 제1종~제4종 코드로 정규화."""
    z = (zone_value or '').strip()
    if z in ('제1종', '제2종', '제3종', '제4종'):
        return z
    zl = z.lower()
    if zl in ('residential', '주거', '주거지역'):
        return '제3종'
    if zl in ('commercial', '상업', '상업지역', 'industrial', '공업', '공업지역'):
        return '제4종'
    if zl in ('agricultural', '농업', '농림', '농림지역'):
        return '제2종'
    if zl in ('nature', '자연', '자연환경', '자연환경보존'):
        return '제1종'
    return '제3종'  # 기본값: 주거지역


# ---- 이미지 측정 유틸 ----
def estimate_luminance_cd_m2(luminance_0_255):
    """이미지 휘도(0~255) → 현장 휘도(cd/m²) 근사 변환."""
    norm = float(np.clip(luminance_0_255 / 255.0, 0.0, 1.0))
    return float((norm ** 1.15) * 1200.0)


def estimate_illuminance_lux(luminance_0_255):
    """이미지 밝기(0~255) → 조도(lux) 근사 변환."""
    norm = float(np.clip(luminance_0_255 / 255.0, 0.0, 1.0))
    return float((norm ** 1.05) * 60.0)


def _srgb_to_linear(channel_0_255):
    """sRGB(0~255)를 선형광(linear light, 0~1)으로 변환."""
    c = np.clip(channel_0_255 / 255.0, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _white_balance(img):
    """Gray World 화이트 밸런스 보정 — 색온도 편향 제거."""
    result = img.astype(np.float32)
    avg_r = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_b = np.mean(result[:, :, 2])
    avg_gray = (avg_r + avg_g + avg_b) / 3.0
    if avg_r > 1:
        result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / avg_r), 0, 255)
    if avg_g > 1:
        result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / avg_g), 0, 255)
    if avg_b > 1:
        result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / avg_b), 0, 255)
    return result.astype(np.uint8)


def preprocess_image(img):
    """
    야간 이미지 탐지용 전처리 파이프라인 (측정값 산출에는 원본 사용):
      1. 화이트 밸런스 보정 (Gray World)
      2. 양방향 필터 노이즈 제거 (엣지 보존)
      3. CLAHE 대비 강화 (LAB 색공간 L채널만 적용)
    """
    img_wb = _white_balance(img)
    img_bgr = cv2.cvtColor(img_wb, cv2.COLOR_RGB2BGR)
    denoised = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b_ch])
    result_bgr = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def analyze_region_metrics(cropped):
    """
    바운딩박스 영역의 휘도·조도 측정.
    평균값(avg)과 최대값(max) 모두 반환 — 장식조명·공간조명 법규 판정에 각각 사용.
    """
    # sRGB를 선형화한 뒤 상대휘도(Rec.709 계수) 계산
    r_lin = _srgb_to_linear(cropped[:, :, 0].astype(np.float32))
    g_lin = _srgb_to_linear(cropped[:, :, 1].astype(np.float32))
    b_lin = _srgb_to_linear(cropped[:, :, 2].astype(np.float32))
    lum_rel = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

    # 기존 API/판정 로직 호환을 위해 0~255 스케일로 환산
    lum_255 = np.clip(lum_rel * 255.0, 0.0, 255.0)
    brightness_avg = float(np.mean(lum_255))
    brightness_max = float(np.max(lum_255))
    brightness_p95 = float(np.percentile(lum_255, 95))

    # 충분히 밝은 픽셀(선형 상대휘도 0.7 이상)의 비율(%)
    bright_pixel_ratio = float(np.mean(lum_rel >= 0.7) * 100.0)

    hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255
    saturation = float(np.mean(s))
    gamma = float(np.clip(1.8 + np.std(lum_255) / 64.0, 1.0, 3.5))
    luminance_cd_m2_avg = estimate_luminance_cd_m2(brightness_avg)
    luminance_cd_m2_max = estimate_luminance_cd_m2(brightness_max)
    illuminance_lux_avg = estimate_illuminance_lux(brightness_avg)
    illuminance_lux_max = estimate_illuminance_lux(brightness_max)
    return (brightness_avg, luminance_cd_m2_avg, illuminance_lux_avg,
            saturation, gamma, luminance_cd_m2_max, illuminance_lux_max,
            brightness_p95, bright_pixel_ratio)


def classify_by_geometry(x1, y1, x2, y2, img_h, img_w):
    """
    YOLO가 탐지했으나 클래스명 불명 시 바운딩박스 모양으로 추측.
    - aspect >= 1.8  → 간판
    - 상단+세로형    → 가로등
    - 그 외          → 조명
    """
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    aspect = bw / bh
    if aspect >= 1.8:
        return '간판'
    if y1 < img_h * 0.45 and aspect <= 0.9:
        return '가로등'
    return '조명'


def _build_pollution_features(x1, y1, x2, y2, img_h, img_w):
    """분류 알고리즘 공통 입력 특징(면적·가장자리 인접도)을 계산한다."""
    th = POLLUTION_THRESHOLDS
    box_area = max(1, (x2 - x1) * (y2 - y1))
    img_area = max(1, img_h * img_w)
    area_ratio = box_area / img_area

    edge_margin_x = min(x1, max(0, img_w - x2)) / max(1, img_w)
    edge_margin_y = min(y1, max(0, img_h - y2)) / max(1, img_h)
    near_edge = edge_margin_x < th['edge_margin'] or edge_margin_y < th['edge_margin']

    return {
        'area_ratio': area_ratio,
        'near_edge': near_edge,
        'is_top_region': y1 <= img_h * 0.25,
    }


def _classify_streetlight_pollution(brightness_avg, brightness_p95,
                                    bright_pixel_ratio, saturation, gamma,
                                    features):
    """가로등 전용 분류: 눈부심/산란광 중심으로 판단한다."""
    th = POLLUTION_THRESHOLDS
    score = {'침입광': 0.0, '눈부심': 0.0, '산란광': 0.0, '군집된빛': 0.0}

    # 가로등은 지점형 고휘도 광원 특성이 강해 눈부심 가중치를 높인다.
    if brightness_p95 >= th['glare_p95']:
        score['눈부심'] += 2.2
    if gamma >= th['glare_gamma']:
        score['눈부심'] += 1.4
    if bright_pixel_ratio <= th['glare_ratio_upper']:
        score['눈부심'] += 0.6

    # 상부 대면적 확산 + 저채도는 산란광 특성으로 본다.
    if features['area_ratio'] >= th['scatter_area']:
        score['산란광'] += 1.6
    if bright_pixel_ratio >= th['scatter_ratio']:
        score['산란광'] += 1.5
    if saturation <= th['scatter_sat_max']:
        score['산란광'] += 0.8
    if features['is_top_region']:
        score['산란광'] += 0.5

    # 주거 경계 방향(프레임 가장자리) 광원은 침입광 가능성을 올린다.
    if features['near_edge']:
        score['침입광'] += 1.4
    if brightness_avg >= th['intrusion_avg']:
        score['침입광'] += 0.8

    return max(score, key=score.get)


def _classify_signboard_pollution(brightness_avg, brightness_p95,
                                  bright_pixel_ratio, saturation, gamma,
                                  features):
    """간판 전용 분류: 군집된빛/침입광 중심으로 판단한다."""
    th = POLLUTION_THRESHOLDS
    score = {'침입광': 0.0, '눈부심': 0.0, '산란광': 0.0, '군집된빛': 0.0}

    # 간판은 다수·고채도·고밀도 발광이 군집된빛 특성으로 이어진다.
    score['군집된빛'] += 1.5
    if saturation >= th['cluster_sat']:
        score['군집된빛'] += 1.3
    if bright_pixel_ratio >= th['cluster_ratio']:
        score['군집된빛'] += 1.1

    # 건물 외곽/창가 경계에 붙은 밝은 간판은 침입광 가능성이 높다.
    if features['near_edge']:
        score['침입광'] += 1.8
    if brightness_avg >= th['intrusion_avg']:
        score['침입광'] += 1.0
    if features['area_ratio'] < 0.25:
        score['침입광'] += 0.5

    # 고휘도 픽셀 집중이 강하면 눈부심도 함께 평가한다.
    if brightness_p95 >= th['glare_p95']:
        score['눈부심'] += 1.2
    if gamma >= th['glare_gamma']:
        score['눈부심'] += 0.8

    return max(score, key=score.get)


def _classify_decorative_pollution(brightness_avg, brightness_p95,
                                   bright_pixel_ratio, saturation, gamma,
                                   features):
    """장식조명 전용 분류: 군집된빛/산란광/침입광을 균형 평가한다."""
    th = POLLUTION_THRESHOLDS
    score = {'침입광': 0.0, '눈부심': 0.0, '산란광': 0.0, '군집된빛': 0.0}

    # 장식조명은 색채/밀도 기반 군집된빛 가능성을 기본 가정으로 둔다.
    if saturation >= th['cluster_sat']:
        score['군집된빛'] += 1.3
    if bright_pixel_ratio >= th['cluster_ratio']:
        score['군집된빛'] += 1.0

    # 상부 대면적의 저채도 확산 조명은 산란광으로 판단한다.
    if features['area_ratio'] >= th['scatter_area']:
        score['산란광'] += 1.6
    if bright_pixel_ratio >= th['scatter_ratio']:
        score['산란광'] += 1.2
    if saturation <= th['scatter_sat_max']:
        score['산란광'] += 0.8
    if features['is_top_region']:
        score['산란광'] += 0.4

    # 경계 인접 + 밝기 높은 장식조명은 침입광 리스크를 가산한다.
    if features['near_edge']:
        score['침입광'] += 1.2
    if brightness_avg >= th['intrusion_avg']:
        score['침입광'] += 0.9

    # 고휘도 포인트가 강하면 눈부심 후보도 남긴다.
    if brightness_p95 >= th['glare_p95']:
        score['눈부심'] += 0.9
    if gamma >= th['glare_gamma']:
        score['눈부심'] += 0.6

    return max(score, key=score.get)


def classify_pollution_category(cat, brightness_avg, brightness_p95,
                                bright_pixel_ratio, saturation, gamma,
                                x1, y1, x2, y2, img_h, img_w):
    """조명 유형별(가로등/간판/조명)로 분리된 분류 알고리즘을 적용한다."""
    features = _build_pollution_features(x1, y1, x2, y2, img_h, img_w)

    # 조명 유형별 전용 분류기 라우팅
    if cat == '가로등':
        return _classify_streetlight_pollution(
            brightness_avg, brightness_p95, bright_pixel_ratio, saturation, gamma, features
        )
    if cat == '간판':
        return _classify_signboard_pollution(
            brightness_avg, brightness_p95, bright_pixel_ratio, saturation, gamma, features
        )
    return _classify_decorative_pollution(
        brightness_avg, brightness_p95, bright_pixel_ratio, saturation, gamma, features
    )


def summarize_pollution_categories(detected):
    """객체별 분류를 집계해 사진의 대표 분류를 산출."""
    counts = {'침입광': 0, '눈부심': 0, '산란광': 0, '군집된빛': 0}
    weighted_counts = {'침입광': 0.0, '눈부심': 0.0, '산란광': 0.0, '군집된빛': 0.0}
    for d in detected:
        cat = d.get('pollutionCategory')
        if cat in counts:
            counts[cat] += 1
            conf = float(d.get('confidence', 0.5))
            # 신뢰도 가중치(최소 0.1)로 과소신뢰 탐지의 영향 축소
            weighted_counts[cat] += max(0.1, min(conf, 1.0))

    if not detected:
        return {
            'overall': '미탐지',
            'counts': counts,
            'weightedCounts': weighted_counts,
            'description': '광원 객체가 탐지되지 않았습니다.'
        }

    # 동률일 때는 법규/민원 우선순위 관점으로 눈부심 > 침입광 > 군집된빛 > 산란광
    tie_break_order = ['눈부심', '침입광', '군집된빛', '산란광']
    max_weight = max(weighted_counts.values())
    winners = [k for k, v in weighted_counts.items() if abs(v - max_weight) < 1e-9]
    if len(winners) > 1:
        max_count = max(counts[k] for k in winners)
        winners = [k for k in winners if counts[k] == max_count]
    overall = next((k for k in tie_break_order if k in winners), winners[0])

    return {
        'overall': overall,
        'counts': counts,
        'weightedCounts': {k: round(v, 3) for k, v in weighted_counts.items()},
        'description': POLLUTION_TYPES.get(overall, ''),
    }


def build_default_zones_summary():
    """EXIF/GPS 미확인 시 기본 4개 구역 시뮬레이션 결과(탐지 실패 대비)."""
    return {
        zc: {
            'zoneLabel': ZONE_LABELS[zc],
            'overall': '준수',
            'totalFineAmount': 0,
            'violationCount': 0,
        }
        for zc in ('제1종', '제2종', '제3종', '제4종')
    }


# ---- EXIF GPS 추출 ----
def extract_gps_from_exif(image_bytes):
    """이미지 바이트에서 EXIF GPS 좌표(위도, 경도)를 추출합니다."""
    try:
        from PIL.ExifTags import TAGS, GPSTAGS
        pil_img = Image.open(io.BytesIO(image_bytes))
        exif_raw = pil_img._getexif()
        if not exif_raw:
            return None
        gps_info = {}
        for tag_id, val in exif_raw.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == 'GPSInfo':
                for gps_tag_id, gps_val in val.items():
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag] = gps_val
        if 'GPSLatitude' not in gps_info or 'GPSLongitude' not in gps_info:
            return None

        def to_deg(v):
            d, m, s = v
            return float(d) + float(m) / 60 + float(s) / 3600

        lat = to_deg(gps_info['GPSLatitude'])
        if gps_info.get('GPSLatitudeRef') == 'S':
            lat = -lat
        lon = to_deg(gps_info['GPSLongitude'])
        if gps_info.get('GPSLongitudeRef') == 'W':
            lon = -lon
        return lat, lon
    except Exception:
        return None


# ---- GPS → 조명환경관리구역 판별 (Nominatim 역지오코딩) ----
def get_zone_from_gps(lat, lon):
    """
    GPS 좌표로 조명환경관리구역 유형을 추정합니다.
    OpenStreetMap Nominatim API 사용 (무료·API키 불필요).
    네트워크 오류 또는 판별 실패 시 None 반환 → 요청 파라미터 폴백.
    """
    try:
        url = 'https://nominatim.openstreetmap.org/reverse'
        params = {'lat': lat, 'lon': lon, 'format': 'json', 'addressdetails': 1}
        headers = {'User-Agent': 'LightPollutionDetector/1.0 (research)'}
        resp = req_lib.get(url, params=params, headers=headers, timeout=5)
        data = resp.json()
        osm_type = data.get('type', '')
        category = data.get('category', '')

        # 제1종: 자연환경 보존지역
        if osm_type in ('nature_reserve', 'national_park', 'protected_area',
                         'wetland', 'natural_reserve'):
            return '제1종'
        if category == 'boundary' and osm_type in ('national_park', 'protected_area'):
            return '제1종'

        # 제2종: 농림지역
        if osm_type in ('farmland', 'farm', 'forest', 'orchard',
                         'vineyard', 'meadow', 'grass', 'wood'):
            return '제2종'
        if category == 'landuse' and osm_type in ('farmland', 'forest', 'agricultural'):
            return '제2종'

        # 제4종: 상업·공업지역
        if osm_type in ('commercial', 'industrial', 'retail',
                         'office', 'mall', 'supermarket'):
            return '제4종'
        if category in ('shop', 'office') or osm_type in ('industrial', 'commercial'):
            return '제4종'

        # 기본값: 제3종 주거지역
        return '제3종'
    except Exception:
        return None


# ---- 과태료 단계 산출 ----
def compute_fine(luminance_cd_m2_avg, luminance_cd_m2_max,
                 illuminance_lux_max, light_type, zone_code):
    """
    조명 유형·지역 구분에 따라 법규 기준과 비교하여 과태료 단계를 산출합니다.

    [측정 기준]
    - 공간조명(가로등): 조도(lux) 최대값
    - 광고물(간판):     휘도(cd/m²) 최대값
    - 장식조명(조명):   휘도(cd/m²) 평균/최대값 중 더 많이 초과된 쪽

    [위반 단계]
    - 준수:   기준값 이하
    - 1단계:  기준값 초과 ~ 1.5배 이내  (1차위반 수준, 100만원)
    - 2단계:  1.5배 초과 ~ 2배 이내    (2차위반 수준, 200만원)
    - 3단계:  2배 초과                  (3차이상 위반 수준, 300만원)
    """
    std = ZONE_STANDARDS.get(light_type, {})
    entry = std.get(zone_code)
    if entry is None:
        return {'compliance': '기준없음', 'violationStage': None, 'fineLabel': None,
                'fineAmount': 0, 'measuredValue': 0, 'threshold': 0,
                'unit': '-', 'basis': '-', 'ratio': 0}

    if light_type == '공간조명':
        measured = illuminance_lux_max
        threshold = entry
        unit, basis = 'lux', '조도 최대값'
    elif light_type == '광고물':
        measured = luminance_cd_m2_max
        threshold = entry
        unit, basis = 'cd/m²', '휘도 최대값'
    elif light_type == '장식조명':
        avg_th, max_th = entry['avg'], entry['max']
        avg_ratio = luminance_cd_m2_avg / max(1.0, avg_th)
        max_ratio = luminance_cd_m2_max / max(1.0, max_th)
        if max_ratio >= avg_ratio:
            measured, threshold, basis = luminance_cd_m2_max, max_th, '휘도 최대값'
        else:
            measured, threshold, basis = luminance_cd_m2_avg, avg_th, '휘도 평균값'
        unit = 'cd/m²'
    else:
        return {'compliance': '기준없음', 'violationStage': None, 'fineLabel': None,
                'fineAmount': 0, 'measuredValue': 0, 'threshold': 0,
                'unit': '-', 'basis': '-', 'ratio': 0}

    ratio = measured / max(1.0, threshold)

    if measured <= threshold:
        return {'compliance': '준수', 'violationStage': None, 'fineLabel': None,
                'fineAmount': 0, 'measuredValue': round(measured, 1),
                'threshold': threshold, 'unit': unit, 'basis': basis,
                'ratio': round(ratio, 2)}

    if ratio <= 1.5:
        stage = '1단계'
    elif ratio <= 2.0:
        stage = '2단계'
    else:
        stage = '3단계'

    fine_info = FINE_AMOUNTS[stage]
    return {
        'compliance':    '위반',
        'violationStage': stage,
        'fineLabel':     fine_info['label'],
        'fineAmount':    fine_info['amount'],
        'measuredValue': round(measured, 1),
        'threshold':     threshold,
        'unit':          unit,
        'basis':         basis,
        'ratio':         round(ratio, 2),
    }


def _normalize_gps_payload(raw_gps):
    if isinstance(raw_gps, dict):
        lat = raw_gps.get('lat')
        lon = raw_gps.get('lon')
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            return {'lat': float(lat), 'lon': float(lon)}
    return None


def _format_gps_text(raw_gps):
    gps = _normalize_gps_payload(raw_gps)
    if not gps:
        return '미확인'
    return f"{gps['lat']:.6f}, {gps['lon']:.6f}"


def _resolve_pdf_font():
    """Korean text가 들어간 PDF를 위한 폰트를 우선순위대로 선택한다."""
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError as exc:
        raise RuntimeError('reportlab 패키지가 설치되어 있지 않습니다.') from exc

    font_candidates = [
        ('MalgunGothic', r'C:\Windows\Fonts\malgun.ttf'),
        ('MalgunGothicBold', r'C:\Windows\Fonts\malgunbd.ttf'),
        ('AppleGothic', '/System/Library/Fonts/Supplemental/AppleGothic.ttf'),
        ('NanumGothic', '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'),
        ('NotoSansKR', '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'),
    ]

    for font_name, font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                return font_name
            except Exception:
                continue

    for cid_font in ('HYGoThic-Medium', 'HYSMyeongJo-Medium'):
        try:
            pdfmetrics.registerFont(UnicodeCIDFont(cid_font))
            return cid_font
        except Exception:
            continue

    return 'Helvetica'


def _build_pdf_report_bytes(report_data):
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError('reportlab 패키지가 설치되어 있지 않습니다.') from exc

    font_name = _resolve_pdf_font()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title='빛 공해 법규 위반 탐지 리포트',
        author='Light Pollution AI System',
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='KrTitle',
        parent=styles['Title'],
        fontName=font_name,
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#12325b'),
    ))
    styles.add(ParagraphStyle(
        name='KrHeading',
        parent=styles['Heading2'],
        fontName=font_name,
        fontSize=11.5,
        leading=14,
        spaceBefore=6,
        spaceAfter=6,
        textColor=colors.HexColor('#12325b'),
    ))
    styles.add(ParagraphStyle(
        name='KrBody',
        parent=styles['BodyText'],
        fontName=font_name,
        fontSize=9,
        leading=12,
    ))
    styles.add(ParagraphStyle(
        name='KrNote',
        parent=styles['BodyText'],
        fontName=font_name,
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#5c667a'),
    ))

    def make_table(rows, col_widths=None, header_fill='#1f4e79'):
        table = Table(rows, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_fill)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 8.5),
            ('LEADING', (0, 0), (-1, -1), 10.5),
            ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#c8d2e3')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor('#f7f9fc')]),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        return table

    def yes_no(flag):
        return '예' if flag else '아니오'

    detected = list(report_data.get('detected') or [])
    zone = report_data.get('zone') or '제3종'
    zone_label = report_data.get('zoneLabel') or ZONE_LABELS.get(zone, zone)
    gps_text = report_data.get('gpsText') or _format_gps_text(report_data.get('rawGps'))
    generated_at = report_data.get('generatedAt') or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    analysis_time = report_data.get('analysisTime') or generated_at
    total_fine = int(report_data.get('totalFineAmount') or 0)
    violation_count = int(report_data.get('violationCount') or 0)
    overall = report_data.get('overall') or '미탐지'
    pollution_overall = report_data.get('pollutionOverall') or overall
    pollution_summary = report_data.get('pollutionSummary') or {}
    counts = pollution_summary.get('counts') or {}
    model_status = report_data.get('modelStatus') or MODEL_STATUS
    file_name = report_data.get('fileName') or '미상'
    file_size = report_data.get('fileSize') or '-'
    risk_summary = report_data.get('riskSummary') or '-'
    all_zones_mode = bool(report_data.get('allZonesMode'))
    capture_summary = report_data.get('captureSummary') or '-'
    capture_context = report_data.get('captureContext') or {}
    capture_label = capture_context.get('cameraLabel') or '-'
    capture_iso = capture_context.get('iso') if capture_context.get('iso') is not None else '-'
    capture_exposure = capture_context.get('exposureMs') if capture_context.get('exposureMs') is not None else '-'
    capture_angle = capture_context.get('angleDeg') if capture_context.get('angleDeg') is not None else '-'

    summary_rows = [
        ['항목', '내용', '항목', '내용'],
        ['파일명', file_name, '파일 크기', file_size],
        ['분석 시간', analysis_time, '생성 시간', generated_at],
        ['종합 판정', overall, '총 과태료', f'{total_fine}만원'],
        ['위반 건수', f'{violation_count}건', '대표 분류', pollution_overall],
        ['조명환경관리구역', f'{zone} ({zone_label})', 'GPS', gps_text],
        ['GPS 판별', yes_no(bool(report_data.get('gpsDetected'))), '구역별 시뮬레이션', yes_no(all_zones_mode)],
        ['모델 상태', model_status, '위험 요약', risk_summary],
        ['촬영 조건', capture_summary, '보정 기종', capture_label],
        ['ISO / 노출', f'{capture_iso} / {capture_exposure} ms', '촬영 각도', f'{capture_angle}°'],
    ]

    story = [
        Paragraph('빛 공해 법규 위반 탐지 및 판정 리포트', styles['KrTitle']),
        Spacer(1, 4 * mm),
        Paragraph('법적 증빙 참고용 자동 생성 문서', styles['KrBody']),
        Spacer(1, 4 * mm),
        make_table(summary_rows, col_widths=[26 * mm, 58 * mm, 28 * mm, 58 * mm]),
        Spacer(1, 5 * mm),
        Paragraph('법규 적용 기준', styles['KrHeading']),
        Paragraph(
            '본 리포트는 인공조명에 의한 빛공해 방지법, 동 시행령 제8조, 동 시행규칙 별표의 조명환경관리구역 기준을 바탕으로 '
            '이미지 기반 추정값을 평가한 결과입니다. 실제 행정 처분은 관할 기관의 공식 측정 및 현장 확인에 따릅니다.',
            styles['KrBody'],
        ),
        Spacer(1, 3 * mm),
    ]

    standards_rows = [[
        '조명환경관리구역', '공간조명·가로등', '광고물·간판', '장식조명·조명'
    ]]
    for zc in ('제1종', '제2종', '제3종', '제4종'):
        light_std = ZONE_STANDARDS['공간조명'][zc]
        ad_std = ZONE_STANDARDS['광고물'][zc]
        deco_std = ZONE_STANDARDS['장식조명'][zc]
        standards_rows.append([
            f'{zc} {ZONE_LABELS[zc]}',
            f'{light_std} lux',
            f'{ad_std} cd/m²',
            f'{deco_std["avg"]} / {deco_std["max"]} cd/m²',
        ])

    story.extend([
        Paragraph('조명환경관리구역별 법규 기준', styles['KrHeading']),
        make_table(standards_rows, col_widths=[42 * mm, 38 * mm, 34 * mm, 46 * mm], header_fill='#28527a'),
        Spacer(1, 5 * mm),
    ])

    if detected:
        detail_rows = [[
            '객체명', '유형', '빛 공해 분류', '법규 유형', '측정값', '기준값', '판정', '과태료'
        ]]
        for item in detected:
            unit = item.get('unit') or ('lux' if item.get('type') == '가로등' else 'cd/m²')
            measured_value = item.get('measuredValue')
            if measured_value is None:
                measured_value = item.get('illuminanceLux') if unit == 'lux' else item.get('luminanceCdM2')
            threshold = item.get('threshold')
            compliance = item.get('compliance') or '미분류'
            stage = item.get('violationStage') or '준수'
            fine_amount = item.get('fineAmount') or 0
            detail_rows.append([
                item.get('name') or '-',
                item.get('type') or '-',
                f"{item.get('pollutionCategory') or '-'}",
                item.get('lightType') or '-',
                f"{measured_value:.1f} {unit}" if isinstance(measured_value, (int, float)) else f'- {unit}',
                f"{threshold} {unit}" if threshold not in (None, '') else '-',
                f'{compliance}{" / " + stage if stage != "준수" else ""}',
                f'{fine_amount}만원' if fine_amount else '없음',
            ])

        story.extend([
            Paragraph('탐지 객체별 상세 분석', styles['KrHeading']),
            make_table(detail_rows, col_widths=[20 * mm, 16 * mm, 20 * mm, 18 * mm, 20 * mm, 20 * mm, 22 * mm, 18 * mm], header_fill='#6b4f2a'),
            Spacer(1, 4 * mm),
        ])
    else:
        story.extend([
            Paragraph('탐지 객체별 상세 분석', styles['KrHeading']),
            Paragraph('광원 객체가 탐지되지 않아 세부 위반 항목은 산출되지 않았습니다.', styles['KrBody']),
            Spacer(1, 4 * mm),
        ])

    counts_text = ', '.join([
        f'침입광 {counts.get("침입광", 0)}건',
        f'눈부심 {counts.get("눈부심", 0)}건',
        f'산란광 {counts.get("산란광", 0)}건',
        f'군집된빛 {counts.get("군집된빛", 0)}건',
    ])
    story.extend([
        Paragraph('종합 해석', styles['KrHeading']),
        Paragraph(
            f'대표 분류는 {pollution_overall}이며, 유형별 집계는 {counts_text}입니다. 총 과태료는 {total_fine}만원, 위반 건수는 {violation_count}건입니다.',
            styles['KrBody'],
        ),
        Spacer(1, 2 * mm),
        Paragraph(
            '주의: 본 문서는 이미지 기반 추정 분석 결과를 시각화한 참고 리포트입니다. 법적 증빙으로 활용 시에는 '
            '현장 측정 기록, 촬영 원본, 위치 정보, 관할 기관의 공식 판정 자료와 함께 제출하는 것을 권장합니다.',
            styles['KrNote'],
        ),
    ])

    def draw_footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont(font_name, 8)
        canvas.setFillColor(colors.HexColor('#6b7280'))
        canvas.drawString(doc_obj.leftMargin, 10 * mm, '인공조명에 의한 빛공해 방지법 · 시행령 제8조 · 시행규칙 별표 기준 참고')
        canvas.drawRightString(A4[0] - doc_obj.rightMargin, 10 * mm, f'Page {canvas.getPageNumber()}')
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    buffer.seek(0)
    return buffer


def _build_pdf_report_filename(report_data):
    base_name = (report_data.get('fileName') or 'light_pollution_report').strip()
    base_name = os.path.splitext(base_name)[0] or 'light_pollution_report'
    safe_name = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in base_name)
    safe_name = safe_name.strip('_') or 'light_pollution_report'
    return f'{safe_name}.pdf'


# ---- Flask 라우트 ----
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analysis')
@app.route('/analysis.html')
def analysis_page():
    return render_template('analysis.html')


@app.route('/result')
@app.route('/result.html')
def result_page():
    return render_template('result.html')


@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({
        'status': 'success',
        'modelStatus': MODEL_STATUS,
        'pdfReportAvailable': True,
        'supportedFormats': ['json', 'pdf'],
        'features': {
            'pollutionCategories': list(POLLUTION_TYPES.keys()),
            'zoneTypes': list(ZONE_LABELS.keys()),
            'lawReferences': [
                '인공조명에 의한 빛공해 방지법',
                '인공조명에 의한 빛공해 방지법 시행령 제8조',
                '인공조명에 의한 빛공해 방지법 시행규칙 별표',
            ],
        },
    })


@app.route('/api/report/pdf', methods=['POST'])
def api_report_pdf():
    report_data = request.get_json(silent=True) or {}
    if not report_data:
        return jsonify({'status': 'error', 'message': 'No report data provided.'}), 400

    try:
        pdf_buffer = _build_pdf_report_bytes(report_data)
        filename = _build_pdf_report_filename(report_data)
    except RuntimeError as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 500
    except Exception as exc:
        return jsonify({'status': 'error', 'message': f'PDF 생성 실패: {exc}'}), 500

    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=filename,
    )


def get_distance_meters(lat1, lon1, lat2, lon2):
    earth_radius_m = 6371000
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return earth_radius_m * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_point_geom(geom):
    if not geom:
        return None
    match = re.search(r'POINT\(([-\d.]+)\s+([-\d.]+)\)', str(geom))
    if not match:
        return None
    try:
        return {'lon': float(match.group(1)), 'lat': float(match.group(2))}
    except ValueError:
        return None


def normalize_items(item):
    if not item:
        return []
    return item if isinstance(item, list) else [item]


def _parse_positive_int(value, default=0):
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _get_nested(data, *keys):
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _normalize_commercial_store(item, origin_lat, origin_lon):
    point = parse_point_geom(item.get('geom'))
    if not point:
        return None

    distance = get_distance_meters(origin_lat, origin_lon, point['lat'], point['lon'])

    return {
        'name': item.get('bplcnm') or '상점명 없음',
        'status': item.get('trdstatenm') or '-',
        'major': item.get('majornm') or '-',
        'minor': item.get('minornm') or '-',
        'businessType': item.get('upjongnm') or '-',
        'address': item.get('rdnwhladdr') or '-',
        'openDate': item.get('apvperymd') or item.get('apvpermymd') or '-',
        'closeDate': item.get('dcbyymd') or item.get('dcbymd') or '-',
        'lat': point['lat'],
        'lon': point['lon'],
        'distanceMeters': round(distance, 2),
    }


def _group_commercial_stores(stores):
    groups = {'within5m': [], 'within10m': [], 'within30m': []}
    for store in sorted(stores, key=lambda x: x['distanceMeters']):
        distance = store['distanceMeters']
        if distance <= 5:
            groups['within5m'].append(store)
        elif distance <= 10:
            groups['within10m'].append(store)
        elif distance <= 30:
            groups['within30m'].append(store)
    return groups


def _append_unique(values, value):
    cleaned = str(value or '').strip()
    if cleaned and cleaned not in values:
        values.append(cleaned)


def _build_busan_commercial_search_terms(lat, lon, fallback_address):
    terms = []
    address_info = {}
    try:
        resp = req_lib.get(
            'https://nominatim.openstreetmap.org/reverse',
            params={
                'lat': lat,
                'lon': lon,
                'format': 'json',
                'addressdetails': 1,
                'accept-language': 'ko',
            },
            headers={'User-Agent': 'LightPollutionDetector/1.0 (research)'},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        address_info = data.get('address') or {}
    except Exception as exc:
        print('Reverse geocode for commercial search failed:', exc, flush=True)

    district = (
        address_info.get('city_district')
        or address_info.get('county')
        or address_info.get('borough')
        or address_info.get('municipality')
    )
    dong = (
        address_info.get('suburb')
        or address_info.get('quarter')
        or address_info.get('neighbourhood')
        or address_info.get('village')
        or address_info.get('town')
    )
    road = address_info.get('road')

    if district and dong:
        _append_unique(terms, f'{district} {dong}')
    _append_unique(terms, dong)
    if district and road:
        _append_unique(terms, f'{district} {road}')
    _append_unique(terms, road)
    _append_unique(terms, district)
    _append_unique(terms, fallback_address)

    return terms, address_info


def _commercial_item_key(item):
    return '|'.join([
        str(item.get('bplcnm') or '').strip(),
        str(item.get('rdnwhladdr') or '').strip(),
        str(item.get('geom') or '').strip(),
        str(item.get('apvperymd') or item.get('apvpermymd') or '').strip(),
    ])


def _fetch_commercial_page(service_key, search_term, page_no):
    params = {
        'serviceKey': service_key,
        'pageNo': page_no,
        'numOfRows': BUSAN_COMMERCIAL_NUM_OF_ROWS,
        'resultType': 'json',
    }
    if search_term:
        params['rdnwhladdr'] = search_term

    upstream = req_lib.get(
        BUSAN_COMMERCIAL_ENDPOINT,
        params=params,
        timeout=8,
    )
    upstream.raise_for_status()
    payload = upstream.json()

    header = _get_nested(payload, 'response', 'header') or {}
    if header.get('resultCode') not in (None, '00'):
        raise RuntimeError(header.get('resultMsg') or 'Busan commercial API returned an error.')

    body = _get_nested(payload, 'response', 'body') or {}
    items = [
        item
        for item in normalize_items(_get_nested(body, 'items', 'item'))
        if isinstance(item, dict)
    ]
    return {
        'payload': payload,
        'totalCount': body.get('totalCount'),
        'items': items,
    }


def _fetch_commercial_items_for_term(service_key, search_term):
    collected_items = []
    total_count = None
    total_pages = None
    requested_pages = 0
    first_payload = None

    page_no = 1
    while True:
        requested_pages += 1
        page = _fetch_commercial_page(service_key, search_term, page_no)
        payload = page['payload']
        if first_payload is None:
            first_payload = payload

        if total_count is None:
            total_count = page['totalCount']
            parsed_total_count = _parse_positive_int(total_count)
            if parsed_total_count:
                total_pages = math.ceil(parsed_total_count / BUSAN_COMMERCIAL_NUM_OF_ROWS)
                if BUSAN_COMMERCIAL_MAX_PAGES > 0:
                    total_pages = min(total_pages, BUSAN_COMMERCIAL_MAX_PAGES)

        page_items = page['items']
        collected_items.extend(page_items)

        if len(page_items) < BUSAN_COMMERCIAL_NUM_OF_ROWS:
            break
        if total_pages is not None and page_no >= total_pages:
            break
        if total_pages is None and BUSAN_COMMERCIAL_MAX_PAGES > 0 and page_no >= BUSAN_COMMERCIAL_MAX_PAGES:
            break
        page_no += 1

    return {
        'term': search_term,
        'items': collected_items,
        'totalCount': total_count,
        'totalPages': total_pages,
        'requestedPages': requested_pages,
        'firstPayload': first_payload,
    }


def _normalize_commercial_cache_item(item):
    point = parse_point_geom(item.get('geom'))
    if not point:
        return None

    open_date = item.get('apvperymd') or item.get('apvpermymd') or '-'
    close_date = item.get('dcbyymd') or item.get('dcbymd') or '-'
    raw_id = '|'.join([
        str(item.get('bplcnm') or '').strip(),
        str(item.get('rdnwhladdr') or '').strip(),
        str(item.get('geom') or '').strip(),
        str(open_date or '').strip(),
    ])

    return {
        'id': hashlib.sha1(raw_id.encode('utf-8')).hexdigest(),
        'name': item.get('bplcnm') or '상점명 없음',
        'status': item.get('trdstatenm') or '-',
        'major': item.get('majornm') or '-',
        'minor': item.get('minornm') or '-',
        'businessType': item.get('upjongnm') or '-',
        'address': item.get('rdnwhladdr') or '-',
        'openDate': open_date,
        'closeDate': close_date,
        'lat': point['lat'],
        'lon': point['lon'],
    }


def _load_busan_commercial_cache():
    global BUSAN_COMMERCIAL_CACHE
    if BUSAN_COMMERCIAL_CACHE is not None:
        return BUSAN_COMMERCIAL_CACHE
    if not os.path.exists(BUSAN_COMMERCIAL_CACHE_PATH):
        BUSAN_COMMERCIAL_CACHE = {'items': [], 'meta': {'exists': False}}
        return BUSAN_COMMERCIAL_CACHE
    try:
        with open(BUSAN_COMMERCIAL_CACHE_PATH, encoding='utf-8') as cache_file:
            BUSAN_COMMERCIAL_CACHE = json.load(cache_file)
    except (OSError, ValueError):
        BUSAN_COMMERCIAL_CACHE = {'items': [], 'meta': {'exists': False, 'loadError': True}}
    return BUSAN_COMMERCIAL_CACHE


def _save_busan_commercial_cache(cache_data):
    global BUSAN_COMMERCIAL_CACHE
    os.makedirs(os.path.dirname(BUSAN_COMMERCIAL_CACHE_PATH), exist_ok=True)
    temp_path = f'{BUSAN_COMMERCIAL_CACHE_PATH}.tmp'
    with open(temp_path, 'w', encoding='utf-8') as cache_file:
        json.dump(cache_data, cache_file, ensure_ascii=False, separators=(',', ':'))
    os.replace(temp_path, BUSAN_COMMERCIAL_CACHE_PATH)
    BUSAN_COMMERCIAL_CACHE = cache_data


def _sync_busan_commercial_cache(service_key, search_term=''):
    start_time = time.time()
    term_result = _fetch_commercial_items_for_term(service_key, search_term)
    seen_ids = set()
    cache_items = []
    geom_count = 0

    for item in term_result['items']:
        cached = _normalize_commercial_cache_item(item)
        if not cached:
            continue
        geom_count += 1
        if cached['id'] in seen_ids:
            continue
        seen_ids.add(cached['id'])
        cache_items.append(cached)

    cache_data = {
        'meta': {
            'exists': True,
            'updatedAt': datetime.now().isoformat(timespec='seconds'),
            'source': 'BusanCommercialHistoryService',
            'searchTerm': search_term or 'ALL',
            'totalCount': term_result['totalCount'],
            'totalPages': term_result['totalPages'],
            'requestedPages': term_result['requestedPages'],
            'rawItemsCount': len(term_result['items']),
            'withGeomCount': geom_count,
            'dedupedCount': len(cache_items),
            'elapsedSeconds': round(time.time() - start_time, 2),
            'numOfRows': BUSAN_COMMERCIAL_NUM_OF_ROWS,
        },
        'items': cache_items,
    }
    _save_busan_commercial_cache(cache_data)
    return cache_data


def _stores_with_distance_from_cache(cache_items, lat, lon):
    stores = []
    for item in cache_items:
        try:
            item_lat = float(item['lat'])
            item_lon = float(item['lon'])
        except (KeyError, TypeError, ValueError):
            continue
        store = dict(item)
        store['distanceMeters'] = round(get_distance_meters(lat, lon, item_lat, item_lon), 2)
        stores.append(store)
    stores.sort(key=lambda x: x['distanceMeters'])
    return stores


@app.route('/api/busan-commercial/nearby-old', methods=['GET'])
def api_busan_commercial_nearby():
    service_key = os.getenv('BUSAN_COMMERCIAL_SERVICE_KEY')
    if not service_key:
        return jsonify({
            'status': 'error',
            'message': 'BUSAN_COMMERCIAL_SERVICE_KEY is not configured.',
        }), 500

    try:
        lat = float(request.args.get('lat', ''))
        lon = float(request.args.get('lon', ''))
        if math.isnan(lat) or math.isnan(lon) or math.isinf(lat) or math.isinf(lon):
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({
            'status': 'error',
            'message': 'Valid lat and lon query parameters are required.',
        }), 400

    address = (request.args.get('address') or '부산진구').strip() or '부산진구'
    page_no = request.args.get('pageNo') or '1'
    num_of_rows = request.args.get('numOfRows') or '100'

    try:
        upstream = req_lib.get(
            BUSAN_COMMERCIAL_ENDPOINT,
            params={
                'serviceKey': service_key,
                'pageNo': page_no,
                'numOfRows': num_of_rows,
                'resultType': 'json',
                'rdnwhladdr': address,
            },
            timeout=8,
        )
        upstream.raise_for_status()
        payload = upstream.json()
    except ValueError:
        return jsonify({
            'status': 'error',
            'message': 'Busan commercial API returned a non-JSON response.',
        }), 502
    except req_lib.RequestException as exc:
        return jsonify({
            'status': 'error',
            'message': f'Busan commercial API request failed: {exc}',
        }), 502

    header = _get_nested(payload, 'response', 'header') or {}
    if header.get('resultCode') not in (None, '00'):
        return jsonify({
            'status': 'error',
            'message': header.get('resultMsg') or 'Busan commercial API returned an error.',
            'resultCode': header.get('resultCode'),
        }), 502

    body = _get_nested(payload, 'response', 'body') or {}
    item_payload = _get_nested(payload, 'response', 'body', 'items', 'item')
    items = [item for item in normalize_items(item_payload) if isinstance(item, dict)]
    total_count = body.get('totalCount')
    items_with_geom = []
    stores = []
    for item in items:
        point = parse_point_geom(item.get('geom'))
        if point:
            items_with_geom.append({**item, 'lat': point['lat'], 'lon': point['lon']})
        store = _normalize_commercial_store(item, lat, lon)
        if store:
            stores.append(store)
    groups = _group_commercial_stores(stores)
    within30m = [store for store in stores if store['distanceMeters'] <= 30]
    within100m = [store for store in stores if store['distanceMeters'] <= 100]
    within300m = [store for store in stores if store['distanceMeters'] <= 300]
    within500m = [store for store in stores if store['distanceMeters'] <= 500]
    distances_sample = [
        {
            'name': store.get('bplcnm'),
            'geom': store.get('geom'),
            'distance': get_distance_meters(lat, lon, store['lat'], store['lon']),
        }
        for store in items_with_geom[:20]
    ]

    print('Busan API totalCount:', total_count, flush=True)
    print('Raw items count:', len(items), flush=True)
    print('Items with geom:', len(items_with_geom), flush=True)
    print('Distances:', distances_sample, flush=True)
    print('Within 30m:', len(within30m), flush=True)
    print('Within 100m:', len(within100m), flush=True)
    print('Within 300m:', len(within300m), flush=True)
    print('Within 500m:', len(within500m), flush=True)

    return jsonify({
        'status': 'success',
        'hasGps': True,
        'origin': {'lat': lat, 'lon': lon},
        'address': address,
        'groups': groups,
        'debug': {
            'apiResponse': payload,
            'totalCount': total_count,
            'itemsCount': len(items),
            'firstItem': items[0] if items else None,
            'itemsWithGeomCount': len(items_with_geom),
            'distances': distances_sample,
            'beforeDistanceFilterCount': len(items_with_geom),
            'afterDistanceFilterCount': len(within30m),
            'within30mCount': len(within30m),
            'within100mCount': len(within100m),
            'within300mCount': len(within300m),
            'within500mCount': len(within500m),
        },
    })


@app.route('/api/busan-commercial/nearby-live', methods=['GET'])
def api_busan_commercial_nearby_paginated():
    service_key = os.getenv('BUSAN_COMMERCIAL_SERVICE_KEY')
    if not service_key:
        return jsonify({
            'status': 'error',
            'message': 'BUSAN_COMMERCIAL_SERVICE_KEY is not configured.',
        }), 500

    try:
        lat = float(request.args.get('lat', ''))
        lon = float(request.args.get('lon', ''))
        if math.isnan(lat) or math.isnan(lon) or math.isinf(lat) or math.isinf(lon):
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({
            'status': 'error',
            'message': 'Valid lat and lon query parameters are required.',
        }), 400

    address = (request.args.get('address') or '부산진구').strip() or '부산진구'
    fallback_address = (request.args.get('address') or '부산진구').strip() or '부산진구'
    address = fallback_address
    fallback_address = (request.args.get('address') or '\ubd80\uc0b0\uc9c4\uad6c').strip() or '\ubd80\uc0b0\uc9c4\uad6c'
    address = fallback_address
    search_terms, reverse_address = _build_busan_commercial_search_terms(lat, lon, fallback_address)
    collected_items = []
    seen_keys = set()
    term_summaries = []
    requested_pages = 0
    first_payload = None
    selected_terms = []
    probe_results = []
    search_filter_ignored = False

    try:
        for search_term in search_terms:
            page = _fetch_commercial_page(service_key, search_term, 1)
            total = _parse_positive_int(page['totalCount'])
            probe_results.append({
                'term': search_term,
                'totalCount': page['totalCount'],
                'parsedTotalCount': total,
                'firstPageItemsCount': len(page['items']),
            })

        positive_terms = [probe for probe in probe_results if probe['parsedTotalCount'] > 0]
        if positive_terms:
            unique_counts = {probe['parsedTotalCount'] for probe in positive_terms}
            if len(unique_counts) == 1 and len(positive_terms) > 1:
                search_filter_ignored = True
                selected_terms = [fallback_address]
            else:
                selected = min(positive_terms, key=lambda x: x['parsedTotalCount'])
                selected_terms = [selected['term']]
        else:
            selected_terms = [fallback_address]

        for search_term in selected_terms:
            term_result = _fetch_commercial_items_for_term(service_key, search_term)
            requested_pages += term_result['requestedPages']
            if first_payload is None:
                first_payload = term_result['firstPayload']
            term_items = term_result['items']
            for item in term_items:
                key = _commercial_item_key(item)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                collected_items.append(item)
            term_summaries.append({
                'term': search_term,
                'totalCount': term_result['totalCount'],
                'totalPages': term_result['totalPages'],
                'requestedPages': term_result['requestedPages'],
                'itemsCount': len(term_items),
            })
    except RuntimeError as exc:
        return jsonify({
            'status': 'error',
            'message': str(exc),
        }), 502
    except ValueError:
        return jsonify({
            'status': 'error',
            'message': 'Busan commercial API returned a non-JSON response.',
        }), 502
    except req_lib.RequestException as exc:
        return jsonify({
            'status': 'error',
            'message': f'Busan commercial API request failed: {exc}',
        }), 502

    items_with_geom = []
    stores = []
    for item in collected_items:
        point = parse_point_geom(item.get('geom'))
        if not point:
            continue
        items_with_geom.append({**item, 'lat': point['lat'], 'lon': point['lon']})
        store = _normalize_commercial_store(item, lat, lon)
        if store:
            stores.append(store)

    stores.sort(key=lambda x: x['distanceMeters'])
    groups = _group_commercial_stores(stores)
    within30m = [store for store in stores if store['distanceMeters'] <= 30]
    nearest_stores = stores[:10]
    nearest_debug = [
        {
            'name': store['name'],
            'address': store['address'],
            'distanceMeters': store['distanceMeters'],
            'lat': store['lat'],
            'lon': store['lon'],
        }
        for store in nearest_stores
    ]

    print('Busan API search terms:', search_terms, flush=True)
    print('Busan API probe results:', probe_results, flush=True)
    print('Selected search terms:', selected_terms, flush=True)
    print('Search filter ignored:', search_filter_ignored, flush=True)
    print('Reverse geocode address:', reverse_address, flush=True)
    print('Busan API term summaries:', term_summaries, flush=True)
    print('Requested pages:', requested_pages, flush=True)
    print('Fetched items count:', len(collected_items), flush=True)
    print('Items with geom:', len(items_with_geom), flush=True)
    print('Nearest stores TOP 10:', nearest_debug, flush=True)
    print('Within 30m:', len(within30m), flush=True)

    return jsonify({
        'status': 'success',
        'hasGps': True,
        'origin': {'lat': lat, 'lon': lon},
        'address': address,
        'summary': {
            'totalFetched': len(collected_items),
            'withGeom': len(items_with_geom),
            'within30mCount': len(within30m),
            'nearestDistanceMeters': nearest_stores[0]['distanceMeters'] if nearest_stores else None,
            'requestedPages': requested_pages,
            'searchTerms': search_terms,
            'selectedTerms': selected_terms,
            'searchFilterIgnored': search_filter_ignored,
            'termSummaries': term_summaries,
        },
        'groups': groups,
        'nearestStores': nearest_stores,
        'debug': {
            'apiResponse': first_payload,
            'searchTerms': search_terms,
            'selectedTerms': selected_terms,
            'searchFilterIgnored': search_filter_ignored,
            'probeResults': probe_results,
            'reverseAddress': reverse_address,
            'termSummaries': term_summaries,
            'requestedPages': requested_pages,
            'itemsCount': len(collected_items),
            'firstItem': collected_items[0] if collected_items else None,
            'itemsWithGeomCount': len(items_with_geom),
            'nearestStores': nearest_debug,
            'within30mCount': len(within30m),
        },
    })


@app.route('/api/busan-commercial/cache/status', methods=['GET'])
def api_busan_commercial_cache_status():
    cache_data = _load_busan_commercial_cache()
    items = cache_data.get('items') or []
    meta = cache_data.get('meta') or {}
    return jsonify({
        'status': 'success',
        'cacheReady': bool(items),
        'meta': meta,
        'itemsCount': len(items),
    })


@app.route('/api/busan-commercial/sync', methods=['GET', 'POST'])
def api_busan_commercial_sync():
    service_key = os.getenv('BUSAN_COMMERCIAL_SERVICE_KEY')
    if not service_key:
        return jsonify({
            'status': 'error',
            'message': 'BUSAN_COMMERCIAL_SERVICE_KEY is not configured.',
        }), 500

    search_term = (request.args.get('address') or request.args.get('searchTerm') or '').strip()
    try:
        cache_data = _sync_busan_commercial_cache(service_key, search_term)
    except RuntimeError as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 502
    except ValueError:
        return jsonify({
            'status': 'error',
            'message': 'Busan commercial API returned a non-JSON response.',
        }), 502
    except req_lib.RequestException as exc:
        return jsonify({
            'status': 'error',
            'message': f'Busan commercial API request failed: {exc}',
        }), 502

    return jsonify({
        'status': 'success',
        'cacheReady': True,
        'meta': cache_data.get('meta') or {},
        'itemsCount': len(cache_data.get('items') or []),
    })


@app.route('/api/busan-commercial/nearby', methods=['GET'])
def api_busan_commercial_nearby_cached():
    try:
        lat = float(request.args.get('lat', ''))
        lon = float(request.args.get('lon', ''))
        if math.isnan(lat) or math.isnan(lon) or math.isinf(lat) or math.isinf(lon):
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({
            'status': 'error',
            'message': 'Valid lat and lon query parameters are required.',
        }), 400

    cache_data = _load_busan_commercial_cache()
    cache_items = cache_data.get('items') or []
    cache_meta = cache_data.get('meta') or {}
    if not cache_items:
        return jsonify({
            'status': 'error',
            'code': 'CACHE_REQUIRED',
            'message': '부산 상권 캐시가 없습니다. 먼저 상점 데이터 업데이트를 실행해주세요.',
            'cacheReady': False,
        }), 409

    stores = _stores_with_distance_from_cache(cache_items, lat, lon)
    groups = _group_commercial_stores(stores)
    within30m = [store for store in stores if store['distanceMeters'] <= 30]
    nearest_stores = stores[:10]

    print('Busan commercial cache items:', len(cache_items), flush=True)
    print('Cache updated at:', cache_meta.get('updatedAt'), flush=True)
    print('Nearest stores TOP 10:', [
        {'name': store['name'], 'distanceMeters': store['distanceMeters']}
        for store in nearest_stores
    ], flush=True)
    print('Within 30m:', len(within30m), flush=True)

    return jsonify({
        'status': 'success',
        'hasGps': True,
        'origin': {'lat': lat, 'lon': lon},
        'cache': {
            'ready': True,
            'meta': cache_meta,
        },
        'summary': {
            'totalFetched': cache_meta.get('rawItemsCount'),
            'withGeom': cache_meta.get('withGeomCount'),
            'cacheItemsCount': len(cache_items),
            'within30mCount': len(within30m),
            'nearestDistanceMeters': nearest_stores[0]['distanceMeters'] if nearest_stores else None,
        },
        'groups': groups,
        'nearestStores': nearest_stores,
        'debug': {
            'cacheMeta': cache_meta,
            'itemsCount': len(cache_items),
            'nearestStores': [
                {
                    'name': store['name'],
                    'address': store['address'],
                    'distanceMeters': store['distanceMeters'],
                    'lat': store['lat'],
                    'lon': store['lon'],
                }
                for store in nearest_stores
            ],
            'within30mCount': len(within30m),
        },
    })


def decode_base64_image(data_url):
    """base64 이미지를 NumPy 배열과 원본 바이트(EXIF 보존)로 디코딩.
    EXIF orientation 태그를 적용해 회전/반전 보정 후 NumPy 배열 반환.
    raw_bytes는 GPS EXIF 추출을 위해 원본 그대로 보존.
    """
    encoded = data_url.split(',')[1] if ',' in data_url else data_url
    raw_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(raw_bytes))
    image = ImageOps.exif_transpose(image)  # EXIF 회전 태그 보정
    image = image.convert('RGB')
    return np.array(image), raw_bytes


@app.route('/api/analyze', methods=['POST'])
def analyze_api():
    data = request.get_json(silent=True) or {}
    image_data = data.get('image') or data.get('imageData')
    if not image_data:
        return jsonify({'status': 'error', 'message': 'No image data provided.'}), 400

    try:
        img, raw_bytes = decode_base64_image(image_data)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Image decode failed: {e}'}), 400

    # 1. 위치 정보 처리
    # 우선순위: 수동선택 > GPS EXIF > 자동감지된 카메라 기종 기본값
    zone_code = None
    gps_coords = None
    
    # 수동 선택된 위치 확인
    location_mode = data.get('locationMode', 'auto')
    location_zone = data.get('locationZone', '')
    if location_mode == 'manual' and location_zone:
        zone_code = normalize_zone_code(location_zone)
    
    # GPS 정보 추출 (수동선택이 없는 경우만)
    if not zone_code:
        req_lat = data.get('gpsLat')
        req_lon = data.get('gpsLon')
        if req_lat is not None and req_lon is not None:
            try:
                gps_coords = (float(req_lat), float(req_lon))
            except (TypeError, ValueError):
                gps_coords = None
        else:
            gps_coords = extract_gps_from_exif(raw_bytes)
        zone_code = get_zone_from_gps(*gps_coords) if gps_coords else None
    
    capture_context = build_capture_context(data)

    # 2. GPS와 위치 선택 모두 없을 때만 4개 구역 전체 시뮬레이션
    # 사용자가 수동으로 지역을 선택했으면 all_zones_mode는 False (우선순위 존중)
    all_zones_mode = False
    if location_mode != 'manual' and not zone_code:
        # 자동감지 모드이면서 GPS도 없는 경우에만 all_zones_mode 활성화
        all_zones_mode = True
        zone_code = '제3종'  # 탐지 필터링용 임시값

    # 전처리
    img_proc = preprocess_image(img)
    h, w = img.shape[:2]
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    detected = []

    if MODEL is not None:
        try:
            results = MODEL(img_proc, imgsz=640, conf=0.25)
            out = results[0]
            boxes = out.boxes
            names = out.names
            for b in boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cls_id = int(b.cls[0])
                conf = float(b.conf[0]) if hasattr(b, 'conf') else 0.5
                label = names.get(cls_id, f'class_{cls_id}')
                cropped = img[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
                if cropped.size == 0:
                    continue

                (brightness, luminance_cd_m2_avg, illuminance_lux_avg,
                 saturation, gamma,
                 luminance_cd_m2_max, illuminance_lux_max,
                 brightness_p95, bright_pixel_ratio) = analyze_region_metrics(cropped)

                raw_metrics = {
                    'brightness': brightness,
                    'brightness_p95': brightness_p95,
                    'brightness_max': float(np.max(cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY))),
                    'bright_pixel_ratio': bright_pixel_ratio,
                    'saturation': saturation,
                    'gamma': gamma,
                    'luminance_cd_m2_avg': luminance_cd_m2_avg,
                    'luminance_cd_m2_max': luminance_cd_m2_max,
                    'illuminance_lux_avg': illuminance_lux_avg,
                    'illuminance_lux_max': illuminance_lux_max,
                }
                adjusted_metrics = apply_capture_adjustment(raw_metrics, capture_context)

                # 카테고리 분류
                if label in DIRECT_LIGHT_CLASSES:
                    cat = label
                elif label in COCO_TO_KR:
                    cat = COCO_TO_KR[label]
                else:
                    cat = classify_by_geometry(x1, y1, x2, y2, h, w)

                ocr_result = {
                    'storeName': None,
                    'ocrText': '',
                    'ocrConfidence': 0.0,
                    'ocrStatus': 'not a signboard',
                }
                if cat == '간판' or label in ('street sign', 'stop sign', 'light_signboard'):
                    ocr_result = recognize_signboard(cropped)

                light_type = LIGHT_TYPE_MAP.get(cat, '장식조명')
                pollution_category = classify_pollution_category(
                    cat,
                    adjusted_metrics['brightness'], adjusted_metrics['brightness_p95'],
                    adjusted_metrics['bright_pixel_ratio'], adjusted_metrics['saturation'], adjusted_metrics['gamma'],
                    x1, y1, x2, y2, h, w
                )
                if all_zones_mode:
                    zone_results = {
                        zc: compute_fine(adjusted_metrics['luminance_cd_m2_avg'], adjusted_metrics['luminance_cd_m2_max'],
                                         adjusted_metrics['illuminance_lux_max'], light_type, zc)
                        for zc in ('제1종', '제2종', '제3종', '제4종')
                    }
                    fine = zone_results['제3종']  # 오버레이 표시용 기본값
                else:
                    zone_results = None
                    fine = compute_fine(
                        adjusted_metrics['luminance_cd_m2_avg'], adjusted_metrics['luminance_cd_m2_max'],
                        adjusted_metrics['illuminance_lux_max'], light_type, zone_code
                    )

                detected.append({
                    'name':             label,
                    'type':             cat,
                    'lightType':        light_type,
                    'confidence':       round(conf, 3),
                    'storeName':        ocr_result['storeName'],
                    'ocrText':          ocr_result['ocrText'],
                    'ocrConfidence':    ocr_result['ocrConfidence'],
                    'ocrStatus':        ocr_result['ocrStatus'],
                    'brightness':       int(round(adjusted_metrics['brightness'])),
                    'rawBrightness':    int(round(brightness)),
                    'luminanceCdM2':    round(adjusted_metrics['luminance_cd_m2_avg'], 1),
                    'rawLuminanceCdM2': round(luminance_cd_m2_avg, 1),
                    'luminanceCdM2Max': round(adjusted_metrics['luminance_cd_m2_max'], 1),
                    'rawLuminanceCdM2Max': round(luminance_cd_m2_max, 1),
                    'illuminanceLux':   round(adjusted_metrics['illuminance_lux_avg'], 1),
                    'rawIlluminanceLux': round(illuminance_lux_avg, 1),
                    'illuminanceLuxMax':round(adjusted_metrics['illuminance_lux_max'], 1),
                    'rawIlluminanceLuxMax': round(illuminance_lux_max, 1),
                    'brightnessP95':    round(adjusted_metrics['brightness_p95'], 1),
                    'rawBrightnessP95': round(brightness_p95, 1),
                    'brightPixelRatio': round(adjusted_metrics['bright_pixel_ratio'], 1),
                    'rawBrightPixelRatio': round(bright_pixel_ratio, 1),
                    'saturation':       round(adjusted_metrics['saturation'], 2),
                    'rawSaturation':    round(saturation, 2),
                    'gamma':            round(adjusted_metrics['gamma'], 2),
                    'rawGamma':         round(gamma, 2),
                    'pollutionCategory': pollution_category,
                    'pollutionCategoryDesc': POLLUTION_TYPES[pollution_category],
                    'measurementNote':  'cd/m², lux 값은 이미지 기반 참고용 추정치입니다.',
                    'compliance':       fine['compliance'],
                    'violationStage':   fine['violationStage'],
                    'fineLabel':        fine.get('fineLabel'),
                    'fineAmount':       fine['fineAmount'],
                    'measuredValue':    fine['measuredValue'],
                    'threshold':        fine['threshold'],
                    'unit':             fine['unit'],
                    'basis':            fine['basis'],
                    'ratio':            fine['ratio'],
                    'zoneResults':      zone_results,
                    'captureContext':   capture_context,
                    'captureApplied':   {
                        'brightnessScale': capture_context['brightnessScale'],
                        'glareScale': capture_context['glareScale'],
                        'pixelRatioScale': capture_context['pixelRatioScale'],
                    },
                    'box': {
                        'x':      int(max(0, x1 / w * 100)),
                        'y':      int(max(0, y1 / h * 100)),
                        'width':  int(max(5, (x2 - x1) / w * 100)),
                        'height': int(max(5, (y2 - y1) / h * 100)),
                    }
                })

            # 광원 관련 객체(간판·가로등·조명)만 유지, 너무 어두운 객체 제외
            detected = [d for d in detected if should_keep_detection(d)]

        except Exception:
            detected = []

    # GPS 없는 경우 전체 구역 요약 계산
    zones_summary = build_default_zones_summary() if all_zones_mode else {}
    if all_zones_mode and detected:
        for zc in ('제1종', '제2종', '제3종', '제4종'):
            zv = [d['zoneResults'][zc] for d in detected if d.get('zoneResults')]
            zv_violations = [r for r in zv if r['compliance'] == '위반']
            zv_stages = [r['violationStage'] for r in zv_violations if r['violationStage']]
            zv_max = next((s for s in ('3단계', '2단계', '1단계') if s in zv_stages), None)
            zones_summary[zc] = {
                'zoneLabel':       ZONE_LABELS[zc],
                'overall':         zv_max or '준수',
                'totalFineAmount': sum(r['fineAmount'] for r in zv_violations),
                'violationCount':  len(zv_violations),
            }

    pollution_summary = summarize_pollution_categories(detected)

    # 탐지 실패
    if not detected:
        return jsonify({
            'status':        'success',
            'overall':       '미탐지',
            'detected':      [],
            'riskSummary':   '광원 객체가 탐지되지 않았습니다.',
            'zone':          zone_code,
            'zoneLabel':     ZONE_LABELS.get(zone_code, zone_code),
            'gpsDetected':   gps_coords is not None,
            'allZonesMode':  all_zones_mode,
            'zonesSummary':  zones_summary,
            'overallPollutionCategory': pollution_summary['overall'],
            'pollutionCategorySummary': pollution_summary,
            'captureContext': capture_context,
            'avgBrightness': int(np.mean(gray_img)),
            'model':         MODEL_STATUS,
            'detectionSource': 'none',
        })

    # ---- 전체 결과 집계 ----
    violations = [d for d in detected if d['compliance'] == '위반']
    stages = [d['violationStage'] for d in violations if d['violationStage']]
    max_stage = next((s for s in ('3단계', '2단계', '1단계') if s in stages), None)
    total_fine = sum(d['fineAmount'] for d in violations)
    avg_brightness = float(np.mean(gray_img))

    return jsonify({
        'status':           'success',
        'overall':          max_stage or '준수',
        'totalFineAmount':  total_fine,
        'violationCount':   len(violations),
        'detected':         detected,
        'riskSummary':      (
            f'GPS 미확인 — 4개 구역 시뮬레이션 (제3종 기준 최대 {total_fine}만원) · 유형: {pollution_summary["overall"]} · cd/m²/lux는 참고용 추정치'
            if all_zones_mode else (
                f'{max_stage} 위반 — 과태료 {total_fine}만원 수준 · 유형: {pollution_summary["overall"]} · cd/m²/lux는 참고용 추정치'
                if max_stage else f'법규 준수 · 유형: {pollution_summary["overall"]} · cd/m²/lux는 참고용 추정치'
            )
        ),
        'avgBrightness':    int(avg_brightness),
        'zone':             zone_code,
        'zoneLabel':        ZONE_LABELS.get(zone_code, zone_code),
        'gpsDetected':      gps_coords is not None,
        'allZonesMode':     all_zones_mode,
        'zonesSummary':     zones_summary,
        'overallPollutionCategory': pollution_summary['overall'],
        'pollutionCategorySummary': pollution_summary,
        'captureContext':   capture_context,
        'model':            MODEL_STATUS,
        'detectionSource':  'yolo',
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
