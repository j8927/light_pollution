from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import base64
import io
import cv2
import numpy as np
from PIL import Image, ImageOps
import requests as req_lib  # Nominatim 역지오코딩용

app = Flask(__name__)
CORS(app)

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

# YOLO 탐지 카테고리 → 조명 법규 유형 매핑
LIGHT_TYPE_MAP = {
    '가로등': '공간조명',   # 조도(lux) 최대값 기준
    '간판':  '광고물',      # 휘도(cd/m²) 최대값 기준
    '조명':  '장식조명',    # 휘도(cd/m²) 평균/최대값 기준
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
    r = cropped[:, :, 0].astype(np.float32)
    g = cropped[:, :, 1].astype(np.float32)
    b = cropped[:, :, 2].astype(np.float32)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    brightness_avg = float(np.mean(lum))
    brightness_max = float(np.max(lum))
    hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255
    saturation = float(np.mean(s))
    gamma = float(np.clip(1.8 + np.std(lum) / 64.0, 1.0, 3.5))
    luminance_cd_m2_avg = estimate_luminance_cd_m2(brightness_avg)
    luminance_cd_m2_max = estimate_luminance_cd_m2(brightness_max)
    illuminance_lux_avg = estimate_illuminance_lux(brightness_avg)
    illuminance_lux_max = estimate_illuminance_lux(brightness_max)
    return (brightness_avg, luminance_cd_m2_avg, illuminance_lux_avg,
            saturation, gamma, luminance_cd_m2_max, illuminance_lux_max)


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

    # 1. EXIF GPS로 지역 자동 판별
    # JS에서 canvas 압축 시 EXIF가 제거되므로, 프론트가 미리 추출한 좌표를 우선 사용
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

    # 2. GPS 없을 때 요청 파라미터 확인
    explicit_zone = data.get('zone') or data.get('regionType')
    all_zones_mode = False
    if not zone_code:
        if explicit_zone:
            zone_code = normalize_zone_code(explicit_zone)
        else:
            # GPS도 없고 zone 파라미터도 없음 → 4개 구역 전체 시뮬레이션
            all_zones_mode = True
            zone_code = '제3종'  # 탐지 필터링용 임시값

    # 전처리
    img_proc = preprocess_image(img)
    h, w = img.shape[:2]
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
                label = names.get(cls_id, f'class_{cls_id}')
                cropped = img[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
                if cropped.size == 0:
                    continue

                (brightness, luminance_cd_m2_avg, illuminance_lux_avg,
                 saturation, gamma,
                 luminance_cd_m2_max, illuminance_lux_max) = analyze_region_metrics(cropped)

                # 카테고리 분류
                if label in DIRECT_LIGHT_CLASSES:
                    cat = label
                elif label in COCO_TO_KR:
                    cat = COCO_TO_KR[label]
                else:
                    cat = classify_by_geometry(x1, y1, x2, y2, h, w)

                light_type = LIGHT_TYPE_MAP.get(cat, '장식조명')
                if all_zones_mode:
                    zone_results = {
                        zc: compute_fine(luminance_cd_m2_avg, luminance_cd_m2_max,
                                         illuminance_lux_max, light_type, zc)
                        for zc in ('제1종', '제2종', '제3종', '제4종')
                    }
                    fine = zone_results['제3종']  # 오버레이 표시용 기본값
                else:
                    zone_results = None
                    fine = compute_fine(
                        luminance_cd_m2_avg, luminance_cd_m2_max,
                        illuminance_lux_max, light_type, zone_code
                    )

                detected.append({
                    'name':             label,
                    'type':             cat,
                    'lightType':        light_type,
                    'brightness':       int(brightness),
                    'luminanceCdM2':    round(luminance_cd_m2_avg, 1),
                    'luminanceCdM2Max': round(luminance_cd_m2_max, 1),
                    'illuminanceLux':   round(illuminance_lux_avg, 1),
                    'illuminanceLuxMax':round(illuminance_lux_max, 1),
                    'saturation':       round(saturation, 2),
                    'gamma':            round(gamma, 2),
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
                    'box': {
                        'x':      int(max(0, x1 / w * 100)),
                        'y':      int(max(0, y1 / h * 100)),
                        'width':  int(max(5, (x2 - x1) / w * 100)),
                        'height': int(max(5, (y2 - y1) / h * 100)),
                    }
                })

            # 광원 관련 객체(간판·가로등·조명)만 유지, 너무 어두운 객체 제외
            detected = [
                d for d in detected
                if d['type'] in ('간판', '가로등', '조명') and d['brightness'] >= 60
            ]
        except Exception:
            detected = []

    # GPS 없는 경우 전체 구역 요약 계산
    zones_summary = {}
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
            'zonesSummary':  {},
            'avgBrightness': int(np.mean(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))),
            'model':         MODEL_STATUS,
            'detectionSource': 'none',
        })

    # ---- 전체 결과 집계 ----
    violations = [d for d in detected if d['compliance'] == '위반']
    stages = [d['violationStage'] for d in violations if d['violationStage']]
    max_stage = next((s for s in ('3단계', '2단계', '1단계') if s in stages), None)
    total_fine = sum(d['fineAmount'] for d in violations)
    avg_brightness = float(np.mean(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)))

    return jsonify({
        'status':           'success',
        'overall':          max_stage or '준수',
        'totalFineAmount':  total_fine,
        'violationCount':   len(violations),
        'detected':         detected,
        'riskSummary':      (
            f'GPS 미확인 — 구역별 시뮬레이션 (제3종 기준 최대 {total_fine}만원)'
            if all_zones_mode else (
                f'{max_stage} 위반 — 과태료 {total_fine}만원 수준'
                if max_stage else '법규 준수'
            )
        ),
        'avgBrightness':    int(avg_brightness),
        'zone':             zone_code,
        'zoneLabel':        ZONE_LABELS.get(zone_code, zone_code),
        'gpsDetected':      gps_coords is not None,
        'allZonesMode':     all_zones_mode,
        'zonesSummary':     zones_summary,
        'model':            MODEL_STATUS,
        'detectionSource':  'yolo',
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
