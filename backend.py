from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import base64
import io
import cv2
import numpy as np
from PIL import Image

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


def normalize_zone(zone_value):
    z = (zone_value or '').strip().lower()
    if z in ('residential', '주거', '주거지역'):
        return '주거지역'
    if z in ('commercial', '상업', '상업지역'):
        return '상업지역'
    return '상업지역'


def estimate_luminance_cd_m2(luminance_0_255):
    # 이미지 휘도(0~255) -> 현장 휘도(cd/m^2) 근사 변환
    # 실제 계측기값과 다를 수 있으며, 운영 시 캘리브레이션 권장
    norm = float(np.clip(luminance_0_255 / 255.0, 0.0, 1.0))
    return float((norm ** 1.15) * 1200.0)


def estimate_illuminance_lux(luminance_0_255):
    # 이미지 밝기(0~255) -> 조도(lux) 근사 변환
    norm = float(np.clip(luminance_0_255 / 255.0, 0.0, 1.0))
    return float((norm ** 1.05) * 60.0)


def _white_balance(img):
    """Gray World 화이트 밸런스 보정 — 색온도 편향 제거"""
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
    # 1. 화이트 밸런스
    img_wb = _white_balance(img)

    # 2. 양방향 필터 — 엣지(간판 테두리)는 유지하면서 배경 노이즈 제거
    img_bgr = cv2.cvtColor(img_wb, cv2.COLOR_RGB2BGR)
    denoised = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)

    # 3. CLAHE — 어두운 영역 대비 강화 (과다 증폭 방지: clipLimit=2.0)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b_ch])
    result_bgr = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def analyze_region_metrics(cropped):
    # 휘도(Y) = 0.2126R + 0.7152G + 0.0722B (광감도 기준)
    r = cropped[:, :, 0].astype(np.float32)
    g = cropped[:, :, 1].astype(np.float32)
    b = cropped[:, :, 2].astype(np.float32)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    brightness = float(np.mean(lum))
    hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255
    saturation = float(np.mean(s))
    gamma = float(np.clip(1.8 + np.std(lum) / 64.0, 1.0, 3.5))
    luminance_cd_m2 = estimate_luminance_cd_m2(brightness)
    illuminance_lux = estimate_illuminance_lux(brightness)
    return brightness, luminance_cd_m2, illuminance_lux, saturation, gamma


def classify_by_geometry(x1, y1, x2, y2, img_h, img_w):
    """
    YOLO가 탐지한 바운딩박스의 기하학적 특성으로 객체 종류를 분류.
    - 가로 길이 > 세로 × 1.8  → 간판 (가로형 직사각형)
    - 이미지 상단 45% + 세로형(aspect ≤ 0.9) → 가로등 (수직 기둥)
    - 그 외                   → 조명
    """
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    aspect = bw / bh
    if aspect >= 1.8:
        return '간판'
    if y1 < img_h * 0.45 and aspect <= 0.9:
        return '가로등'
    return '조명'


def detect_light_objects_cv(img_proc, img_orig, zone):
    # YOLO가 놓친 경우를 대비한 자동 탐지 fallback (수동 아님)
    # img_proc: 전처리된 이미지 (컨투어 탐지용)
    # img_orig: 원본 이미지 (휘도/측정값 산출용)
    h, w = img_proc.shape[:2]
    lum = (0.2126 * img_proc[:, :, 0].astype(np.float32)
           + 0.7152 * img_proc[:, :, 1].astype(np.float32)
           + 0.0722 * img_proc[:, :, 2].astype(np.float32))
    gray = np.clip(lum, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # 이미지마다 다른 밝기 조건을 맞추기 위한 동적 임계값
    th = max(160, int(np.percentile(blur, 92)))
    _, mask = cv2.threshold(blur, th, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    detected = []
    min_area = (h * w) * 0.0008

    for c in contours[:10]:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 8 or bh < 8:
            continue

        # 측정값은 원본 이미지에서 추출 (전처리로 인한 휘도 왜곡 방지)
        cropped = img_orig[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]
        if cropped.size == 0:
            continue

        brightness, luminance_cd_m2, illuminance_lux, saturation, gamma = analyze_region_metrics(cropped)
        aspect = bw / max(1, bh)

        if aspect >= 1.8:
            cat = '간판'
            name = 'cv_signboard'
        elif y < h * 0.45 and aspect <= 0.9:
            cat = '가로등'
            name = 'cv_streetlight'
        else:
            cat = '조명'
            name = 'cv_light'

        risk_score, risk_level, compliance, threshold, law_unit, measured_value = compute_law_risk(
            luminance_cd_m2,
            illuminance_lux,
            saturation,
            gamma,
            cat,
            zone
        )
        detected.append({
            'name': name,
            'type': cat,
            'brightness': int(brightness),
            'luminanceCdM2': round(luminance_cd_m2, 1),
            'illuminanceLux': round(illuminance_lux, 1),
            'measuredValue': round(measured_value, 1),
            'saturation': round(saturation, 2),
            'gamma': round(gamma, 2),
            'riskLevel': risk_level,
            'compliance': compliance,
            'lawUnit': law_unit,
            'lawThreshold': threshold,
            'lawThresholdCdM2': threshold,
            'box': {
                'x': int(max(0, x / w * 100)),
                'y': int(max(0, y / h * 100)),
                'width': int(max(5, bw / w * 100)),
                'height': int(max(5, bh / h * 100))
            }
        })

    return detected[:5]


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
    if ',' in data_url:
        data_url = data_url.split(',')[1]
    b = base64.b64decode(data_url)
    image = Image.open(io.BytesIO(b)).convert('RGB')
    return np.array(image)


def compute_law_risk(luminance_cd_m2, illuminance_lux, saturation, gamma, category, zone):
    # 휘도(cd/m^2) 중심 법규 판단
    # - luminance_cd_m2: 추정 휘도(cd/m^2)
    # - saturation: 0-1
    # - gamma: 1.0-3.5

    # 간판/전광판은 휘도(cd/m²), 가로등은 조도(lux) 기준
    cd_thresholds = {
        '상업지역': {'간판': 800, '조명': 650, '구조물': 450},
        '주거지역': {'간판': 300, '조명': 220, '구조물': 160}
    }
    lux_thresholds = {
        '상업지역': {'가로등': 30},
        '주거지역': {'가로등': 20}
    }
    current_zone = normalize_zone(zone)

    if category == '가로등':
        threshold = lux_thresholds[current_zone]['가로등']
        measured_value = illuminance_lux
        law_unit = 'lux'
    else:
        threshold = cd_thresholds.get(current_zone, cd_thresholds['상업지역']).get(category, cd_thresholds[current_zone]['조명'])
        measured_value = luminance_cd_m2
        law_unit = 'cd/m²'

    base = 20
    norm_value = float(np.clip(measured_value / max(1.0, threshold), 0.0, 3.0))
    base += saturation * 8
    base += max(0, gamma - 1.8) * 8

    if category in ('간판', '조명', '가로등', '구조물'):
        base += 4
    elif category == '사람' or category == '차량':
        base += 8

    # 휘도 기준 적합 여부
    compliance = '준수' if measured_value <= threshold else '위반'

    if compliance == '준수':
        # 기준 이하면 위험 상한 제한 (과대 판정 방지)
        base += min(12, norm_value * 12)
        base = min(base, 58)
    else:
        exceed_ratio = min(2.0, (measured_value - threshold) / max(1.0, threshold))
        base = 60 + min(35, exceed_ratio * 40) + saturation * 6 + max(0, gamma - 1.8) * 6

    if base >= 85:
        level = '고위험'
    elif base >= 60:
        level = '주의'
    else:
        level = '관찰'

    return min(100, int(base)), level, compliance, threshold, law_unit, measured_value


@app.route('/api/analyze', methods=['POST'])
def analyze_api():
    data = request.get_json(silent=True) or {}
    image_data = data.get('image') or data.get('imageData')
    zone = normalize_zone(data.get('zone') or data.get('regionType') or '상업지역')
    if not image_data:
        return jsonify({'status': 'error', 'message': 'No image data provided.'}), 400

    try:
        img = decode_base64_image(image_data)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Image decode failed: {e}'}), 400

    # 탐지 정확도 향상을 위한 전처리 (원본은 측정값 산출에 유지)
    img_proc = preprocess_image(img)
    h, w = img.shape[:2]
    detected = []
    detection_source = 'none'

    if MODEL is not None:
        try:
            results = MODEL(img_proc, imgsz=640, conf=0.25)
            out = results[0]
            boxes = out.boxes
            names = out.names
            for b in boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                label = names.get(cls_id, f'class_{cls_id}')
                # YOLO 박스 좌표는 전처리 이미지 기준이나 크기 동일 → 원본에서 crop
                cropped = img[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
                if cropped.size == 0:
                    continue
                brightness, luminance_cd_m2, illuminance_lux, saturation, gamma = analyze_region_metrics(cropped)
                # 재학습한 커스텀 모델이 한국어 클래스를 직접 반환하면 그대로 사용한다.
                # 단일 클래스(light_object)나 COCO 미매핑 클래스만 기하학 fallback으로 분류한다.
                if label in DIRECT_LIGHT_CLASSES:
                    cat = label
                elif label in COCO_TO_KR:
                    cat = COCO_TO_KR[label]
                else:
                    cat = classify_by_geometry(x1, y1, x2, y2, h, w)
                risk_score, risk_level, compliance, threshold, law_unit, measured_value = compute_law_risk(
                    luminance_cd_m2,
                    illuminance_lux,
                    saturation,
                    gamma,
                    cat,
                    zone
                )
                detected.append({
                    'name': f'{label}',
                    'type': cat,
                    'brightness': int(brightness),
                    'luminanceCdM2': round(luminance_cd_m2, 1),
                    'illuminanceLux': round(illuminance_lux, 1),
                    'measuredValue': round(measured_value, 1),
                    'saturation': round(saturation, 2),
                    'gamma': round(gamma, 2),
                    'riskLevel': risk_level,
                    'compliance': compliance,
                    'lawUnit': law_unit,
                    'lawThreshold': threshold,
                    'lawThresholdCdM2': threshold,
                    'box': {
                        'x': int(max(0, x1 / w * 100)),
                        'y': int(max(0, y1 / h * 100)),
                        'width': int(max(5, (x2 - x1) / w * 100)),
                        'height': int(max(5, (y2 - y1) / h * 100))
                    }
                })
            # 사람/차량 등 비광원 객체만 탐지되는 경우를 제외하고 광원 관련 객체 우선 사용
            detected = [d for d in detected if d['type'] in ('간판', '가로등', '조명') and d['brightness'] >= 60]
            if detected:
                detection_source = 'yolo'
        except Exception as e:
            detected = []

    # YOLO 결과가 없으면 OpenCV 기반 자동 탐지 fallback 수행
    if not detected:
        detected = detect_light_objects_cv(img_proc, img, zone)
        if detected:
            detection_source = 'cv-fallback'

    if not detected:
        # 객체가 아예 검출되지 않으면 기본 관찰 low risk
        brightness = float(np.mean(0.2126 * img[:, :, 0].astype(np.float32)
                                   + 0.7152 * img[:, :, 1].astype(np.float32)
                                   + 0.0722 * img[:, :, 2].astype(np.float32)))
        gamma = 1.8
        base_cd = estimate_luminance_cd_m2(brightness)
        base_lux = estimate_illuminance_lux(brightness)
        default_threshold = 800 if zone == '상업지역' else 300
        risk_score, risk_level, compliance, threshold = 20, '관찰', '준수', default_threshold
        detection_source = 'none'
        detected = [{
            'name': '객체 미검출',
            'type': '조명',
            'brightness': int(brightness),
            'luminanceCdM2': round(base_cd, 1),
            'illuminanceLux': round(base_lux, 1),
            'measuredValue': round(base_cd, 1),
            'saturation': 0.2,
            'gamma': round(gamma, 2),
            'riskLevel': risk_level,
            'compliance': compliance,
            'lawUnit': 'cd/m²',
            'lawThreshold': threshold,
            'lawThresholdCdM2': threshold,
            'box': {'x': 10, 'y': 10, 'width': 80, 'height': 80}
        }]

    avg_brightness = float(np.mean(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)))
    gamma = float(np.mean([obj.get('gamma', 1.8) for obj in detected]))
    risk_score = int(np.mean([
        compute_law_risk(
            obj.get('luminanceCdM2', estimate_luminance_cd_m2(obj.get('brightness', 0))),
            obj.get('illuminanceLux', estimate_illuminance_lux(obj.get('brightness', 0))),
            obj.get('saturation', 0.0),
            obj.get('gamma', 1.8),
            obj.get('type', '조명'),
            zone
        )[0]
        for obj in detected
    ]))
    risk_level = '고위험' if risk_score >= 80 else '주의' if risk_score >= 60 else '관찰'
    avg_luminance_cd_m2 = float(np.mean([obj.get('luminanceCdM2', estimate_luminance_cd_m2(obj.get('brightness', 0))) for obj in detected]))

    return jsonify({
        'status': 'success',
        'confidence': min(99, 60 + risk_score // 2),
        'violation': risk_score,
        'overall': risk_level,
        'detected': detected,
        'riskSummary': f'{risk_level} 위험 ({risk_score}%)',
        'avgBrightness': int(avg_brightness),
        'avgLuminanceCdM2': round(avg_luminance_cd_m2, 1),
        'zone': zone,
        'gamma': round(gamma, 2),
        'model': MODEL_STATUS,
        'detectionSource': detection_source
    })


@app.route('/api/status', methods=['GET'])
def status_api():
    return jsonify({
        'status': 'success',
        'model': MODEL_STATUS,
        'customModel': os.path.exists('models/light_pollution_best.pt')
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', '5000')), debug=True)
