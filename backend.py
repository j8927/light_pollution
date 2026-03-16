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
    'parking meter': '조명',
    'bench': '구조물',
    'person': '사람',
    'car': '차량'
}


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


def compute_law_risk(brightness, saturation, gamma, category):
    # 실제 야간 간판/조명 규제 초안 기반 유사 계산
    # (실제 법규 단위값은 측정 장비/거리에 따라 다름, 여기서는 이미지 기반 활성화 예시)
    # - brightness: 0-255 (이미지 픽셀 밝기) -> 가중치
    # - saturation: 0-1
    # - gamma: 1.0-3.5

    # 법규 임계치 예시
    legal_thresholds = {
        '간판': {'lux': 120, 'score': 45},
        '조명': {'lux': 90, 'score': 35},
        '가로등': {'lux': 100, 'score': 38},
        '구조물': {'lux': 80, 'score': 30}
    }
    base = 20
    # 밝기는 normalized luminance
    norm_brightness = brightness / 255.0
    base += norm_brightness * 40
    base += saturation * 20
    base += max(0, gamma - 1.5) * 15

    if category in legal_thresholds:
        base += legal_thresholds[category]['score'] * (norm_brightness > 0.45)
    elif category == '사람' or category == '차량':
        base += 15

    # 법규 기준 적합 여부
    threshold = legal_thresholds.get(category, {'lux': 100})['lux']
    # 간단히 이미지 기준으로 luminance 기준 비교
    compliance = '준수' if brightness < threshold * 2 else '위반'  # 이미지 밝기 to lux 근사

    if base >= 85:
        level = '고위험'
    elif base >= 60:
        level = '주의'
    else:
        level = '관찰'

    return min(100, int(base)), level, compliance, threshold


@app.route('/api/analyze', methods=['POST'])
def analyze_api():
    data = request.get_json(silent=True) or {}
    image_data = data.get('image') or data.get('imageData')
    if not image_data:
        return jsonify({'status': 'error', 'message': 'No image data provided.'}), 400

    try:
        img = decode_base64_image(image_data)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Image decode failed: {e}'}), 400

    h, w = img.shape[:2]
    detected = []

    if MODEL is not None:
        try:
            results = MODEL(img, imgsz=640, conf=0.25)
            out = results[0]
            boxes = out.boxes
            names = out.names
            for b in boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                label = names.get(cls_id, f'class_{cls_id}')
                cropped = img[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
                if cropped.size == 0:
                    continue
                # 휘도(Y) = 0.2126R + 0.7152G + 0.0722B (광감도 기준)
                r, g, b = cropped[:, :, 0].astype(np.float32), cropped[:, :, 1].astype(np.float32), cropped[:, :, 2].astype(np.float32)
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                brightness = float(np.mean(lum))
                hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
                s = hsv[:, :, 1].astype(np.float32) / 255
                saturation = float(np.mean(s))
                gamma = float(np.clip(1.8 + np.std(lum) / 64.0, 1.0, 3.5))
                cat = COCO_TO_KR.get(label, '조명')
                risk_score, risk_level, compliance, threshold = compute_law_risk(brightness, saturation, gamma, cat)
                detected.append({
                    'name': f'{label}',
                    'type': cat,
                    'brightness': int(brightness),
                    'saturation': round(saturation, 2),
                    'gamma': round(gamma, 2),
                    'riskLevel': risk_level,
                    'compliance': compliance,
                    'lawThreshold': threshold,
                    'box': {
                        'x': int(max(0, x1 / w * 100)),
                        'y': int(max(0, y1 / h * 100)),
                        'width': int(max(5, (x2 - x1) / w * 100)),
                        'height': int(max(5, (y2 - y1) / h * 100))
                    }
                })
        except Exception as e:
            detected = []
    
    if not detected:
        # 객체가 아예 검출되지 않으면 기본 관찰 low risk
        brightness = float(np.mean(0.2126 * img[:, :, 0].astype(np.float32)
                                   + 0.7152 * img[:, :, 1].astype(np.float32)
                                   + 0.0722 * img[:, :, 2].astype(np.float32)))
        gamma = 1.8
        risk_score, risk_level, compliance, threshold = 20, '관찰', '준수', 90
        detected = [{
            'name': '객체 미검출',
            'type': '조명',
            'brightness': int(brightness),
            'saturation': 0.2,
            'gamma': round(gamma, 2),
            'riskLevel': risk_level,
            'compliance': compliance,
            'lawThreshold': threshold,
            'box': {'x': 10, 'y': 10, 'width': 80, 'height': 80}
        }]

    avg_brightness = float(np.mean(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)))
    gamma = float(np.mean([obj.get('gamma', 1.8) for obj in detected]))
    risk_score = int(np.mean([compute_law_risk(obj['brightness'], obj['saturation'], obj['gamma'], obj['type'])[0] for obj in detected]))
    risk_level = '고위험' if risk_score >= 80 else '주의' if risk_score >= 60 else '관찰'

    return jsonify({
        'status': 'success',
        'confidence': min(99, 60 + risk_score // 2),
        'violation': risk_score,
        'overall': risk_level,
        'detected': detected,
        'riskSummary': f'{risk_level} 위험 ({risk_score}%)',
        'avgBrightness': int(avg_brightness),
        'gamma': round(gamma, 2),
        'model': MODEL_STATUS
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
