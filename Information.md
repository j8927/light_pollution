# 빛 공해 법규 위반 탐지 시스템 — 프로젝트 정보

> 발표 준비·계획 검토·기술 설명을 위한 종합 문서

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [빛 공해 개념 정리](#2-빛-공해-개념-정리)
3. [요구사항 분석](#3-요구사항-분석)
4. [구현 계획 및 아키텍처](#4-구현-계획-및-아키텍처)
5. [현재 구현된 핵심 기술 설명](#5-현재-구현된-핵심-기술-설명)
6. [시스템 흐름도](#6-시스템-흐름도)
7. [테스트 및 검증 계획](#7-테스트-및-검증-계획)
8. [기대효과 및 활용방안](#8-기대효과-및-활용방안)
9. [향후 개선 방향](#9-향후-개선-방향)
10. [객체탐지 모델 비교 실험](#10-객체탐지-모델-비교-실험)
11. [참고문헌 및 법령](#11-참고문헌-및-법령)

---

## 1. 프로젝트 개요

### 프로젝트 명
**실시간 인공조명 빛 공해 법규 위반 탐지 및 판정 시스템 구현**

### 프로젝트 목적

최근 도심 지역의 무분별한 인공 조명 사용은 단순한 시각적 불편을 넘어 수면 장애, 생태계 교란, 농작물 수확량 감소 등 심각한 사회적·환경적 문제를 야기하고 있다.

한국은 세계적으로 빛 공해 노출도가 매우 높은 국가로 분류되어 **'인공조명에 의한 빛공해 방지법'** 을 시행 중이나, 실제 단속 현장에서는 고가의 휘도계 장비를 지참한 인력이 일일이 수동 측정해야 하는 행정적 한계가 존재한다.

본 프로젝트는 이 문제를 해결하기 위해 아래 과정을 구현한다.

- 최신 객체 탐지 모델(YOLOv8)로 야간 이미지에서 간판·조명을 자동 탐지
- 탐지된 광원 영역의 픽셀 강도를 휘도값(cd/m²) 또는 조도값(lux)으로 변환
- 법적 기준치와 자동 비교하여 위반 단계(1단계 / 2단계 / 3단계) 및 과태료 산출
- 웹 인터페이스를 통해 사진 한 장으로 누구나 결과 확인

### 개발 환경 및 도구

| 분류 | 도구 |
|---|---|
| 언어 | Python 3.9+, JavaScript (ES6+) |
| AI / 이미지 처리 | PyTorch, OpenCV, Pillow, ultralytics (YOLOv8) |
| 백엔드 | Flask, Flask-CORS, Gunicorn |
| 프론트엔드 | HTML5, CSS3, Vanilla JS |
| 배포 | Render (gunicorn + render.yaml) |
| 협업 및 관리 | GitHub, Discord, VS Code, Roboflow, Weights & Biases |

---

## 2. 빛 공해 개념 정리

빛 공해는 과도한 인공조명이 밤하늘·환경·인간 생활을 방해하는 현상이다. 4가지 유형으로 분류된다.

| 유형 | 설명 | 예시 |
|---|---|---|
| **침입광** (Light Trespass) | 원치 않는 장소로 빛이 들어오는 현상 | 가로등 불빛이 주거지 창문으로 유입 |
| **눈부심** (Glare) | 강렬한 빛이 눈에 직접 들어와 시각 마비 | 자동차 전조등, 강한 보안등 |
| **산란광 / 하늘 밝아짐** (Sky Glow) | 대기 중 수증기·오염물질에 빛이 산란돼 밤하늘이 낮처럼 밝아지는 현상 | 도심 야경 — 별이 보이지 않음 |
| **군집된 빛 / 과도한 조명** (Over-illumination) | 상업지구 등에서 무질서한 조명 과다 사용 | 번화가 네온·간판 밀집 |

---

## 3. 요구사항 분석

### 기능적 요구사항

- 외부 조명 및 간판 이미지 데이터를 입력(업로드)받을 수 있어야 한다
- 입력된 이미지에 대해 밝기 분석 및 전처리를 수행할 수 있어야 한다
- 이미지 데이터를 기반으로 빛 공해 법규 위반 여부를 판단하는 모델을 적용할 수 있어야 한다
- 분석 결과를 사용자에게 직관적으로 제공할 수 있어야 한다

### 비기능적 요구사항

- 다양한 환경에서 촬영된 이미지 데이터를 처리할 수 있어야 한다
- 분석 결과는 일정 수준 이상의 정확도를 유지해야 한다
- 사용자가 쉽게 접근할 수 있도록 웹 기반 인터페이스를 제공해야 한다

---

## 4. 구현 계획 및 아키텍처

### 전체 구조

```
사용자 브라우저
    │ 이미지 업로드
    ▼
index.html  →  analysis.html  →  result.html
    │                │                │
    └─────── Flask 서버 (backend.py) ──┘
                     │
            /api/analyze (POST)
                     │
          ┌──────────┴──────────┐
          │  YOLO 객체 탐지      │
          │  휘도·조도 계산       │
          │  법규 기준치 비교     │
          │  과태료 단계 산출     │
          └─────────────────────┘
```

### 단계별 구현 계획

**1단계 — 데이터 전처리 및 분석**
- 외부 조명·간판 이미지 데이터 수집 및 정리
- OpenCV 활용: 노이즈 제거, 밝기 분석

**2단계 — AI 모델 개발**
- YOLOv8 아키텍처를 야간 특화 데이터로 전이 학습
- 탐지된 광원 영역의 RGB → 휘도값 변환 수식 구현
- 법적 기준치와 대조하는 판정 알고리즘 구현

**3단계 — 웹 서비스 구현**
- Flask 기반 서버: 이미지 업로드 및 API 제공
- HTML/CSS/JS: 3-페이지 UI (홈 → 분석 중 → 결과)
- 실시간 분석 결과 시각화

**4단계 — 시스템 통합 및 배포**
- 이미지 업로드 → 판정 결과 파이프라인 검증
- Render 플랫폼 배포 (Gunicorn + render.yaml)

---

## 5. 현재 구현된 핵심 기술 설명

### 5-1. YOLOv8 객체 탐지 (ultralytics)

```python
from ultralytics import YOLO

MODEL = YOLO('yolov8n.pt')          # 기본 모델
# 또는
MODEL = YOLO('models/light_pollution_best.pt')  # 커스텀 학습 모델
```

- **YOLOv8 (You Only Look Once v8)**: 이미지 한 장을 한 번만 보고 모든 객체를 동시에 탐지하는 실시간 객체 탐지 모델
- `imgsz=640`, `conf=0.25` 기준으로 추론 수행
- COCO 데이터셋 레이블을 한국어 카테고리로 매핑 (`traffic light` → 가로등, `stop sign` → 간판 등)
- 커스텀 모델(`light_pollution_best.pt`) 존재 시 우선 적용

### 5-2. 휘도(Luminance) 계산

사람 눈의 색감도(광감도) 기준 수식을 적용한다.

$$Y = 0.2126 \cdot R + 0.7152 \cdot G + 0.0722 \cdot B$$

```python
r = cropped[:, :, 0].astype(np.float32)
g = cropped[:, :, 1].astype(np.float32)
b = cropped[:, :, 2].astype(np.float32)
lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
brightness = float(np.mean(lum))
```

- 단순 평균 밝기(`mean(R+G+B)/3`)보다 실제 사람 눈이 느끼는 밝기에 더 근접
- ITU-R BT.709 표준 기반 계수 사용

### 5-3. 채도 및 감마 계산

```python
hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
saturation = float(np.mean(hsv[:, :, 1].astype(np.float32) / 255))

gamma = float(np.clip(1.8 + np.std(lum) / 64.0, 1.0, 3.5))
```

- **채도(Saturation)**: 빛의 색 선명도 — 높을수록 강한 네온/컬러 간판일 가능성 큼
- **감마(Gamma)**: 밝기 분산값으로 추정 — 밝기 편차가 클수록 고감마, 눈부심 위험 증가

### 5-4. 법규 기반 위험 점수 및 과태료 산출 (`compute_fine`)

법규 기준 단위는 조명 유형(공간조명, 광고물, 장식조명)과 지역구분(제1종~제4종)에 따라 다르게 적용된다.

**조명 유형별 측정 기준:**
- **공간조명(가로등)**: 조도(lux) 최대값 기준
- **광고물(간판)**: 휘도(cd/m²) 최대값 기준
- **장식조명(조명)**: 휘도(cd/m²) 평균값/최대값 중 더 많이 초과된 쪽 기준

```python
# ---- 빛 공해 4대 분류 ----
POLLUTION_TYPES = {
    '침입광': '원치 않는 공간(창가/주거 방향)으로 유입되는 조명',
    '눈부심': '고휘도 광원이 시야 불편을 유발하는 상태',
    '산란광': '하늘 방향으로 퍼지는 확산광/배경 밝아짐',
    '군집된빛': '간판·장식 조명이 과도하게 밀집된 상태',
}

# ---- 4대 분류 튜닝 파라미터 ----
POLLUTION_THRESHOLDS = {
    'edge_margin': 0.14,        # 가장자리 여백
    'intrusion_avg': 52.0,     # 침입광 평균 임계값
    'glare_p95': 230.0,        # 눈부심 p95 밝기
    'glare_gamma': 2.45,       # 눈부심 감마
    'glare_ratio_upper': 14.0, # 눈부심 비율 상한
    'scatter_area': 0.22,      # 산란광 영역
    'scatter_ratio': 4.2,     # 산란광 비율
    'scatter_sat_max': 0.40,  # 산란광 채도 최대
    'cluster_sat': 0.30,       # 군집 채도
    'cluster_ratio': 2.0,     # 군집 비율
    'cluster_count_bonus': 2,  # 군집 수 보너스
    # 탐지 유지 임계값
    'min_confidence': 0.24,
    'min_brightness': 60.0,
    'min_p95': 175.0,
    'min_bright_ratio': 1.2,
}
```

**조명환경관리구역별 법규 기준치:**

| 구분 | 제1종 (자연환경) | 제2종 (농림지역) | 제3종 (주거지역) | 제4종 (상업·공업) |
|---|---|---|---|---|
| 공간조명 (lux) | 10 | 10 | 10 | 25 |
| 광고물 (cd/m²) | 50 | 400 | 800 | 1000 |
| 장식조명 평균 (cd/m²) | 5 | 5 | 15 | 25 |
| 장식조명 최대 (cd/m²) | 20 | 60 | 180 | 300 |

**과태료 단계 산출:**
- 기준값 이하: **준수** (과태료 0원)
- 기준값 초과 ~ 1.5배: **1단계** (50만원)
- 1.5배 초과 ~ 2배: **2단계** (75만원)
- 2배 초과: **3단계** (100만원 - 1차 위반 상한)

```python
def compute_fine(luminance_cd_m2_avg, luminance_cd_m2_max, illuminance_lux_max, light_type, zone_code):
    # 조명 유형과 지역구에 따른 기준치 비교
    # 초과 배율에 따라 1단계/2단계/3단계 산출
    ratio = measured / max(1.0, threshold)
    if ratio <= 1.5:
        stage = '1단계'
    elif ratio <= 2.0:
        stage = '2단계'
    else:
        stage = '3단계'
```

### 5-4-1. 4대 분류 알고리즘 상세

탐지된 광원 객체는 `가로등`, `간판`, `조명` 세 가지 유형으로 분류된 후, 각 유형별로 다음의 4대 분류(침입광/눈부심/산란광/군집된빛)를 판단합니다.

- **가로등**: 지점형 고휘도 광원의 특성을 반영하여 `눈부심`을 우선 강화하고, 상부 확산/저채도 특성은 `산란광`으로, 프레임 경계 인접성과 평균 밝기 상승은 `침입광`으로 해석합니다.
- **간판**: 다수·고채도·고밀도 발광이 `군집된빛` 특성으로 연결되며, 가장자리 위치와 높은 평균 밝기는 `침입광`으로, 고휘도 포인트는 `눈부심`으로 추가 평가됩니다.
- **장식조명(조명)**: 색채·밀도 기반 `군집된빛`을 기본으로 하고, 대면적 저채도 확산은 `산란광`, 경계 인접과 밝기 상승은 `침입광`, 고휘도 포인트는 `눈부심`으로 함께 평가합니다.

#### 조명 유형별 분류 기준 요약

| 유형 | 주요 특징 | 가중치 기준 |
|---|---|---|
| 가로등 | 지점형 고휘도 | `brightness_p95 >= 230`, `gamma >= 2.45`, `bright_pixel_ratio <= 14.0` → 눈부심 강화 |
| 간판 | 고채도·밀도 | `saturation >= 0.30`, `bright_pixel_ratio >= 2.0` → 군집된빛 강화 |
| 장식조명 | 색채/확산 혼합 | `area_ratio >= 0.22`, `bright_pixel_ratio >= 4.2`, `saturation <= 0.40` → 산란광/군집된빛 평가 |

#### 4대 분류 판단 요소

- `침입광`: `brightness_avg >= 52.0`, 프레임 가장자리 인접(`edge_margin < 0.14`) 등
- `눈부심`: `brightness_p95 >= 230.0`, `gamma >= 2.45`, `bright_pixel_ratio` 기준
- `산란광`: 면적 비율(`area_ratio >= 0.22`), 밝은 픽셀 비율(`bright_pixel_ratio >= 4.2`), 채도 낮음(`saturation <= 0.40`)
- `군집된빛`: 채도 높음(`saturation >= 0.30`), 밝은 픽셀 밀집(`bright_pixel_ratio >= 2.0`)

최종적으로 각 객체에 대해 점수화된 4개 분류 값 중 최고 점수를 선택합니다. 전체 사진의 대표 분류는 탐지 객체별 결과를 집계하여 `눈부심 > 침입광 > 군집된빛 > 산란광` 순으로 동률 해소합니다.

### 5-4-2. EXIF 기반 카메라 보정 알고리즘

이미지 EXIF에서 추출한 카메라 모델을 `smartphone`, `iphone`, `galaxy`, `dslr`, `action`, `cctv` 등으로 분류하고, ISO/노출/촬영 각도에 따라 밝기 보정 계수를 계산합니다.

- **EXIF 추출**: `Model` 태그에서 원본 모델명 획득 후 기종 키로 매핑
  - `iphone` → `iphone`
  - `galaxy`, `sm-` → `galaxy`
  - `canon`, `nikon`, `sony`, `pentax` → `dslr`
  - `gopro`, `dji`, `action` → `action`
  - `cctv`, `hikvision` → `cctv`
  - 그 외 → `smartphone`
- **추출 정보**: ISO(`0x8827`), ExposureTime(`0x829A`)도 함께 사용

각 카메라 프로필은 다음 보정 파라미터를 가집니다.

| 카메라 키 | label | iso_ref | exposure_ref_ms | brightness_bias | glare_bias | angle_power |
|---|---|---|---|---|---|---|
| smartphone | 스마트폰 기본 | 100 | 12 | 1.00 | 1.02 | 0.82 |
| iphone | 아이폰 계열 | 80 | 10 | 0.98 | 1.00 | 0.80 |
| galaxy | 갤럭시 계열 | 90 | 11 | 1.00 | 1.03 | 0.83 |
| dslr | 디지털카메라/DSLR | 200 | 20 | 0.95 | 0.97 | 0.76 |
| action | 액션캠/드론 | 160 | 16 | 1.06 | 1.08 | 0.90 |
| cctv | CCTV/고정형 | 140 | 18 | 1.08 | 1.10 | 0.88 |
| default | 기본값 | 100 | 12 | 1.00 | 1.00 | 0.82 |

#### 보정 공식

- `iso_scale = iso_ref / iso`
- `exposure_scale = sqrt(exposure_ref_ms / exposure_ms)`
- `angle_scale = 1 / max(0.55, cos(angle_deg) ** angle_power)`
- `brightness_scale = brightness_bias * iso_scale * exposure_scale * angle_scale` (0.55~2.40 제한)
- `glare_scale = glare_bias * (0.92 + 0.08 * angle_scale)` (0.85~1.25 제한)
- `pixel_ratio_scale = 1 / max(0.82, brightness_scale ** 0.35)` (0.74~1.12 제한)

이 보정 계수는 객체별 밝기/눈부심 지표를 정규화하는 `apply_capture_adjustment()`에 적용되어, `brightness`, `brightness_p95`, `brightness_max`, `bright_pixel_ratio` 값을 실제 촬영 환경에 맞게 보정합니다.

### 5-5. 적응형 탐지 필터 (`should_keep_detection`)

어두운 야간 환경에서의 탐지 누락을 줄이기 위한 적응형 필터.

```python
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
```

### 5-6. EXIF GPS 추출 및 지역 자동 판별

이미지 파일에 저장된 EXIF GPS 정보를 추출하여 위도·경도를 얻고, OpenStreetMap Nominatim API를 통해 조명환경관리구역(제1종~제4종)을 자동 판별합니다.

```python
def extract_gps_from_exif(image_bytes):
    """이미지 바이트에서 EXIF GPS 좌표(위도, 경도)를 추출합니다."""
    from PIL.ExifTags import TAGS, GPSTAGS
    pil_img = Image.open(io.BytesIO(image_bytes))
    exif_raw = pil_img._getexif()
    # GPS 정보 추출 로직
    ...

def get_zone_from_gps(lat, lon):
    """GPS 좌표로 조명환경관리구역 유형을 추정합니다."""
    # Nominatim API 호출
    # 제1종: 자연환경 보존지역
    # 제2종: 농림지역
    # 제3종: 주거지역
    # 제4종: 상업·공업지역
```

- GPS가 없을 경우 사용자가 지역구를 직접 선택하거나, 전체 4개 구역에 대한 시뮬레이션 결과를 제공

### 5-7. 이미지 전처리 파이프라인

야간 이미지 탐지를 위한 전처리 파이프라인.

```python
def preprocess_image(img):
    """
    야간 이미지 탐지용 전처리 파이프라인:
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
```

### 5-8. 프론트엔드 — sessionStorage 기반 상태 전달

```
index.html ──── sessionStorage에 이미지/파일명 저장 ────► analysis.html
                                                           │
                                              sessionStorage에 분석 결과 저장
                                                           │
                                                           ▼
                                                       result.html
```

- 페이지 간 상태 전달에 `sessionStorage` 사용 (서버 세션 불필요)
- 분석 결과(JSON)는 `light_detected`, `light_risk`, `light_confidence` 키로 저장
- `pageshow` 이벤트 + 버전 파라미터(`?v=`)로 캐시 문제 방지

### 5-7. Flask 라우팅 구조

```python
@app.route('/')                  # 홈
@app.route('/analysis')          # 분석 진행 중
@app.route('/result')            # 분석 결과
@app.route('/api/analyze', methods=['POST'])   # 이미지 분석 API
@app.route('/api/status', methods=['GET'])     # 서버·모델 상태 확인
```

- 정적 파일: `static/css`, `static/js`, `static/assets`
- HTML 템플릿: `templates/index.html` 등 (`render_template` 사용)

### 5-8. 배포 구조 (Render)

```
gunicorn  →  backend.py 의 app 객체
```

- `Procfile`: `web: gunicorn backend:app`
- `render.yaml`: 빌드/시작 명령, 환경변수 설정
- `runtime.txt`: Python 버전 고정
- `requirements.txt`: 의존성 전체 명시

### 5-9. PDF 리포트 다운로드

분석 결과를 PDF로 다운로드하는 기능은 프론트엔드 버튼 클릭으로 `/api/report/pdf`에 JSON payload를 POST하고, 백엔드가 PDF 바이트를 응답하는 구조로 구현되어 있습니다.

```python
@app.route('/api/report/pdf', methods=['POST'])
def api_report_pdf():
    report_data = request.get_json(silent=True) or {}
    if not report_data:
        return jsonify({'status': 'error', 'message': 'No report data provided.'}), 400

    pdf_buffer = _build_pdf_report_bytes(report_data)
    filename = _build_pdf_report_filename(report_data)
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=filename,
    )
```

- `_build_pdf_report_bytes(report_data)`는 `reportlab`을 사용해 PDF 문서를 생성합니다.
- `_build_pdf_report_filename(report_data)`는 원본 파일명을 기반으로 안전한 PDF 파일명을 만듭니다.
- 반환된 PDF는 `application/pdf` MIME 타입으로 다운로드됩니다.

프론트엔드 `static/js/main.js`에서는 다음과 같은 흐름으로 PDF 요청을 처리합니다.

```javascript
async function downloadReport() {
  const payload = buildReportPayload();
  const response = await fetch('/api/report/pdf', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`PDF generation failed: ${response.status}`);
  }

  const blob = await response.blob();
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = buildPdfFileName(payload.fileName);
  link.click();
  URL.revokeObjectURL(link.href);
}
```

- `buildReportPayload()`는 분석 결과 및 EXIF/촬영 정보를 JSON으로 구성합니다.
- 응답이 성공적이지 않으면 예외가 발생하고, 현재 구현에서는 TXT 파일로 대체 다운로드하는 폴백 경로가 동작합니다.

---

## 6. 시스템 흐름도

```
[사용자]
   │  JPG/PNG 이미지 업로드
   │
[index.html]
   │  sessionStorage에 이미지 저장
   │  → analysis.html 이동
   │
[analysis.html]
   │  단계별 진행 UI 표시 (5단계)
   │  POST /api/analyze 호출
   │
[Flask /api/analyze]
   │  base64 디코드
   │  YOLO 객체 탐지
   │  휘도·채도·감마 계산
   │  compute_fine() 실행
   │  JSON 응답 반환
   │
[analysis.html]
   │  sessionStorage에 결과 저장
   │  → result.html 이동
   │
[result.html]
      분석 결과 카드, 오버레이 박스, 개선 권장사항 표시
```

---

## 7. 테스트 및 검증 계획

### AI 모델 성능 검증
- 테스트 데이터셋으로 조명 기구·간판 탐지 정밀도 측정
- 정상 조명의 오검출(False Positive) / 위반 대상 미검출(False Negative) 분석
- 야간 저조도, 기상 악화(비·안개) 환경 이미지에서 탐지 안정성 확인

### 법규 위반 판정 로직 정확도
- 물리 휘도계 측정값 vs 모델 픽셀 분석 추정값 오차율 계산
- 환경부 조명환경관리구역별 허용 기준치 알고리즘 반영 여부 검증
- 위반 단계(1단계/2단계/3단계) 및 과태료 정확성 확인

### 시스템 통합 테스트
- 업로드 → 분석 → 결과 전 과정 오류 없는 동작 확인
- 분석 응답 시간 측정 (실시간 서비스 가능 여부 판단)
- 예외 처리: 조명 없는 사진, 깨진 파일, 미지원 형식 업로드 시 안내 메시지 출력 확인

### 사용자 테스트
- 학교 주변 실제 간판·조명 촬영 후 앱 판정 결과와 육안 점검 결과 비교
- 대시보드 직관성 및 리포트 확인 편의성 내·외부 피드백 수집

---

## 8. 기대효과 및 활용방안

### 기대효과

| 효과 | 설명 |
|---|---|
| **객관적 빛 공해 관리 기반** | 이미지+AI 기반으로 주관적 판단이 아닌 데이터 중심의 객관적 분석 가능 |
| **행정 효율성 극대화** | 인력 현장 점검 → 이미지 기반 자동 탐지로 전환, 업무 효율 향상 |
| **사회적 인식 제고** | 직접 분석 결과 확인 서비스로 시민들의 빛 공해 인식 향상 |
| **기술의 실용적 응용 입증** | 이미지 처리·AI 기술을 실 환경 문제에 적용한 4차 산업 응용 사례 |

### 활용 방안

- **지자체·환경 기관** 모니터링 도구 — 빛 공해 발생 지역 파악 및 정책 기초 자료
- **스마트시티** 통합 관제 시스템 환경 모니터링 모듈 연동
- **대국민 빛 환경 자가 진단 서비스** — 공익 앱으로 배포
- **차세대 스마트 조명 제어 기술** 연계 — 기준치 초과 시 자동 밝기 조절

---

## 9. 향후 개선 방향

- **실 데이터 라벨링 + 재학습**: 간판·조명 실사 이미지 10,000건 이상 수집 및 YOLO 파인튜닝으로 탐지 정밀도 향상
- **거리/휘도 보정식 도입**: 카메라 거리·각도 보정을 통한 물리 lux 값 근사 계산
- **다중 이미지 배치 분석**: 여러 장을 한 번에 업로드하여 구역별 위험 지도 생성
- **리포트 PDF 다운로드**: 분석 결과를 공문 형식으로 출력하는 기능
- **YOLOv12 전환**: 어텐션 메커니즘 강화 버전으로 야간 미세 광원 탐지 성능 극대화
- **모바일 앱 포팅**: React Native 또는 Flutter 기반 모바일 서비스 확장

---

## 10. 객체탐지 모델 비교 실험

### 10-1. 목적

기존 시스템은 YOLOv8 단일 모델로 광원을 탐지한다. 빛 공해 탐지에 어떤 객체탐지 모델이 더 적합한지를
주관적 인상이 아니라 **동일 데이터 · 동일 조건 · 동일 평가 기준의 실측치**로 확인하기 위해
4개 모델을 같은 데이터셋으로 학습하고 비교하는 실험 환경을 추가했다.

기존 학습 스크립트(`train_model.py`)와 웹 서비스(`backend.py`), 기존 데이터셋·weight 는 그대로 유지되며,
비교 실험 코드는 별도 폴더(`lpcompare/`, `training/`, `evaluation/`, `tools/`)에 추가되었다.

### 10-2. 비교 대상 모델과 Framework

| 모델 | 기본 weight | Framework | 비고 |
|---|---|---|---|
| YOLOv8 | `yolov8n.pt` | Ultralytics (`YOLO`) | 기존 시스템이 사용하는 계열 |
| YOLO11 | `yolo11n.pt` | Ultralytics (`YOLO`) | YOLOv8 후속 세대 |
| RT-DETR | `rtdetr-l.pt` | Ultralytics (`RTDETR`) | Transformer 기반, NMS 불필요 |
| Faster R-CNN | `fasterrcnn_resnet50_fpn_v2` (COCO pretrained) | torchvision | 2-stage 검출기 |

모델 규모가 서로 다르므로(nano 모델 vs. ResNet50 백본) 정확도 단독 비교가 아니라
파라미터 수 · 모델 크기 · GFLOPs · GPU 메모리 · FPS 를 함께 기록한다.

### 10-3. 데이터셋 구성

- 데이터셋 루트: `config/experiment.yaml` 의 `dataset.root` (현재 값 `data`)
- 지원 구조: `<root>/<split>/images` + `<root>/<split>/labels` (현재 프로젝트 구조) 및 `<root>/images/<split>` 구조 자동 판별
- `data.yaml`: 데이터셋 루트의 `data.yaml` 또는 `light_pollution.yaml` 을 **읽기만** 한다
- 라벨 형식: `class_id x_center y_center width height` (0~1 정규화)

실제 연결된 데이터셋 (Roboflow `light_pollution` v8 export, 2026-09-29 `tools/validate_dataset.py` 실측)

| 항목 | 값 |
|---|---|
| 경로 | `data/train`, `data/valid`, `data/test` (+ `data/data.yaml`) |
| 이미지 | train 4,462 / val 1,161 / test 579 = **6,202장** |
| bbox | train 22,971 / val 5,634 / test 3,201 = **31,806개** |
| 클래스 3개 | `light_signboard`(0) / `lighting`(1) / `streetlight`(2) |
| 라벨 없는 이미지 | 0장 (빈 라벨 파일 184개는 배경 이미지) |
| 이미지 없는 라벨 | 0개 |

클래스명은 데이터셋 `data.yaml` 의 영문 이름을 그대로 사용한다 (원본을 수정하지 않기 위함).
웹 서비스(`backend.py`)는 이 이름을 `light_signboard → 간판`, `lighting → 조명`, `streetlight → 가로등`
로 매핑해 사용한다.

#### 폴리곤(세그멘테이션) 라벨 혼재 처리

이 데이터셋에는 일반 bbox 라벨과 YOLO 폴리곤 라벨(`class x1 y1 x2 y2 ...`)이 섞여 있다.

| 항목 | 수량 |
|---|---|
| 전체 라벨 파일 | 6,202개 |
| bbox 전용 파일 | 5,774개 |
| 폴리곤 전용 파일 | 70개 |
| bbox + 폴리곤 혼재 파일 | 174개 |
| 폴리곤 라인 | 490줄 |
| 혼재 파일 안의 bbox 라인 | 821줄 |

Ultralytics 는 `verify_image_label()` 에서 `any(len(x) > 6 for x in lb)` 조건을 쓰기 때문에,
**파일 안에 폴리곤 라인이 한 줄이라도 있으면 그 파일의 모든 라인을 폴리곤으로 간주**해 좌표를 재해석한다.
그 결과 혼재 파일의 정상 bbox 821줄이 전혀 다른 박스로 바뀐다. (실측 예: 정답 `0.585 0.902 0.173 0.197`
→ 잘못 해석된 값 `0.379 0.549 0.413 0.705`)

따라서 `tools/prepare_splits.py` 가 라벨을 **파생 데이터셋으로 정규화**한다
(`config/experiment.yaml` 의 `splits.normalize_polygon_labels`, 기본값 `auto`).

```text
derived_data/dataset_normalized/<split>/labels/*.txt   폴리곤 -> 외접 bbox 로 변환한 라벨 (새로 생성)
derived_data/dataset_normalized/<split>/images         원본 이미지 폴더로의 Windows junction (복사 아님)
```

- 폴리곤 라인은 Ultralytics `segments2boxes` 와 동일하게 폴리곤 점들의 최소/최대값으로 외접 bbox 를 만든다
- 일반 bbox 라인은 좌표를 그대로 보존한다
- 정규화 후 모든 라인이 5필드이므로 Ultralytics 가 폴리곤으로 오인하지 않는다 (검증: `segments=0`)
- 박스 총 개수는 원본과 동일한 31,806개이며 손실이 없다
- **원본 `data/` 의 이미지·라벨·data.yaml 은 어떤 경우에도 수정하지 않는다**

이 정규화 덕분에 YOLO 계열과 Faster R-CNN 이 **완전히 동일한 정답 좌표**를 사용한다.

#### Train / Validation / Test 구성

모든 모델이 **완전히 동일한 이미지 목록**을 사용하도록, 원본을 복사·이동하지 않고
이미지 경로 목록 파일만 생성한다.

```text
splits/train.txt        splits/val.txt        splits/test.txt
splits/split_info.json          (어떤 방식으로 나눴는지 기록)
derived_data/data_compare.yaml  (4개 모델 공용 data.yaml — 원본 data.yaml 은 수정하지 않음)
```

split 전략은 `config/experiment.yaml` 의 `splits.test_strategy` 로 정한다.

| 전략 | 동작 |
|---|---|
| `existing` | 데이터셋에 `test` 폴더가 있으면 그대로 사용 |
| `split_val` (기본값) | `test` 가 없으면 기존 `val` 을 seed 42 로 val/test 로 나눈다. **train 은 손대지 않는다** |
| `use_val_as_test` | `val` 을 `test` 로 재사용 (val == test 이므로 성능이 과대평가되며 그 사실이 기록된다) |
| `resplit_all` | 전체를 70/15/15 로 재분할 (기존 split 이 깨지므로 기본값 아님) |

`test` 폴더가 이미 있으면 전략과 무관하게 기존 split 을 그대로 쓴다.

### 10-4. 공통 실험 조건

| 항목 | 값 |
|---|---|
| Image Size | 640 × 640 |
| Epoch | 100 |
| Random Seed | 42 |
| Pretrained | True (COCO 사전학습 weight) |
| Device | `cuda` — GPU 사용 전제. `common.require_cuda: true` 이므로 CUDA 를 쓸 수 없으면 CPU 로 대체하지 않고 **학습을 중단**한다 (`--allow-cpu` 로만 강행) |
| DataLoader workers | 0 (Windows multiprocessing 문제 회피 기본값) |

구조상 완전히 통일할 수 없는 항목은 억지로 맞추지 않고 **기록**한다.

| 항목 | YOLOv8 / YOLO11 / RT-DETR | Faster R-CNN |
|---|---|---|
| optimizer | Ultralytics `optimizer=auto` (AdamW/SGD 자동 선택) | SGD (lr 0.005, momentum 0.9, weight_decay 0.0005, cosine + warmup) |
| augmentation | mosaic / HSV / scale / fliplr 등 Ultralytics 기본값 | 좌우 반전(p=0.5)만 적용 |
| loss | box / cls / dfl (RT-DETR 은 giou / cls / l1) | RPN 2종 + ROI head 2종 |

#### Batch Size

설정값은 YOLOv8 · YOLO11 = 16, RT-DETR = 2, Faster R-CNN = 2 이다.
RT-DETR(`rtdetr-l`, 32M 파라미터)과 Faster R-CNN(ResNet50-FPN-V2)은 RTX 5060 8GB VRAM 에서
batch 4 로는 OOM 가능성이 커 2 로 낮춰 시작하도록 설정했다.
CUDA Out Of Memory 가 발생하면 batch 를 절반씩 낮춰 재시도하며(최소 1),
**실제로 사용된 batch 와 재시도 이력**이 `runs/<model>/training_meta.json` 과
`results/model_comparison.csv` 의 `Batch_Size` / `Requested_Batch` / `Note` 열에 그대로 남는다.

### 10-5. Faster R-CNN 의 YOLO annotation 처리

`training/yolo_dataset.py` 의 `YoloDetectionDataset` 이 원본 `.txt` 라벨을 **실행 시점에** 변환한다.
COCO JSON 등 별도 포맷을 만들지 않으며 원본 라벨 파일은 수정하지 않는다.

```text
xmin = (x_center - width  / 2) * 이미지너비
ymin = (y_center - height / 2) * 이미지높이
xmax = (x_center + width  / 2) * 이미지너비
ymax = (y_center + height / 2) * 이미지높이
```

- 변환 후 YOLO 와 동일한 letterbox(비율 유지 + 패딩)로 640×640 에 맞춘다
- torchvision 규약에 맞춰 class id 를 0-based → 1-based 로 올린다 (0 = background)
- 모델 내부 `GeneralizedRCNNTransform` 은 `min_size = max_size = 640` 으로 고정해 추가 리사이즈를 막는다
- 추론 시에는 letterbox 변환을 역으로 적용해 **원본 이미지 좌표계**로 되돌린 뒤 평가한다

### 10-6. 학습 결과 / weight 위치

```text
runs/yolov8/       runs/yolo11/       runs/rtdetr/       runs/faster_rcnn/
  weights/best.pt, weights/last.pt
  results.csv            (epoch 별 지표)
  training_meta.json     (실제 batch, 학습 시간, device, GPU, 시작·종료 시각, 성공 여부, 오류 메시지)
```

기존 웹 서비스가 쓰는 `models/light_pollution_best.pt` 와 `models/light_pollution/` 은 건드리지 않는다.

### 10-7. 평가 지표와 계산 방식

4개 모델의 예측을 모두 같은 형식(pixel `xyxy` + score + class id)으로 모은 뒤,
`lpcompare/metrics.py` 의 **단일 코드**로 지표를 계산한다. 프레임워크별 평가기를 쓰지 않으므로
평가 기준 차이로 인한 왜곡이 없다.

| 지표 | 계산 방식 |
|---|---|
| mAP@0.5, mAP@0.5:0.95 | COCO 방식 (IoU 0.50:0.05:0.95, 101-point interpolation) |
| Precision / Recall / F1 | conf ≥ 0.25 예측을 IoU 0.5 로 greedy 매칭 |
| Small / Medium / Large AP | COCO 면적 기준 (small < 32², medium < 96² px). `evaluation.area_criterion: relative` 로 상대 면적 기준(이미지 면적의 0.1% / 1%) 사용 가능 |
| Confusion Matrix | (클래스 수 + 1) × (클래스 수 + 1), 마지막 행·열은 background (미탐 / 오탐) |
| FPS / Inference Time | warm-up 후 반복 추론 평균. preprocess / inference / postprocess 분리 기록, 이미지 로딩 시간은 별도 측정 |
| Parameters / Model Size / GFLOPs | 모델 파라미터 수, weight 파일 크기(MB), GFLOPs |
| GPU Memory | 추론 중 `torch.cuda.max_memory_allocated` 최대값 (CPU 실행 시 측정 불가로 공란) |
| Training Time | 학습 시작~종료 실측 시간 |

- 모든 모델에 동일한 임계값을 적용한다: mAP 계산 conf 0.001, P/R/F1 conf 0.25, 매칭 IoU 0.5, NMS IoU 0.7, max detections 300
- `torchmetrics` 가 설치돼 있으면 mAP 를 한 번 더 계산해 **교차 검증 값**을 로그에 남긴다 (선택 사항)
- 측정하지 못한 값은 임의로 채우지 않고 공란으로 두며, 사유를 로그와 `Note` 열에 남긴다

### 10-8. 결과 파일 위치

```text
results/model_comparison.csv          4개 모델 통합 비교표
results/<model>_class_metrics.csv     클래스별 Precision/Recall/F1/AP50/AP50-95
results/object_size_metrics.csv       작은 광원(Small)/Medium/Large AP
results/dataset_report.txt | .csv     데이터셋 검사 리포트
results/evaluation_summary.json       평가 원본 수치
results/graphs/                       지표별 비교 막대그래프
results/training_curves/              모델별 학습 곡선 + Ultralytics 원본 그래프 복사본
results/confusion_matrix/             모델별 혼동행렬
results/predictions/                  original / ground_truth / 모델별 예측 이미지
results/comparison_images/            같은 이미지에 대한 정답 + 4개 모델 결과 한 장 비교
logs/                                 모델별 로그 (yolov8.log, yolo11.log, rtdetr.log, faster_rcnn.log, experiment.log 등)
```

### 10-9. 실행 환경 (2026-09-29 `tools/check_environment.py` 실측)

| 항목 | 값 |
|---|---|
| OS | Windows 10 Pro (10.0.19045) |
| Python | 3.11.9 (`.venv`) |
| PyTorch | 2.14.0+cpu |
| Torchvision | 0.29.0+cpu |
| Ultralytics | 8.3.36 |
| GPU (nvidia-smi) | NVIDIA GeForce RTX 5060, 8151 MiB, 드라이버 616.56 (CUDA UMD 13.4) |
| CUDA Available (PyTorch) | **False** |

PyTorch 공식 휠 채널 조회 결과(2026-09-29, 설치는 하지 않음): `cu128` 채널은 torch 2.11.0 까지만 제공하며,
현재 설치된 버전과 동일한 `torch 2.14.0` / `torchvision 0.29.0` 의 CUDA 빌드는 **`cu130` 채널**에 있다.
드라이버가 CUDA 13.4 를 지원하므로 cu130 을 사용할 수 있다.

### 10-10. 실행 시 주의사항 / 알려진 제한사항

- **현재 설치된 PyTorch 는 CPU 전용 빌드(`2.14.0+cpu`, `torch.version.cuda = None`)다.**
  이 빌드에는 CUDA 커널이 없어 설정이나 코드 수정만으로는 GPU 를 쓸 수 없으며,
  RTX 5060 이 장착돼 있어도 4개 모델 × 100 epoch × 4,462장 학습을 현실적인 시간 안에 끝낼 수 없다.
  따라서 `common.require_cuda: true` 상태에서 학습을 실행하면 CPU 로 대체하지 않고 즉시 중단하며,
  원인과 해결 방법을 출력한다 (`run_all.py` 는 종료 코드 2, 개별 학습 스크립트는 1).
  `python tools/check_environment.py` 가 확인 절차를 안내하며, 기존 환경을 깨지 않기 위해
  코드가 자동으로 재설치하지는 않는다. CUDA 빌드 설치 후 `CUDA Available : True` 를 확인한 뒤 학습한다.
  (평가만 하려면 `python run_all.py --evaluate-only`, CPU 로 강행하려면 `--allow-cpu`)
- 데이터셋에 폴리곤 라벨이 섞여 있어 `derived_data/dataset_normalized` 의 정규화 라벨을 사용한다
  (10-3 참고). `splits/*.txt` 도 이 파생 경로를 가리킨다. 파생 폴더를 지우면
  `python tools/prepare_splits.py --force` 로 다시 만들 수 있다.
- 파생 데이터셋의 `images` 는 원본 폴더를 가리키는 **Windows junction** 이다. 원본 `data/` 폴더를
  옮기거나 이름을 바꾸면 링크가 끊어지므로, 그때는 split 을 다시 생성해야 한다.
- 이미 학습된 weight 가 있으면 기본적으로 학습을 건너뛴다. 다시 학습하려면 `--overwrite` 를 명시해야 한다.
- 한 모델이 실패해도 나머지 모델은 계속 실행되며, 최종 요약에 `[SUCCESS] / [FAILED]` 로 표시된다.
- Ultralytics 가 자체 생성하는 그래프(`labels.jpg`, `confusion_matrix.png` 등)는 내부적으로 Arial 폰트를
  사용하므로 한글 클래스명이 깨질 수 있다. 본 실험이 생성하는 그래프·이미지는 맑은 고딕을 사용해 정상 표시된다.
- Faster R-CNN 의 GFLOPs 는 기록하지 않는다. torchvision 구현이 `list[Tensor]` 입력과 내부 transform 을
  사용해 thop 표준 프로파일링이 적용되지 않기 때문이며, 그 사유가 `Note` 열에 남는다.
- YOLO 계열/RT-DETR 은 Ultralytics 내부 계측값을, Faster R-CNN 은 직접 계측값을 사용한다.
  Faster R-CNN 은 NMS 와 box decoding 이 forward 내부에서 수행되므로 `Inference_ms` 가 포함하는
  연산 범위가 완전히 같지는 않다 (로그에 함께 기록된다).

---

## 11. 참고문헌 및 법령

- 인공조명에 의한 빛공해 방지법  
  https://www.law.go.kr/법령/인공조명에의한빛공해방지법

- 빛공해 방지를 위한 조명기구 설치·관리 권고기준 가이드라인  
  https://www.mcee.go.kr/home/web/policy_data/read.do?menuId=10276&seq=7933

- ITU-R BT.709 — 휘도 변환 계수 표준  
  Y = 0.2126R + 0.7152G + 0.0722B

- YOLOv8 공식 문서  
  https://docs.ultralytics.com

---

> 작성일: 2026년 3월 | 동아대학교 승학캠퍼스  
> 팀장: 조태승 (2143412@donga.ac.kr) | 010-8603-8271  
> 팀원: 곽승우 (2353660@donga.ac.kr) | 김동규 (2353695@donga.ac.kr) | 김승주 (2353716@donga.ac.kr)
