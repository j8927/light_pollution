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
10. [참고문헌 및 법령](#10-참고문헌-및-법령)

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

## 10. 참고문헌 및 법령

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
