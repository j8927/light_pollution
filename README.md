# 빛 공해(light_pollution)
#### 빛 공해는 과도한 인공조명이 밤하늘, 환경, 인간의 생활을 방해하는 현상으로, 주요 종류는 침입광(빛 침해), 눈부심(Glair), 산란광(하늘 밝아짐), 군집된 빛(Over-illumination) 4가지로 나뉩니다. 이로 인해 수면 장애, 생태계 교란, 농작물 피해 등이 발생하며, 한국은 세계적으로 빛 공해가 심각한 편입니다.
### 종류
    1. 침입광 (Light Trespass): 가로등이나 간판 불빛이 원치 않는 주거지 창문으로 들어와 수면 방해나 생활 불편을 주는 경우.
    2. 눈부심 (Glare): 강렬한 빛이 눈에 직접 들어와 잠시 시각을 마비시키거나 불쾌감을 주는 현상 (예: 자동차 전조등, 강한 보안등).
    3. 산란광/하늘 밝아짐 (Sky Glow): 인공조명 빛이 대기 중의 수증기나 오염물질에 굴절/산란되어 밤하늘이 낮처럼 밝아져 별이 보이지 않는 현상.
    4. 군집된 빛/과도한 조명 (Over-illumination/Clutter): 상업지구 등에서 너무 많은 조명이 무질서하게 켜져 있어 시각적 혼란과 에너지 낭비를 초래하는 현상.

# 1. 과제 개요

## 1) 프로젝트 명
실시간 인공조명 빛 공해 법규 위반 탐지 및 판정 시스템 구현


## 2) 프로젝트 목적

최근 도심 지역의 무분별한 인공 조명 사용은 단순한 시각적 불편을 넘어 수면 장애, 생태계 교란, 농작물 수확량 감소 등 심각한 사회적·환경적 문제를 야기하고 있습니다. 한국은 세계적으로 빛 공해 노출도가 매우 높은 국가로 분류되어 '인공조명에 의한 빛공해 방지법'을 시행 중이나, 실제 단속 현장에서는 고가의 휘도계 장비를 지참한 인력이 일일이 수동 측정해야 하는 행정적 한계가 존재한다.

본 프로젝트는 이러한 문제를 해결하기 위해 최신 객체 탐지 모델을 기반으로 강화된 어텐션 메커니즘을 통해 야간의 복잡한 조명 노이즈 속에서 실제 단속 대상이 되는 간판 및 장식 조명을 정밀하게 분리하고, 탐지된 광원을 대상으로 픽셀 강도 분석 알고리즘을 적용하여 법적 기준치(휘도) 초과 여부를 실시간으로 판정한다.

결과적으로 약 10,000개의 실사 데이터를 학습한 인공지능이 복잡한 단속 과정을 자동화함으로써, 누구나 사진 한 장으로 빛 공해 위반 여부를 객관적으로 확인할 수 있는 기술적 토대를 마련하고자 한다.


## 3) 개발 환경 및 도구

Python, PyTorch, OpenCV

Flask, HTML/CSS, JavaScript

GitHub, Discord, VS Code, Roboflow, Weights & Biases



# 2. 수행 계획
## 1) 요구사항 분석

### 기능적 요구사항
⦁ 외부 조명 및 간판 이미지 데이터를 입력받을 수 있어야 한다.

⦁ 입력된 이미지에 대해 밝기 분석 및 전처리를 수행할 수 있어야 한다.

⦁ 이미지 데이터를 기반으로 빛 공해 법규 위반 여부를 판단하는 모델을 적용할 수 있어야 한다.

⦁ 분석 결과를 사용자에게 직관적으로 제공할 수 있어야 한다.

### 비기능적 요구사항
⦁ 다양한 환경에서 촬영된 이미지 데이터를 처리할 수 있어야 한다.

⦁ 분석 결과는 일정 수준 이상의 정확도를 유지해야 한다.

⦁ 사용자가 쉽게 접근할 수 있도록 웹 기반 인터페이스를 제공해야 한다.

## 2) 구현 계획

### 1. 데이터 전처리 및 분석
⦁ 외부 조명 및 간판 이미지 데이터를 수집하고 정리한다.

⦁ OpenCV를 활용하여 이미지 전처리(노이즈 제거, 밝기 분석 등)를 수행한다.

### 2. AI 모델 개발
⦁ TensorFlow 또는 PyTorch를 활용하여 이미지 기반 빛 공해 탐지 모델을 설계한다.

⦁ YOLOv8 아키텍처를 야간 특화 데이터로 전이 학습하여 미세 광원 탐지 성능을 극대화한다.

⦁ 동일한 데이터셋으로 YOLOv8·YOLO11·RT-DETR·Faster R-CNN을 학습·비교하여 빛 공해 탐지에 적합한 모델을 실측 수치로 확인한다. (→ [5. 객체탐지 모델 비교 실험](#5-객체탐지-모델-비교-실험-yolov8--yolo11--rt-detr--faster-r-cnn))

⦁ 탐지된 광원 영역의 RGB 값을 휘도 값으로 변환하는 수식을 구현하고 법적 기준치와 대조한다.

### 3. 웹 서비스 구현
⦁ Flask 기반 서버를 구축하여 이미지 업로드 기능을 구현한다.

⦁ HTML을 이용하여 사용자 인터페이스를 개발한다.

⦁ 업로드된 이미지에 대해 분석 결과를 사용자에게 제공한다.

### 4. 시스템 통합
⦁ 이미지 업로드부터 판정 결과 출력까지의 파이프라인 정상 동작 확인하고
   AI 모델, 웹 서비스와 통합한다.
   
⦁ 전체 시스템의 동작을 확인하고 성능을 개선한다.

---

## 적용된 알고리즘 및 기술 상세 + 만들어진 주소

본 프로젝트에 적용된 알고리즘과 기술에 대한 상세한 설명은 [Information.md](./Information.md)를 참조하세요.

적용된 코드의 웹페이지 주소: https://j8927-light-pollution-ai.hf.space/


## 3) 테스트(검증) 계획

### 1. AI 모델 성능 검증

⦁테스트 데이터셋을 활용하여 조명 기구 및 간판에 대한 탐지 정밀도를 측정합니다.

⦁'정상 조명'을 '위반'으로 오검출하거나, 반대로 위반 대상을 놓치는 경우를 분석하여 모델의 신뢰성을 확인합니다.

⦁노이즈가 심한 아주 어두운 환경이나 기상 악화(비, 안개) 상황의 이미지에서도 객체를 놓치지 않고 탐지하는지 확인합니다.


### 2. 법규 위반 판정 로직 정확도 테스트

⦁실제 물리적 휘도계로 측정한 값과 모델이 픽셀 분석을 통해 추정한 값 사이의 오차율을 계산합니다.

⦁환경부 지침에 따른 조명환경관리구역별 허용 기준치가 알고리즘 내에 정확히 반영되어 위반 등급(정상/주의/위반)이 올바르게 출력되는지 확인합니다.

### 3. 시스템 통합 테스트

⦁사용자가 웹 인터페이스를 통해 이미지를 업로드한 시점부터 최종 분석 결과가 화면에 출력되기까지의 전 과정이 오류 없이 동작하는지 확인합니다.

⦁이미지 업로드 후 분석 결과가 나오기까지의 시간을 측정하여 실시간 서비스 가능 여부를 판단합니다.

⦁조명이 없는 사진, 깨진 파일, 혹은 지원하지 않는 형식의 파일 업로드 시 시스템이 다운되지 않고 적절한 안내 메시지를 출력하는지 검증합니다.

### 4. 사용자 테스트

⦁학교 주변의 실제 간판 및 조명을 촬영하여 앱의 판정 결과와 실제 육안 점검 결과를 비교합니다.

⦁대시보드가 일반 사용자 입장에서 정보를 직관적으로 전달하는지, 리포트 확인 과정이 간편한지 팀 내외부 인원을 대상으로 피드백을 수집합니다.

# 3. 기대효과 및 활용방안

## 1. 기대효과

객관적 빛 공해 관리 기반 마련: 이미지 분석과 YOLOv8 인공지능 기술을 활용하여 외부 조명 및 간판의 밝기 상태를 분석함으로써, 주관적 판단이 아닌 데이터 중심의 객관적 분석 기반을 마련할 수 있습니다.

행정 및 관리 효율성 극대화: 빛 공해 법규 위반 여부를 자동으로 탐지함으로써, 기존에 인력이 직접 현장을 점검해야 했던 문제를 데이터 기반으로 전환하여 관리 기관의 업무 효율성을 획기적으로 향상시킬 수 있습니다.

사회적 인식 제고 및 환경 보호: 사용자가 직접 조명 사진을 분석하여 결과를 확인할 수 있는 서비스를 제공함으로써 빛 공해 문제의 심각성을 인식시키고, 환경 보호에 대한 시민들의 자발적 관심을 유도할 수 있습니다.

컴퓨터공학 기술의 실용적 응용 입증: 이미지 처리 및 인공지능 기술을 실제 환경 문제 해결에 적용함으로써, 4차 산업 핵심 기술의 사회적 응용 가능성과 실무적 가치를 보여줄 수 있습니다.


## 2. 활용 방안

지자체 및 환경 기관의 모니터링 도구: 지방자치단체에서 도시 내 조명 시설이나 간판 조명을 상시 모니터링하는 도구로 도입하여, 빛 공해 발생 지역 파악 및 관련 관리 정책 수립을 위한 기초 자료로 활용할 수 있습니다.

스마트시티 환경 관리 시스템 연동: 도시 내 실시간 조명 데이터를 수집 및 분석하여 환경 친화적인 스마트 조명 체계를 구축하고, 스마트시티 통합 관제 시스템의 환경 모니터링 모듈로 활용할 수 있습니다.

대국민 빛 환경 자가 진단 서비스: 일반 사용자들이 자신의 생활권 내 조명 환경을 직접 촬영하여 분석해볼 수 있는 서비스로 배포하여, 개인의 생활 환경 개선을 돕는 공익적 도구로 활용할 수 있습니다.

차세대 스마트 조명 제어 기술과의 연계: 향후 조명 제어 시스템과 연동하여, 법적 기준치를 초과할 경우 자동으로 밝기를 조절하는 지능형 조명 시스템 및 자동 제어 기술 개발의 핵심 알고리즘으로 활용할 수 있습니다.



# 4. 로컬 실행 방법 (개인 개발 환경)

## 1) 프로젝트 폴더 이동

```powershell
cd e:\visual.code\Light_Pollution
```

## 2) 가상환경 생성 및 활성화

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

## 3) 패키지 설치

Render 배포 기준(경량):

```powershell
pip install -r requirements.txt
```

로컬 AI 모델까지 포함(전체):

```powershell
pip install -r requirements-full.txt
```

## 4) Flask 서버 실행

```powershell
python backend.py
```

## 5) 브라우저 접속

- 메인 페이지: http://127.0.0.1:5000/
- 분석 페이지: http://127.0.0.1:5000/analysis
- 결과 페이지: http://127.0.0.1:5000/result
- 상태 API: http://127.0.0.1:5000/api/status

## 6) 확인 체크리스트

- 메인 화면 정상 표시
- CSS/JS 깨짐 없음
- 예시 이미지 또는 업로드 후 분석 페이지 이동
- 결과 페이지 정상 표시

# 5. 객체탐지 모델 비교 실험 (YOLOv8 / YOLO11 / RT-DETR / Faster R-CNN)

기존 YOLOv8 학습 데이터셋을 그대로 사용하여 4개 객체탐지 모델을 **같은 조건으로 학습**하고,
정확도·속도·모델 크기·GPU 사용량을 **실제 측정값**으로 비교하는 실험 환경입니다.

기존 학습 스크립트(`train_model.py`)와 웹 서비스(`backend.py`), 기존 데이터셋과 `models/light_pollution_best.pt`는
그대로 유지되며, 비교 실험 코드만 추가되어 있습니다.

기술적인 상세 설명(평가 기준, split 방식, Faster R-CNN의 라벨 변환 등)은
[Information.md — 10. 객체탐지 모델 비교 실험](./Information.md#10-객체탐지-모델-비교-실험)을 참조하세요.

## 1) 비교 대상 모델

| 모델 | 기본 weight | Framework |
|---|---|---|
| YOLOv8 | `yolov8n.pt` | Ultralytics |
| YOLO11 | `yolo11n.pt` | Ultralytics |
| RT-DETR | `rtdetr-l.pt` | Ultralytics |
| Faster R-CNN | `fasterrcnn_resnet50_fpn_v2` (COCO pretrained) | torchvision |

모델 규모가 서로 다르므로 정확도만 보지 않고 파라미터 수·모델 크기·GFLOPs·GPU 메모리·FPS를 함께 기록합니다.

## 2) 개발 환경

- Windows 10 + PowerShell + Visual Studio Code
- Python 가상환경 `.venv` (Python 3.11)
- 학습·평가는 NVIDIA GPU(CUDA)를 사용합니다. `config/experiment.yaml`의 `common.device: cuda`, `common.require_cuda: true`가 기본값이며, CUDA를 쓸 수 없으면 CPU로 조용히 넘어가지 않고 **학습을 중단하고 원인을 출력**합니다. CPU로 강행하려면 `--allow-cpu`를 사용합니다.

패키지 설치(로컬 전체):

```powershell
pip install -r requirements-full.txt
```

## 3) 환경 / CUDA 확인

```powershell
python tools/check_environment.py
```

Python·PyTorch·torchvision·Ultralytics 버전, CUDA 사용 가능 여부, GPU 이름과 VRAM을 출력합니다.
NVIDIA GPU가 있는데 CUDA를 쓸 수 없으면(예: CPU 전용 PyTorch가 설치된 경우) 경고와 함께 필요한 설치 명령을 안내합니다.
이 스크립트는 패키지를 자동으로 설치하거나 제거하지 않습니다.

## 4) 공통 설정 파일

모든 실험 조건은 `config/experiment.yaml` 한 곳에서 관리합니다.

```yaml
dataset:
  root: data/images        # 기존 YOLO 데이터셋 (읽기 전용)
common:
  epochs: 100
  image_size: 640
  seed: 42
  device: auto
  pretrained: true
models:
  yolov8:      { weights: yolov8n.pt,  batch: 16, run_dir: runs/yolov8 }
  yolo11:      { weights: yolo11n.pt,  batch: 16, run_dir: runs/yolo11 }
  rtdetr:      { weights: rtdetr-l.pt, batch: 4,  run_dir: runs/rtdetr }
  faster_rcnn: { weights: fasterrcnn_resnet50_fpn_v2, batch: 4, run_dir: runs/faster_rcnn }
```

## 5) Dataset 위치와 검사

데이터셋은 기존 구조를 그대로 사용합니다. (Roboflow `light_pollution` v8 export)

```text
data/
├─ train/images/   train/labels/     4,462장
├─ valid/images/   valid/labels/     1,161장
├─ test/images/    test/labels/        579장
└─ data.yaml       (클래스 3개: light_signboard / lighting / streetlight)
```

경로는 `config/experiment.yaml`의 `dataset.root`로 지정합니다(현재 값 `data`).

검사 실행:

```powershell
python tools/validate_dataset.py
```

이미지 수, bbox 수, 클래스별 분포, 이미지-라벨 1:1 대응, 잘못된 class id, bbox 좌표 오류,
라벨 없는 이미지, 이미지 없는 라벨, 클래스 불균형을 확인하고 결과를 저장합니다.

```text
results/dataset_report.txt
results/dataset_report.csv
```

원본 데이터셋은 검사만 하며 수정하지 않습니다.

## 6) 공통 Dataset split

4개 모델이 **정확히 같은 train / val / test 이미지**를 쓰도록 목록 파일을 생성합니다.
원본 이미지를 복사하거나 이동하지 않습니다.

```powershell
python tools/prepare_splits.py
```

```text
splits/train.txt   splits/val.txt   splits/test.txt
splits/split_info.json             (어떤 방식으로 나눴는지 기록)
derived_data/data_compare.yaml     (4개 모델 공용 data.yaml — 원본 data.yaml은 수정하지 않음)
```

`test` 폴더가 이미 있으면 기존 split을 그대로 사용하고, 없으면 기존 `val`을 seed 42로 val/test로 나눕니다
(`train`은 손대지 않음). 이미 만들어진 split이 있으면 그대로 재사용하며, 다시 만들려면 `--force`를 사용합니다.

현재 데이터셋에는 YOLO bbox 라벨과 폴리곤(세그멘테이션) 라벨이 섞여 있습니다. Ultralytics는 파일 안에
폴리곤 라인이 하나라도 있으면 그 파일 전체를 폴리곤으로 간주해 정상 bbox까지 잘못 읽으므로,
이 스크립트가 라벨을 `derived_data/dataset_normalized/`에 bbox로 정규화한 뒤 4개 모델이 모두
그 데이터를 쓰도록 연결합니다(이미지는 원본 폴더로의 junction, 복사 아님). **원본 `data/`는 수정하지 않습니다.**

## 7) 모델별 학습

```powershell
python training/train_yolov8.py
python training/train_yolo11.py
python training/train_rtdetr.py
python training/train_faster_rcnn.py
```

주요 옵션(4개 공통): `--epochs`, `--batch`, `--imgsz`, `--device`, `--overwrite`, `--allow-cpu`, `--config`

- 이미 학습된 weight가 있으면 **기본적으로 학습을 건너뜁니다.** 다시 학습하려면 `--overwrite`를 붙입니다.
- CUDA Out Of Memory가 발생하면 batch를 절반씩 낮춰 재시도하고, 실제 사용된 값을 결과에 기록합니다.

학습 결과 위치:

```text
runs/yolov8/   runs/yolo11/   runs/rtdetr/   runs/faster_rcnn/
  weights/best.pt, weights/last.pt
  results.csv             (epoch별 지표)
  training_meta.json      (실제 batch, 학습 시간, device, 성공 여부, 오류 메시지)
```

## 8) 전체 일괄 실행

```powershell
python run_all.py
```

환경 검사 → 데이터셋 검사 → split 확인 → 4개 모델 학습 → 동일 test dataset 평가 →
비교 CSV → 그래프 → 예측/비교 이미지 → 최종 요약 순으로 진행합니다.
한 모델이 실패해도 나머지는 계속 진행되며, 마지막에 `[SUCCESS] / [FAILED]`로 정리해 보여줍니다.

평가만 실행(기존 weight 재사용):

```powershell
python run_all.py --evaluate-only
```

특정 모델만 실행:

```powershell
python run_all.py --model yolov8
python run_all.py --model yolo11
python run_all.py --model rtdetr
python run_all.py --model faster_rcnn
```

그 밖의 옵션: `--overwrite`, `--epochs`, `--device`, `--allow-cpu`, `--no-speed`, `--skip-graphs`, `--skip-predictions`, `--num-predictions`, `--val-map-interval`, `--force-splits`

본 학습 전에 2 epoch로 전체 흐름을 먼저 확인할 수 있습니다(설정 파일을 고치지 않아도 됩니다).

```powershell
python run_all.py --epochs 2
python run_all.py --overwrite          # 확인 후 100 epoch 본 학습
```

## 9) 평가 · 결과 파일

평가만 따로 실행할 수도 있습니다.

```powershell
python evaluation/evaluate_models.py
python evaluation/benchmark_speed.py
python evaluation/make_graphs.py
python evaluation/make_predictions.py
```

결과 위치:

```text
results/model_comparison.csv          4개 모델 통합 비교표
results/<model>_class_metrics.csv     클래스별 Precision / Recall / F1 / AP50 / AP50-95
results/object_size_metrics.csv       작은 광원(Small) / Medium / Large AP
results/evaluation_summary.json       평가 원본 수치
results/graphs/                       지표별 비교 그래프
results/training_curves/              모델별 학습 곡선
results/confusion_matrix/             모델별 혼동행렬
results/predictions/                  original / ground_truth / 모델별 예측 이미지
results/comparison_images/            정답 + 4개 모델 결과를 한 장에 비교
logs/                                 모델별 로그
```

Precision·Recall·F1·mAP@0.5·mAP@0.5:0.95·FPS·추론 시간·파라미터 수·모델 크기·GPU 메모리·학습 시간을
**모든 모델에 동일한 임계값과 동일한 계산 코드**로 산출합니다. 측정하지 못한 값은 임의로 채우지 않고
빈칸으로 두며 사유를 로그와 CSV의 `Note` 열에 남깁니다.

2026-09-30 ~ 10-01에 RTX 5060에서 4개 모델 100 epoch 본 학습·평가를 완료했습니다(총 25.6시간, 실패 없음).
실측 수치와 학습 곡선 해석은 [Information.md — 10-11. 본 학습 결과](./Information.md#10-11-본-학습-결과-2026-09-30--10-01-실측)에 정리돼 있습니다.

## 10) 단일 이미지 추론

```powershell
python inference.py --model yolov8 --image test.jpg
python inference.py --model yolo11 --image test.jpg
python inference.py --model rtdetr --image test.jpg
python inference.py --model faster_rcnn --image test.jpg
```

검출된 클래스, confidence, bbox 좌표, 추론 시간을 출력하고 bbox가 그려진 결과 이미지를
`results/inference/`에 저장합니다.

## 11) 향후 영상 / 카메라 적용

학습 코드(`training/`)와 추론 코드(`inference.py`)가 분리되어 있어, 같은 진입점으로
동영상과 카메라 입력도 처리할 수 있습니다.

```powershell
python inference.py --model yolov8 --video sample.mp4
python inference.py --model yolov8 --webcam 0
```

# 6. 참고문헌

인공조명에 의한 빛공해 방지법
https://www.law.go.kr/법령/인공조명에의한빛공해방지법

빛공해 방지를 위한 조명기구 설치·관리 권고기준 가이드라인
https://www.mcee.go.kr/home/web/policy_data/read.do?menuId=10276&seq=7933
