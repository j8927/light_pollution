# 빛 공해 분석 시스템 — 전체 알고리즘 흐름도

```mermaid
flowchart TD
    subgraph SERVER["🚀 서버 시작 시 1회 실행 (backend.py 로딩)"]
        DM{"models/light_pollution_best.pt\n파일 존재 여부"}
        D1A["🟢 light_pollution_best.pt 로드\n커스텀 파인튜닝 모델\n→ 한국어 3개 클래스만 출력"]
        D1B["🟡 yolov8n.pt (COCO) 로드\nlight_pollution_best.pt 없을 때만 사용\n→ 영어 80개 클래스 출력"]
        DM -->|"파일 있음"| D1A
        DM -->|"파일 없음 (fallback)"| D1B
    end

    subgraph TRAIN["🏋️ 학습 파이프라인 (train_model.py)"]
        T1["📁 데이터셋 준비\n(train/val/test 폴더)"]
        T2["📄 YAML Config 생성\n학습 클래스: 간판 · 조명 · 가로등"]
        T3["🤖 YOLOv8n 베이스 모델 로드\n(Ultralytics 공식 yolov8n.pt)"]
        T4["⚙️ 파인튜닝 학습\nepochs=50, imgsz=640, batch=8"]
        T5["💾 light_pollution_best.pt 저장\n출력 클래스: 간판 · 조명 · 가로등 (한국어)"]
        T1 --> T2 --> T3 --> T4 --> T5
    end

    subgraph INFER["🔍 추론 파이프라인 (요청마다 실행)"]
        I1["📷 이미지 입력\nBase64 → NumPy 배열 + 원본 바이트 (EXIF 보존)"]

        subgraph GPS_ZONE["📍 촬영 위치 → 조명환경관리구역 판별"]
            G1["🛰️ EXIF GPS 추출\n(PIL EXIF → 위도/경도)"]
            G2{"GPS\n추출 성공?"}
            G3["🌏 Nominatim 역지오코딩\nOSM type 기반 자동 분류\n자연환경 → 제1종\n농림지   → 제2종\n주거지   → 제3종\n상업·공업 → 제4종"]
            G4["📋 요청 파라미터 사용\nzone 파라미터 → 정규화\n미전송 시 → 제3종(주거) 기본값"]
            G1 --> G2
            G2 -->|"GPS 있음"| G3
            G2 -->|"GPS 없음"| G4
        end

        subgraph PRE["🎨 야간 이미지 전처리"]
            P1["⚖️ 화이트 밸런스 보정 (Gray World)"]
            P2["🔇 노이즈 제거 (Bilateral Filter)"]
            P3["💡 대비 강화 (CLAHE, LAB L채널)"]
            P1 --> P2 --> P3
        end

        subgraph YOLO_PATH["🤖 YOLO 탐지 경로"]
            D1["YOLOv8 추론\nconf=0.25, imgsz=640\n→ 바운딩 박스 + 클래스명"]
            D2{"광원 객체\n탐지됨?"}
            D1 --> D2

            subgraph CLS["🏷️ 클래스명 → 카테고리 변환"]
                C1{"클래스명 확인"}
                C2["✅ 한국어 클래스\n간판·조명·가로등 등\n그대로 사용\n← light_pollution_best.pt 출력"]
                C3["🔄 COCO 영어 클래스\ntraffic light·lighting 등\n딕셔너리로 한국어 번역\n← yolov8n.pt 출력"]
                C4["📐 미매핑 클래스\n바운딩박스 모양으로 추측\naspect≥1.8 → 간판\n위쪽+세로형 → 가로등\n그 외 → 조명"]
                C1 -->|"한국어"| C2
                C1 -->|"COCO 영어"| C3
                C1 -->|"둘 다 아님"| C4
            end
            D2 -->|"Yes"| CLS
        end

        NOTFOUND["🚫 미탐지 결과 반환\ndetected: []\noverall: 미탐지\n광원 객체가 탐지되지 않았습니다."]

        subgraph METRICS["📊 휘도·조도 측정 (탐지 객체별)"]
            M1["휘도(Y) = 0.2126R + 0.7152G + 0.0722B\n(픽셀 평균 lum_avg + 픽셀 최대 lum_max 모두 산출)"]
            M2["cd/m²(avg) = (norm_avg^1.15) × 1200"]
            M3["cd/m²(max) = (norm_max^1.15) × 1200"]
            M4["lux(max)   = (norm_max^1.05) × 60"]
            M5["채도(saturation) · 감마\n눈부심 보정 보조 지표"]
            M1 --> M2 & M3 & M4 & M5
        end

        subgraph FINE["💸 과태료 단계 산출\n인공조명에 의한 빛공해 방지법 시행규칙 별표 + 시행령 제8조"]
            F1{"조명 유형 분류\nLIGHT_TYPE_MAP"}
            F2["🏮 공간조명 · 가로등\n판정 기준: 조도 최대값 lux\n제1·2·3종 ≤10 / 제4종 ≤25"]
            F3["📢 광고물 · 간판\n판정 기준: 휘도 최대값 cd/m²\n제1종 ≤50 / 제2종 ≤400\n제3종 ≤800 / 제4종 ≤1000"]
            F4["💡 장식조명 · 조명\n판정 기준: 휘도 평균/최대값 cd/m²\n제1종 5/20 · 제2종 5/60\n제3종 15/180 · 제4종 25/300"]
            F5["📐 초과 배율 계산\nratio = 측정값 ÷ 기준값"]
            F6{"ratio 비교"}
            F7["✅ 준수\nfineAmount = 0"]
            F8["⚠️ 1단계 위반\n기준 초과 ~ 1.5배\n→ 과태료 100만원"]
            F9["🔶 2단계 위반\n1.5배 ~ 2배\n→ 과태료 200만원"]
            F10["🔴 3단계 위반\n2배 초과\n→ 과태료 300만원"]
            F1 -->|"가로등"| F2
            F1 -->|"간판"| F3
            F1 -->|"조명"| F4
            F2 & F3 & F4 --> F5 --> F6
            F6 -->|"≤1 (기준 이하)"| F7
            F6 -->|"1 초과 ~ 1.5"| F8
            F6 -->|"1.5 초과 ~ 2"| F9
            F6 -->|"2 초과"| F10
        end

        OUT["📤 결과 반환 (JSON)\n▸ overall: 준수 · 1단계 · 2단계 · 3단계 · 미탐지\n▸ totalFineAmount: 총 과태료 합계(만원)\n▸ violationCount: 위반 객체 수\n▸ zone / zoneLabel: 제1~4종 · 지역명\n▸ gpsDetected: GPS 자동판별 여부\n▸ detected[]: 탐지 객체별 상세 측정값 + 과태료"]

        I1 --> GPS_ZONE
        I1 --> PRE --> YOLO_PATH
        D2 -->|"No (광원 없음)"| NOTFOUND
        G3 & G4 -.->|"zone_code 제공"| FINE
        CLS --> METRICS --> FINE --> OUT
    end

    D1A & D1B -.->|"로드된 모델 사용"| D1
    T5 -.->|"커스텀 모델 로드"| D1A

    style SERVER fill:#e8eaf6,stroke:#3F51B5,stroke-width:2px
    style TRAIN fill:#e8f4fd,stroke:#2196F3,stroke-width:2px
    style INFER fill:#f1f8e9,stroke:#4CAF50,stroke-width:2px
    style GPS_ZONE fill:#e3f2fd,stroke:#1976D2,stroke-width:1.5px
    style PRE fill:#fff3e0,stroke:#FF9800,stroke-width:1.5px
    style YOLO_PATH fill:#fce4ec,stroke:#E91E63,stroke-width:1.5px
    style CLS fill:#f3e5f5,stroke:#9C27B0,stroke-width:1.5px
    style METRICS fill:#e0f2f1,stroke:#009688,stroke-width:1.5px
    style FINE fill:#fff8e1,stroke:#FF6F00,stroke-width:1.5px
    style NOTFOUND fill:#fafafa,stroke:#9E9E9E,stroke-width:1.5px
```
