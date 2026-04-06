# 빛 공해 분석 시스템 — 전체 알고리즘 흐름도

```mermaid
flowchart TD
    subgraph TRAIN["🏋️ 학습 파이프라인 (train_model.py)"]
        T1["📁 데이터셋 준비\n(train/val/test 폴더)"]
        T2["📄 YAML Config 자동 생성\n(3 클래스: 간판·조명·가로등)"]
        T3["🤖 YOLOv8n 모델 로드\n(yolov8n.pt)"]
        T4["⚙️ 학습 실행\nepochs=50, imgsz=640\nbatch=8, GPU/CPU 자동 선택"]
        T5["💾 Best Checkpoint 저장\nmodels/light_pollution_best.pt"]
        T1 --> T2 --> T3 --> T4 --> T5
    end

    subgraph INFER["🔍 추론 파이프라인 (backend.py)"]
        I1["📷 이미지 입력\n(Base64 → NumPy Array)"]

        subgraph PRE["🎨 야간 이미지 전처리"]
            P1["⚖️ 화이트 밸런스 보정\n(Gray World Algorithm)"]
            P2["🔇 노이즈 제거\n(Bilateral Filter\nd=9, σ_color=75, σ_space=75)"]
            P3["💡 대비 강화\n(CLAHE, LAB L채널\nclipLimit=2.0, tile 8×8)"]
            P1 --> P2 --> P3
        end

        subgraph DET["🎯 객체 탐지"]
            D1["🤖 YOLOv8 탐지\n(conf=0.25, imgsz=640)\n→ 바운딩 박스 + 클래스"]
            D2{"광원 객체\n존재?"}
            D3["🔦 OpenCV Fallback 탐지\n(Luminance Mask → Contour)"]
            D1 --> D2
            D2 -->|"No"| D3
        end

        subgraph CLS["🏷️ 객체 분류"]
            C1{"YOLO 클래스\n확인"}
            C2["직접 매핑\n(간판·조명·가로등)"]
            C3["COCO → 한국어 매핑\n(traffic light → 가로등 등)"]
            C4["📐 기하학적 분류 Fallback\n가로폭/높이 비율(aspect) 기반\naspect≥1.8 → 간판\ny<45%+aspect≤0.9 → 가로등\n그 외 → 조명"]
            C1 -->|"한국어 클래스"| C2
            C1 -->|"COCO 클래스"| C3
            C1 -->|"미매핑"| C4
        end

        subgraph METRICS["📊 휘도·조도 측정"]
            M1["🌟 휘도(Y) 계산\nY = 0.2126R + 0.7152G + 0.0722B"]
            M2["📡 휘도 추정\n(cd/m²)\n= (norm^1.15) × 1200"]
            M3["🔆 조도 추정\n(lux)\n= (norm^1.05) × 60"]
            M4["🎨 채도(Saturation)\n& 감마(Gamma) 계산"]
            M1 --> M2 & M3 & M4
        end

        subgraph RISK["⚠️ 빛 공해 위험도 판단"]
            R0["🏢 상업지역 기준 고정\n(프론트엔드가 zone 미전송\n→ 항상 상업지역 기본값 사용)\n간판: 800 cd/m²\n조명: 650 cd/m²\n가로등: 30 lux"]
            R4{"측정값 ≤ 기준값?"}
            R5["✅ 준수\n위험점수 ≤ 58점\n= 20 + 채도보정 + 감마보정"]
            R6["❌ 위반\n위험점수 60~95점\n= 60 + 초과비율×40\n+ 채도보정 + 감마보정"]
            R7["🎯 위험 등급 판정\n≥85점: 고위험\n≥60점: 주의\n< 60점: 관찰"]
            R0 --> R4
            R4 -->|"Yes"| R5
            R4 -->|"No"| R6
            R5 & R6 --> R7
        end

        OUT["📤 분석 결과 반환\n탐지 객체 목록 · 위험 등급\n위험 점수 · 준수 여부\n측정 휘도/조도값"]

        I1 --> PRE --> DET
        D2 -->|"Yes"| CLS
        D3 --> CLS
        CLS --> METRICS --> RISK --> OUT
    end

    T5 -.->|"모델 로드"| INFER

    style TRAIN fill:#e8f4fd,stroke:#2196F3,stroke-width:2px
    style INFER fill:#f1f8e9,stroke:#4CAF50,stroke-width:2px
    style PRE fill:#fff3e0,stroke:#FF9800,stroke-width:1.5px
    style DET fill:#fce4ec,stroke:#E91E63,stroke-width:1.5px
    style CLS fill:#f3e5f5,stroke:#9C27B0,stroke-width:1.5px
    style METRICS fill:#e0f2f1,stroke:#009688,stroke-width:1.5px
    style RISK fill:#fff8e1,stroke:#FF6F00,stroke-width:1.5px
```
