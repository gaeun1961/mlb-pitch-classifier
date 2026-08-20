# MLB Pitch Classifier

MLB Statcast 투구 데이터를 기반으로 MLP 신경망이 7가지 구종(포심, 싱커, 슬라이더, 커브, 체인지업, 커터, 스플리터)을 분류하는 딥러닝 애플리케이션입니다.

가천대학교 영상처리 수업 과제로 시작해, 데스크톱 GUI → 웹 데모 → FastAPI 백엔드 분리 → Gemini 기반 자연어 설명까지 단계적으로 확장했습니다.

## 🌐 웹 데모

**[Live Demo →](https://mlb-pitch-classifier-tgtrdgldkletpsw43j26sz.streamlit.app/)**

설치 없이 브라우저에서 바로 슬라이더로 가상의 투구를 만들거나, 실제 Statcast 데이터를 업로드하거나, **선수 이름으로 실제 투수의 최근 투구를 검색**해 모델의 예측과 예측 근거를 자연어로 확인할 수 있습니다.

> 예측 API는 Render 무료 플랜에서 실행됩니다. 트래픽이 없으면 슬립 상태가 되어 첫 예측 요청이 최대 50초 정도 걸릴 수 있습니다(웹 데모에 관련 안내가 표시됩니다).

<p>
  <img src="results/confusion_matrix.png" width="48%" />
  <img src="results/loss_accuracy.png" width="48%" />
</p>

## 모델 성능

7개 구종에 대한 weighted F1-score **0.94** (테스트 세트 64,853건 기준).

| 구종 | F1-Score |
|---|---|
| FF (포심) | 0.97 |
| CU (커브) | 0.96 |
| SI (싱커) | 0.95 |
| CH (체인지업) | 0.94 |
| SL (슬라이더) | 0.92 |
| FS (스플리터) | 0.87 |
| FC (커터) | 0.85 |

FC(커터)와 SL(슬라이더)·FF(포심) 사이의 혼동이 가장 큰 한계로, 세 구종의 물리적 특성이 유사하기 때문입니다.

## 아키텍처

```
Streamlit (UI)  ──HTTP──▶  FastAPI (backend/, Render 배포)
                                │
                                ├── MLP 모델 예측 (TensorFlow)
                                └── Gemini API 자연어 설명
                                    (실패 시 규칙 기반 설명으로 폴백)
```

Streamlit은 모델을 직접 로드하지 않고, 17개 Statcast 피처를 FastAPI 백엔드의 `/predict`로
보내 예측 구종·신뢰도·구종별 확률·자연어 설명을 받아 렌더링만 담당합니다. 백엔드를
분리해두었기 때문에 모델 교체나 설명 로직 변경이 웹 UI 배포와 독립적으로 이루어집니다.

## 프로젝트 구조

```
├── streamlit_app/       # 웹 UI (Streamlit + Plotly), FastAPI 백엔드를 HTTP로 호출
│   ├── app.py
│   ├── utils.py
│   └── README.md         # 실행 및 배포 방법
│
├── backend/              # 예측 API (FastAPI), Render에 배포
│   ├── main.py            # /predict, /health 엔드포인트
│   ├── model_utils.py     # 모델 로딩 및 추론
│   ├── explain.py         # Gemini API 기반 자연어 설명 (규칙 기반 폴백 포함)
│   ├── model/              # pitch_model.h5, scaler.pkl, label_encoder.pkl
│   └── README.md          # 실행 및 Render 배포 방법
│
├── desktop_app/          # 데스크톱 GUI (tkinter, 과제 원본)
│   └── src/
│       ├── data_loader.py
│       ├── model.py
│       ├── train.py
│       ├── predict.py
│       └── MLB_pitch_classifier.py
│   └── README.md
│
├── render.yaml            # backend/ Render Blueprint 배포 설정
└── results/                # 학습 곡선, 혼동 행렬
```

| | `streamlit_app/` | `backend/` | `desktop_app/` |
|---|---|---|---|
| 용도 | 웹 UI, 포트폴리오 | 예측 API 서버 | 수업 과제 제출본 |
| 실행 환경 | Streamlit Cloud | Render | 로컬 Python GUI |
| 모델 접근 | 없음 (HTTP로 backend 호출) | 직접 로딩 (TensorFlow) | 직접 로딩 (TensorFlow) |
| 자연어 설명 | backend가 내려준 텍스트 표시 | Gemini API 호출 + 폴백 | 없음 |

각 폴더의 README에서 자세한 설치/실행/배포 방법을 확인할 수 있습니다.

## 기술 스택

- **언어/환경**: Python 3.9+ (backend는 Python 3.11)
- **딥러닝**: TensorFlow / Keras (3층 MLP, BatchNorm + Dropout)
- **백엔드**: FastAPI, Render (무료 플랜)
- **자연어 설명**: Gemini API (`gemini-3.6-flash`), 실패 시 규칙 기반 설명으로 자동 폴백
- **실제 투수 데이터**: [pybaseball](https://github.com/jldbc/pybaseball) (MLB Statcast 조회, 웹 UI의 투수 검색 기능에 사용)
- **웹 UI**: Streamlit + Plotly
- **데스크톱 UI**: tkinter + matplotlib

## 구현 포인트

- **17개 Statcast 피처**(구속, 회전수, 가속도, 릴리스 위치 등) 기반 MLP 분류
- **모델/UI 분리**: 예측 로직을 FastAPI 백엔드로 분리해 Streamlit UI가 TensorFlow에 의존하지 않도록 경량화
- **자연어 설명 + 안전한 폴백**: 예측 근거를 Gemini API로 자연스러운 한국어 문장으로 생성하되, API 키가 없거나 호출이 실패/타임아웃되어도 규칙 기반 설명으로 즉시 대체해 `/predict`가 항상 응답
- **실제 투수 검색**: 이름으로 실제 MLB 투수를 찾아 최근 Statcast 투구 중 하나를 골라 바로 예측
- **좌완/우완 처리**: `p_throws`와 가속도(`ax`) 부호를 함께 반영해 상단뷰 궤적의 좌우 비대칭을 정확히 시각화
- **동적 변수 연동**: 슬라이더의 `vz0`, `vx0`를 `az`, `ax` 값에 비례시켜, 고정값으로 인해 예측이 한 클래스로 쏠리는 문제를 해결

## 한계 및 개선 방향

- 학습 데이터가 2024 MLB 정규시즌으로 한정되어 타 리그(KBO 등) 일반화는 제한적
- FS(스플리터)는 다른 구종 대비 샘플 수가 적어 상대적으로 낮은 성능
- backend가 Render 무료 플랜에서 실행되어 콜드 스타트(최대 ~50초)가 발생할 수 있음
- desktop_app은 예측마다 모델을 재로드해 평균 638ms 소요

## License

MIT
