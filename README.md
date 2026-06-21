# MLB Pitch Classifier

MLB Statcast 투구 데이터를 기반으로 MLP 신경망이 7가지 구종(포심, 싱커, 슬라이더, 커브, 체인지업, 커터, 스플리터)을 분류하는 딥러닝 애플리케이션입니다.

가천대학교 영상처리 수업 과제로 시작해, 데스크톱 GUI에 이어 웹 데모까지 확장했습니다.

## 🌐 웹 데모

**[Live Demo →](https://share.streamlit.io)** *(배포 후 실제 링크로 교체)*

설치 없이 브라우저에서 바로 슬라이더로 가상의 투구를 만들거나, 실제 Statcast 데이터를 업로드해 모델의 예측을 확인할 수 있습니다.

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

## 프로젝트 구조

이 레포는 같은 모델을 두 가지 인터페이스로 제공합니다.

```
├── streamlit_app/      # 웹 데모 (Streamlit + Plotly)
│   ├── app.py
│   ├── utils.py
│   └── README.md        # 실행 및 배포 방법
│
├── desktop_app/         # 데스크톱 GUI (tkinter, 과제 원본)
│   └── src/
│       ├── data_loader.py
│       ├── model.py
│       ├── train.py
│       ├── predict.py
│       └── MLB_pitch_classifier.py
│   └── README.md
│
└── results/              # 학습 곡선, 혼동 행렬
```

| | `streamlit_app/` | `desktop_app/` |
|---|---|---|
| 용도 | 웹 데모, 포트폴리오 | 수업 과제 제출본 |
| UI | Plotly 인터랙티브 차트 | matplotlib 정적 차트 |
| 실행 | 브라우저 (`streamlit run`) | Python GUI (`tkinter`) |
| 모델 로딩 | 1회 캐싱 (`st.cache_resource`) | 예측마다 재로드 |

각 폴더의 README에서 자세한 설치/실행 방법을 확인할 수 있습니다.

## 기술 스택

- **언어/환경**: Python 3.9+
- **딥러닝**: TensorFlow / Keras (3층 MLP, BatchNorm + Dropout)
- **데이터**: [pybaseball](https://github.com/jldbc/pybaseball) (MLB Statcast)
- **웹 UI**: Streamlit + Plotly
- **데스크톱 UI**: tkinter + matplotlib

## 구현 포인트

- **17개 Statcast 피처**(구속, 회전수, 가속도, 릴리스 위치 등) 기반 MLP 분류
- **좌완/우완 처리**: `p_throws`와 가속도(`ax`) 부호를 함께 반영해 상단뷰 궤적의 좌우 비대칭을 정확히 시각화
- **동적 변수 연동**: 슬라이더의 `vz0`, `vx0`를 `az`, `ax` 값에 비례시켜, 고정값으로 인해 예측이 한 클래스로 쏠리는 문제를 해결

## 한계 및 개선 방향

- 학습 데이터가 2024 MLB 정규시즌으로 한정되어 타 리그(KBO 등) 일반화는 제한적
- FS(스플리터)는 다른 구종 대비 샘플 수가 적어 상대적으로 낮은 성능
- desktop_app은 예측마다 모델을 재로드해 평균 638ms 소요 (streamlit_app은 캐싱으로 개선됨)

## License

MIT
