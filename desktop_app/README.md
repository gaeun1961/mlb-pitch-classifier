# MLB Pitch Classifier — Desktop (tkinter)

가천대학교 영상처리 수업 과제로 만든 tkinter 데스크톱 버전입니다. 웹 데모는 상위 폴더의 [`streamlit_app/`](../streamlit_app)을 참고하세요.

## 실행 환경
- OS: Windows 10/11
- Python: 3.9
- IDE: Spyder (Anaconda 환경)

## 라이브러리 설치
```
conda install scikit-learn matplotlib pandas numpy
conda install -c conda-forge pip=23.3
pip install tensorflow
pip install pybaseball
```

## 실행 순서 (반드시 이 순서대로)

### 1단계. 데이터 수집
```
src/data_loader.py 실행
```
- `data/statcast_2024.csv`, `data/statcast_2025.csv` 생성 (처음 실행 시 20~30분, 이후 캐시 사용)

### 2단계. 모델 학습
```
src/train.py 실행
```
- `model/pitch_model.h5`, `model/scaler.pkl`, `model/label_encoder.pkl` 생성
- `results/confusion_matrix.png`, `results/loss_accuracy.png` 생성
- 학습 시간 약 5~10분

### 3단계. GUI 실행
```
src/MLB_pitch_classifier.py 실행
```

## 파일 구조

```
desktop_app/
├── src/
│   ├── data_loader.py             # 데이터 수집 및 전처리
│   ├── model.py                   # MLP 모델 정의
│   ├── train.py                   # 모델 학습 및 결과 저장
│   ├── predict.py                 # 예측 함수
│   └── MLB_pitch_classifier.py    # 메인 GUI
├── model/                          # 학습된 모델 (자동 생성, gitignore)
├── data/                           # Statcast 원본 데이터 (자동 생성, gitignore)
└── results/                        # 학습 곡선, 혼동 행렬 이미지
```

## 분류 구종 (7종)

| 코드 | 구종명 |
|------|--------|
| FF | 포심 패스트볼 |
| SI | 싱커 |
| SL | 슬라이더 |
| CU | 커브 |
| CH | 체인지업 |
| FC | 커터 |
| FS | 스플리터 |

## GUI 사용 방법

### 수치 예측 탭
1. 투구 수치 17개 직접 입력 또는
2. CSV 불러오기 → 투구 선택으로 자동 입력 (`p_throws` 컬럼 있으면 좌완/우완 자동 반영)
3. 구종 예측 버튼 클릭
4. 예측 결과 + 궤적 그래프 + 확률 막대그래프 확인

### 슬라이더 예측 탭
1. 4개 슬라이더(구속, 회전수, 수직 가속도 az, 수평 가속도 ax) 조절
2. 궤적/스트라이크존/예측 결과 실시간 자동 업데이트
3. 투구 손(우완/좌완) 선택 시 상단뷰 궤적 방향 자동 반영
4. 스트라이크존 클릭으로 공 위치 직접 지정 가능
5. 하단 "구종별 만들기 가이드" 표 참고 가능

## 주의사항
- `train.py` 실행 전 `data_loader.py`를 반드시 먼저 실행할 것
- `data/statcast_2024.csv`, `data/statcast_2025.csv`가 있으면 `data_loader.py`는 재실행 불필요
