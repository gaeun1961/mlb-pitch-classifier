# Backend (FastAPI)

MLB Pitch Classifier의 예측 API 서버입니다. 모델 로딩, 예측, 자연어 설명 생성을 담당하며
`streamlit_app`은 이 서버를 HTTP로 호출합니다.

## 실행

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## API

### `POST /predict`

17개 Statcast 피처를 받아 예측 구종, 신뢰도, 구종별 확률, 자연어 설명을 반환합니다.

요청 바디 예시:

```json
{
  "release_speed": 94.5, "release_spin_rate": 2280.0,
  "release_extension": 6.2, "release_pos_x": -1.5,
  "release_pos_z": 6.1, "pfx_x": 0.8, "pfx_z": 1.2,
  "plate_x": 0.3, "plate_z": 2.8, "vx0": 5.2,
  "vy0": -138.0, "vz0": -5.1, "ax": 8.3, "ay": 28.5,
  "az": -14.2, "effective_speed": 93.1, "spin_axis": 210.0
}
```

응답 예시:

```json
{
  "predicted_label": "SL",
  "confidence": 0.843,
  "probabilities": {"FF": 0.02, "SI": 0.05, "SL": 0.843, "...": 0.0},
  "explanation": "구속 94.5mph로 매우 빠른 편, ... 등의 특징을 종합하여 모델은 이 투구를 SL(슬라이더)로 분류했습니다 (신뢰도 84.3%)."
}
```

### `GET /health`

헬스체크용 엔드포인트.

## 배포 (Render)

리포지토리 루트의 `render.yaml`이 Render Blueprint 설정을 정의합니다
(`rootDir: backend`, 빌드/시작 커맨드, 무료 플랜). 배포 절차:

1. [Render 대시보드](https://dashboard.render.com)에서 **New > Blueprint**를 선택하고
   이 GitHub 저장소(`gaeun1961/mlb-pitch-classifier`)를 연결합니다.
2. Render가 `render.yaml`을 읽어 `mlb-pitch-classifier-api` 서비스를 자동 구성합니다.
   그대로 **Apply**를 눌러 배포를 시작합니다.
3. 첫 빌드는 `tensorflow-cpu` 설치 때문에 몇 분 정도 걸릴 수 있습니다. 빌드가
   끝나면 `https://<서비스 이름>.onrender.com` 형태의 공개 URL이 발급됩니다.
4. 배포 후 `https://<서비스 URL>/health`에 접속해 `{"status": "ok"}`가 반환되는지 확인합니다.

> Render Blueprint 대신 수동으로 만들 경우: New Web Service → 이 저장소 선택 →
> Root Directory `backend` → Build Command `pip install -r requirements.txt` →
> Start Command `uvicorn main:app --host 0.0.0.0 --port $PORT` → Instance Type Free.

무료 플랜은 일정 시간 트래픽이 없으면 슬립 상태로 전환되며, 다음 요청 시
콜드 스타트로 응답이 수십 초 지연될 수 있습니다. TensorFlow 모델 로딩 자체도
콜드 스타트를 늘리는 요인이니 참고하세요.

### Streamlit Cloud에 BACKEND_URL 연결

`streamlit_app`이 이 서버를 호출하려면 `BACKEND_URL`을 서버의 공개 URL로
설정해야 합니다. 로컬 실행 시 기본값은 `http://localhost:8000`입니다.

Streamlit Cloud 앱 설정 → **Settings > Secrets**에 다음을 추가합니다:

```toml
BACKEND_URL = "https://<서비스 이름>.onrender.com"
```

저장하면 앱이 재시작되며 이후 예측 요청은 배포된 FastAPI 백엔드로 전달됩니다.
