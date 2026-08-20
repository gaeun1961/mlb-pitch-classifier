# Backend (FastAPI)

MLB Pitch Classifier의 예측 API 서버입니다. 모델 로딩, 예측, 자연어 설명 생성을 담당하며
`streamlit_app`은 이 서버를 HTTP로 호출합니다.

## 실행

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

`GEMINI_API_KEY`를 설정하면 자연어 설명을 Gemini API로 생성합니다. 설정하지
않으면(또는 API 호출이 실패하면) 규칙 기반 설명으로 자동 폴백합니다.

로컬 개발 시 `backend/.env.example`을 복사해 `backend/.env`를 만들고 키를 채워주세요
(`.env`는 `.gitignore`에 포함되어 있어 커밋되지 않습니다). `main.py`가 시작 시
`.env`를 자동으로 읽습니다.

```bash
cp .env.example .env
# .env를 열어 GEMINI_API_KEY=발급받은키 로 채우기
```

## API

### `POST /predict`

17개 Statcast 피처와 투구 손(`p_throws`, 생략 시 `"R"`)을 받아 예측 구종, 신뢰도,
구종별 확률, 자연어 설명을 반환합니다. `p_throws`는 예측 자체에는 쓰이지 않고,
자연어 설명에서 좌우 움직임을 올바르게 해석하는 데만 사용됩니다.

요청 바디 예시:

```json
{
  "release_speed": 94.5, "release_spin_rate": 2280.0,
  "release_extension": 6.2, "release_pos_x": -1.5,
  "release_pos_z": 6.1, "pfx_x": 0.8, "pfx_z": 1.2,
  "plate_x": 0.3, "plate_z": 2.8, "vx0": 5.2,
  "vy0": -138.0, "vz0": -5.1, "ax": 8.3, "ay": 28.5,
  "az": -14.2, "effective_speed": 93.1, "spin_axis": 210.0, "p_throws": "R"
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

## 자연어 설명 (Gemini API)

`explain.py`는 예측 구종, 신뢰도, 2순위 구종, 핵심 피처(구속/az/ax/회전수)로
프롬프트를 구성해 `gemini-3.6-flash`를 호출하고 2~3문장짜리 한국어 설명을 받습니다.
(`gemini-2.0-flash`는 서비스 종료되어 API가 이 모델로 마이그레이션을 안내합니다.)

- API 키는 [Google AI Studio](https://aistudio.google.com)에서 발급받습니다.
- 키는 절대 코드나 저장소에 커밋하지 않고 `GEMINI_API_KEY` 환경변수로만 전달합니다.
- 키가 없거나 Gemini 호출이 실패/타임아웃(10초)되면 기존 규칙 기반 설명으로 자동
  폴백하므로 `/predict`는 Gemini 장애와 무관하게 항상 응답합니다.

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

`render.yaml`에 `GEMINI_API_KEY`가 `sync: false`로 선언되어 있어 Blueprint 적용 시
값 입력을 요구합니다. Blueprint 생성 시 바로 입력하거나, 이후 Render 대시보드의
서비스 → **Environment** 탭에서 `GEMINI_API_KEY`를 추가/수정할 수 있습니다.

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
