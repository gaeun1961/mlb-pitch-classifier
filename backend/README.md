# Backend (FastAPI)

MLB Pitch Classifier의 예측 API 서버입니다. 모델 로딩, 예측, 자연어 설명 생성을 담당합니다.
`model_utils.py`·`explain.py`가 추론 코드의 원본이며, 배포된 웹 데모(`streamlit_app`)는
이 두 모듈을 그대로 import 해 재사용합니다. 이 FastAPI 서버는 로컬 실행·구조 참고용입니다.

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

`explain.py`는 예측 구종, 신뢰도, 2순위 구종, 핵심 피처(구속/az/ax/pfx/회전수/투구 손)로
프롬프트를 구성해 `gemini-3.5-flash-lite`를 호출하고 2~3문장짜리 한국어 설명을 받습니다.
(`gemini-2.0-flash`는 서비스 종료되었고, `gemini-3.6-flash`는 무료 티어가 하루 20회로
제한돼 있어 `-lite` 계열로 사용합니다.)

- API 키는 [Google AI Studio](https://aistudio.google.com)에서 발급받습니다.
- 키는 절대 코드나 저장소에 커밋하지 않고 `GEMINI_API_KEY` 환경변수로만 전달합니다.
- 키가 없거나 Gemini 호출이 실패/타임아웃(15초)되면 기존 규칙 기반 설명으로 자동
  폴백하므로 `/predict`는 Gemini 장애와 무관하게 항상 응답합니다.
- Gemini 무료 티어는 모델별로 일일 요청 한도가 있습니다(`gemini-3.6-flash`는 하루 20회로
  실사용에는 부족했습니다). 한도를 초과해도 위 폴백 덕분에 서비스는 계속되지만, 응답이
  자연어 설명 대신 규칙 기반 문장으로 조용히 바뀝니다.

## 과거 배포 (Render, 현재 미사용)

초기에는 이 서버를 Render 무료 플랜에 배포하고 `streamlit_app`이 `BACKEND_URL`로
HTTP 호출했습니다. 무료 플랜의 콜드 스타트(최대 ~50초, TensorFlow 로딩 포함)로
데모 UX가 나빠, 현재는 Streamlit Cloud가 위 추론 코드를 직접 import 하는 구조로
전환했고 Render 배포는 중단했습니다.

리포지토리 루트의 `render.yaml`은 당시 Blueprint 설정(`rootDir: backend`, 빌드/시작
커맨드, 무료 플랜)을 참고용으로 남겨둔 것입니다. 다시 API 서버로 띄우려면 Render에서
**New > Blueprint**로 이 저장소를 연결하면 `render.yaml`이 그대로 적용됩니다
(`GEMINI_API_KEY`는 `sync: false`라 배포 시 값 입력을 요구).
