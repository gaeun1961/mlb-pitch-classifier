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

## 배포

`streamlit_app`에서 이 서버를 호출하려면 `BACKEND_URL` 환경변수(또는 Streamlit
secrets의 `BACKEND_URL`)를 이 서버의 공개 URL로 설정해야 합니다. 기본값은
`http://localhost:8000`이며 로컬 개발용입니다.
