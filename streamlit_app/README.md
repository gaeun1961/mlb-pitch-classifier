# MLB Pitch Classifier — Web Demo (Streamlit)

MLB 투구 구종 분류 모델의 웹 UI입니다. 모델을 직접 로드하지 않고, [`backend/`](../backend)의
FastAPI 서버를 HTTP로 호출해 예측 구종·신뢰도·구종별 확률·자연어 설명을 받아 표시합니다.

## 기능

- **슬라이더로 직접 조절**: 구속/회전수/가속도를 조절해 가상의 투구를 만들고 즉시 예측
- **CSV에서 실제 투구 선택**: Statcast CSV를 업로드해 실제 투구 중 하나를 골라 예측
- **투수 이름으로 검색**: 실제 MLB 투수 이름을 입력하면 [pybaseball](https://github.com/jldbc/pybaseball)로
  해당 선수의 최근 Statcast 투구를 가져와 같은 방식으로 선택·예측
- 예측 결과 아래에 **왜 이 구종으로 분류됐는지 자연어 설명**을 표시 (backend가 Gemini API로 생성)
- backend가 Render 무료 플랜에서 슬립 상태일 수 있어, 첫 예측 요청 시 콜드 스타트 안내와
  로딩 스피너를 표시

## 로컬 실행

1. 이 앱은 자체적으로 모델을 로드하지 않으므로, 먼저 [`backend/`](../backend)를 로컬에서 실행해야
   합니다 (자세한 방법은 `backend/README.md` 참고).
   ```bash
   cd ../backend
   pip install -r requirements.txt
   uvicorn main:app --reload --port 8000
   ```

2. 패키지 설치
   ```bash
   cd streamlit_app
   pip install -r requirements.txt
   ```

3. `BACKEND_URL` 환경변수로 백엔드 주소를 지정하고 실행 (지정하지 않으면 기본값
   `http://localhost:8000`을 사용합니다)
   ```bash
   export BACKEND_URL=http://localhost:8000  # 로컬 backend를 쓸 때는 생략 가능
   streamlit run app.py
   ```
   브라우저가 자동으로 열리며 `http://localhost:8501`에서 확인할 수 있습니다.

## Streamlit Community Cloud 배포

1. 이 레포를 GitHub에 푸시합니다 (모델 파일은 `backend/model/`에 있으며 이 앱은 더 이상
   모델을 직접 포함하지 않습니다).
2. [share.streamlit.io](https://share.streamlit.io) 에 GitHub 계정으로 로그인합니다.
3. "New app" → 레포 선택 → Main file path에 `streamlit_app/app.py` 지정 → Deploy.
4. 배포 후 앱 설정 → **Settings > Secrets**에 배포된 FastAPI 백엔드 주소를 추가합니다:
   ```toml
   BACKEND_URL = "https://mlb-pitch-classifier-api.onrender.com"
   ```
   (backend를 Render에 배포하는 방법은 `backend/README.md` 참고)
5. 몇 분 후 `https://[앱이름].streamlit.app` 형태의 공개 링크가 생성됩니다.

> `pybaseball`이 의존성에 포함되어 있어(matplotlib, scipy, lxml 등) 빌드에 다소 시간이
> 걸릴 수 있습니다.

## backend와의 관계

- `utils.py`의 `predict()`가 `BACKEND_URL`로 `POST /predict`를 호출해 예측·설명을 받아옵니다.
- backend 연결이 실패하면(서버 다운, 타임아웃 등) 화면에 에러 메시지를 표시할 뿐 앱이
  죽지 않습니다.
- 궤적 계산(`compute_trajectory_side`, `compute_trajectory_top`)은 backend 호출 없이
  이 앱에서 순수 물리 공식으로 직접 계산합니다.
