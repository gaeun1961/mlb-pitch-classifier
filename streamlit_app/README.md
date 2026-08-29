# MLB Pitch Classifier — Web Demo (Streamlit)

MLB 투구 구종 분류 모델의 웹 UI입니다. Streamlit이 모델을 직접 로드해 예측하며,
추론 코드는 [`backend/`](../backend)의 `model_utils.py`·`explain.py`를 그대로 import 해
재사용합니다 (`backend/model/`의 아티팩트도 이 경로에서 로드).

## 기능

- **슬라이더로 직접 조절**: 구속/회전수/가속도를 조절해 가상의 투구를 만들고 즉시 예측
- **CSV에서 실제 투구 선택**: Statcast CSV를 업로드해 실제 투구 중 하나를 골라 예측
- **투수 이름으로 검색**: 실제 MLB 투수 이름을 입력하면 [pybaseball](https://github.com/jldbc/pybaseball)로
  해당 선수의 최근 Statcast 투구를 가져와 같은 방식으로 선택·예측
- 예측 결과 아래에 **왜 이 구종으로 분류됐는지 자연어 설명**을 표시 (Gemini API로 생성, 실패 시 규칙 기반 폴백)
- 모델은 `lru_cache`로 프로세스당 1회만 로드하며, 첫 예측에만 로딩 안내와 스피너를 표시

## 로컬 실행

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

브라우저가 자동으로 열리며 `http://localhost:8501`에서 확인할 수 있습니다.
`requirements.txt`에 `tensorflow-cpu`가 포함돼 있어 최초 설치에 다소 시간이 걸립니다.
Gemini 설명을 쓰려면 `GEMINI_API_KEY` 환경변수를 지정합니다 (없으면 규칙 기반 설명으로 폴백).

## Streamlit Community Cloud 배포

1. 이 레포를 GitHub에 푸시합니다 (모델 파일은 `backend/model/`에 포함돼 있습니다).
2. [share.streamlit.io](https://share.streamlit.io) 에 GitHub 계정으로 로그인합니다.
3. "New app" → 레포 선택 → Main file path에 `streamlit_app/app.py` 지정 → Deploy.
4. 앱 설정 → **Settings > Secrets**에 Gemini 키를 추가합니다 (선택):
   ```toml
   GEMINI_API_KEY = "..."
   ```
   Streamlit secrets는 `os.environ`으로도 노출되어 `explain.py`가 그대로 읽습니다.
5. 몇 분 후 `https://[앱이름].streamlit.app` 형태의 공개 링크가 생성됩니다.

> `pybaseball`이 의존성에 포함되어 있고(matplotlib, scipy, lxml 등) `tensorflow-cpu`도
> 설치하므로 빌드에 다소 시간이 걸릴 수 있습니다.

## backend와의 관계

- `utils.py`가 `sys.path`에 `../backend`를 추가하고 `model_utils.predict`·
  `explain.generate_explanation`을 import 해 예측·설명을 계산합니다.
- 동일 코드를 로컬에서 FastAPI로도 띄울 수 있습니다 (`backend/README.md` 참고).
- 궤적 계산(`compute_trajectory_side`, `compute_trajectory_top`)은 이 앱에서 순수 물리
  공식으로 직접 계산합니다.
