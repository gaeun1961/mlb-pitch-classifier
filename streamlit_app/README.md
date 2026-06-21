# MLB Pitch Classifier — Web Demo (Streamlit)

tkinter 데스크톱 버전과 동일한 모델·로직을 사용하는 웹 데모입니다. 설치 없이 브라우저에서 바로 체험할 수 있도록 Streamlit + Plotly로 다시 구현했습니다.

## 로컬 실행

1. 이 폴더(`streamlit_app/`)에 `model/` 디렉토리를 만들고, 학습 완료된 3개 파일을 넣습니다.
   ```
   streamlit_app/
   └── model/
       ├── pitch_model.h5
       ├── scaler.pkl
       └── label_encoder.pkl
   ```
   (`desktop_app/model/` 폴더에서 그대로 복사하면 됩니다. 모델이 없다면 `desktop_app/src/train.py`를 먼저 실행하세요.)

2. 패키지 설치
   ```bash
   pip install -r requirements.txt
   ```

3. 실행
   ```bash
   streamlit run app.py
   ```
   브라우저가 자동으로 열리며 `http://localhost:8501`에서 확인할 수 있습니다.

## Streamlit Community Cloud 배포 (무료)

1. 이 `streamlit_app/` 폴더를 GitHub 레포에 푸시합니다. (`model/*.h5`, `model/*.pkl`은 용량이 작다면 함께 올리고, 크다면 Git LFS 또는 별도 호스팅을 고려하세요.)
2. [share.streamlit.io](https://share.streamlit.io) 에 GitHub 계정으로 로그인합니다.
3. "New app" → 레포 선택 → Main file path에 `streamlit_app/app.py` 지정 → Deploy.
4. 몇 분 후 `https://[앱이름].streamlit.app` 형태의 공개 링크가 생성됩니다.

## tkinter 버전과의 차이

- `predict()` 호출마다 모델을 재로드하던 구조를 `@st.cache_resource`로 바꿔 최초 1회만 로드하도록 개선했습니다.
- matplotlib 정적 차트 대신 Plotly로 인터랙티브 차트(호버 시 수치 표시)를 사용합니다.
- 야구장 다크 테마 기반의 커스텀 CSS를 적용했습니다.
