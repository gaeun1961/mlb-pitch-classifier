"""app.py - MLB Pitch Classifier 웹 데모 (Streamlit) · "Pitch Workbench" 라이트 테마"""

import concurrent.futures
import re
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

from utils import (
    predict, explain, compute_trajectory_side, compute_trajectory_top,
    FEATURE_COLS, PITCH_NAMES, SAMPLE_VALUES, PITCH_DIST,
)

st.set_page_config(
    page_title="Pitch Workbench",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT_HEX = "#c53637"      # 빨강 = 지금 예측된 구종
INK = "#1c1f24"             # 본문
GRAY = "#8a8f98"            # 중립 바/마커
GRAY_SOFT = "rgba(138,143,152,0.28)"   # 배경 레퍼런스 포인트
GRAY_TRAIL = "rgba(80,86,94,0.5)"      # 세션 이력 포인트
ZONE_LINE = "#7d8086"
PLOT_BG = "#ffffff"
GRID = "#eceef0"
AXIS_TEXT = "#50565e"

FEAT_KR = {
    'release_speed': '구속', 'release_spin_rate': '스핀레이트',
    'release_extension': '릴리스 익스텐션', 'release_pos_x': '릴리스 좌우',
    'release_pos_z': '릴리스 높이', 'pfx_x': '수평 변화량', 'pfx_z': '수직 변화량',
    'plate_x': '로케이션 좌우', 'plate_z': '로케이션 높이', 'vx0': '초기 수평속도',
    'vy0': '초기 전진속도', 'vz0': '초기 수직속도', 'ax': '수평 가속도',
    'ay': '전진 가속도', 'az': '수직 가속도', 'effective_speed': '체감 구속',
    'spin_axis': '회전축',
}

PITCH_ORDER = ['FF', 'SI', 'SL', 'CU', 'CH', 'FC', 'FS']

# 무브먼트 차트 참조 클러스터용 구종별 대표 (구속, ax, az) — 가이드 테이블 기반 근사.
# 예측된 점과 같은 식으로 유도 무브먼트(inch)로 변환해 배경에 깐다.
# data/statcast_2026.csv.gz(61만 건, 우완 투수만) 실측 평균 — 예전엔 가이드 테이블 기반
# 손 추정치였는데, ax 부호가 여러 구종에서 실제와 반대였다(예: SI를 +9로 잘못 추정했지만
# 실측은 -17.5). 패스트볼 계열(FF·SI·CH·FS)은 ax가 음수, 변화구 계열(SL·CU·FC)은
# 양수로 갈리는 게 실측 데이터의 일관된 패턴이라 이쪽이 맞다.
PITCH_REF = {
    'FF': (95, -10, -15), 'SI': (95, -18, -23), 'SL': (87, 2, -30),
    'CU': (81, 7, -40), 'CH': (87, -14, -28), 'FC': (90, 1, -24), 'FS': (87, -11, -29),
}

# 구종별 만들기 가이드 = (코드, 이름, 구속, az 힌트, ax 힌트) — 위 PITCH_REF 실측치 기반.
# ax 힌트는 우완 기준 · 좌완은 좌우(+/−)가 반전된다.
GUIDE_ROWS = [
    ('FF', '포심', '93+', '↑', '−'),
    ('SI', '싱커', '92+', '↓ 살짝', '−−'),
    ('SL', '슬라이더', '82-88', '−−', '~0'),
    ('CU', '커브', '75-82', '−−−', '+'),
    ('CH', '체인지업', '80-87', '−−', '−−'),
    ('FC', '커터', '88-94', '↓ 살짝', '~0'),
    ('FS', '스플리터', '83-90', '−−', '−'),
]

# ── 디자인 토큰 · 라이트 테마 CSS ────────────────────────
# Streamlit은 매 rerun마다 DOM을 새로 만들므로 이 <style>도 매번 재주입돼야 한다.
# 문자열은 모듈 상수라 재생성 비용은 없다(= 성능 병목 아님).
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Public+Sans:wght@400;500;600;700&display=swap');

:root{
  --card:#ffffff;
  --card-border:#e3e5e8;
  --text:#1c1f24;
  --text-2:#565b63;
  --text-3:#7a8089;
  --label:#6b7078;
  --accent:#c53637;
  --chip-gray:#f4f5f6;
  --track:#e9eaec;
}

html,body,[class*="css"]{font-family:'Public Sans',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;}
.stApp{background:#ffffff;color:var(--text);}
h1,h2,h3,h4{font-family:'Sora',sans-serif !important;color:var(--text);}
#MainMenu,footer{visibility:hidden;}
header[data-testid="stHeader"]{height:0 !important;visibility:hidden;}
hr{border-color:var(--card-border) !important;}

/* 상단 여백 축소 */
[data-testid="stMainBlockContainer"],.block-container{padding-top:1.1rem !important;}

/* 사이드바 컴팩트화 */
section[data-testid="stSidebar"]{background:var(--card);border-right:1px solid var(--card-border);}
section[data-testid="stSidebar"] .block-container{padding:0.7rem 1rem 1rem;}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:0.5rem;}
section[data-testid="stSidebar"] hr{margin:0.45rem 0 !important;}
section[data-testid="stSidebar"] [data-testid="stSlider"]{padding-bottom:0.15rem;}
section[data-testid="stSidebar"] [data-testid="stSliderTickBar"]{display:none;}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"]{margin-bottom:0.1rem;}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{font-size:0.82rem;}
section[data-testid="stSidebar"] .stExpander{margin-top:0.2rem;}

div[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--card);
  border:1px solid var(--card-border) !important;
  border-radius:16px !important;
  box-shadow:0 1px 2px rgba(16,24,40,0.03);
}
div[data-testid="stVerticalBlockBorderWrapper"] > div{padding:14px 18px;}

.pw-label{font-size:11px;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;color:var(--label);margin-bottom:10px;}

.pw-logo{display:flex;align-items:center;gap:9px;margin:0 0 10px;}
.pw-logo span{font-family:'Sora';font-weight:800;font-size:1.05rem;color:var(--text);}

/* 결과 요약 스트립 — 4칸 균등, 각 칸 왼쪽 정렬(라벨 위 / 큰 값 아래). 예측 구종만 빨간 테두리 */
.pw-summary{display:flex;gap:10px;align-items:stretch;margin-bottom:16px;}
.pw-sum-pred,.pw-sum-chip{flex:1 1 0;display:flex;flex-direction:column;align-items:flex-start;justify-content:center;gap:3px;background:#fff;border-radius:12px;padding:8px 14px;}
.pw-sum-pred{border:1.5px solid var(--accent);}
.pw-sum-chip{border:1px solid var(--card-border);}
.pw-sum-k{font-size:0.66rem;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;color:var(--text-3);}
.pw-sum-v{font-family:'Sora';font-size:1.2rem;font-weight:700;color:var(--text);line-height:1.1;}
.pw-sum-pred .pw-sum-v{display:inline-flex;align-items:center;gap:6px;}
.pw-sum-pred i{width:8px;height:8px;border-radius:50%;background:var(--accent);display:inline-block;}

/* 구종별 확률 바 — 전부 회색, 예측 구종만 빨강 */
.pw-bar-row{display:flex;align-items:center;gap:10px;margin:6px 0;}
.pw-bar-name{width:82px;font-size:0.82rem;color:var(--text-2);flex:none;}
.pw-bar-track{flex:1;height:8px;background:var(--track);border-radius:999px;overflow:hidden;}
.pw-bar-fill{display:block;height:100%;border-radius:999px;}
.pw-bar-pct{width:40px;text-align:right;font-size:0.8rem;font-weight:600;color:var(--text);flex:none;}

.pw-shap-head{display:flex;align-items:center;gap:8px;font-family:'Sora';font-weight:700;font-size:1rem;color:var(--text);}
.pw-shap-sub{font-size:0.78rem;color:var(--text-3);margin:2px 0 12px;}
.pw-shap-expl{font-size:0.82rem;line-height:1.6;color:var(--text-2);background:var(--chip-gray);padding:10px 12px;border-radius:10px;margin:0 0 14px;}
.pw-wf-ends{display:flex;justify-content:space-between;font-size:0.72rem;color:var(--text-3);margin:-6px 2px 14px;}
.pw-fcard{border:1px solid var(--card-border);border-radius:12px;padding:11px 13px;margin-bottom:9px;}
.pw-fcard-top{display:flex;justify-content:space-between;align-items:center;}
.pw-fname{font-size:0.85rem;font-weight:600;color:var(--text);}
.pw-fbadge{font-size:0.75rem;font-weight:700;padding:2px 8px;border-radius:999px;background:#fff;border:1px solid var(--card-border);color:var(--text-2);}
.pw-fbadge.pos{color:var(--accent);border-color:var(--accent);}
.pw-fmag{height:3px;background:var(--track);border-radius:999px;margin:9px 0 8px;overflow:hidden;}
.pw-fmag span{display:block;height:100%;border-radius:999px;}
.pw-fdesc{font-size:0.78rem;line-height:1.55;color:var(--text-3);margin:0;}

.pw-guide{width:100%;table-layout:fixed;border-collapse:collapse;font-size:0.76rem;color:var(--text-2);}
.pw-guide th{font-size:0.6rem;font-weight:700;letter-spacing:0.03em;text-transform:uppercase;color:var(--text-3);text-align:left;padding:0 3px 5px;border-bottom:1px solid var(--card-border);}
.pw-guide td{padding:6px 3px;border-bottom:1px solid #f1f2f4;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.pw-guide td:first-child{color:var(--text);font-weight:500;white-space:normal;}
.pw-guide tbody tr:last-child td{border-bottom:none;}
.pw-gcode{display:inline-block;font-family:'Sora';font-weight:700;font-size:0.64rem;background:var(--chip-gray);color:var(--text-2);padding:2px 5px;border-radius:5px;margin-right:6px;}
.pw-gnote{font-size:9px;color:var(--text-3);margin-top:8px;line-height:1.4;}

div[data-baseweb="tab-list"]{gap:6px;border-bottom:none !important;}
button[data-baseweb="tab"]{background:var(--chip-gray);border-radius:999px;padding:5px 15px !important;font-family:'Sora';font-weight:600;color:var(--text-2);min-height:0;}
button[data-baseweb="tab"][aria-selected="true"]{background:var(--text);color:#fff;}
button[data-baseweb="tab"][aria-selected="true"] p{color:#fff !important;}
div[data-baseweb="tab-highlight"],div[data-baseweb="tab-border"]{display:none !important;}
button[data-baseweb="tab"] p{font-size:0.8rem !important;font-weight:600 !important;}

div[role="radiogroup"]{gap:6px;flex-wrap:wrap;}
div[role="radiogroup"] > label{background:var(--chip-gray);border:1px solid var(--card-border);border-radius:999px;padding:4px 12px;margin:0;}
div[role="radiogroup"] > label:has(input:checked){background:var(--text);border-color:var(--text);}
div[role="radiogroup"] > label:has(input:checked) div{color:#fff;}
div[role="radiogroup"] > label > div:first-child{display:none;}

/* 입력창 — 테두리는 config.toml showWidgetBorder가 그려주고, 여기선 연한 회색
   채움을 더해 흰 배경에서도 필드 경계가 확실히 보이게 한다 */
[data-testid="stTextInputRootElement"],
.stTextInput div[data-baseweb="base-input"],
.stDateInput div[data-baseweb="input"],
.stNumberInput div[data-baseweb="input"],
.stSelectbox div[data-baseweb="select"] > div{background:#f4f5f7 !important;}
.stTextInput input,.stDateInput input,.stNumberInput input{background:#f4f5f7 !important;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── 세션 상태 초기화 ─────────────────────────────────────
if 'inp' not in st.session_state:
    st.session_state.inp = dict(SAMPLE_VALUES)
if 'p_throws' not in st.session_state:
    st.session_state.p_throws = 'R'
if 'history' not in st.session_state:
    st.session_state.history = []          # 이번 세션에서 예측해 본 (ax, az) 이력
if 'movement_bg' not in st.session_state:
    st.session_state.movement_bg = None    # (pfx_x_in[], pfx_z_in[]) — 실제 데이터 로드 시에만
if 'label' not in st.session_state:
    st.session_state.label = None
    st.session_state.conf = None
    st.session_state.proba = None
    st.session_state.attribution = None
    st.session_state.explanation = None
    st.session_state.inf_ms = 0.0
if 'warmed_up' not in st.session_state:
    st.session_state.warmed_up = False
if 'favorites' not in st.session_state:
    st.session_state.favorites = []          # [{"id":, "name":, "label":}] — 이번 세션 한정(로그인 없어 서버 저장 불가)

MAX_FAVORITES = 10


def toggle_favorite(pid, name, label):
    favs = st.session_state.favorites
    if any(f['id'] == pid for f in favs):
        st.session_state.favorites = [f for f in favs if f['id'] != pid]
    elif len(favs) < MAX_FAVORITES:
        favs.append({'id': pid, 'name': name, 'label': label})

# ax/az/릴리스좌우 슬라이더는 key 기반(클릭·손 전환으로도 값이 바뀜). 최초 1회만 시드.
st.session_state.setdefault("ax_slider", float(SAMPLE_VALUES['ax']))
st.session_state.setdefault("az_slider", float(SAMPLE_VALUES['az']))
st.session_state.setdefault("rx_slider", float(SAMPLE_VALUES['release_pos_x']))


def _snap(v, lo, hi, step=0.5):
    return float(np.clip(round(v / step) * step, lo, hi))


def _raw_pfx(spd, ax, az, ext):
    t = (PITCH_DIST - ext) / max(spd * 1.467, 1)
    return (0.20 * ax * t + 0.5 * ax * t * t,
            (-3.0 - 0.15 * az) * t + 0.5 * az * t * t)


# 앱의 궤적식은 유도 무브먼트에 일정한 오프셋이 있어(모든 구종이 음수) Savant처럼
# 0을 기준으로 안 퍼진다. PITCH_REF 7개의 평균을 빼서 "포심 위 / 커브 아래" 로 재중심화.
_PFX0 = [sum(c) / len(c)
         for c in zip(*(_raw_pfx(s, a, z, 6.2) for s, a, z in PITCH_REF.values()))]


def _pfx_inches(spd, ax, az, ext=6.2):
    """앱 궤적식 유도 무브먼트를 재중심화해 inch로. 예측 점·참조 클러스터·이력이 모두
    이 함수를 거쳐 같은 좌표계(위 = 라이징/포심, 아래 = 드롭/커브)를 쓴다."""
    rx, rz = _raw_pfx(spd, ax, az, ext)
    return (rx - _PFX0[0]) * 12.0, (rz - _PFX0[1]) * 12.0


def in_data_mode():
    """실제 투구 히스토리가 로드된 모드인지. 릴리스포인트·구속분포·히트맵처럼 여러 투구가
    있어야 그릴 수 있는 보조 차트는 이때만 렌더한다(아니면 안내 placeholder)."""
    return st.session_state.get('movement_bg') is not None


def _apply_zone_click():
    """스트라이크존 클릭 좌표를 입력에 반영한다.

    - 좌우(px): 사이드바 plate_x 식(vx0 = 0.20·ax)을 뒤집어 ax 슬라이더를 맞춘다.
    - 높이(pz): plate_z는 az에 거의 둔감해서(계수가 작다) az 대신 vz0을 직접 풀어
      맞춘다. `_click_vz0`/`_click_ctx`로 넘겨서, 클릭 이후 슬라이더를 건드리기
      전까지만 이 vz0을 쓰고 그 뒤엔 일반 식으로 돌아간다.
    구속·회전수·익스텐션은 사용자가 맞춘 값을 그대로 둔다.
    """
    sel = st.session_state.get("zone_click")
    if not sel:
        return
    try:
        pts = sel["selection"]["points"]
    except (TypeError, KeyError):
        pts = getattr(getattr(sel, "selection", None), "points", None) or []
    if not pts:
        return
    px, pz = float(pts[-1]["x"]), float(pts[-1]["y"])
    sig = (round(px, 2), round(pz, 2))
    if st.session_state.get("_zone_sig") == sig:
        return
    st.session_state["_zone_sig"] = sig

    spd = round(float(st.session_state.inp['release_speed']), 2)
    ext = round(float(st.session_state.inp['release_extension']), 2)
    rz = float(st.session_state.inp['release_pos_z'])
    rx = float(st.session_state.inp['release_pos_x'])
    t = (PITCH_DIST - ext) / max(spd * 1.467, 1)

    ax = _snap((px - rx) / (0.2 * t + 0.5 * t * t), -30.0, 30.0)
    az = st.session_state["az_slider"]
    vz0 = _snap((pz - rz - 0.5 * az * t * t) / t, -14.0, 3.0, step=0.01)

    st.session_state["ax_slider"] = ax
    st.session_state["_click_vz0"] = vz0
    st.session_state["_click_ctx"] = (ax, az, spd, ext)


_apply_zone_click()


@st.cache_data(show_spinner=False)
def _predict_cached(items):
    """동일 입력이면 모델 추론을 다시 하지 않도록 결과를 캐시한다 (빠름, 네트워크 호출 없음).

    슬라이더 값이 안 바뀐 rerun(탭 전환, 필터 클릭 등)에서는 즉시 반환된다.
    반환 튜플 끝에 최초 계산 시 측정한 추론 시간(ms)을 함께 담는다.
    """
    t0 = time.perf_counter()
    label, conf, proba, attribution = predict(dict(items))
    return label, conf, proba, attribution, (time.perf_counter() - t0) * 1000


@st.cache_resource
def _explain_executor():
    """설명 생성용 백그라운드 스레드 풀 (프로세스당 1개, rerun에도 유지)."""
    return concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _explain_worker(items, p_throws):
    """스레드에서 실행 — st.cache_data는 메인 스크립트 컨텍스트에 의존하므로 여기서는 안 씀."""
    label, conf, proba, _ = predict(dict(items))
    return explain(dict(items), label, conf, proba, p_throws)


def run_prediction(inp):
    """빠른 모델 예측을 실행(캐시 경유)해 세션 상태를 갱신하고, 느린 자연어 설명은
    백그라운드 스레드에 맡긴다. 실패 시 에러 메시지를 표시한다.

    설명 생성(Gemini API, 최대 15초)을 슬라이더 조작 등과 같은 스레드에서 동기 호출하면
    그 인터랙션 자체가 최대 15초씩 막힌다. 스레드 풀에 맡기고 render_shap_panel의
    자동 새로고침 프래그먼트가 완료 여부를 폴링하면, 예측·차트는 항상 즉시 반응하면서도
    설명은 준비되는 대로 자동으로 표시된다.
    """
    spinner_msg = (
        "예측 중입니다..." if st.session_state.warmed_up
        else "예측 중입니다... (첫 예측은 모델 로딩으로 잠시 걸릴 수 있어요)"
    )
    try:
        with st.spinner(spinner_msg):
            key = tuple(sorted(inp.items()))
            label, conf, proba, attribution, inf_ms = _predict_cached(key)
        st.session_state.label = label
        st.session_state.conf = conf
        st.session_state.proba = proba
        st.session_state.attribution = attribution
        # run_prediction은 입력이 그대로여도 매 rerun마다 불린다 — 실제로 입력이
        # 바뀌었을 때만 새 설명 작업을 스레드 풀에 새로 제출한다.
        if st.session_state.get('_last_pred_key') != key:
            st.session_state.explanation = None
            st.session_state['_explain_future'] = _explain_executor().submit(
                _explain_worker, key, st.session_state.p_throws)
        st.session_state['_last_pred_key'] = key
        st.session_state.inf_ms = inf_ms
        st.session_state.warmed_up = True

        hist = st.session_state.history
        pt = tuple(round(v, 2) for v in _pfx_inches(
            inp['release_speed'], inp['ax'], inp['az'], inp['release_extension']))
        if not hist or hist[-1] != pt:
            hist.append(pt)
            del hist[:-40]                  # 최근 40개만 유지
    except Exception as e:
        st.error(f"예측에 실패했습니다: {e}")


def select_and_predict_from_df(df):
    """DataFrame에서 결측치 없는 행만 골라 선택 UI를 보여주고, 선택된 투구로 예측을 실행한다."""
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        st.error(f"필요한 컬럼이 없습니다: {missing}")
        return

    df_valid = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    if df_valid.empty:
        st.warning("선택 가능한 투구가 없습니다.")
        return

    st.success(f"{len(df_valid):,}개 투구 로드 완료")

    # 무브먼트 차트 배경 = 이 데이터셋의 실제 유도 무브먼트(inch). 실제 Statcast pfx_z는
    # 이미 위=라이징 관례라 부호 반전 없이 그대로 × 12.
    bg = df_valid[['pfx_x', 'pfx_z']]
    if len(bg) > 500:
        bg = bg.sample(500, random_state=0)
    st.session_state.movement_bg = ((bg['pfx_x'] * 12.0).tolist(),
                                    (bg['pfx_z'] * 12.0).tolist())

    show_cols = [c for c in ['player_name', 'p_throws', 'release_speed',
                              'release_spin_rate', 'game_date'] if c in df_valid.columns]
    idx = st.selectbox(
        "투구 선택",
        options=df_valid.index,
        format_func=lambda i: " | ".join(str(df_valid.loc[i, c]) for c in show_cols)
    )
    row = df_valid.loc[idx]
    inp = {col: float(row[col]) for col in FEATURE_COLS}
    st.session_state.inp = inp
    if 'p_throws' in df_valid.columns:
        st.session_state.p_throws = str(row.get('p_throws', 'R'))
    run_prediction(inp)


def contains_korean(text):
    return bool(re.search(r'[가-힣]', text))


@st.cache_data(show_spinner=False, ttl=1800)
def search_pitchers(query):
    """MLB 공식 Stats API로 이름을 검색해 투수만 필터링한 후보 목록을 반환한다.

    pybaseball의 playerid_lookup()은 포지션 정보가 없고 성(last name) 기준 검색만
    지원해 자동완성에 부적합하다. MLB Stats API는 이름 부분 일치 검색과 포지션·소속팀·
    데뷔년도를 함께 제공해 이 용도에 더 적합하다.
    """
    resp = requests.get(
        "https://statsapi.mlb.com/api/v1/people/search",
        params={"names": query, "hydrate": "currentTeam"},
        timeout=10,
    )
    resp.raise_for_status()

    candidates = []
    for p in resp.json().get("people", []):
        # "P" = 투수, "TWP" = 투타겸업(예: 오타니 쇼헤이) — 포지션을 완전히 바꾼
        # 선수라도 이 두 포지션 이력이 있으면 과거 투구 기록이 있을 수 있다.
        if p.get("primaryPosition", {}).get("abbreviation") not in ("P", "TWP"):
            continue
        debut = p.get("mlbDebutDate")
        if not debut:
            continue  # MLB 출전 기록이 없으면 Statcast 데이터도 없다
        last_played = p.get("lastPlayedDate")
        candidates.append({
            "id": p["id"],
            "name": p["fullName"],
            "team": p.get("currentTeam", {}).get("name", "소속팀 미상"),
            "debut_year": debut[:4],
            "debut_date": debut[:10],
            "last_year": last_played[:4] if last_played else None,
            "last_played_date": last_played[:10] if last_played else None,
            "active": p.get("active", False),
            "throws": p.get("pitchHand", {}).get("code", "R"),
        })

    candidates.sort(key=lambda c: (not c["active"], -int(c["debut_year"])))
    return candidates[:15]


STATCAST_START_YEAR = 2015
STATCAST_START_DATE = date(STATCAST_START_YEAR, 1, 1)
MAX_DATE_ONLY_DAYS = 5      # 이름 없이 날짜만으로 검색할 때 허용하는 최대 기간(리그 전체 조회라 무거움)
MAX_DATE_ONLY_ROWS = 500    # 날짜만 검색 시 화면에 표시할 최대 투구 수


def career_label(c):
    """'이름 (팀, 데뷔~현재)' 또는 은퇴 선수는 '이름 (데뷔~은퇴, 은퇴)' 형식으로 표기한다."""
    if c["active"]:
        return f"{c['name']} ({c['team']}, {c['debut_year']}~현재)"
    end = c["last_year"] or c["debut_year"]
    return f"{c['name']} ({c['debut_year']}~{end}, 은퇴)"


def pitcher_date_bounds(c):
    """이 투수의 Statcast 조회 가능 날짜 범위(최소·최대)와 기본 조회 범위(최근 시즌)를 반환한다.

    Statcast 시작일(2015-01-01) 이전에 은퇴한 선수는 None을 반환한다.
    """
    debut = date.fromisoformat(c["debut_date"])
    today = date.today()
    min_d = max(debut, STATCAST_START_DATE)

    if c["active"]:
        max_d = today
    else:
        last = c["last_played_date"]
        max_d = min(date.fromisoformat(last) if last else debut, today)

    if min_d > max_d:
        return None

    def_start = max(date(max_d.year, 1, 1), min_d)
    return min_d, max_d, def_start, max_d


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_pitcher_statcast(player_id, start_dt, end_dt):
    """해당 투수의 기간 내 Statcast 투구 데이터를 가져온다."""
    from pybaseball import statcast_pitcher

    return statcast_pitcher(start_dt, end_dt, player_id)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_statcast_range(start_dt, end_dt):
    """투수 이름 없이 날짜 범위만으로 리그 전체 Statcast 투구 데이터를 가져온다."""
    from pybaseball import statcast

    return statcast(start_dt, end_dt)


# ── 렌더 헬퍼 ────────────────────────────────────────────
LOGO_HTML = """
<div class="pw-logo">
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#c53637"
       stroke-width="1.6" stroke-linecap="round">
    <circle cx="12" cy="12" r="9"/>
    <path d="M6.2 4.8c2.7 2.9 2.7 11.5 0 14.4"/>
    <path d="M17.8 4.8c-2.7 2.9-2.7 11.5 0 14.4"/>
  </svg>
  <span>Pitch Workbench</span>
</div>
"""


def _flip_ax(hint):
    """ax 힌트의 +/− 를 서로 뒤집는다 (좌완 표시용)."""
    return hint.replace('+', '\x00').replace('−', '+').replace('\x00', '−')


def render_guide(p_throws):
    is_left = p_throws == 'L'
    rows = ""
    for code, name, spd, az_hint, ax_hint in GUIDE_ROWS:
        ax_disp = _flip_ax(ax_hint) if is_left else ax_hint
        _, ref_ax, ref_az = PITCH_REF[code]
        ref_ax_disp = -ref_ax if is_left else ref_ax
        rows += (
            f'<tr><td><span class="pw-gcode">{code}</span>{name}</td>'
            f'<td>{spd}</td>'
            f'<td>{az_hint} ({ref_az:+.0f})</td>'
            f'<td>{ax_disp} ({ref_ax_disp:+.0f})</td></tr>'
        )
    side = "좌완" if is_left else "우완"
    other = "우완" if is_left else "좌완"
    st.markdown(
        '<table class="pw-guide"><colgroup>'
        '<col style="width:22%"><col style="width:22%"><col style="width:28%"><col style="width:28%">'
        '</colgroup><thead><tr>'
        '<th>구종</th><th>구속 (mph)</th><th>수직 az</th><th>수평 ax</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
        f'<div class="pw-gnote">괄호 안은 슬라이더에 넣어볼 대략적인 az·ax 값(실제 예측은 17개 피처 전체로 계산) · '
        f'수평 ax는 {side} 기준 · {other}은 +/− 반대</div>',
        unsafe_allow_html=True,
    )


def render_summary(label, conf, inf_ms):
    st.markdown(
        '<div class="pw-summary">'
        '<span class="pw-sum-pred"><span class="pw-sum-k">예측 구종</span>'
        f'<span class="pw-sum-v"><i></i>{PITCH_NAMES.get(label, label)}</span></span>'
        f'<span class="pw-sum-chip"><span class="pw-sum-k">신뢰도</span>'
        f'<span class="pw-sum-v">{conf * 100:.0f}%</span></span>'
        '<span class="pw-sum-chip"><span class="pw-sum-k">모델 정확도</span>'
        '<span class="pw-sum-v">93.4%</span></span>'
        f'<span class="pw-sum-chip"><span class="pw-sum-k">추론 시간</span>'
        f'<span class="pw-sum-v">{inf_ms:.0f}ms</span></span>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_prob_dist(proba, label):
    st.markdown('<div class="pw-label">구종별 확률 분포</div>', unsafe_allow_html=True)
    rows = ""
    for k in PITCH_ORDER:
        if k not in proba:
            continue
        v = proba[k] * 100
        color = ACCENT_HEX if k == label else GRAY
        rows += (
            '<div class="pw-bar-row">'
            f'<span class="pw-bar-name">{PITCH_NAMES[k]}</span>'
            '<span class="pw-bar-track">'
            f'<span class="pw-bar-fill" style="width:{v:.1f}%;background:{color}"></span>'
            '</span>'
            f'<span class="pw-bar-pct">{v:.0f}%</span>'
            '</div>'
        )
    st.markdown(rows, unsafe_allow_html=True)


def render_movement(inp):
    """유도 무브먼트(inch) 평면. 원점 중심 12·24인치 동심원 + 구종별 참조 클러스터
    (회색 원, 라벨 없음, 예측 구종만 빨간 테두리) 위에 현재 예측 투구를 빨간 링으로.
    세로는 Statcast 관례(위 = 라이징/포심, 아래 = 드롭/커브). 배경 회색 점은 실제
    데이터 로드 시 그 분포, 아니면 세션 예측 이력. 예측 점·클러스터·이력은 모두
    같은 식(_pfx_inches)이라 서로 비교 가능하다."""
    st.markdown('<div class="pw-label">무브먼트 차트 (유도 무브먼트, in)</div>',
                unsafe_allow_html=True)
    bg = st.session_state.get('movement_bg')
    hist = st.session_state.history
    label = st.session_state.label

    fig = go.Figure()
    for r in (12, 24):
        fig.add_shape(type="circle", x0=-r, y0=-r, x1=r, y1=r, layer="below",
                      line=dict(color="#e5e5e5", width=1, dash="dot"))
    fig.add_hline(y=0, line_color="#e5e5e5", line_width=1)
    fig.add_vline(x=0, line_color="#e5e5e5", line_width=1)

    for code, ref in PITCH_REF.items():
        cx, cy = _pfx_inches(*ref)
        hot = code == label
        # add_shape는 호버를 못 받으므로, 마우스오버 시 구종명이 뜨도록 큰 마커의
        # scatter로 그린다(반경 4.5인치 ≈ 이 차트 스케일에서 지름 50px 근사).
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode='markers', showlegend=False,
            marker=dict(size=50, color="rgba(140,143,152,0.10)",
                        line=dict(color=ACCENT_HEX if hot else "rgba(140,143,152,0.45)",
                                  width=2 if hot else 1)),
            hovertemplate=f"{PITCH_NAMES.get(code, code)}<extra></extra>",
        ))

    if bg:
        fig.add_trace(go.Scatter(x=bg[0], y=bg[1], mode='markers',
                                 marker=dict(size=5, color=GRAY_SOFT), hoverinfo='skip'))
        ctx = "회색 점 = 불러온 실제 투구 분포"
    elif hist:
        fig.add_trace(go.Scatter(x=[h[0] for h in hist], y=[h[1] for h in hist],
                                 mode='markers', marker=dict(size=7, color=GRAY_TRAIL),
                                 hoverinfo='skip'))
        ctx = "회색 점 = 이번 세션에서 예측해 본 투구"
    else:
        ctx = "슬라이더를 움직이면 예측 이력이 쌓입니다"

    px, py = _pfx_inches(inp['release_speed'], inp['ax'], inp['az'], inp['release_extension'])
    fig.add_trace(go.Scatter(x=[px], y=[py], mode='markers',
                             marker=dict(size=22, color='rgba(0,0,0,0)',
                                         line=dict(color=ACCENT_HEX, width=3))))
    fig.add_trace(go.Scatter(x=[px], y=[py], mode='markers',
                             marker=dict(size=7, color=ACCENT_HEX)))
    fig.update_layout(
        height=380, margin=dict(l=44, r=20, t=10, b=38),
        plot_bgcolor=PLOT_BG, paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=AXIS_TEXT, size=11), showlegend=False,
        xaxis=dict(title='수평 무브먼트 (in)', range=[-30, 30], gridcolor=GRID,
                   zeroline=False, scaleanchor='y'),
        yaxis=dict(title='수직 무브먼트 (in) · 위 라이징 / 아래 드롭', range=[-30, 30],
                   gridcolor=GRID, zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.caption(f"점선 원 = 12·24인치 · 회색 원 = 구종별 대략 위치(예측 구종 빨강) · {ctx}")
    st.caption("⚠️ 회색 원은 2026시즌 실측 평균의 대략적인 경향일 뿐, 실제 예측은 이 차트에 없는 "
               "회전축 등 15개 피처까지 합쳐 계산합니다 — 그래서 예측 구종이 자기 회색 원과 "
               "떨어져 보일 수 있습니다.")


def render_strikezone(inp, clickable=False):
    """주심(포수 뒤)이 마운드를 바라본 시점. 회색 점 = 릴리스, 빨간 점 = 플레이트
    통과 위치. clickable이면 존을 클릭해 공 위치를 지정할 수 있다."""
    st.markdown('<div class="pw-label">로케이션 · 스트라이크존</div>', unsafe_allow_html=True)
    fig = go.Figure()
    # 평균 존: 홈플레이트 폭 ±0.83ft(17인치 절반 + 볼 반경), 무릎~겨드랑이 1.5~3.5ft 근사
    fig.add_shape(type="rect", x0=-0.83, x1=0.83, y0=1.5, y1=3.5,
                  line=dict(color=ZONE_LINE, width=2),
                  fillcolor="rgba(138,143,152,0.06)", layer="below")
    if clickable:
        # 옅은 격자 = 클릭 히트타깃(≈0.15ft 간격). 클릭하면 가장 가까운 점이 선택된다.
        # 완전 투명 + hoverinfo='skip'이면 Plotly가 클릭 이벤트를 안 잡으므로 옅게라도 그린다.
        gx, gy = np.meshgrid(np.arange(-1.8, 1.81, 0.15), np.arange(0.3, 4.81, 0.15))
        fig.add_trace(go.Scatter(x=gx.ravel(), y=gy.ravel(), mode='markers',
                                 name='click', marker=dict(size=16, color='rgba(140,143,152,0.16)'),
                                 hoverinfo='none'))
    # 릴리스 지점(회색). plate_x/release_pos_x는 Statcast와 같은 포수·주심 시점 좌표
    fig.add_trace(go.Scatter(
        x=[inp['release_pos_x']], y=[inp['release_pos_z']], mode='markers',
        marker=dict(color=GRAY, size=13, line=dict(color='white', width=1.5)),
        hoverinfo='skip',
    ))
    fig.add_annotation(x=inp['release_pos_x'], y=inp['release_pos_z'], yshift=13,
                       text="릴리스", showarrow=False, font=dict(size=9, color=AXIS_TEXT))
    fig.add_trace(go.Scatter(
        x=[inp['plate_x']], y=[inp['plate_z']], mode='markers',
        marker=dict(color=ACCENT_HEX, size=20, line=dict(color='white', width=1.5)),
        hoverinfo='skip',
    ))
    fig.update_layout(
        height=400, margin=dict(l=40, r=10, t=10, b=32),
        plot_bgcolor=PLOT_BG, paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=AXIS_TEXT, size=11), showlegend=False,
        xaxis=dict(title='좌우 (ft) · 주심 시점', range=[-2.5, 2.5], gridcolor=GRID,
                   zeroline=False, scaleanchor='y'),
        yaxis=dict(title='높이 (ft)', range=[0, 7.5], gridcolor=GRID, zeroline=False),
    )
    if clickable:
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False},
                        on_select="rerun", selection_mode="points", key="zone_click")
        st.caption("존 안을 클릭하면 그 위치로 공이 이동합니다 · 구속·회전수·익스텐션은 유지 · "
                   "회색 점 = 릴리스 지점")
        st.caption("좌우는 수평 무브먼트(ax) 슬라이더에 반영되지만, 높이는 수직 무브먼트(az)가 "
                   "아니라 화면에 없는 초기 수직속도로 맞춥니다 — az로 높이를 맞추려 하면 az 변화가 "
                   "초기 수직속도에도 반대로 영향을 줘 서로 상쇄돼서, 슬라이더 범위를 넘길 만큼 크게 "
                   "움직여야 하기 때문입니다. 그래서 클릭해도 '수직 무브먼트' 슬라이더 값 자체는 그대로예요.")
    else:
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.caption("주심이 마운드를 바라본 시점 · 회색 점 = 릴리스 · 빨간 점 = 플레이트 통과 위치")


def render_paths(inp, p_throws):
    """측면뷰·상단뷰 궤적. 홈플레이트에 스트라이크존 박스로 존 대비 위치를 표시한다."""
    st.markdown('<div class="pw-label">투구 궤적</div>', unsafe_allow_html=True)
    side_x, side_z = compute_trajectory_side(inp)
    top_y, top_x = compute_trajectory_top(inp, p_throws)

    # 궤적 끝점을 모델이 실제 쓰는 plate 좌표(= 로케이션·스트라이크존의 빨간 점)에 맞춘다.
    # 정상 입력이면 이미 거의 같아 티 안 남. 극단값이라 plate_x/z가 clip된 경우에만 꼬리 보정.
    w = np.linspace(0.0, 1.0, len(top_x))
    top_x = np.asarray(top_x) + (inp['plate_x'] - top_x[-1]) * w
    side_z = np.asarray(side_z) + (inp['plate_z'] - side_z[-1]) * w

    fig = go.Figure()
    # 스트라이크존 높이 1.5~3.5ft: 전 구간 가로 점선 + 홈플레이트에 채운 존 박스
    fig.add_hline(y=1.5, line_dash="dot", line_color=ZONE_LINE, line_width=1)
    fig.add_hline(y=3.5, line_dash="dot", line_color=ZONE_LINE, line_width=1)
    fig.add_shape(type="rect", x0=PITCH_DIST - 1.5, x1=PITCH_DIST, y0=1.5, y1=3.5,
                  line=dict(color=ZONE_LINE, width=2),
                  fillcolor="rgba(140,143,152,0.14)", layer="below")
    # 투수판(거리 0) · 홈플레이트(거리 60.5)
    fig.add_shape(type="line", x0=0, x1=0, y0=0, y1=0.6, line=dict(color=ZONE_LINE, width=5))
    fig.add_annotation(x=0, y=0.6, yshift=9, text="투수판", showarrow=False,
                       font=dict(size=9, color=AXIS_TEXT))
    fig.add_annotation(x=PITCH_DIST, y=3.5, yshift=10, text="홈플레이트", showarrow=False,
                       font=dict(size=9, color=AXIS_TEXT))
    fig.add_trace(go.Scatter(x=side_x, y=side_z, mode='lines',
                             line=dict(color=ACCENT_HEX, width=3)))
    fig.add_trace(go.Scatter(x=[side_x[0], side_x[-1]], y=[side_z[0], side_z[-1]],
                             mode='markers', marker=dict(color=ACCENT_HEX, size=8)))
    fig.update_layout(
        height=300, margin=dict(l=40, r=10, t=10, b=30),
        plot_bgcolor=PLOT_BG, paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=AXIS_TEXT, size=11), showlegend=False,
        xaxis=dict(title='거리 (ft) · 측면뷰', range=[0, PITCH_DIST], gridcolor=GRID, zeroline=False),
        yaxis=dict(title='높이 (ft)', range=[0, 8], gridcolor=GRID, zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # 상단뷰: 위에서 내려다본 시점. y축을 뒤집어(홈플레이트 아래·투수판 위) 로케이션
    # 차트와 같은 "주심 시점" 좌우 방향이 되게 한다.
    fig2 = go.Figure()
    fig2.add_shape(type="rect", x0=0.83, x1=5, y0=PITCH_DIST - 3, y1=PITCH_DIST + 1.5,
                   fillcolor="rgba(140,143,152,0.08)", line_width=0, layer="below")
    fig2.add_shape(type="rect", x0=-5, x1=-0.83, y0=PITCH_DIST - 3, y1=PITCH_DIST + 1.5,
                   fillcolor="rgba(140,143,152,0.08)", line_width=0, layer="below")
    fig2.add_shape(type="line", x0=-1, x1=1, y0=0, y1=0, line=dict(color=ZONE_LINE, width=5))
    fig2.add_shape(type="rect", x0=-0.83, x1=0.83, y0=PITCH_DIST - 1.7, y1=PITCH_DIST,
                   line=dict(color=ZONE_LINE, width=1.8), fillcolor="rgba(0,0,0,0)", layer="below")
    fig2.add_annotation(x=0, y=0, yshift=-12, text="투수판", showarrow=False,
                        font=dict(size=9, color=AXIS_TEXT))
    fig2.add_annotation(x=0, y=PITCH_DIST + 3, text="타석 · 홈플레이트", showarrow=False,
                        font=dict(size=9, color=AXIS_TEXT))
    fig2.add_trace(go.Scatter(x=top_x, y=top_y, mode='lines', line=dict(color=ACCENT_HEX, width=3)))
    fig2.add_trace(go.Scatter(x=[top_x[0], top_x[-1]], y=[top_y[0], top_y[-1]],
                              mode='markers', marker=dict(color=ACCENT_HEX, size=8)))
    fig2.add_vline(x=0, line_dash='dash', line_color=GRID)
    fig2.update_layout(
        height=440, margin=dict(l=40, r=10, t=10, b=30),
        plot_bgcolor=PLOT_BG, paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=AXIS_TEXT, size=11), showlegend=False,
        xaxis=dict(title='좌우 (ft) · 상단뷰 (주심 시점)', range=[-4, 4], gridcolor=GRID,
                   zeroline=False),
        yaxis=dict(title='거리 (ft) · 아래 홈플레이트 → 위 투수판',
                   range=[PITCH_DIST + 6, -4], gridcolor=GRID, zeroline=False),
    )
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    st.caption("회색 사각형 = 홈플레이트 스트라이크존 · 빨간 점 = 릴리스·플레이트 통과 지점")


def render_model_info():
    st.markdown('<div class="pw-label">모델 정보</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Weighted F1", "0.93")
    m2.metric("Test Accuracy", "93.4%")
    m3.metric("Pitch Types", "7")
    st.caption(
        "17개 Statcast 피처를 입력하면 MLP 신경망이 7가지 구종 중 하나를 예측합니다. "
        "MLB Statcast 2024~2025 정규시즌 데이터로 학습 · Weighted F1 0.93 · "
        "모델은 lru_cache로 프로세스당 1회만 로드됩니다."
    )


def render_pitch_mix(df):
    """최근 투구의 구종별 사용 비율을 가로 막대 차트로 보여준다 (단색 회색)."""
    if 'pitch_type' not in df.columns:
        return
    counts = df['pitch_type'].dropna()
    counts = counts[counts.isin(PITCH_NAMES.keys())]
    if counts.empty:
        return

    ratio = (counts.value_counts(normalize=True) * 100).sort_values()
    fig = go.Figure(go.Bar(
        x=ratio.values, y=[PITCH_NAMES.get(k, k) for k in ratio.index], orientation='h',
        marker_color=GRAY,
        text=[f'{v:.0f}%' for v in ratio.values], textposition='outside',
    ))
    fig.update_layout(
        height=max(120, 40 * len(ratio)), margin=dict(l=10, r=30, t=10, b=30),
        plot_bgcolor=PLOT_BG, paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=AXIS_TEXT, size=11), showlegend=False,
        xaxis=dict(title='%', range=[0, max(ratio.values) * 1.25], gridcolor=GRID, zeroline=False),
        yaxis=dict(gridcolor='rgba(0,0,0,0)'),
    )
    st.caption("최근 구종 사용 비율")
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def _waterfall_contribs(attribution, conf):
    """gradient×input 기여도를 base→예측 확률 구간에 맞춰 선형 정규화한다.

    base(균등 사전확률) + Σ기여도 = 예측 확률 이 정확히 성립한다. 부호와
    피처 간 상대 크기는 원본 gradient×input 그대로 유지된다.
    """
    base = 1.0 / len(PITCH_NAMES)
    total = conf - base
    s = sum(attribution.values())
    if abs(s) > 1e-9:
        k = total / s
        contribs = {f: v * k for f, v in attribution.items()}
    else:
        contribs = {f: total / len(attribution) for f in attribution}
    return base, contribs


def render_waterfall(base, ordered):
    """기준값 → 긍정/부정 기여 합 → 예측값을 가로 바 하나로. 세그먼트를 크게 잡아
    (피처별 세부는 아래 카드에서) 마우스오버가 잘 잡히게 한다."""
    pos = sum(c for _, c in ordered if c > 0)
    neg = sum(c for _, c in ordered if c < 0)
    segs = [("기준값", base, "#dfe1e4")]
    if pos > 1e-6:
        segs.append(("긍정 기여 합", pos, ACCENT_HEX))
    if neg < -1e-6:
        segs.append(("부정 기여 합", neg, GRAY))

    fig = go.Figure()
    cursor = 0.0
    for name, val, color in segs:
        left = cursor if val >= 0 else cursor + val
        fig.add_trace(go.Bar(
            x=[abs(val)], y=[""], base=left, orientation='h', width=0.62,
            marker=dict(color=color, line=dict(color="#ffffff", width=1.5)),
            hovertemplate=f"{name}: {val:+.3f}<extra></extra>",
        ))
        cursor += val
    fig.update_layout(
        barmode='overlay', height=64, margin=dict(l=2, r=2, t=2, b=2),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
        hovermode='closest',
        xaxis=dict(range=[0, max(cursor, base) * 1.05], visible=False),
        yaxis=dict(visible=False),
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_feature_card(f, c, mx, label):
    name = FEAT_KR.get(f, f)
    pos = c >= 0
    width = abs(c) / mx * 100 if mx else 0
    ref = SAMPLE_VALUES.get(f)
    cur = st.session_state.inp.get(f)
    if ref is not None and cur is not None and abs(cur - ref) > 1e-6:
        level = "높아" if cur > ref else "낮아"
    else:
        level = "작용해"
    direction = "끌어올렸" if pos else "끌어내렸"
    pitch = PITCH_NAMES.get(label, label)
    sent = f"{name}이(가) 일반적인 수준보다 {level} {pitch} 예측 확률을 {direction}습니다 (기여도 {c:+.2f})."
    st.markdown(
        '<div class="pw-fcard">'
        '<div class="pw-fcard-top">'
        f'<span class="pw-fname">{name}</span>'
        f'<span class="pw-fbadge {"pos" if pos else "neg"}">{c:+.2f}</span>'
        '</div>'
        f'<div class="pw-fmag"><span style="width:{width:.0f}%;'
        f'background:{ACCENT_HEX if pos else GRAY}"></span></div>'
        f'<p class="pw-fdesc">{sent}</p>'
        '</div>',
        unsafe_allow_html=True,
    )


@st.fragment(run_every=1)
def _explanation_fragment():
    """설명이 백그라운드 스레드에서 준비되면 자동으로 표시한다. 1초마다 확인하지만
    이 프래그먼트만 다시 그려서 슬라이더 등 나머지 UI는 전혀 영향받지 않는다."""
    future = st.session_state.get('_explain_future')
    if future is not None and future.done():
        try:
            st.session_state.explanation = future.result()
        except Exception:
            st.session_state.explanation = None
        st.session_state['_explain_future'] = None

    explanation = st.session_state.explanation
    if explanation:
        st.markdown(f'<p class="pw-shap-expl">{explanation}</p>', unsafe_allow_html=True)
    elif st.session_state.get('_explain_future') is not None:
        st.caption("💭 설명 생성 중...")


def render_shap_panel(attribution, conf, label):
    st.markdown(
        '<div class="pw-shap-head">'
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#c53637" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M13 2 4 14h7l-1 8 9-12h-7z"/></svg>'
        'SHAP 인사이트</div>'
        '<div class="pw-shap-sub">예측에 영향을 준 특성 기여도</div>',
        unsafe_allow_html=True,
    )
    if not attribution or not conf or not label:
        st.info("사이드바에서 값을 조절하면 기여도가 표시됩니다.")
        return

    _explanation_fragment()

    base, contribs = _waterfall_contribs(attribution, conf)
    ordered = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)

    render_waterfall(base, ordered)
    st.markdown(
        f'<div class="pw-wf-ends"><span>기준값 {base:.2f}</span>'
        f'<span>예측값 {conf:.2f}</span></div>',
        unsafe_allow_html=True,
    )

    flt = st.radio("필터", ["전체", "긍정 기여", "부정 기여"], horizontal=True,
                   label_visibility="collapsed", key="shap_filter")

    mx = max((abs(v) for _, v in ordered), default=1.0)
    for f, c in ordered[:8]:
        if flt == "긍정 기여" and c <= 0:
            continue
        if flt == "부정 기여" and c >= 0:
            continue
        render_feature_card(f, c, mx, label)


# ── 사이드바: 입력 컨트롤 ────────────────────────────────
with st.sidebar:
    st.markdown(LOGO_HTML, unsafe_allow_html=True)

    _c_mode, _c_hand = st.columns(2)
    with _c_mode:
        st.markdown('<div class="pw-label">입력 방식</div>', unsafe_allow_html=True)
        input_mode = st.radio(
            "입력 방식", ["직접 조작", "투수 검색"],
            horizontal=True, label_visibility="collapsed",
        )
    _manual = input_mode == "직접 조작"
    _mode_changed = st.session_state.get("_prev_mode") != input_mode
    st.session_state["_prev_mode"] = input_mode

    # 투구 손 ↔ 릴리스 좌우 양방향 연동. 손 라디오 렌더 전에 session_state를 맞춰둔다.
    _sync_hand = st.session_state.get("_sync_hand")
    _cur_rx = float(st.session_state.get("rx_slider", SAMPLE_VALUES['release_pos_x']))
    if _manual and _mode_changed:
        st.session_state["ax_slider"] = _snap(st.session_state.inp['ax'], -30.0, 30.0)
        st.session_state["az_slider"] = _snap(st.session_state.inp['az'], -50.0, 15.0)
        st.session_state["rx_slider"] = _snap(st.session_state.inp['release_pos_x'], -3.0, 3.0, 0.1)
        _cur_rx = st.session_state["rx_slider"]
    elif _manual and st.session_state.get("_sync_rx") is not None \
            and abs(_cur_rx - st.session_state["_sync_rx"]) > 1e-6:
        # 릴리스 좌우를 움직였고 부호가 손과 안 맞으면 → 손을 부호에 맞춘다
        _want = 'L' if _cur_rx > 0 else 'R'
        if st.session_state.get("hand") != _want:
            st.session_state["hand"] = _want
            _sync_hand = _want   # 아래 "손 변경" 체크가 이걸 사용자 조작으로 오인 안 하게

    with _c_hand:
        st.markdown('<div class="pw-label">투구 손</div>', unsafe_allow_html=True)
        hand = st.radio(
            "투구 손", options=['R', 'L'],
            format_func=lambda v: '우완 (R)' if v == 'R' else '좌완 (L)',
            horizontal=True, key='hand', label_visibility="collapsed",
            disabled=not _manual,
            help="릴리스 방향·가이드 표시에 반영 (예측은 17개 수치로 계산)"
            if _manual else "투수 검색 시 데이터에서 자동 반영됩니다.",
        )

    if _manual:
        st.session_state.p_throws = hand
        # 손을 사용자가 바꿨으면 좌우를 거울상으로: 릴리스 좌우 부호를 손에 맞추고
        # 수평 무브먼트(ax)도 반전한다 (크기는 유지). az·구속·스핀은 손과 무관.
        if not _mode_changed and _sync_hand is not None and hand != _sync_hand:
            _mag = abs(_cur_rx) if abs(_cur_rx) > 0.2 else abs(SAMPLE_VALUES['release_pos_x'])
            st.session_state["rx_slider"] = _snap(_mag * (1.0 if hand == 'L' else -1.0),
                                                  -3.0, 3.0, 0.1)
            st.session_state["ax_slider"] = _snap(-float(st.session_state["ax_slider"]),
                                                  -30.0, 30.0)
        st.session_state["_sync_hand"] = hand
        st.session_state["_sync_rx"] = float(st.session_state["rx_slider"])

    st.markdown("---")

    if _manual:
        spd = st.slider("구속 (mph)", 60.0, 105.0, float(st.session_state.inp['release_speed']),
                        0.5, help="빠를수록 FF/FC 계열")
        spin = st.slider("스핀레이트 (rpm)", 1500.0, 3500.0,
                         float(st.session_state.inp['release_spin_rate']), 10.0,
                         help="높을수록 포심/커터 계열")
        spin_axis = st.slider("회전축 (°)", 0.0, 360.0,
                              float(st.session_state.inp['spin_axis']), 5.0,
                              help="공이 회전하는 축의 방향(시계 방향, 0~360°). 무브먼트 차트의 "
                                   "ax·az와 별개로 구종 판별에 큰 영향을 줌 — 2026시즌 실측 평균: "
                                   "포심 215° · 싱커 225° · 체인지업 242° · 스플리터 238°"
                                   "(패스트볼 계열이 210~245°에 몰림) · 커터 185° · "
                                   "슬라이더 122°(선수마다 편차 큼) · 커브 44°(패스트볼과 거의 반대 방향)")
        ax_ = st.slider("수평 무브먼트 (ax)", -30.0, 30.0, step=0.5, key="ax_slider",
                        help="스트라이크존을 클릭하면 이 값이 자동으로 맞춰집니다")
        az = st.slider("수직 무브먼트 (az)", -50.0, 15.0, step=0.5, key="az_slider",
                       help="양수=포심(떠오름), 음수=싱커(가라앉음) · 스트라이크존 클릭으로는 "
                            "이 값이 안 바뀝니다(높이는 내부 초기 수직속도로 맞춤) — 직접 움직여야 반영됩니다")
        ext = st.slider("릴리스 익스텐션 (ft)", 5.0, 7.5,
                        float(st.session_state.inp['release_extension']), 0.1,
                        help="릴리스 지점이 홈플레이트에 얼마나 가까운지")
        rx = st.slider("릴리스 좌우 (ft)", -3.0, 3.0, step=0.1, key="rx_slider",
                       help="투구 손과 자동 연동 · 음수 = 우완(1루쪽), 양수 = 좌완(3루쪽)")
        rz = st.slider("릴리스 높이 (ft)", 4.0, 7.0,
                       float(st.session_state.inp['release_pos_z']), 0.1,
                       help="릴리스 암슬롯 높이")

        vz0 = round(-3.0 - 0.15 * az, 2)
        # 존 클릭 직후(슬라이더 미조작)면 클릭 높이를 맞춘 vz0을 사용
        if st.session_state.get("_click_ctx") == (ax_, az, round(spd, 2), round(ext, 2)):
            vz0 = st.session_state["_click_vz0"]
        vx0 = round(0.20 * ax_, 2)

        vy0 = -(spd * 1.467)
        dist = PITCH_DIST - ext
        t = dist / max(abs(vy0), 1)

        inp = dict(SAMPLE_VALUES)
        inp.update({
            'release_speed': spd, 'release_spin_rate': spin, 'spin_axis': spin_axis,
            'release_extension': ext,
            'az': az, 'ax': ax_, 'vz0': vz0, 'vx0': vx0,
            'release_pos_x': rx, 'release_pos_z': rz, 'vy0': round(vy0, 2),
            'effective_speed': round(spd * 0.984, 2),
            'pfx_z': round(vz0 * t + 0.5 * az * t**2, 3),
            'pfx_x': round(vx0 * t + 0.5 * ax_ * t**2, 3),
            'plate_z': round(float(np.clip(rz + vz0*t + 0.5*az*t**2, 0, 5)), 3),
            'plate_x': round(float(np.clip(rx + vx0*t + 0.5*ax_*t**2, -2, 2)), 3),
        })
        st.session_state.inp = inp
        st.session_state.movement_bg = None   # 슬라이더 모드엔 실제 데이터 배경 없음
        run_prediction(inp)   # 위젯 변경 시 Streamlit이 자동 rerun → 여기서 즉시 재예측

    else:
        with st.expander("CSV로 실제 투구 업로드"):
            uploaded = st.file_uploader("Statcast CSV 업로드", type=["csv"])
            if uploaded is not None:
                df = pd.read_csv(uploaded, low_memory=False)
                select_and_predict_from_df(df)

        st.caption(
            f"Statcast 데이터는 {STATCAST_START_YEAR}년부터 제공됩니다. "
            f"{STATCAST_START_YEAR}년 이전 활동 선수는 그 이후 시즌 기록만 조회할 수 있어요. "
            "이름을 비워두면 날짜만으로도 검색할 수 있습니다."
        )
        favs = st.session_state.favorites
        if favs:
            st.markdown('<div class="pw-label">⭐ 즐겨찾기 (이번 세션)</div>', unsafe_allow_html=True)
            for f in favs:
                # 구버전 세션엔 'label' 없이 {'id','name'}만 있을 수 있음 — 배포 중 세션이
                # 유지되면 즐겨찾기 스키마가 바뀌어도 session_state는 그대로 남는다.
                if st.button(f.get('label', f['name']), key=f"favbtn_{f['id']}", use_container_width=True):
                    st.session_state.pitcher_query = f['name']
                    st.rerun()

        query = st.text_input(
            "투수 이름 (2글자 이상, 영어 · 비워두면 날짜로만 검색)",
            placeholder="Gerrit Cole", key="pitcher_query",
        ).strip()

        if query and contains_korean(query):
            st.warning("영어로 입력해주세요 (예: Gerrit Cole).")
        elif len(query) >= 2:
            with st.spinner(f"'{query}' 검색 중..."):
                try:
                    candidates = search_pitchers(query)
                except requests.exceptions.RequestException as e:
                    candidates = None
                    st.error(f"선수 검색에 실패했습니다: {e}")

            if candidates is not None and not candidates:
                st.error("선수를 찾을 수 없습니다.")
            elif candidates:
                options = {career_label(c): c for c in candidates}
                c_star, c_select = st.columns([1, 11])
                choice = c_select.selectbox("검색 결과", options=list(options.keys()))
                chosen = options[choice]
                st.session_state.p_throws = chosen['throws']

                is_fav = any(f['id'] == chosen['id'] for f in st.session_state.favorites)
                c_star.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
                if c_star.button("⭐" if is_fav else "☆", key=f"fav_toggle_{chosen['id']}",
                                  help="즐겨찾기 해제" if is_fav else "즐겨찾기에 추가"):
                    toggle_favorite(chosen['id'], chosen['name'], choice)
                    st.rerun()

                bounds = pitcher_date_bounds(chosen)
                if bounds is None:
                    st.warning(
                        f"{chosen['name']} 선수는 Statcast 데이터 제공({STATCAST_START_YEAR}년) "
                        "이전에 활동을 마쳐 투구 데이터를 조회할 수 없습니다."
                    )
                else:
                    min_d, max_d, def_start, def_end = bounds
                    dc1, dc2 = st.columns(2)
                    start_d = dc1.date_input("시작일", value=def_start,
                                              min_value=min_d, max_value=max_d, key="pitcher_start")
                    end_d = dc2.date_input("종료일", value=def_end,
                                            min_value=min_d, max_value=max_d, key="pitcher_end")

                    if start_d > end_d:
                        st.error("시작일이 종료일보다 늦을 수 없습니다.")
                    else:
                        with st.spinner(
                            f"{chosen['name']} 선수의 {start_d}~{end_d} 투구 데이터를 가져오는 중..."
                        ):
                            df = fetch_pitcher_statcast(chosen['id'], str(start_d), str(end_d))

                        if df.empty:
                            st.warning("해당 기간 투구 기록이 없습니다. "
                                       "타자로만 뛴 기간일 수 있어요.")
                        else:
                            render_pitch_mix(df)
                            select_and_predict_from_df(df)
        elif query:
            st.caption("2글자 이상 입력하면 검색됩니다.")
        else:
            st.markdown('<div class="pw-label">날짜로 검색</div>', unsafe_allow_html=True)
            dc1, dc2 = st.columns(2)
            default_end = date.today()
            default_start = default_end - timedelta(days=1)
            start_d = dc1.date_input("시작일", value=default_start,
                                      min_value=STATCAST_START_DATE, max_value=default_end, key="range_start")
            end_d = dc2.date_input("종료일", value=default_end,
                                    min_value=STATCAST_START_DATE, max_value=default_end, key="range_end")

            if start_d > end_d:
                st.error("시작일이 종료일보다 늦을 수 없습니다.")
            elif (end_d - start_d).days > MAX_DATE_ONLY_DAYS:
                st.error(
                    f"이름 없이 날짜만으로 검색할 때는 리그 전체 데이터를 불러오므로 "
                    f"최대 {MAX_DATE_ONLY_DAYS}일 범위까지만 지원합니다. 범위를 좁혀주세요."
                )
            else:
                with st.spinner(f"{start_d}~{end_d} 전체 투구 데이터를 가져오는 중..."):
                    df = fetch_statcast_range(str(start_d), str(end_d))

                if df.empty:
                    st.warning("해당 기간에 투구 기록이 없습니다 (시즌 오프일 수 있습니다).")
                else:
                    total = len(df)
                    if total > MAX_DATE_ONLY_ROWS:
                        st.info(
                            f"총 {total:,}건 중 {MAX_DATE_ONLY_ROWS}건만 표시합니다. "
                            "날짜 범위를 좁히면 더 정확히 볼 수 있어요."
                        )
                        df = df.head(MAX_DATE_ONLY_ROWS)
                    render_pitch_mix(df)
                    select_and_predict_from_df(df)

    if input_mode == "직접 조작":
        st.markdown("---")
        with st.expander("구종별 만들기 가이드"):
            render_guide(st.session_state.p_throws)

# ── 메인: 요약 스트립 + 대시보드 + SHAP 패널 ────────────
inp = st.session_state.inp
label, conf, proba = st.session_state.label, st.session_state.conf, st.session_state.proba
attribution = st.session_state.attribution
inf_ms = st.session_state.get('inf_ms', 0.0)

def summary_card():
    """예측 요약 스트립 — 각 탭 바 바로 아래에 반복 렌더해 어느 탭에서도 보이게 한다.
    칩 자체가 테두리를 가지므로 바깥 카드로 한 번 더 감싸지 않는다(이중 박스 방지)."""
    if label:
        render_summary(label, conf, inf_ms)


left, right = st.columns([2.2, 1], gap="large")

with left:
    tab_dash, tab_traj, tab_model = st.tabs(["대시보드", "궤적 분석", "모델 정보"])

    with tab_dash:
        summary_card()
        if label and proba:
            with st.container(border=True):
                render_prob_dist(proba, label)
            with st.container(border=True):
                render_strikezone(inp, clickable=(input_mode == "직접 조작"))
            with st.container(border=True):
                render_movement(inp)
        else:
            with st.container(border=True):
                st.markdown("사이드바에서 값을 조절하거나 투수를 선택하면 "
                            "여기에 예측 결과가 표시됩니다.")

    with tab_traj:
        summary_card()
        with st.container(border=True):
            render_paths(inp, st.session_state.p_throws)

    with tab_model:
        summary_card()
        with st.container(border=True):
            render_model_info()

with right:
    with st.container(border=True):
        render_shap_panel(attribution, conf, label)
