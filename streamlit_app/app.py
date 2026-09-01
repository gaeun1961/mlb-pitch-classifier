"""app.py - MLB Pitch Classifier 웹 데모 (Streamlit) · "Pitch Workbench" 라이트 테마"""

import re
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

from utils import (
    predict, compute_trajectory_side, compute_trajectory_top,
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

# 구종별 만들기 가이드 = (코드, 이름, 구속, az 힌트, ax 힌트)
# ax 힌트는 우완 기준 · 좌완은 좌우(+/−)가 반전된다.
GUIDE_ROWS = [
    ('FF', '포심패스트볼', '93+', 'az↑', '~0'),
    ('SI', '싱커', '92+', 'az↓살짝', '+'),
    ('SL', '슬라이더', '82-88', 'az0/−', '++'),
    ('CU', '커브', '75-82', 'az−−', '−'),
    ('CH', '체인지업', '80-87', 'az−', '+'),
    ('FC', '커터', '88-94', 'az~0', '−'),
    ('FS', '스플리터', '83-90', 'az−−−', '~0'),
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

/* 결과 요약 칩 — 예측 구종만 빨간 외곽선, 나머지는 회색 보더 */
.pw-summary{display:flex;flex-wrap:wrap;gap:8px;align-items:center;}
.pw-sum-pred{display:inline-flex;align-items:center;gap:7px;font-family:'Sora';font-weight:700;font-size:0.95rem;color:var(--text);background:#fff;border:1.5px solid var(--accent);padding:5px 12px;border-radius:999px;}
.pw-sum-pred i{width:8px;height:8px;border-radius:50%;background:var(--accent);display:inline-block;}
.pw-sum-chip{font-size:0.8rem;font-weight:600;color:var(--text-2);background:#fff;border:1px solid var(--card-border);padding:5px 12px;border-radius:999px;}

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

.pw-grow{display:flex;align-items:center;gap:8px;font-size:0.8rem;color:var(--text-2);padding:3px 0;}
.pw-gcode{font-family:'Sora';font-weight:700;font-size:0.68rem;background:var(--chip-gray);color:var(--text-2);padding:2px 6px;border-radius:5px;flex:none;}
.pw-gnote{font-size:9px;color:var(--text-3);margin-top:6px;}

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
    st.session_state.movement_bg = None    # (ax[], az[]) — 실제 데이터 로드 시에만 채움
if 'label' not in st.session_state:
    st.session_state.label = None
    st.session_state.conf = None
    st.session_state.proba = None
    st.session_state.attribution = None
    st.session_state.explanation = None
    st.session_state.inf_ms = 0.0
if 'warmed_up' not in st.session_state:
    st.session_state.warmed_up = False


@st.cache_data(show_spinner=False)
def _predict_cached(items, p_throws):
    """동일 입력이면 모델 추론·설명 생성을 다시 하지 않도록 결과를 캐시한다.

    슬라이더 값이 안 바뀐 rerun(탭 전환, 필터 클릭 등)에서는 즉시 반환된다.
    반환 튜플 끝에 최초 계산 시 측정한 추론 시간(ms)을 함께 담는다.
    """
    t0 = time.perf_counter()
    label, conf, proba, attribution, explanation = predict(dict(items), p_throws)
    return label, conf, proba, attribution, explanation, (time.perf_counter() - t0) * 1000


def run_prediction(inp):
    """예측을 실행(캐시 경유)하고 세션 상태를 갱신한다. 실패 시 에러 메시지를 표시한다."""
    spinner_msg = (
        "예측 중입니다..." if st.session_state.warmed_up
        else "예측 중입니다... (첫 예측은 모델 로딩으로 잠시 걸릴 수 있어요)"
    )
    try:
        with st.spinner(spinner_msg):
            key = tuple(sorted(inp.items()))
            label, conf, proba, attribution, explanation, inf_ms = _predict_cached(
                key, st.session_state.p_throws
            )
        st.session_state.label = label
        st.session_state.conf = conf
        st.session_state.proba = proba
        st.session_state.attribution = attribution
        st.session_state.explanation = explanation
        st.session_state.inf_ms = inf_ms
        st.session_state.warmed_up = True

        hist = st.session_state.history
        pt = (float(inp['ax']), float(inp['az']))
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

    # 무브먼트 차트 배경 = 이 데이터셋의 실제 ax/az 분포 (합성 클러스터 아님)
    bg = df_valid[['ax', 'az']]
    if len(bg) > 500:
        bg = bg.sample(500, random_state=0)
    st.session_state.movement_bg = (bg['ax'].tolist(), bg['az'].tolist())

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
        rows += (
            f'<div class="pw-grow"><span class="pw-gcode">{code}</span>'
            f'<span>{name} · {spd} · {az_hint} · ax{ax_disp}</span></div>'
        )
    side = "좌완" if is_left else "우완"
    other = "우완" if is_left else "좌완"
    st.markdown(
        rows + f'<div class="pw-gnote">※ ax 방향은 {side} 기준 · {other}은 좌우 반대</div>',
        unsafe_allow_html=True,
    )


def render_summary(label, conf, inf_ms):
    st.markdown(
        '<div class="pw-summary">'
        f'<span class="pw-sum-pred"><i></i>{PITCH_NAMES.get(label, label)}</span>'
        f'<span class="pw-sum-chip">신뢰도 {conf * 100:.0f}%</span>'
        '<span class="pw-sum-chip">모델 정확도 94.2%</span>'
        f'<span class="pw-sum-chip">추론 시간 {inf_ms:.0f}ms</span>'
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
    """수평(ax)·수직(az) 평면 위 현재 투구(빨간 링). 배경 회색 점은 실제 데이터가
    로드됐을 때만 그 데이터셋의 실제 분포로 표시하고, 슬라이더 모드에서는 이번 세션에
    예측해 본 이력만 찍는다. ax·az는 모델이 쓰는 17개 피처 중 2개일 뿐이라 위치가
    구종을 결정하지 않는다 — 그래서 합성 클러스터/구종 라벨은 두지 않는다.
    """
    st.markdown('<div class="pw-label">궤적 · 무브먼트 차트</div>', unsafe_allow_html=True)
    bg = st.session_state.get('movement_bg')
    hist = st.session_state.history

    fig = go.Figure()
    fig.add_hline(y=0, line_color=GRID)
    fig.add_vline(x=0, line_color=GRID)

    if bg:
        fig.add_trace(go.Scatter(x=bg[0], y=bg[1], mode='markers',
                                 marker=dict(size=5, color=GRAY_SOFT), hoverinfo='skip'))
        ctx = "회색 점 = 불러온 실제 투구 분포"
    elif hist:
        fig.add_trace(go.Scatter(
            x=[h[0] for h in hist], y=[h[1] for h in hist], mode='markers',
            marker=dict(size=7, color=GRAY_TRAIL), hoverinfo='skip',
        ))
        ctx = "회색 점 = 이번 세션에서 예측해 본 투구"
    else:
        ctx = "슬라이더를 움직이면 예측 이력이 쌓입니다"

    fig.add_trace(go.Scatter(x=[inp['ax']], y=[inp['az']], mode='markers',
                             marker=dict(size=22, color='rgba(0,0,0,0)',
                                         line=dict(color=ACCENT_HEX, width=3))))
    fig.add_trace(go.Scatter(x=[inp['ax']], y=[inp['az']], mode='markers',
                             marker=dict(size=7, color=ACCENT_HEX)))
    fig.update_layout(
        height=320, margin=dict(l=44, r=20, t=10, b=38),
        plot_bgcolor=PLOT_BG, paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=AXIS_TEXT, size=11), showlegend=False,
        xaxis=dict(title='수평 무브먼트 (ax)', range=[-30, 30], gridcolor=GRID, zeroline=False),
        yaxis=dict(title='수직 무브먼트 (az)', range=[-50, 15], gridcolor=GRID, zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.caption(f"빨간 링 = 현재 예측 투구 · {ctx}. ax·az는 모델이 쓰는 17개 피처 중 2개입니다.")


def render_strikezone(inp, p_throws):
    """정면(포수) 시점: 흐린 궤적이 존 안 어디로 들어오는지 + 플레이트 통과 지점."""
    st.markdown('<div class="pw-label">로케이션 · 스트라이크존</div>', unsafe_allow_html=True)
    # 정면뷰 궤적 = 같은 시간축으로 뽑은 (좌우, 높이). 두 배열 모두 길이 50, 인덱스로 정렬됨.
    _, front_lr = compute_trajectory_top(inp, p_throws)
    _, front_h = compute_trajectory_side(inp)

    fig = go.Figure()
    # 평균 존: 홈플레이트 폭 ±0.83ft(17인치 절반 + 볼 반경), 무릎~겨드랑이 1.5~3.5ft 근사
    fig.add_shape(type="rect", x0=-0.83, x1=0.83, y0=1.5, y1=3.5,
                  line=dict(color=ZONE_LINE, width=2),
                  fillcolor="rgba(138,143,152,0.06)", layer="below")
    fig.add_trace(go.Scatter(x=front_lr, y=front_h, mode='lines',
                             line=dict(color="rgba(120,124,133,0.35)", width=6),
                             hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=[front_lr[-1]], y=[front_h[-1]], mode='markers',
        marker=dict(color=ACCENT_HEX, size=20, line=dict(color='white', width=1.5)),
    ))
    fig.update_layout(
        height=340, margin=dict(l=40, r=10, t=10, b=32),
        plot_bgcolor=PLOT_BG, paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=AXIS_TEXT, size=11), showlegend=False,
        xaxis=dict(title='좌우 (ft)', range=[-2, 2], gridcolor=GRID, zeroline=False,
                   scaleanchor='y'),
        yaxis=dict(title='높이 (ft)', range=[0, 5], gridcolor=GRID, zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.caption("회색 사각형 = 평균 스트라이크존 · 흐린 선 = 정면뷰 궤적 · 빨간 점 = 플레이트 통과 위치")


def render_paths(inp, p_throws):
    """측면뷰·상단뷰 궤적. 홈플레이트에 스트라이크존 박스로 존 대비 위치를 표시한다."""
    st.markdown('<div class="pw-label">투구 궤적</div>', unsafe_allow_html=True)
    side_x, side_z = compute_trajectory_side(inp)
    top_y, top_x = compute_trajectory_top(inp, p_throws)

    fig = go.Figure()
    fig.add_shape(type="rect", x0=PITCH_DIST - 1.7, x1=PITCH_DIST, y0=1.5, y1=3.5,
                  line=dict(color=ZONE_LINE, width=1.8), fillcolor="rgba(0,0,0,0)", layer="below")
    fig.add_trace(go.Scatter(x=side_x, y=side_z, mode='lines',
                             line=dict(color=ACCENT_HEX, width=3)))
    fig.add_trace(go.Scatter(x=[side_x[0], side_x[-1]], y=[side_z[0], side_z[-1]],
                             mode='markers', marker=dict(color=ACCENT_HEX, size=8)))
    fig.update_layout(
        height=280, margin=dict(l=40, r=10, t=10, b=30),
        plot_bgcolor=PLOT_BG, paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=AXIS_TEXT, size=11), showlegend=False,
        xaxis=dict(title='거리 (ft) · 측면뷰', range=[0, PITCH_DIST], gridcolor=GRID, zeroline=False),
        yaxis=dict(title='높이 (ft)', range=[0, 8], gridcolor=GRID, zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # 상단뷰: 좌우를 가로축, 거리를 세로축으로 (위에서 내려다본 시점, 홈플레이트가 위쪽)
    fig2 = go.Figure()
    fig2.add_shape(type="rect", x0=-0.83, x1=0.83, y0=PITCH_DIST - 1.7, y1=PITCH_DIST,
                   line=dict(color=ZONE_LINE, width=1.8), fillcolor="rgba(0,0,0,0)", layer="below")
    fig2.add_trace(go.Scatter(x=top_x, y=top_y, mode='lines', line=dict(color=ACCENT_HEX, width=3)))
    fig2.add_trace(go.Scatter(x=[top_x[0], top_x[-1]], y=[top_y[0], top_y[-1]],
                              mode='markers', marker=dict(color=ACCENT_HEX, size=8)))
    fig2.add_vline(x=0, line_dash='dash', line_color=GRID)
    fig2.update_layout(
        height=320, margin=dict(l=40, r=10, t=10, b=30),
        plot_bgcolor=PLOT_BG, paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=AXIS_TEXT, size=11), showlegend=False,
        xaxis=dict(title='좌우 (ft) · 상단뷰', range=[-3, 3], gridcolor=GRID, zeroline=False),
        yaxis=dict(title='거리 (ft)', range=[0, PITCH_DIST], gridcolor=GRID, zeroline=False),
    )
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    st.caption("회색 사각형 = 홈플레이트 스트라이크존 · 빨간 점 = 릴리스·플레이트 통과 지점")


def render_model_info():
    st.markdown('<div class="pw-label">모델 정보</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Weighted F1", "0.94")
    m2.metric("Test Accuracy", "94.2%")
    m3.metric("Pitch Types", "7")
    st.caption(
        "17개 Statcast 피처를 입력하면 MLP 신경망이 7가지 구종 중 하나를 예측합니다. "
        "MLB Statcast 2024 정규시즌 데이터로 학습 · Weighted F1 0.94 · "
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
    """기준값 → 피처 기여도 누적 → 예측값을 가로 스택 바 하나로 그린다."""
    top = ordered[:6]
    rest = sum(c for _, c in ordered[6:])
    segs = [("기준값", base, "#dfe1e4")]
    for f, c in top:
        segs.append((FEAT_KR.get(f, f), c, ACCENT_HEX if c >= 0 else GRAY))
    if abs(rest) > 1e-6:
        segs.append(("기타", rest, ACCENT_HEX if rest >= 0 else GRAY))

    fig = go.Figure()
    cursor = 0.0
    for name, val, color in segs:
        left = cursor if val >= 0 else cursor + val
        fig.add_trace(go.Bar(
            x=[abs(val)], y=[""], base=left, orientation='h', width=0.6,
            marker=dict(color=color, line=dict(color="#ffffff", width=1)),
            hovertemplate=f"{name}: {val:+.3f}<extra></extra>",
        ))
        cursor += val
    fig.update_layout(
        barmode='overlay', height=64, margin=dict(l=2, r=2, t=2, b=2),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
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


def render_shap_panel(attribution, conf, label, explanation):
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

    if explanation:
        st.markdown(f'<p class="pw-shap-expl">{explanation}</p>', unsafe_allow_html=True)

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

    st.markdown('<div class="pw-label">입력 방식</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "입력 방식", ["직접 조작", "투수 검색"],
        horizontal=True, label_visibility="collapsed",
    )
    st.markdown("---")

    if input_mode == "직접 조작":
        spd = st.slider("구속 (mph)", 60.0, 105.0, float(st.session_state.inp['release_speed']),
                        0.5, help="빠를수록 FF/FC 계열")
        spin = st.slider("스핀레이트 (rpm)", 1500.0, 3500.0,
                         float(st.session_state.inp['release_spin_rate']), 10.0,
                         help="높을수록 포심/커터 계열")
        ax_ = st.slider("수평 무브먼트 (ax)", -30.0, 30.0, float(st.session_state.inp['ax']), 0.5,
                        help="좌완/우완에 따라 휘는 방향이 반대로 표시됩니다")
        az = st.slider("수직 무브먼트 (az)", -50.0, 15.0, float(st.session_state.inp['az']), 0.5,
                       help="양수=포심(떠오름), 음수=싱커(가라앉음)")
        ext = st.slider("릴리스 익스텐션 (ft)", 5.0, 7.5,
                        float(st.session_state.inp['release_extension']), 0.1,
                        help="릴리스 지점이 홈플레이트에 얼마나 가까운지")

        base_rx = abs(SAMPLE_VALUES['release_pos_x'])
        rx = base_rx if st.session_state.p_throws == 'L' else -base_rx

        vz0 = round(-3.0 - 0.15 * az, 2)
        vx0 = round(0.20 * ax_, 2)
        rz = SAMPLE_VALUES['release_pos_z']

        vy0 = -(spd * 1.467)
        dist = PITCH_DIST - ext
        t = dist / max(abs(vy0), 1)

        inp = dict(SAMPLE_VALUES)
        inp.update({
            'release_speed': spd, 'release_spin_rate': spin,
            'release_extension': ext,
            'az': az, 'ax': ax_, 'vz0': vz0, 'vx0': vx0,
            'release_pos_x': rx, 'vy0': round(vy0, 2),
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
        query = st.text_input(
            "투수 이름 (2글자 이상, 영어 · 비워두면 날짜로만 검색)",
            placeholder="Gerrit Cole",
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
                choice = st.selectbox("검색 결과", options=list(options.keys()))
                chosen = options[choice]
                st.session_state.p_throws = chosen['throws']

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

    st.markdown("---")
    st.markdown('<div class="pw-label">투구 손</div>', unsafe_allow_html=True)
    st.radio(
        "투구 손", options=['R', 'L'],
        format_func=lambda v: '우완 (R)' if v == 'R' else '좌완 (L)',
        horizontal=True, key='p_throws', label_visibility="collapsed",
        disabled=(input_mode != "직접 조작"),
        help=None if input_mode == "직접 조작" else "실제 투구 데이터에서 자동으로 반영됩니다.",
    )

    if input_mode == "직접 조작":
        st.markdown("---")
        with st.expander("구종별 만들기 가이드"):
            render_guide(st.session_state.p_throws)

# ── 메인: 요약 스트립 + 대시보드 + SHAP 패널 ────────────
inp = st.session_state.inp
label, conf, proba = st.session_state.label, st.session_state.conf, st.session_state.proba
attribution = st.session_state.attribution
explanation = st.session_state.explanation
inf_ms = st.session_state.get('inf_ms', 0.0)

def summary_card():
    """예측 요약 칩 — 각 탭 바 바로 아래에 반복 렌더해 어느 탭에서도 보이게 한다."""
    if label:
        with st.container(border=True):
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
                render_strikezone(inp, st.session_state.p_throws)
            # 무브먼트 산점도는 실제 데이터를 불러온 경우에만 (실제 분포가 있어 의미 있음)
            if st.session_state.get('movement_bg'):
                with st.container(border=True):
                    render_movement(inp)
        else:
            st.info("사이드바에서 값을 조절해 예측을 실행하세요.")

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
        render_shap_panel(attribution, conf, label, explanation)
