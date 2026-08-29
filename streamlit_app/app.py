"""app.py - MLB Pitch Classifier 웹 데모 (Streamlit) · "Pitch Workbench" 라이트 테마"""

import re
import time
from datetime import date

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

from utils import (
    predict, compute_trajectory_side, compute_trajectory_top,
    FEATURE_COLS, PITCH_NAMES, PITCH_COLORS, SAMPLE_VALUES, PITCH_DIST,
)

st.set_page_config(
    page_title="Pitch Workbench",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT_HEX = "#c53637"
NEUTRAL_HEX = "#7d8086"
PLOT_BG = "#ffffff"
GRID = "#eef0f2"
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

# ── 디자인 토큰 · 라이트 테마 CSS ────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Public+Sans:wght@400;500;600;700&display=swap');

:root{
  --bg: oklch(96% 0.003 260);
  --card:#ffffff;
  --card-border: oklch(90% 0.006 260);
  --text: oklch(18% 0.01 260);
  --text-2: oklch(45% 0.015 260);
  --text-3: oklch(56% 0.02 260);
  --label: oklch(50% 0.02 265);
  --accent: oklch(55% 0.18 25);
  --accent-hover: oklch(46% 0.18 25);
  --chip-bg: oklch(94% 0.045 25);
  --chip-text: oklch(45% 0.16 25);
  --chip-gray: oklch(94% 0.005 260);
  --track: oklch(91% 0.006 260);
}

html,body,[class*="css"]{font-family:'Public Sans',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;}
.stApp{background:var(--bg);color:var(--text);}
h1,h2,h3,h4{font-family:'Sora',sans-serif !important;color:var(--text);}
#MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden;}
hr{border-color:var(--card-border) !important;}

section[data-testid="stSidebar"]{background:var(--card);border-right:1px solid var(--card-border);}
section[data-testid="stSidebar"] .block-container{padding-top:1rem;}

div[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--card);
  border:1px solid var(--card-border) !important;
  border-radius:16px !important;
  box-shadow:0 1px 2px rgba(16,24,40,0.03);
}
div[data-testid="stVerticalBlockBorderWrapper"] > div{padding:16px 18px;}

.pw-label{font-size:11px;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;color:var(--label);margin-bottom:10px;}

.pw-logo{display:flex;align-items:center;gap:9px;margin:2px 0 14px;}
.pw-logo span{font-family:'Sora';font-weight:800;font-size:1.05rem;color:var(--text);}

.pw-summary{display:flex;flex-wrap:wrap;gap:8px;align-items:center;}
.pw-sum-pred{display:inline-flex;align-items:center;gap:7px;font-family:'Sora';font-weight:700;font-size:0.95rem;color:var(--text);background:var(--chip-bg);padding:6px 12px;border-radius:999px;}
.pw-sum-pred i{width:8px;height:8px;border-radius:50%;display:inline-block;}
.pw-sum-chip{font-size:0.8rem;font-weight:600;color:var(--text-2);background:var(--chip-gray);padding:6px 12px;border-radius:999px;}

.pw-bar-row{display:flex;align-items:center;gap:10px;margin:7px 0;}
.pw-bar-name{width:82px;font-size:0.82rem;color:var(--text-2);flex:none;}
.pw-bar-track{flex:1;height:8px;background:var(--track);border-radius:999px;overflow:hidden;}
.pw-bar-fill{display:block;height:100%;border-radius:999px;}
.pw-bar-pct{width:40px;text-align:right;font-size:0.8rem;font-weight:600;color:var(--text);flex:none;}

.pw-legend{display:flex;flex-wrap:wrap;gap:12px;margin-top:8px;}
.pw-leg{display:inline-flex;align-items:center;gap:6px;font-size:0.75rem;color:var(--text-3);}
.pw-leg i{width:8px;height:8px;border-radius:50%;}

.pw-shap-head{display:flex;align-items:center;gap:8px;font-family:'Sora';font-weight:700;font-size:1rem;color:var(--text);}
.pw-shap-sub{font-size:0.78rem;color:var(--text-3);margin:2px 0 12px;}
.pw-shap-expl{font-size:0.82rem;line-height:1.6;color:var(--text-2);background:var(--chip-gray);padding:10px 12px;border-radius:10px;margin:0 0 14px;}
.pw-wf-ends{display:flex;justify-content:space-between;font-size:0.72rem;color:var(--text-3);margin:-6px 2px 14px;}
.pw-fcard{border:1px solid var(--card-border);border-radius:12px;padding:11px 13px;margin-bottom:9px;}
.pw-fcard-top{display:flex;justify-content:space-between;align-items:center;}
.pw-fname{font-size:0.85rem;font-weight:600;color:var(--text);}
.pw-fbadge{font-size:0.75rem;font-weight:700;padding:2px 8px;border-radius:999px;}
.pw-fbadge.pos{background:var(--chip-bg);color:var(--chip-text);}
.pw-fbadge.neg{background:var(--chip-gray);color:var(--text-2);}
.pw-fmag{height:3px;background:var(--track);border-radius:999px;margin:9px 0 8px;overflow:hidden;}
.pw-fmag span{display:block;height:100%;border-radius:999px;}
.pw-fdesc{font-size:0.78rem;line-height:1.55;color:var(--text-3);margin:0;}

.pw-grow{display:flex;align-items:center;gap:8px;font-size:0.8rem;color:var(--text-2);padding:4px 0;}
.pw-gcode{font-family:'Sora';font-weight:700;font-size:0.68rem;background:var(--chip-bg);color:var(--chip-text);padding:2px 6px;border-radius:5px;flex:none;}
.pw-gnote{font-size:9px;color:var(--text-3);margin-top:8px;}

.stButton > button{background:var(--accent);color:#fff;border:none;border-radius:10px;font-weight:700;width:100%;padding:9px 0;}
.stButton > button:hover{background:var(--accent-hover);color:#fff;}
.stButton > button:focus{color:#fff;box-shadow:none;}

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
if 'label' not in st.session_state:
    st.session_state.label = None
    st.session_state.conf = None
    st.session_state.proba = None
    st.session_state.attribution = None
    st.session_state.explanation = None
    st.session_state.inf_ms = 0.0
if 'warmed_up' not in st.session_state:
    st.session_state.warmed_up = False


def run_prediction(inp):
    """백엔드 예측을 호출하고 세션 상태를 갱신한다. 실패 시 에러 메시지를 표시한다."""
    spinner_msg = (
        "예측 중입니다..." if st.session_state.warmed_up
        else "예측 중입니다... (첫 예측은 모델 로딩으로 잠시 걸릴 수 있어요)"
    )
    try:
        with st.spinner(spinner_msg):
            t0 = time.perf_counter()
            label, conf, proba, attribution, explanation = predict(inp, st.session_state.p_throws)
            st.session_state.inf_ms = (time.perf_counter() - t0) * 1000
        st.session_state.label = label
        st.session_state.conf = conf
        st.session_state.proba = proba
        st.session_state.attribution = attribution
        st.session_state.explanation = explanation
        st.session_state.warmed_up = True
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
        if p.get("primaryPosition", {}).get("abbreviation") != "P":
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
            "last_year": last_played[:4] if last_played else None,
            "active": p.get("active", False),
            "throws": p.get("pitchHand", {}).get("code", "R"),
        })

    candidates.sort(key=lambda c: (not c["active"], -int(c["debut_year"])))
    return candidates[:15]


STATCAST_START_YEAR = 2015


def career_label(c):
    """'이름 (팀, 데뷔~현재)' 또는 은퇴 선수는 '이름 (데뷔~은퇴, 은퇴)' 형식으로 표기한다."""
    if c["active"]:
        return f"{c['name']} ({c['team']}, {c['debut_year']}~현재)"
    end = c["last_year"] or c["debut_year"]
    return f"{c['name']} ({c['debut_year']}~{end}, 은퇴)"


def available_seasons(c):
    """Statcast가 제공되는 2015년 이후 범위로 제한한 이 투수의 조회 가능 시즌 목록(최신순)."""
    this_year = date.today().year
    start = max(int(c["debut_year"]), STATCAST_START_YEAR)
    end = this_year if c["active"] else int(c["last_year"] or c["debut_year"])
    end = min(end, this_year)
    if start > end:
        return []
    return list(range(end, start - 1, -1))


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_pitcher_statcast(player_id, start_dt, end_dt):
    """해당 투수의 기간 내 Statcast 투구 데이터를 가져온다."""
    from pybaseball import statcast_pitcher

    return statcast_pitcher(start_dt, end_dt, player_id)


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

PITCH_ORDER = ['FF', 'SI', 'SL', 'CU', 'CH', 'FC', 'FS']

GUIDE_ROWS = [
    ('FF', '포심패스트볼', '93+ · az↑ · ax~0'),
    ('SI', '싱커', '92+ · az↓살짝 · ax+'),
    ('SL', '슬라이더', '82-88 · az0/− · ax++'),
    ('CU', '커브', '75-82 · az−− · ax−'),
    ('CH', '체인지업', '80-87 · az− · ax+'),
    ('FC', '커터', '88-94 · az~0 · ax−'),
    ('FS', '스플리터', '83-90 · az−−− · ax~0'),
]


def render_guide():
    rows = "".join(
        f'<div class="pw-grow"><span class="pw-gcode">{code}</span>'
        f'<span>{name} · {rule}</span></div>'
        for code, name, rule in GUIDE_ROWS
    )
    st.markdown(
        rows + '<div class="pw-gnote">※ ax 방향은 우완 기준 · 좌완은 좌우 반대</div>',
        unsafe_allow_html=True,
    )


def render_summary(label, conf, inf_ms):
    dot = PITCH_COLORS.get(label, ACCENT_HEX)
    st.markdown(
        '<div class="pw-summary">'
        f'<span class="pw-sum-pred"><i style="background:{dot}"></i>{PITCH_NAMES.get(label, label)}</span>'
        f'<span class="pw-sum-chip">신뢰도 {conf * 100:.0f}%</span>'
        '<span class="pw-sum-chip">모델 정확도 94.2%</span>'
        f'<span class="pw-sum-chip">추론 시간 {inf_ms:.0f}ms</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_prob_dist(proba):
    st.markdown('<div class="pw-label">구종별 확률 분포</div>', unsafe_allow_html=True)
    rows = ""
    for k in PITCH_ORDER:
        if k not in proba:
            continue
        v = proba[k] * 100
        rows += (
            '<div class="pw-bar-row">'
            f'<span class="pw-bar-name">{PITCH_NAMES[k]}</span>'
            '<span class="pw-bar-track">'
            f'<span class="pw-bar-fill" style="width:{v:.1f}%;background:{PITCH_COLORS[k]}"></span>'
            '</span>'
            f'<span class="pw-bar-pct">{v:.0f}%</span>'
            '</div>'
        )
    st.markdown(rows, unsafe_allow_html=True)


def render_movement(inp, label):
    """현재 투구의 수평·수직 변화량(pfx, inch)을 무브먼트 평면 위 링 마커로 표시한다."""
    hx = PITCH_COLORS.get(label, ACCENT_HEX) if label else ACCENT_HEX
    x, y = inp['pfx_x'] * 12, inp['pfx_z'] * 12
    fig = go.Figure()
    fig.add_hline(y=0, line_color=GRID)
    fig.add_vline(x=0, line_color=GRID)
    fig.add_trace(go.Scatter(
        x=[x], y=[y], mode='markers',
        marker=dict(size=22, color='rgba(0,0,0,0)', line=dict(color=hx, width=3)),
    ))
    fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', marker=dict(size=7, color=hx)))
    fig.update_layout(
        height=300, margin=dict(l=44, r=20, t=10, b=38),
        plot_bgcolor=PLOT_BG, paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=AXIS_TEXT, size=11), showlegend=False,
        xaxis=dict(title='수평 변화량 (in)', range=[-24, 24], gridcolor=GRID, zeroline=False),
        yaxis=dict(title='수직 변화량 (in)', range=[-24, 24], gridcolor=GRID, zeroline=False),
    )
    st.markdown('<div class="pw-label">궤적 · 무브먼트 차트</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    legend = "".join(
        f'<span class="pw-leg"><i style="background:{PITCH_COLORS[k]}"></i>{PITCH_NAMES[k]}</span>'
        for k in PITCH_ORDER
    )
    st.markdown(f'<div class="pw-legend">{legend}</div>', unsafe_allow_html=True)


def render_trajectory(inp, p_throws, label):
    """측면뷰·상단뷰 궤적과 스트라이크존을 라이트 테마로 그린다."""
    color = PITCH_COLORS.get(label, ACCENT_HEX) if label else ACCENT_HEX
    c1, c2 = st.columns([1.3, 1])

    with c1:
        st.markdown('<div class="pw-label">투구 궤적</div>', unsafe_allow_html=True)
        side_x, side_z = compute_trajectory_side(inp)
        top_y, top_x = compute_trajectory_top(inp, p_throws)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=side_x, y=side_z, mode='lines',
                                 line=dict(color=color, width=3)))
        fig.add_trace(go.Scatter(x=[side_x[0]], y=[side_z[0]], mode='markers',
                                 marker=dict(color=color, size=8)))
        fig.update_layout(
            height=250, margin=dict(l=40, r=10, t=10, b=30),
            plot_bgcolor=PLOT_BG, paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=AXIS_TEXT, size=11), showlegend=False,
            xaxis=dict(title='거리 (ft)', range=[0, PITCH_DIST], gridcolor=GRID, zeroline=False),
            yaxis=dict(title='높이 (ft)', range=[0, 8], gridcolor=GRID, zeroline=False),
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=top_y, y=top_x, mode='lines', line=dict(color=color, width=3)))
        fig2.add_trace(go.Scatter(x=[top_y[0]], y=[top_x[0]], mode='markers',
                                  marker=dict(color=color, size=8)))
        fig2.add_hline(y=0, line_dash='dash', line_color=GRID)
        fig2.update_layout(
            height=200, margin=dict(l=40, r=10, t=10, b=30),
            plot_bgcolor=PLOT_BG, paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=AXIS_TEXT, size=11), showlegend=False,
            xaxis=dict(title='거리 (ft)', range=[0, PITCH_DIST], gridcolor=GRID, zeroline=False),
            yaxis=dict(title='좌우 (ft)', range=[-3, 3], gridcolor=GRID, zeroline=False),
        )
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    with c2:
        st.markdown('<div class="pw-label">스트라이크존</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_shape(type="rect", x0=-0.708, x1=0.708, y0=1.5, y1=3.5,
                       line=dict(color="#c9ccd2", width=2), fillcolor="rgba(197,54,55,0.06)")
        fig3.add_trace(go.Scatter(
            x=[inp['plate_x']], y=[inp['plate_z']], mode='markers',
            marker=dict(color=color, size=18, line=dict(color='white', width=1)),
        ))
        fig3.update_layout(
            height=470, margin=dict(l=40, r=10, t=10, b=30),
            plot_bgcolor=PLOT_BG, paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=AXIS_TEXT, size=11), showlegend=False,
            xaxis=dict(title='좌우 (ft)', range=[-2, 2], gridcolor=GRID, zeroline=False,
                       scaleanchor='y'),
            yaxis=dict(title='높이 (ft)', range=[0, 5], gridcolor=GRID, zeroline=False),
        )
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})


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
    """최근 투구의 구종별 사용 비율을 가로 막대 차트로 보여준다."""
    if 'pitch_type' not in df.columns:
        return
    counts = df['pitch_type'].dropna()
    counts = counts[counts.isin(PITCH_NAMES.keys())]
    if counts.empty:
        return

    ratio = (counts.value_counts(normalize=True) * 100).sort_values()
    fig = go.Figure(go.Bar(
        x=ratio.values, y=[PITCH_NAMES.get(k, k) for k in ratio.index], orientation='h',
        marker_color=[PITCH_COLORS.get(k, NEUTRAL_HEX) for k in ratio.index],
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
    """gradient×input 기여도를 base→예측 확률 구간에 맞춰 정규화한다.

    base(균등 사전확률) + Σ기여도 = 예측 확률 이 정확히 성립하도록 선형 스케일한다.
    부호와 피처 간 상대 크기는 원본 gradient×input 그대로 유지된다.
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


def render_waterfall(base, ordered, pred):
    """기준값 → 피처 기여도 누적 → 예측값을 가로 스택 바 하나로 그린다."""
    top = ordered[:6]
    rest = sum(c for _, c in ordered[6:])
    segs = [("기준값", base, "#e9ebef")]
    for f, c in top:
        segs.append((FEAT_KR.get(f, f), c, ACCENT_HEX if c >= 0 else NEUTRAL_HEX))
    if abs(rest) > 1e-6:
        segs.append(("기타", rest, ACCENT_HEX if rest >= 0 else NEUTRAL_HEX))

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
        f'background:{ACCENT_HEX if pos else NEUTRAL_HEX}"></span></div>'
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
        st.info("사이드바에서 **예측 실행**을 누르면 기여도가 표시됩니다.")
        return

    if explanation:
        st.markdown(f'<p class="pw-shap-expl">{explanation}</p>', unsafe_allow_html=True)

    base, contribs = _waterfall_contribs(attribution, conf)
    ordered = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)

    render_waterfall(base, ordered, conf)
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
        run_prediction(inp)

    else:
        with st.expander("CSV로 실제 투구 업로드"):
            uploaded = st.file_uploader("Statcast CSV 업로드", type=["csv"])
            if uploaded is not None:
                df = pd.read_csv(uploaded, low_memory=False)
                select_and_predict_from_df(df)

        st.caption(
            f"Statcast 데이터는 {STATCAST_START_YEAR}년부터 제공됩니다. "
            f"{STATCAST_START_YEAR}년 이전 활동 선수는 그 이후 시즌 기록만 조회할 수 있어요."
        )
        query = st.text_input("투수 이름 (2글자 이상, 영어)", placeholder="Gerrit Cole").strip()

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

                seasons = available_seasons(chosen)
                if not seasons:
                    st.warning(
                        f"{chosen['name']} 선수는 Statcast 데이터 제공({STATCAST_START_YEAR}년) "
                        "이전에 활동을 마쳐 투구 데이터를 조회할 수 없습니다."
                    )
                else:
                    season = st.selectbox("시즌 선택", options=seasons)
                    with st.spinner(f"{chosen['name']} 선수의 {season}시즌 투구 데이터를 가져오는 중..."):
                        df = fetch_pitcher_statcast(chosen['id'], f"{season}-01-01", f"{season}-12-31")

                    if df.empty:
                        st.warning(f"{season}시즌 투구 기록이 없습니다.")
                    else:
                        render_pitch_mix(df)
                        select_and_predict_from_df(df)
        elif query:
            st.caption("2글자 이상 입력하면 검색됩니다.")

    st.markdown("---")
    st.markdown('<div class="pw-label">투구 손</div>', unsafe_allow_html=True)
    st.radio(
        "투구 손", options=['R', 'L'],
        format_func=lambda v: '우완 (R)' if v == 'R' else '좌완 (L)',
        horizontal=True, key='p_throws', label_visibility="collapsed",
        disabled=(input_mode != "직접 조작"),
        help=None if input_mode == "직접 조작" else "실제 투구 데이터에서 자동으로 반영됩니다.",
    )

    st.markdown("---")
    with st.expander("구종별 만들기 가이드"):
        render_guide()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("예측 실행"):
        run_prediction(st.session_state.inp)

# ── 메인: 대시보드 + SHAP 패널 ──────────────────────────
inp = st.session_state.inp
label, conf, proba = st.session_state.label, st.session_state.conf, st.session_state.proba
attribution = st.session_state.attribution
explanation = st.session_state.explanation
inf_ms = st.session_state.get('inf_ms', 0.0)

left, right = st.columns([2.2, 1], gap="large")

with left:
    tab_dash, tab_traj, tab_model = st.tabs(["대시보드", "궤적 분석", "모델 정보"])

    with tab_dash:
        if label and proba:
            with st.container(border=True):
                render_summary(label, conf, inf_ms)
            with st.container(border=True):
                render_prob_dist(proba)
            with st.container(border=True):
                render_movement(inp, label)
        else:
            st.info("사이드바에서 값을 조절하고 **예측 실행**을 눌러보세요.")

    with tab_traj:
        with st.container(border=True):
            render_trajectory(inp, st.session_state.p_throws, label)

    with tab_model:
        with st.container(border=True):
            render_model_info()

with right:
    with st.container(border=True):
        render_shap_panel(attribution, conf, label, explanation)
