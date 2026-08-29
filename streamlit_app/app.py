"""app.py - MLB Pitch Classifier 웹 데모 (Streamlit)"""

import re
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
    page_title="MLB Pitch Classifier",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 디자인 토큰 ──────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

:root {
    --navy: #0B1929;
    --navy-light: #142840;
    --cream: #F5F1E8;
    --red: #C8443C;
    --green: #2D5A4A;
    --amber: #E8B84B;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(180deg, var(--navy) 0%, #0E2238 100%);
}

h1, h2, h3 { font-family: 'Oswald', sans-serif !important; letter-spacing: 0.02em; }

.hero-title {
    font-family: 'Oswald', sans-serif;
    font-weight: 700;
    font-size: 3rem;
    color: var(--cream);
    letter-spacing: 0.03em;
    margin-bottom: 0;
    line-height: 1.1;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    color: var(--amber);
    font-size: 0.95rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 4px;
}
.hero-desc {
    color: #9CACC0;
    font-size: 1.05rem;
    margin-top: 14px;
    max-width: 640px;
    line-height: 1.6;
}

.metric-card {
    background: var(--navy-light);
    border: 1px solid rgba(232,184,75,0.15);
    border-radius: 4px;
    padding: 18px 20px;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #6B7B91;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--cream);
    margin-top: 2px;
}

.pred-banner {
    background: var(--navy-light);
    border-left: 4px solid var(--amber);
    border-radius: 4px;
    padding: 22px 28px;
}
.pred-code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #6B7B91;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.pred-name {
    font-family: 'Oswald', sans-serif;
    font-size: 2.1rem;
    font-weight: 600;
    color: var(--cream);
    margin: 2px 0 0 0;
}
.pred-conf {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    color: var(--amber);
    font-weight: 600;
}

.guide-table { font-family: 'Inter', sans-serif; font-size: 0.88rem; }
.guide-table td { padding: 7px 14px; border-bottom: 1px solid rgba(255,255,255,0.06); color: #C5D0DD; }
.guide-code { font-family: 'JetBrains Mono', monospace; font-weight: 600; color: var(--amber); }

section[data-testid="stSidebar"] {
    background: var(--navy-light);
    border-right: 1px solid rgba(232,184,75,0.1);
}
section[data-testid="stSidebar"] * { color: #C5D0DD; }

div[data-baseweb="tab-list"] { gap: 4px; }
button[data-baseweb="tab"] {
    font-family: 'Oswald', sans-serif;
    letter-spacing: 0.04em;
}

hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── 히어로 영역 ──────────────────────────────────────────
col_hero, col_strike = st.columns([3, 2])

with col_hero:
    st.markdown('<p class="hero-sub">MLB STATCAST · DEEP LEARNING</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-title">PITCH CLASSIFIER</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-desc">17개 Statcast 피처를 입력하면 MLP 신경망이 7가지 구종 중 '
        '하나를 예측합니다. 슬라이더로 가상의 투구를 직접 만들어보거나, '
        '실제 2025 시즌 데이터를 불러와 모델의 판단을 확인해보세요.</p>',
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="metric-card"><div class="metric-label">Weighted F1</div>'
                     '<div class="metric-value">0.94</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card"><div class="metric-label">Test Accuracy</div>'
                     '<div class="metric-value">94.2%</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card"><div class="metric-label">Pitch Types</div>'
                     '<div class="metric-value">7</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

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
            label, conf, proba, attribution, explanation = predict(inp, st.session_state.p_throws)
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


def render_attribution(attribution, label):
    """이 예측에 각 피처가 기여한 방향·크기를 상위 7개만 가로 막대로 보여준다.

    값은 예측 클래스 확률을 표준화된 입력으로 편미분한 gradient×input 이라
    절대 단위가 아니라 부호와 상대 크기로 읽는다.
    """
    items = sorted(attribution.items(), key=lambda kv: abs(kv[1]))[-7:]
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = ['#2D9D6F' if v >= 0 else '#C8443C' for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation='h', marker_color=colors,
        text=[f'{v:+.2f}' for v in vals], textposition='outside',
        textfont=dict(color="#9CACC0", size=11),
    ))
    fig.update_layout(
        height=max(180, 42 * len(items)), margin=dict(l=10, r=44, t=10, b=30),
        plot_bgcolor="#142840", paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#9CACC0", size=11),
        xaxis=dict(title='기여도 (gradient×input · 표준화 피처)',
                   gridcolor="rgba(255,255,255,0.07)", zeroline=True, zerolinecolor="#6B7B91"),
        yaxis=dict(gridcolor='rgba(0,0,0,0)'),
        showlegend=False,
    )
    st.caption(f"모델이 이 투구를 {label}로 예측하는 데 각 피처가 기여한 방향 · "
               "초록=예측을 밀어올림, 빨강=끌어내림")
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


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
        marker_color=[PITCH_COLORS.get(k, '#9CACC0') for k in ratio.index],
        text=[f'{v:.0f}%' for v in ratio.values],
        textposition='outside',
    ))
    fig.update_layout(
        height=max(120, 40 * len(ratio)), margin=dict(l=10, r=30, t=10, b=30),
        plot_bgcolor="#142840", paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#9CACC0", size=11),
        xaxis=dict(title='%', range=[0, max(ratio.values) * 1.25],
                   gridcolor="rgba(255,255,255,0.07)", zeroline=False),
        yaxis=dict(gridcolor='rgba(0,0,0,0)'),
        showlegend=False,
    )
    st.caption("최근 구종 사용 비율")
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# ── 사이드바: 입력 컨트롤 ────────────────────────────────
with st.sidebar:
    st.markdown("### 투구 만들기")
    st.caption("⏱️ 첫 예측은 모델 로딩으로 잠시 걸릴 수 있어요. 이후에는 즉시 응답합니다.")
    mode = st.radio(
        "입력 방식",
        ["슬라이더로 직접 조절", "CSV에서 실제 투구 선택", "투수 이름으로 검색"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if mode == "슬라이더로 직접 조절":
        spd = st.slider("구속 (mph)", 60.0, 105.0, SAMPLE_VALUES['release_speed'], 0.5,
                         help="빠를수록 FF/FC 계열")
        spin = st.slider("회전수 (rpm)", 1500.0, 3500.0, SAMPLE_VALUES['release_spin_rate'], 10.0,
                          help="높을수록 포심/커터 계열")
        az = st.slider("수직 가속도 az", -50.0, 15.0, SAMPLE_VALUES['az'], 0.5,
                        help="양수=포심(떠오름), 음수=싱커(가라앉음)")
        ax_ = st.slider("수평 가속도 ax", -30.0, 30.0, SAMPLE_VALUES['ax'], 0.5,
                         help="좌완/우완에 따라 휘는 방향이 반대로 표시됩니다")

        base_rx = abs(SAMPLE_VALUES['release_pos_x'])
        rx = base_rx if st.session_state.p_throws == 'L' else -base_rx

        vz0 = round(-3.0 - 0.15 * az, 2)
        vx0 = round(0.20 * ax_, 2)
        ext = SAMPLE_VALUES['release_extension']
        rz = SAMPLE_VALUES['release_pos_z']

        vy0 = -(spd * 1.467)
        dist = PITCH_DIST - ext
        t = dist / max(abs(vy0), 1)

        inp = dict(SAMPLE_VALUES)
        inp.update({
            'release_speed': spd, 'release_spin_rate': spin,
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

    elif mode == "CSV에서 실제 투구 선택":
        uploaded = st.file_uploader("Statcast CSV 업로드", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded, low_memory=False)
            select_and_predict_from_df(df)
        else:
            st.info("CSV 파일을 업로드하면 실제 투구 데이터에서 선택할 수 있습니다.")

    else:
        st.caption(
            f"Statcast 데이터는 {STATCAST_START_YEAR}년부터 제공됩니다. "
            f"{STATCAST_START_YEAR}년 이전 활동 선수는 그 이후 시즌 기록만 조회할 수 있어요."
        )
        query = st.text_input("투수 이름 (2글자 이상, 영어)", placeholder="Gerrit Cole")
        query = query.strip()

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
    p_throws_choice = st.radio(
        "투구 손", options=['R', 'L'],
        format_func=lambda v: '우완 (R)' if v == 'R' else '좌완 (L)',
        horizontal=True, key='p_throws',
        disabled=(mode != "슬라이더로 직접 조절"),
        help=None if mode == "슬라이더로 직접 조절" else "실제 투구 데이터에서 자동으로 반영됩니다.",
    )

    st.markdown("---")
    with st.expander("구종별 만들기 가이드"):
        guide_rows = [
            ('FF', '포심패스트볼', '구속↑(93+)', 'az 양수 크게(떠오름)', 'ax 0 근처'),
            ('SI', '싱커', '구속↑(92+)', 'az 음수(살짝 가라앉음)', 'ax 양수(몸쪽으로 휨)'),
            ('SL', '슬라이더', '구속중간(82-88)', 'az 0~음수 약간', 'ax 양수 크게(바깥쪽으로 휨)'),
            ('CU', '커브', '구속↓(75-82)', 'az 음수 크게(뚝 떨어짐)', 'ax 음수(바깥쪽 반대로 휨)'),
            ('CH', '체인지업', '구속↓(80-87)', 'az 음수 약간', 'ax 양수 약간'),
            ('FC', '커터', '구속↑(88-94)', 'az 0 근처', 'ax 음수 약간(살짝 반대로 휨)'),
            ('FS', '스플리터', '구속중간(83-90)', 'az 음수 매우 크게(급격히 떨어짐)', 'ax 0 근처'),
        ]
        html = '<table class="guide-table">'
        for code, name, spd_d, az_d, ax_d in guide_rows:
            html += (f'<tr><td class="guide-code">{code}</td><td>{name}</td>'
                     f'<td>{spd_d}</td><td>{az_d}</td><td>{ax_d}</td></tr>')
        html += '</table>'
        st.markdown(html, unsafe_allow_html=True)
        st.caption("※ ax 방향은 우완 기준입니다. 좌완은 좌우가 반대로 적용됩니다.")

# ── 예측 결과 배너 ───────────────────────────────────────
inp = st.session_state.inp
label, conf, proba = st.session_state.label, st.session_state.conf, st.session_state.proba

if label:
    color = PITCH_COLORS.get(label, '#E8B84B')
    st.markdown(
        f'<div class="pred-banner">'
        f'<span class="pred-code">예측 구종</span>'
        f'<p class="pred-name" style="color:{color}">{label} · {PITCH_NAMES.get(label, label)}</p>'
        f'<span class="pred-conf">신뢰도 {conf*100:.1f}%</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    if st.session_state.explanation:
        st.caption(st.session_state.explanation)
    if st.session_state.attribution:
        render_attribution(st.session_state.attribution, label)

st.markdown("<br>", unsafe_allow_html=True)

# ── 시각화: 궤적 + 스트라이크존 + 확률 ───────────────────
col1, col2, col3 = st.columns([1.3, 1, 1.1])

plot_bg = "#142840"
grid_color = "rgba(255,255,255,0.07)"
text_color = "#9CACC0"

with col1:
    st.markdown("**투구 궤적**")
    side_x, side_z = compute_trajectory_side(inp)
    top_y, top_x = compute_trajectory_top(inp, st.session_state.p_throws)
    color = PITCH_COLORS.get(label, '#E8B84B') if label else '#E8B84B'

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=side_x, y=side_z, mode='lines', line=dict(color=color, width=3),
                              name='측면뷰', xaxis='x', yaxis='y'))
    fig.add_trace(go.Scatter(x=[side_x[0]], y=[side_z[0]], mode='markers',
                              marker=dict(color=color, size=8), showlegend=False))
    fig.update_layout(
        height=260, margin=dict(l=40, r=10, t=10, b=30),
        plot_bgcolor=plot_bg, paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=text_color, size=11),
        xaxis=dict(title='거리 (ft)', range=[0, PITCH_DIST], gridcolor=grid_color, zeroline=False),
        yaxis=dict(title='높이 (ft)', range=[0, 8], gridcolor=grid_color, zeroline=False),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=top_y, y=top_x, mode='lines', line=dict(color=color, width=3)))
    fig2.add_trace(go.Scatter(x=[top_y[0]], y=[top_x[0]], mode='markers',
                               marker=dict(color=color, size=8)))
    fig2.add_hline(y=0, line_dash='dash', line_color=grid_color)
    fig2.update_layout(
        height=200, margin=dict(l=40, r=10, t=10, b=30),
        plot_bgcolor=plot_bg, paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=text_color, size=11),
        xaxis=dict(title='거리 (ft)', range=[0, PITCH_DIST], gridcolor=grid_color, zeroline=False),
        yaxis=dict(title='좌우 (ft)', range=[-3, 3], gridcolor=grid_color, zeroline=False),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

with col2:
    st.markdown("**스트라이크존**")
    fig3 = go.Figure()
    fig3.add_shape(type="rect", x0=-0.708, x1=0.708, y0=1.5, y1=3.5,
                    line=dict(color="#6B7B91", width=2), fillcolor="rgba(232,184,75,0.08)")
    fig3.add_trace(go.Scatter(
        x=[inp['plate_x']], y=[inp['plate_z']], mode='markers',
        marker=dict(color=color, size=18, line=dict(color='white', width=1))
    ))
    fig3.update_layout(
        height=480, margin=dict(l=40, r=10, t=10, b=30),
        plot_bgcolor=plot_bg, paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=text_color, size=11),
        xaxis=dict(title='좌우 (ft)', range=[-2, 2], gridcolor=grid_color, zeroline=False,
                   scaleanchor='y'),
        yaxis=dict(title='높이 (ft)', range=[0, 5], gridcolor=grid_color, zeroline=False),
        showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})

with col3:
    st.markdown("**구종별 예측 확률**")
    if proba:
        sorted_items = sorted(proba.items(), key=lambda x: x[1])
        labels = [k for k, v in sorted_items]
        values = [v * 100 for k, v in sorted_items]
        colors = [PITCH_COLORS.get(k, '#9CACC0') for k in labels]

        fig4 = go.Figure(go.Bar(
            x=values, y=labels, orientation='h',
            marker_color=colors,
            text=[f'{v:.1f}%' for v in values],
            textposition='outside',
            textfont=dict(color=text_color, size=11),
        ))
        fig4.update_layout(
            height=480, margin=dict(l=40, r=40, t=10, b=30),
            plot_bgcolor=plot_bg, paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=text_color, size=11),
            xaxis=dict(title='%', range=[0, 115], gridcolor=grid_color, zeroline=False),
            yaxis=dict(gridcolor='rgba(0,0,0,0)'),
        )
        st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption(
    "MLB Statcast 2024 정규시즌 데이터로 학습된 MLP 분류기 · "
    "Weighted F1 0.94 · GitHub에서 전체 코드와 학습 과정을 확인할 수 있습니다."
)
