"""app.py - MLB Pitch Classifier 웹 데모 (Streamlit) · "Pitch Workbench" 라이트 테마"""

import concurrent.futures
import html
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
    FEATURE_COLS, PITCH_NAMES, PITCH_COLORS, SAMPLE_VALUES, PITCH_DIST, games_from_pitches,
)

st.set_page_config(
    page_title="Pitch Workbench",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="auto",   # 모바일에선 접힌 채 시작(펼치면 메인을 덮음)
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

# 슬라이더 min/max. 실측 데이터(data/statcast_2026.csv.gz, 7개 구종·구속 40~110mph
# 기준) 백분위수로 정한 것 — 여러 곳에서 같은 값을 써야 해서 상수로 뺐다.
# az만 예외: 실제 az는 -50~0에 거의 다 들어가지만, 존 클릭 시 vz0과의 상쇄 때문에
# 특정 높이에 정확히 도달하려면 (기본 설정 기준 존 하단만 해도) -190 근처까지 필요해
# 훨씬 넓게 잡았다 — 클릭 없이 직접 만질 땐 -50 아래는 실제 MLB엔 없는 값이다.
AZ_RANGE = (-220.0, 20.0)
RX_RANGE = (-4.0, 4.0)

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

# 구종별 정밀도·재현율·F1. 이전 배포 모델(2024~2025 학습)을 당시 학습 때 보지 못한
# 2026시즌 데이터(549,360건, data/statcast_2026.csv.gz)로 평가한 값. 현재 모델(#88)은
# 2026시즌까지 학습에 포함하므로 이 값은 현재 모델의 미학습 성능이 아니다 — 모델 정보 탭
# 요약 지표(Weighted F1 0.859)는 2024~2025 투수 단위 5-fold 기준이라 평가 방식이 달라
# 서로 직접 비교하면 안 된다.
CLASS_METRICS = {
    'FF': (0.953, 0.974, 0.964), 'SI': (0.945, 0.933, 0.939),
    'SL': (0.846, 0.884, 0.865), 'CU': (0.938, 0.908, 0.923),
    'CH': (0.885, 0.878, 0.882), 'FC': (0.801, 0.745, 0.772),
    'FS': (0.684, 0.638, 0.660),
}

# 전역 특성 중요도 = 예측 클래스 확률에 대한 gradient×input(오른쪽 SHAP 인사이트와
# 같은 방식)을 위 2026시즌 평가셋 6만 건에 평균 낸 값. "이번 한 번의 예측 기여도"가
# 아니라 모델 전반의 평균적인 중요도 순위다. 비중(%)으로 정규화해 저장.
GLOBAL_IMPORTANCE = [
    ('spin_axis', 20.6), ('release_pos_x', 10.1), ('pfx_x', 9.2),
    ('release_spin_rate', 9.1), ('ax', 9.1), ('pfx_z', 8.1), ('az', 5.4),
    ('release_pos_z', 4.9), ('vx0', 4.6), ('release_extension', 3.7),
    ('release_speed', 3.6), ('vy0', 2.8), ('effective_speed', 2.4),
    ('vz0', 1.9), ('plate_z', 1.8), ('ay', 1.5), ('plate_x', 1.3),
]

# ── 디자인 토큰 · 라이트 테마 CSS ────────────────────────
# Streamlit은 매 rerun마다 DOM을 새로 만들므로 이 <style>도 매번 재주입돼야 한다.
# 문자열은 모듈 상수라 재생성 비용은 없다(= 성능 병목 아님).
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap');

:root{
  --font:'Public Sans',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  --bg:#ffffff;
  --card:#ffffff;
  --card-border:#e3e5e8;
  --rule:#eceef0;
  --text:#1c1f24;
  --text-2:#50565e;
  --text-3:#62676f;          /* 흰 배경 5.7:1 · 칩 배경 5.2:1 (작은 글씨 AA 4.5:1 이상) */
  --accent:#c53637;
  --on-accent:#ffffff;
  --chip-gray:#f4f5f6;
  --field:#f4f5f7;
  --track:#e9eaec;
  --tip-bg:#3f444b;
  --radius:8px;
}

html,body,[class*="css"]{font-family:var(--font);}
/* 한글은 글자마다 줄바꿈되어 끝 글자만 다음 줄로 넘어가므로 단어 단위로 */
[data-testid="stMarkdownContainer"] p,[data-testid="stCaptionContainer"] p,[data-testid="stCaptionContainer"],.pw-shap-sub,.pw-shap-expl,.pw-gnote{word-break:keep-all;overflow-wrap:break-word;}
/* 선택 목록(드롭다운)은 사이드바 폭과 무관하게 내용만큼 넓게 펼친다 */
/* 가상 스크롤 목록은 줄이 겹쳐 배치돼 max-content가 0으로 계산되므로 고정 최소 폭(가장 긴 항목 ≈ 25rem) */
[data-testid="stSelectboxVirtualDropdown"]{width:auto !important;min-width:28rem;max-width:92vw;}
[data-testid="stSelectboxVirtualDropdown"] [role="listbox"]{width:auto !important;min-width:100%;}
.stApp{background:var(--bg);color:var(--text);}
h1,h2,h3,h4{font-family:var(--font) !important;color:var(--text);text-wrap:balance;}
#MainMenu,footer{visibility:hidden;}
header[data-testid="stHeader"]{height:0 !important;visibility:hidden;}
/* 헤더는 숨기되 사이드바 펼침 버튼은 살린다 — 없으면 접힌 사이드바(모바일 기본)를 다시 열 수 없다 */
[data-testid="stExpandSidebarButton"]{visibility:visible;}
hr{border-color:var(--card-border) !important;}

/* 상단 여백 축소 */
[data-testid="stMainBlockContainer"],.block-container{padding-top:1.1rem !important;}

/* 사이드바 컴팩트화 */
section[data-testid="stSidebar"]{background:var(--card);border-right:1px solid var(--card-border);}
section[data-testid="stSidebar"] .block-container{padding:0.75rem 1rem 1rem;}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:0.5rem;}
section[data-testid="stSidebar"] hr{margin:0.5rem 0 !important;}
section[data-testid="stSidebar"] [data-testid="stSliderTickBar"]{display:none;}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{font-size:0.82rem;}

div[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--card);
  border:1px solid var(--card-border) !important;
  border-radius:var(--radius) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div{padding:16px 20px;}

/* 라벨 = 본문 글꼴 한 단계 작게, 대문자·자간 없음 */
.pw-label{font-size:13px;font-weight:600;color:var(--text-2);margin-bottom:12px;}
.pw-help{position:relative;display:inline-flex;align-items:center;justify-content:center;cursor:help;
  font-weight:700;color:var(--text-3);border:1px solid var(--text-3);border-radius:50%;
  width:16px;height:16px;font-size:11px;margin-left:6px;vertical-align:middle;}
.pw-help::before{content:"";position:absolute;inset:-4px;}   /* 클릭·탭 영역 24px */
.pw-help:focus-visible{outline:2px solid var(--accent);outline-offset:2px;}
.pw-help::after{content:attr(data-tip);position:absolute;z-index:50;bottom:150%;left:-8px;width:280px;max-width:70vw;
  padding:12px;border-radius:var(--radius);background:var(--tip-bg);color:var(--on-accent);font-size:13px;font-weight:400;
  line-height:1.5;text-align:left;white-space:normal;opacity:0;visibility:hidden;pointer-events:none;}
.pw-help:hover::after,.pw-help:focus-visible::after{opacity:1;visibility:visible;}

div[role="radiogroup"] label:has(input:focus-visible){outline:2px solid var(--accent);outline-offset:2px;border-radius:4px;}

[data-testid="stMultiSelectTagsContainer"] [data-tag]{background:var(--chip-gray) !important;color:var(--text) !important;}
.pw-logo{display:flex;align-items:center;gap:8px;margin:0 0 12px;}
.pw-logo span{font-weight:700;font-size:1.05rem;color:var(--text);}

/* 결과 요약 — 예측 구종 하나만 크게, 모델 정보는 작은 한 줄 */
.pw-summary{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 16px;margin:4px 0 16px;
  padding-bottom:12px;border-bottom:1px solid var(--rule);}
.pw-sum-pred{font-size:1.9rem;font-weight:700;color:var(--accent);line-height:1.1;}
.pw-sum-conf{font-size:1.1rem;font-weight:600;color:var(--text);font-variant-numeric:tabular-nums;}
.pw-sum-actual{font-size:0.9rem;font-weight:600;color:var(--text-2);}
.pw-sum-actual.bad{color:var(--accent);}
/* 투구 이동 줄은 좁은 화면에서도 한 줄 유지 */
.st-key-pitch_nav [data-testid="stHorizontalBlock"]{flex-wrap:nowrap;align-items:center;}
.st-key-pitch_nav [data-testid="stColumn"]{min-width:0;}
.pw-nav{text-align:center;font-size:0.85rem;color:var(--text-2);font-variant-numeric:tabular-nums;}
.pw-sum-meta{flex-basis:100%;font-size:12px;color:var(--text-3);font-variant-numeric:tabular-nums;}
.pw-sum-meta span + span{margin-left:12px;}

/* 구종별 확률 바 — 전부 회색, 예측 구종만 빨강 */
.pw-bars{padding-bottom:12px;display:grid;grid-template-columns:1fr 1fr;column-gap:24px;}
@media (max-width:640px){.pw-bars{grid-template-columns:1fr;}}
.pw-bar-row{display:flex;align-items:center;gap:12px;margin:6px 0;}
.pw-bar-name{width:84px;font-size:0.82rem;color:var(--text-2);flex:none;}
.pw-bar-track{flex:1;height:8px;background:var(--track);border-radius:2px;overflow:hidden;}
.pw-bar-fill{display:block;height:100%;border-radius:2px;}
.pw-bar-pct{width:40px;text-align:right;font-size:0.8rem;font-weight:600;color:var(--text);flex:none;font-variant-numeric:tabular-nums;}

.pw-shap-head{font-weight:700;font-size:1rem;color:var(--text);}
.pw-shap-sub{font-size:0.8rem;color:var(--text-3);margin:4px 0 12px;}
.pw-shap-expl{font-size:0.85rem;line-height:1.6;color:var(--text-2);margin:0 0 16px;padding-left:12px;border-left:2px solid var(--accent);}
.pw-wf-ends{display:flex;justify-content:space-between;font-size:12px;color:var(--text-3);margin:-8px 0 16px;font-variant-numeric:tabular-nums;}

/* 특성 기여도 — 카드 대신 구분선 목록 */
.pw-frow{padding:10px 0;border-top:1px solid var(--rule);}
.pw-frow-top{display:flex;justify-content:space-between;align-items:baseline;gap:8px;}
.pw-fname{font-size:0.85rem;font-weight:600;color:var(--text);}
.pw-fval{font-size:12px;color:var(--text-3);font-variant-numeric:tabular-nums;margin-left:8px;font-weight:400;}
.pw-fbadge{font-size:0.8rem;font-weight:700;color:var(--text-2);font-variant-numeric:tabular-nums;}
.pw-fbadge.pos{color:var(--accent);}
.pw-fmag{height:4px;background:var(--track);border-radius:2px;margin-top:8px;overflow:hidden;}
.pw-fmag span{display:block;height:100%;}

.pw-guide{width:100%;table-layout:fixed;border-collapse:collapse;font-size:0.8rem;color:var(--text-2);font-variant-numeric:tabular-nums;}
.pw-guide th{font-size:12px;font-weight:600;color:var(--text-3);text-align:left;padding:0 4px 4px;border-bottom:1px solid var(--card-border);}
.pw-guide td{padding:6px 4px;border-bottom:1px solid var(--rule);overflow-wrap:anywhere;}
.pw-guide td:first-child{color:var(--text);font-weight:500;}
.pw-guide tbody tr:last-child td{border-bottom:none;}
.pw-gcode{display:inline-block;font-weight:700;font-size:0.7rem;background:var(--chip-gray);color:var(--text-2);padding:2px 4px;border-radius:4px;margin-right:8px;}
.pw-gnote{font-size:12px;color:var(--text-3);margin-top:8px;line-height:1.5;}

/* 입력창 — 테두리는 config.toml showWidgetBorder가 그려주고, 여기선 연한 회색
   채움을 더해 흰 배경에서도 필드 경계가 확실히 보이게 한다 */
[data-testid="stTextInputRootElement"],
.stTextInput div[data-baseweb="base-input"],
.stDateInput div[data-baseweb="input"],
.stNumberInput div[data-baseweb="input"],
.stSelectbox div[data-baseweb="select"] > div{background:var(--field) !important;}
.stTextInput input,.stDateInput input,.stNumberInput input{background:var(--field) !important;}
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


def _pfx_inches(spd, ax, az, ext=6.2):
    """ax·az(가속도, ft/s²)와 비행시간으로 계산한 유도 무브먼트(inch).
    pfx = 0.5·a·t² (z는 중력 32.174 제거), t = 릴리스~홈플레이트 거리 / 초기 구속.
    2026시즌 실측 Statcast pfx_x·pfx_z(×12)와 절대오차 약 1in(x)·0.5in(z)로 일치해,
    구종별 참조 원(PITCH_REF)·가상 입력 링·실제 투구 점이 같은 좌표계에 놓인다
    (위 = 라이징/포심, 아래 = 드롭/커브)."""
    t = (PITCH_DIST - ext) / max(spd * 1.467, 1)
    return 6.0 * ax * t * t, 6.0 * (az + 32.174) * t * t


def in_data_mode():
    """실제 투구 히스토리가 로드된 모드인지. 릴리스포인트·구속분포·히트맵처럼 여러 투구가
    있어야 그릴 수 있는 보조 차트는 이때만 렌더한다(아니면 안내 placeholder)."""
    return st.session_state.get('movement_bg') is not None


def _apply_zone_click():
    """스트라이크존 클릭 좌표를 ax·az 슬라이더와 실제 궤적에 반영한다.

    구속·회전수·익스텐션·릴리스 위치는 고정. 어떤 구종으로 예측되는지는 신경 쓰지
    않는다 — 그 위치에 실제로 도달하는 데 필요한 값을 그대로 반영하고, 예측은 바뀐
    값에 맞춰 다시 계산된다.

    - 좌우(ax): plate_x 식을 뒤집어 정확한 ax를 바로 구한다.
    - 높이(az): az가 vz0 = -3.0-0.15·az에도 연동된다는 걸 감안해, plate_z 식을
      그대로 뒤집어 az를 직접 정확히 구한다. 이 연동 때문에 az가 plate_z에 미치는
      실질 영향이 매우 작아(전형적 t에서 az 1단위당 0.02ft 안팎), 존 안 어디든
      정확히 도달하려면 실제 MLB az 범위(-50~0)를 한참 벗어나는 값이 필요할 수
      있다 — AZ_RANGE를 넉넉히 잡아 두었다. 그래도 필요한 az가 슬라이더 범위를
      넘는 경우(아주 빠른 구속·짧은 익스텐션 조합 등)에만 클램프하고, 그때는
      클램프된 az를 고정값으로 두고 vz0을 다시 풀어 최대한 가깝게 맞춘다 — 정상
      범위 안이면 슬라이더 값과 실제 궤적이 항상 정확히 일치한다.
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

    az_exact = (pz - rz + 3.0 * t) / (0.5 * t * t - 0.15 * t)
    az = _snap(az_exact, *AZ_RANGE)
    # az가 정상 범위면 표준 공식(vz0 = -3.0-0.15az)이 이미 클릭 높이에 정확히 맞는다.
    # 존 위/아래 끝을 클릭해 az가 AZ_RANGE 밖으로 튀어(진짜 클램프) 나갈 때만, 그
    # 클램프된 az를 고정하고 vz0을 다시 풀어서 최대한 가깝게 맞춘다 — 어느 쪽이든
    # az 슬라이더 값과 실제 궤적이 항상 서로 일치한다.
    # (0.5 step으로 반올림(_snap)된 오차만으로는 "클램프"로 치면 안 된다 — 그 정도
    # 오차는 항상 있고, 그때마다 폴백으로 넘어가면 vz0이 [-14,3] 범위에 부당하게
    # 잘려서 오히려 결과가 크게 어긋난다.)
    clamped = az_exact < AZ_RANGE[0] or az_exact > AZ_RANGE[1]
    vz0 = round(-3.0 - 0.15 * az, 2)
    if clamped:
        vz0 = _snap((pz - rz - 0.5 * az * t * t) / t, -14.0, 3.0, step=0.01)

    st.session_state["ax_slider"] = ax
    st.session_state["az_slider"] = az
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


def _explain_worker(items, p_throws, label, conf, proba):
    """스레드에서 실행 — 메인에서 이미 계산한 예측 결과를 받아 설명만 생성한다(모델 재추론 없음)."""
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
        "예측 중입니다…" if st.session_state.warmed_up
        else "예측 중입니다… (첫 예측은 모델 로딩으로 잠시 걸릴 수 있어요)"
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
                _explain_worker, key, st.session_state.p_throws, label, conf, proba)
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


def prepare_pitches(df):
    """결측 검사 후 예측 가능한 행만 남기고 무브먼트 배경(movement_bg)을 채운다.
    선택할 투구가 없으면 None."""
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        st.error(f"필요한 컬럼이 없습니다: {missing}")
        return None

    df_valid = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    if df_valid.empty:
        st.warning("선택 가능한 투구가 없습니다.")
        return None

    st.success(f"{len(df_valid):,}개 투구 로드 완료")

    # 무브먼트 차트 배경 = 이 데이터셋의 실제 유도 무브먼트(inch). 실제 Statcast pfx_z는
    # 이미 위=라이징 관례라 부호 반전 없이 그대로 × 12.
    bg = df_valid[['pfx_x', 'pfx_z']]
    if len(bg) > 500:
        bg = bg.sample(500, random_state=0)
    st.session_state.movement_bg = ((bg['pfx_x'] * 12.0).tolist(),
                                    (bg['pfx_z'] * 12.0).tolist())
    return df_valid


EXTRA_PITCH_NAMES = {'ST': '스위퍼', 'KC': '너클커브', 'SV': '슬러브', 'FO': '포크볼',
                     'KN': '너클볼', 'EP': '이퓨스', 'CS': '슬로 커브'}
RESULT_KR = {'ball': '볼', 'blocked_ball': '볼', 'called_strike': '스트라이크', 'swinging_strike': '헛스윙',
             'swinging_strike_blocked': '헛스윙', 'foul': '파울', 'foul_tip': '파울팁', 'foul_bunt': '파울',
             'hit_into_play': '인플레이', 'hit_by_pitch': '몸에 맞는 공', 'automatic_ball': '볼',
             'automatic_strike': '스트라이크', 'missed_bunt': '헛스윙', 'pitchout': '피치아웃'}


def pitch_name(code):
    return PITCH_NAMES.get(code) or EXTRA_PITCH_NAMES.get(code) or str(code)


def pitch_table(df):
    """투구 목록 표. 경기 데이터(이닝·타석 등)가 있으면 그 열, 없으면(CSV 등) 있는 열만."""
    c, t = df.columns, {}

    def num(col):
        return pd.to_numeric(df[col], errors='coerce').astype('Int64')

    game = 'inning' in c
    if game:
        half = df['inning_topbot'].map({'Top': '초', 'Bot': '말'}).fillna('') if 'inning_topbot' in c else ''
        t['이닝'] = num('inning').astype(str) + '회' + half
    if 'at_bat_number' in c:
        t['타석'] = num('at_bat_number')
    if 'pitch_number' in c:
        t['구'] = num('pitch_number')
    if 'balls' in c and 'strikes' in c:
        t['카운트'] = num('balls').astype(str) + '-' + num('strikes').astype(str)
    if 'stand' in c:
        t['타자'] = df['stand'].map({'L': '좌', 'R': '우'}).fillna('')
    if not game and 'player_name' in c:
        t['투수'] = df['player_name']
    if not game and 'game_date' in c:
        t['날짜'] = df['game_date'].astype(str).str[:10]
    if 'pitch_type' in c:
        t['실제 구종'] = df['pitch_type'].map(lambda x: pitch_name(x) if pd.notna(x) else '')
    t['구속'] = df['release_speed'].round(1)
    if not game and 'release_spin_rate' in c:
        t['회전수'] = df['release_spin_rate'].round(0).astype('Int64')
    if 'description' in c:
        t['결과'] = df['description'].map(lambda x: RESULT_KR.get(x, x) if pd.notna(x) else '')
    return pd.DataFrame(t)


def _step_pitch(d, n, tkey):
    """이전·다음 버튼: 표의 선택 행을 옮긴다(콜백이라 이번 실행의 위젯 생성 전에 반영됨)."""
    s = st.session_state.get(tkey)
    cur = s["selection"]["rows"][0] if s and s["selection"]["rows"] else 0
    st.session_state[tkey] = {"selection": {"rows": [min(max(cur + d, 0), n - 1)], "columns": []}}


def _pitch_info(row):
    """'5회초, 우타자, 카운트 2-1' — 있는 정보만."""
    out = []
    if pd.notna(row.get('inning')):
        out.append(f"{int(row['inning'])}회{'초' if row.get('inning_topbot') == 'Top' else '말'}")
    if row.get('stand') in ('L', 'R'):
        out.append('좌타자' if row['stand'] == 'L' else '우타자')
    if pd.notna(row.get('balls')) and pd.notna(row.get('strikes')):
        out.append(f"카운트 {int(row['balls'])}-{int(row['strikes'])}")
    return ", ".join(out)


def select_and_predict(df_valid, key=None):
    """prepare_pitches를 거친 DataFrame에서 현재 투구를 정하고 예측을 실행한다.
    현재 투구 = '투구 목록' 표의 선택 행(표 클릭·이전/다음 버튼 모두 이 한 곳을 바꾼다).
    key를 바꾸면(예: 경기별) 선택이 첫 투구로 초기화된다."""
    if {'at_bat_number', 'pitch_number'} <= set(df_valid.columns):
        df_valid = df_valid.sort_values(['at_bat_number', 'pitch_number']).reset_index(drop=True)
    n = len(df_valid)
    sig = key or f"df{n}"
    tkey = f"pitch_tbl_{sig}"
    state = st.session_state.get(tkey)
    rows = state["selection"]["rows"] if state else []
    if st.session_state.get("_pitch_sig") != sig:
        idx = 0
    else:   # 선택 해제(같은 행 재클릭)는 직전 선택 유지
        idx = rows[0] if rows else st.session_state.get("_pitch_idx", 0)
    idx = min(max(idx, 0), n - 1)
    st.session_state["_pitch_sig"], st.session_state["_pitch_idx"] = sig, idx
    st.session_state[tkey] = {"selection": {"rows": [idx], "columns": []}}

    row = df_valid.iloc[idx]
    st.session_state["_pitch_nav"] = dict(n=n, idx=idx, tkey=tkey, table=pitch_table(df_valid),
                                          info=_pitch_info(row), df=df_valid)
    pt = row.get('pitch_type')
    st.session_state["actual_pitch"] = pt if isinstance(pt, str) else None
    inp = {col: float(row[col]) for col in FEATURE_COLS}
    st.session_state.inp = inp
    if 'p_throws' in df_valid.columns:
        st.session_state.p_throws = str(row.get('p_throws', 'R'))
    st.caption(f"현재 투구 {idx + 1} / {n}. 메인 화면 위의 이전·다음 버튼이나 '투구 목록' 탭에서 바꿉니다.")
    run_prediction(inp)


def select_and_predict_from_df(df):
    """DataFrame에서 결측치 없는 행만 골라 선택 UI를 보여주고, 선택된 투구로 예측을 실행한다."""
    df_valid = prepare_pitches(df)
    if df_valid is not None:
        select_and_predict(df_valid)


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


@st.cache_data(show_spinner=False, ttl=3600)
def last_game_date(player_id, from_year, to_year):
    """MLB 경기 기록(정규시즌)에서 이 투수가 마지막으로 등판한 날. from_year부터 to_year까지
    거꾸로 훑어 처음 등판 기록이 나오는 시즌의 마지막 날을 돌려준다(없거나 조회 실패면 None).

    검색 API의 lastPlayedDate는 현역 선수에겐 비어 있어서 못 쓴다.
    ponytail: 정규시즌만 — 포스트시즌 등판이 더 뒤여도 기본 종료일은 정규시즌 마지막 날(직접 늘릴 수 있음).
    """
    for year in range(from_year, to_year - 1, -1):
        try:
            r = requests.get(
                f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats",
                params={"stats": "gameLog", "group": "pitching", "season": year}, timeout=10)
            r.raise_for_status()
            splits = (r.json().get("stats") or [{}])[0].get("splits", [])
        except requests.exceptions.RequestException:
            return None
        if splits:
            return max(sp["date"] for sp in splits)
    return None


def pitcher_date_bounds(c):
    """이 투수의 Statcast 조회 가능 날짜 범위(최소·최대)와 기본 조회 범위를 반환한다.

    기본 범위 = 마지막으로 등판한 시즌 처음~마지막 등판일(등판 기록을 못 구하면 최근 시즌 전체).
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

    def_end = max_d
    last = last_game_date(c["id"], max_d.year, min_d.year)
    if last:
        def_end = min(max(date.fromisoformat(last), min_d), max_d)
    def_start = max(date(def_end.year, 1, 1), min_d)
    return min_d, max_d, def_start, def_end


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


def help_icon(tip):
    """라벨 옆 도움말 ? — 키보드 포커스·화면 낭독기(aria-label)로도 읽히게."""
    t = html.escape(tip, quote=True)
    return f'<span class="pw-help" tabindex="0" role="img" aria-label="도움말: {t}" data-tip="{t}">?</span>'


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


def render_summary(label, conf, inf_ms, actual=None):
    actual_html = ''
    if actual:
        verdict = ("일치" if actual == label else "불일치") if actual in PITCH_NAMES else "모델 7구종 밖"
        actual_html = (f'<span class="pw-sum-actual {"bad" if actual in PITCH_NAMES and actual != label else ""}">'
                       f'실제 구종 {pitch_name(actual)} ({verdict})</span>')
    st.markdown(
        '<div class="pw-summary">'
        f'<span class="pw-sum-pred">{PITCH_NAMES.get(label, label)}</span>'
        f'<span class="pw-sum-conf">신뢰도 {conf * 100:.0f}%</span>'
        f'{actual_html}'
        '<span class="pw-sum-meta"><span>모델 정확도 86.0% (처음 보는 투수 기준)</span>'
        f'<span>추론 {inf_ms:.0f}ms</span></span>'
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
    st.markdown(f'<div class="pw-bars">{rows}</div>', unsafe_allow_html=True)


def render_movement(inp):
    """유도 무브먼트(inch) 평면. 원점 중심 12·24인치 동심원 + 구종별 참조 클러스터
    (회색 원, 라벨 없음, 예측 구종만 빨간 테두리) 위에 현재 예측 투구를 빨간 링으로.
    세로는 Statcast 관례(위 = 라이징/포심, 아래 = 드롭/커브).
    - 실제 투구 모드: 회색 점(투구 분포)과 빨간 링 모두 Statcast 실측 pfx_x·pfx_z(×12).
    - 가상 입력 모드: 빨간 링·회색 점(세션 이력)은 _pfx_inches 계산값.
    구종별 회색 원도 _pfx_inches 계산값이며, 계산식이 실측 pfx와 같은 좌표계라 모두 겹쳐 비교할 수 있다."""
    label_slot = st.empty()   # 도움말에 아래에서 정해지는 범례(ctx)를 넣으려고 자리만 먼저 잡는다
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
        # scatter로 그린다(반경 4.5인치 ≈ 이 차트 스케일에서 지름 40px 근사).
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode='markers', showlegend=False,
            marker=dict(size=40, color="rgba(140,143,152,0.10)",
                        line=dict(color=ACCENT_HEX if hot else "rgba(140,143,152,0.45)",
                                  width=2 if hot else 1)),
            hovertemplate=f"{PITCH_NAMES.get(code, code)}<extra></extra>",
        ))

    if bg:
        fig.add_trace(go.Scatter(x=bg[0], y=bg[1], mode='markers',
                                 marker=dict(size=5, color=GRAY_SOFT), hoverinfo='skip'))
        ctx = "회색 점·빨간 링 = 실제 투구의 Statcast 실측 무브먼트"
    elif hist:
        fig.add_trace(go.Scatter(x=[h[0] for h in hist], y=[h[1] for h in hist],
                                 mode='markers', marker=dict(size=7, color=GRAY_TRAIL),
                                 hoverinfo='skip'))
        ctx = "회색 점 = 이번 세션에서 예측해 본 투구"
    else:
        ctx = "슬라이더를 움직이면 예측 이력이 쌓입니다"

    if bg:
        # 실제 투구: 회색 점과 같은 실측 좌표(Statcast pfx, ft → in)
        px, py = inp['pfx_x'] * 12.0, inp['pfx_z'] * 12.0
    else:
        # 가상 입력: 실측 pfx가 없으므로 ax·az·구속으로 계산(실측 대비 오차 ~1in)
        px, py = _pfx_inches(inp['release_speed'], inp['ax'], inp['az'], inp['release_extension'])
        ctx += " · 빨간 링 = ax·az·구속으로 계산한 무브먼트"
    fig.add_trace(go.Scatter(x=[px], y=[py], mode='markers',
                             marker=dict(size=22, color='rgba(0,0,0,0)',
                                         line=dict(color=ACCENT_HEX, width=3))))
    fig.add_trace(go.Scatter(x=[px], y=[py], mode='markers',
                             marker=dict(size=7, color=ACCENT_HEX)))
    fig.update_layout(
        height=300, margin=dict(l=44, r=8, t=8, b=38),
        plot_bgcolor=PLOT_BG, paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=AXIS_TEXT, size=11), showlegend=False,
        xaxis=dict(title='수평 (in)', range=[-30, 30], gridcolor=GRID,
                   zeroline=False, scaleanchor='y'),
        yaxis=dict(title='수직 (in, 위가 라이징)', range=[-30, 30],
                   gridcolor=GRID, zeroline=False),
    )
    label_slot.markdown(
        '<div class="pw-label">무브먼트 (유도 무브먼트, in) '
        + help_icon(f"점선 원 = 12·24인치, 회색 원 = 구종별 대략 위치(예측 구종은 빨강). {ctx}. "
                    "회색 원은 2026시즌 실측 평균의 대략적인 경향일 뿐, 실제 예측은 이 차트에 없는 "
                    "회전축 등 15개 피처까지 합쳐 계산합니다 — 그래서 예측 구종이 자기 회색 원과 "
                    "떨어져 보일 수 있습니다.") + '</div>',
        unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_strikezone(inp, clickable=False):
    """주심(포수 뒤)이 마운드를 바라본 시점. 회색 점 = 릴리스, 빨간 점 = 플레이트
    통과 위치. clickable이면 존을 클릭해 공 위치를 지정할 수 있다."""
    tip = (
        "좌우(ax)·수직(az) 무브먼트 슬라이더가 클릭한 위치에 실제로 도달하도록 역산되어 "
        "바뀝니다 — 구종이 무엇이 되는지는 신경 쓰지 않고, 그 위치에 맞는 움직임 값을 그대로 "
        "반영해요. 그래서 클릭 한 번에 예측 구종이 크게 바뀔 수 있습니다. "
        "구속·회전수·익스텐션은 그대로 유지됩니다. 회색 점 = 릴리스, 빨간 점 = 플레이트 통과 위치."
        if clickable else
        "주심이 마운드를 바라본 시점 · 회색 점 = 릴리스 · 빨간 점 = 플레이트 통과 위치"
    )
    st.markdown(
        f'<div class="pw-label">스트라이크존{" (클릭해서 위치 지정)" if clickable else ""} {help_icon(tip)}</div>',
        unsafe_allow_html=True)
    fig = go.Figure()
    # 평균 존: 홈플레이트 폭 ±0.83ft(17인치 절반 + 볼 반경), 무릎~겨드랑이 1.5~3.5ft 근사
    fig.add_shape(type="rect", x0=-0.83, x1=0.83, y0=1.5, y1=3.5,
                  line=dict(color=ZONE_LINE, width=2),
                  fillcolor="rgba(138,143,152,0.06)", layer="below")
    if clickable:
        # 옅은 격자 = 클릭 히트타깃(≈0.15ft 간격 ≈ 이 차트 크기에서 5px). 클릭하면 가장 가까운 점이 선택된다.
        # 완전 투명 + hoverinfo='skip'이면 Plotly가 클릭 이벤트를 안 잡으므로 옅게라도 그린다.
        gx, gy = np.meshgrid(np.arange(-1.8, 1.81, 0.15), np.arange(0.3, 4.81, 0.15))
        fig.add_trace(go.Scatter(x=gx.ravel(), y=gy.ravel(), mode='markers',
                                 name='click', marker=dict(size=6, color='rgba(140,143,152,0.10)'),
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
        height=300, margin=dict(l=40, r=8, t=8, b=32),
        plot_bgcolor=PLOT_BG, paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=AXIS_TEXT, size=11), showlegend=False,
        xaxis=dict(title='좌우 (ft, 주심 시점)', range=[-2.5, 2.5], gridcolor=GRID, constrain='domain',
                   zeroline=False, scaleanchor='y'),
        yaxis=dict(title='높이 (ft)', range=[0, 7.5], gridcolor=GRID, zeroline=False),
    )
    if clickable:
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False},
                        on_select="rerun", selection_mode="points", key="zone_click")
    else:
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_paths(inp, p_throws):
    """측면뷰·상단뷰 궤적. 홈플레이트에 스트라이크존 박스로 존 대비 위치를 표시한다."""
    st.markdown('<div class="pw-label">투구 궤적 '
                + help_icon("위 = 측면뷰, 아래 = 위에서 내려다본 상단뷰(주심 시점). 회색 사각형 = 홈플레이트 "
                            "스트라이크존, 빨간 점 = 릴리스·플레이트 통과 지점.") + '</div>',
                unsafe_allow_html=True)
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
    fig.add_annotation(x=PITCH_DIST, y=3.5, yshift=10, xanchor='right', text="홈플레이트", showarrow=False,
                       font=dict(size=9, color=AXIS_TEXT))
    fig.add_trace(go.Scatter(x=side_x, y=side_z, mode='lines',
                             line=dict(color=ACCENT_HEX, width=3)))
    fig.add_trace(go.Scatter(x=[side_x[0], side_x[-1]], y=[side_z[0], side_z[-1]],
                             mode='markers', marker=dict(color=ACCENT_HEX, size=8)))
    fig.update_layout(
        height=230, margin=dict(l=40, r=10, t=8, b=30),
        plot_bgcolor=PLOT_BG, paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=AXIS_TEXT, size=11), showlegend=False,
        xaxis=dict(title='거리 (ft, 측면뷰)', range=[0, PITCH_DIST], gridcolor=GRID, zeroline=False),
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
    fig2.add_annotation(x=0, y=PITCH_DIST + 3, text="타석 (홈플레이트)", showarrow=False,
                        font=dict(size=9, color=AXIS_TEXT))
    fig2.add_trace(go.Scatter(x=top_x, y=top_y, mode='lines', line=dict(color=ACCENT_HEX, width=3)))
    fig2.add_trace(go.Scatter(x=[top_x[0], top_x[-1]], y=[top_y[0], top_y[-1]],
                              mode='markers', marker=dict(color=ACCENT_HEX, size=8)))
    fig2.add_vline(x=0, line_dash='dash', line_color=GRID)
    fig2.update_layout(
        height=290, margin=dict(l=40, r=10, t=8, b=30),
        plot_bgcolor=PLOT_BG, paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=AXIS_TEXT, size=11), showlegend=False,
        xaxis=dict(title='좌우 (ft, 상단뷰·주심 시점)', range=[-4, 4], gridcolor=GRID,
                   zeroline=False),
        yaxis=dict(title='거리 (ft, 아래가 홈플레이트)',
                   range=[PITCH_DIST + 6, -4], gridcolor=GRID, zeroline=False),
    )
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})


def render_model_info():
    st.markdown('<div class="pw-label">모델 정보</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Weighted F1", "0.859")
    m2.metric("Test Accuracy", "86.0%")
    m3.metric("Pitch Types", "7")
    st.caption(
        "17개 Statcast 피처를 입력하면 MLP 신경망이 7가지 구종 중 하나를 예측합니다. "
        "MLB Statcast 2024~2025 정규시즌 데이터로 학습 · 위 지표는 투수 단위 5-fold 평균(처음 보는 투수 기준) · "
        "모델은 lru_cache로 프로세스당 1회만 로드됩니다."
    )
    st.caption("참고: 투구 단위 랜덤 분할 기준(같은 투수가 학습·테스트에 함께 포함)으로는 Weighted F1 0.93 · 정확도 93.4%")

    st.divider()
    st.markdown(
        '<div class="pw-label">구종별 정밀도, 재현율, F1 '
        + help_icon("이전 모델(2024~2025 학습)을 당시 학습 때 보지 못한 2026시즌 "
                    "데이터(549,360건)에 평가한 값입니다. 현재 모델은 2026시즌까지 학습에 포함해 아직 재평가 전입니다. "
                    "위 요약 지표(투수 단위 5-fold, 처음 보는 투수 기준)와는 "
                    "평가 방식이 달라 서로 직접 비교하지 마세요.") + '</div>',
        unsafe_allow_html=True)
    rows = "".join(
        f'<tr><td><span class="pw-gcode">{code}</span>{PITCH_NAMES.get(code, code)}</td>'
        f'<td>{p:.2f}</td><td>{r:.2f}</td><td>{f1:.2f}</td></tr>'
        for code, (p, r, f1) in CLASS_METRICS.items()
    )
    st.markdown(
        '<table class="pw-guide"><colgroup>'
        '<col style="width:34%"><col style="width:22%"><col style="width:22%"><col style="width:22%">'
        '</colgroup><thead><tr>'
        '<th>구종</th><th>정밀도</th><th>재현율</th><th>F1</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>',
        unsafe_allow_html=True)
    st.caption("이전 모델(2024~2025 학습)의 2026시즌 평가 기준 · 현재 모델은 2026 포함 학습, 재평가 전 · "
               "FC·FS가 상대적으로 약함")

    st.divider()
    st.markdown(
        '<div class="pw-label">전역 특성 중요도 '
        + help_icon("오른쪽 예측 근거(SHAP)는 이번 한 번의 예측에 대한 기여도이고, "
                    "이 차트는 2026시즌 평가셋 전체에서 평균 낸 모델 전반의 평균적인 중요도 순위입니다.") + '</div>',
        unsafe_allow_html=True)
    names = [FEAT_KR.get(f, f) for f, _ in reversed(GLOBAL_IMPORTANCE)]
    vals = [v for _, v in reversed(GLOBAL_IMPORTANCE)]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation='h', marker_color=GRAY,
        text=[f'{v:.0f}%' for v in vals], textposition='outside',
    ))
    fig.update_layout(
        height=460, margin=dict(l=10, r=30, t=10, b=30),
        plot_bgcolor=PLOT_BG, paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=AXIS_TEXT, size=11), showlegend=False,
        xaxis=dict(title='평균 기여도 비중 (%)', range=[0, max(vals) * 1.2],
                   gridcolor=GRID, zeroline=False),
        yaxis=dict(gridcolor='rgba(0,0,0,0)'),
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.caption("gradient×input 기여도의 절댓값을 평가셋 전체에 평균 · 회전축(spin_axis)이 가장 크게 기여")


def game_filters(g, game_pk):
    """한 경기의 구종·이닝·타자 손 필터(사이드바, 비워 두면 전체). (필터링한 df, 선택 상태 서명)을 돌려준다.
    위젯 key에 경기를 넣어 경기를 바꾸면 필터도 초기화된다."""
    types = [t for t in g['pitch_type'].value_counts().index if isinstance(t, str)]
    innings = sorted(g['inning'].dropna().astype(int).unique())
    with st.expander("필터 (구종, 이닝, 타자)"):
        sel_t = st.multiselect("구종", types, format_func=pitch_name, placeholder="전체", key=f"flt_t_{game_pk}")
        sel_i = st.multiselect("이닝", innings, format_func=lambda i: f"{i}회", placeholder="전체",
                               key=f"flt_i_{game_pk}")
        stand = st.radio("타자 손", ["전체", "좌타자", "우타자"], horizontal=True, key=f"flt_s_{game_pk}")
    out = g
    if sel_t:   # 비어 있으면 전체
        out = out[out['pitch_type'].isin(sel_t)]
    if sel_i:
        out = out[out['inning'].isin(sel_i)]
    if stand != "전체":
        out = out[out['stand'] == ('L' if stand == "좌타자" else 'R')]
    return out, f"{','.join(sel_t)}_{','.join(map(str, sel_i))}_{stand}"


def _by_type_scatter(d, x, y, cur, xaxis, yaxis, shapes):
    """구종별 색 산점도. cur = 지금 보고 있는 투구(빨간 링으로 표시)."""
    fig = go.Figure()
    for sh in shapes:
        fig.add_shape(**sh)
    for t, g in d.groupby('pitch_type'):
        fig.add_trace(go.Scatter(x=g[x], y=g[y], mode='markers', name=pitch_name(t),
                                 marker=dict(size=7, color=PITCH_COLORS.get(t, GRAY), opacity=0.8),
                                 hovertemplate=f"{pitch_name(t)}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[cur[x]], y=[cur[y]], mode='markers', showlegend=False, hoverinfo='skip',
                             marker=dict(size=16, color='rgba(0,0,0,0)', line=dict(color=ACCENT_HEX, width=2.5))))
    fig.update_layout(
        height=400, margin=dict(l=40, r=8, t=8, b=110), plot_bgcolor=PLOT_BG, paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=AXIS_TEXT, size=11), xaxis=xaxis, yaxis=yaxis,
        legend=dict(orientation='h', yanchor='top', y=-0.2, x=0, font=dict(size=10), tracegroupgap=2))
    return fig


def render_game_summary(df, idx):
    """지금 보고 있는 투구 묶음(경기, 필터 적용)의 요약: 구종별 표 + 무브먼트·존 분포. 빨간 링 = 현재 투구."""
    d = df[df['pitch_type'].notna()].copy()
    if d.empty:
        st.info("구종 정보가 있는 투구가 없습니다.")
        return
    cur = df.iloc[idx]
    d['mx'], d['mz'] = d['pfx_x'] * 12.0, d['pfx_z'] * 12.0
    cur = cur.copy()
    cur['mx'], cur['mz'] = cur['pfx_x'] * 12.0, cur['pfx_z'] * 12.0
    st_ = d.groupby('pitch_type').agg(n=('release_speed', 'size'), v=('release_speed', 'mean'),
                                      mx=('mx', 'mean'), mz=('mz', 'mean')).sort_values('n', ascending=False)
    table = pd.DataFrame({'구종': [pitch_name(c) for c in st_.index], '투구수': st_['n'].values,
                          '비율': st_['n'].values / len(d) * 100, '평균 구속': st_['v'].values,
                          '수평(in)': st_['mx'].values, '수직(in)': st_['mz'].values})
    st.dataframe(table, hide_index=True, use_container_width=True, height=min(36 * (len(table) + 1) + 3, 330),
                 column_config={
                     '비율': st.column_config.ProgressColumn('비율', format="%.0f%%", min_value=0, max_value=100),
                     '평균 구속': st.column_config.NumberColumn(format="%.1f"),
                     '구종': st.column_config.TextColumn(width=120),
                     '투구수': st.column_config.NumberColumn(width="small"),
                     '수평(in)': st.column_config.NumberColumn(format="%.1f", width="small"),
                     '수직(in)': st.column_config.NumberColumn(format="%.1f", width="small")})
    st.caption("무브먼트는 Statcast 유도 무브먼트(수평, 수직) 평균입니다.")
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown('<div class="pw-label">구종별 무브먼트</div>', unsafe_allow_html=True)
        fig = _by_type_scatter(
            d, 'mx', 'mz', cur,
            dict(title='수평 (in)', range=[-30, 30], gridcolor=GRID, zeroline=False, scaleanchor='y'),
            dict(title='수직 (in, 위가 라이징)', range=[-30, 30], gridcolor=GRID, zeroline=False),
            [dict(type="circle", x0=-r, y0=-r, x1=r, y1=r, layer="below", line=dict(color="#e5e5e5", width=1, dash="dot"))
             for r in (12, 24)])
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    with c2, st.container(border=True):
        st.markdown('<div class="pw-label">구종별 투구 위치 (주심 시점)</div>', unsafe_allow_html=True)
        fig = _by_type_scatter(
            d, 'plate_x', 'plate_z', cur,
            dict(title='좌우 (ft)', range=[-2.5, 2.5], gridcolor=GRID, zeroline=False, scaleanchor='y'),
            dict(title='높이 (ft)', range=[0, 5], gridcolor=GRID, zeroline=False),
            [dict(type="rect", x0=-0.83, x1=0.83, y0=1.5, y1=3.5, line=dict(color=ZONE_LINE, width=2),
                  fillcolor="rgba(138,143,152,0.06)", layer="below")])
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


def _fmt(v):
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}"


def render_feature_card(f, c, mx, label):
    """특성 하나의 기여도 행: 이름 · 현재값(보통값) · 막대 · 기여도."""
    name = FEAT_KR.get(f, f)
    pos = c >= 0
    width = abs(c) / mx * 100 if mx else 0
    ref = SAMPLE_VALUES.get(f)
    cur = st.session_state.inp.get(f)
    val = f'{_fmt(cur)} (보통 {_fmt(ref)})' if ref is not None and cur is not None else ''
    st.markdown(
        '<div class="pw-frow">'
        '<div class="pw-frow-top">'
        f'<span class="pw-fname">{name}<span class="pw-fval">{val}</span></span>'
        f'<span class="pw-fbadge {"pos" if pos else "neg"}">{c:+.2f}</span>'
        '</div>'
        f'<div class="pw-fmag"><span style="width:{width:.0f}%;'
        f'background:{ACCENT_HEX if pos else GRAY}"></span></div>'
        '</div>',
        unsafe_allow_html=True,
    )


def _explanation_body():
    """설명이 백그라운드 스레드에서 준비되면 자동으로 표시한다. 설명 작업이 진행 중일 때만
    1초마다 이 프래그먼트를 다시 그려 확인하고(나머지 UI는 영향 없음), 완료되면 앱을 한 번
    다시 실행해 폴링 없는 프래그먼트로 바꾼다."""
    future = st.session_state.get('_explain_future')
    if future is not None and future.done():
        try:
            st.session_state.explanation = future.result()
        except Exception:
            st.session_state.explanation = None
        st.session_state['_explain_future'] = None
        if st.session_state.get('_explain_polling'):
            st.rerun()          # 폴링 중지(run_every 없는 프래그먼트로 다시 그림)

    explanation = st.session_state.explanation
    if explanation:
        st.markdown(f'<p class="pw-shap-expl">{html.escape(explanation)}</p>', unsafe_allow_html=True)
    elif st.session_state.get('_explain_future') is not None:
        st.caption("설명 생성 중…")
    else:
        st.caption("설명을 불러오지 못했습니다. 아래 기여도 목록으로 확인해 주세요.")


def render_shap_panel(attribution, conf, label):
    st.markdown(
        '<div class="pw-shap-head">예측 근거 (SHAP)</div>'
        '<div class="pw-shap-sub">각 특성이 예측 확률을 올리고 내린 정도</div>',
        unsafe_allow_html=True,
    )
    if not attribution or not conf or not label:
        st.info("사이드바에서 값을 조절하면 기여도가 표시됩니다.")
        return

    pending = st.session_state.get('_explain_future') is not None
    st.session_state['_explain_polling'] = pending
    with st.expander("AI 설명 보기"):
        st.fragment(_explanation_body, run_every=1 if pending else None)()

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


# 투구 탐색 상태는 매 실행마다 사이드바(select_and_predict)가 다시 채운다 — 직접 조작 모드에선 비어 있음
st.session_state["_pitch_nav"] = None
st.session_state["actual_pitch"] = None

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
        st.session_state["az_slider"] = _snap(st.session_state.inp['az'], *AZ_RANGE)
        st.session_state["rx_slider"] = _snap(st.session_state.inp['release_pos_x'], *RX_RANGE, 0.1)
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
            help="릴리스 좌우 방향과 구종별 가이드의 ax 부호 표시에 반영됩니다 "
                 "(예측 자체는 17개 수치로 계산)"
            if _manual else "투수 검색 시 데이터에서 자동 반영됩니다.",
        )

    if _manual:
        st.session_state.p_throws = hand
        # 손을 사용자가 바꿨으면 좌우를 거울상으로: 릴리스 좌우 부호를 손에 맞추고
        # 수평 무브먼트(ax)도 반전한다 (크기는 유지). az·구속·스핀은 손과 무관.
        if not _mode_changed and _sync_hand is not None and hand != _sync_hand:
            _mag = abs(_cur_rx) if abs(_cur_rx) > 0.2 else abs(SAMPLE_VALUES['release_pos_x'])
            st.session_state["rx_slider"] = _snap(_mag * (1.0 if hand == 'L' else -1.0),
                                                  *RX_RANGE, 0.1)
            st.session_state["ax_slider"] = _snap(-float(st.session_state["ax_slider"]),
                                                  -30.0, 30.0)
        st.session_state["_sync_hand"] = hand
        st.session_state["_sync_rx"] = float(st.session_state["rx_slider"])

    st.markdown("---")

    if _manual:
        spd = st.slider("구속 (mph)", 60.0, 105.0, float(st.session_state.inp['release_speed']),
                        0.5, format="%.1f", help="공이 손을 떠나는 순간의 속도 · 빠를수록 FF/FC 계열, "
                                  "느릴수록 커브 등 변화구 계열로 기울어짐")
        spin = st.slider("스핀레이트 (rpm)", 1000.0, 3500.0,
                         float(st.session_state.inp['release_spin_rate']), 10.0, format="%.0f",
                         help="공의 분당 회전수 · 높을수록 포심/커터 계열로 기울어짐 "
                              "(회전 방향은 아래 회전축 슬라이더가 따로 담당)")
        spin_axis = st.slider("회전축 (°)", 0.0, 360.0,
                              float(st.session_state.inp['spin_axis']), 5.0, format="%.0f",
                              help="회전축 방향(시계 방향, 0–360°) · ax·az와 별개로 판별에 큰 영향 · "
                                   "2026 실측 평균: 포심 215° 싱커 225° 체인지업 242° 스플리터 238° "
                                   "커터 185° 슬라이더 122°(편차 큼) 커브 44°")
        ax_ = st.slider("수평 무브먼트 (ax)", -30.0, 30.0, step=0.5, format="%.1f", key="ax_slider",
                        help="우완 기준 패스트볼 계열은 음수, 변화구 계열(SL·CU·FC)은 양수 쪽 · "
                             "스트라이크존을 클릭하면 자동으로 맞춰집니다")
        az = st.slider("수직 무브먼트 (az)", *AZ_RANGE, step=0.5, format="%.1f", key="az_slider",
                       help="양수=포심(떠오름), 음수=싱커(가라앉음) · 실측은 약 -50에서 0 사이 · "
                            "존 클릭 시엔 그 위치에 닿도록 범위 밖 값 가능")
        ext = st.slider("릴리스 익스텐션 (ft)", 4.5, 8.0,
                        float(st.session_state.inp['release_extension']), 0.1, format="%.1f",
                        help="투수판에서 릴리스 지점까지 앞으로 나온 거리 · 클수록 "
                             "홈플레이트에 더 가까이서 공을 놓음")
        rx = st.slider("릴리스 좌우 (ft)", *RX_RANGE, step=0.1, format="%.1f", key="rx_slider",
                       help="홈플레이트 중심 기준 릴리스 지점의 좌우 위치 · 투구 손과 자동 연동 · "
                            "음수 = 우완(1루쪽), 양수 = 좌완(3루쪽)")
        rz = st.slider("릴리스 높이 (ft)", 3.5, 7.2,
                       float(st.session_state.inp['release_pos_z']), 0.1, format="%.1f",
                       help="공을 놓는 지점의 지면 기준 높이 · 암슬롯을 반영하며 높을수록 "
                            "오버핸드, 낮을수록 사이드암 쪽")

        vz0 = round(-3.0 - 0.15 * az, 2)
        # 존 클릭 직후(슬라이더 미조작)면 클릭 높이를 정확히 맞춘 vz0을 사용 — az
        # 슬라이더의 표시값이 클램프됐을 때도 실제 궤적은 클릭한 높이에 정확히 도달한다.
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
            # Statcast pfx(ft) = 0.5·a·t², z는 중력 제거 — _pfx_inches와 같은 식(÷12)
            'pfx_z': round(0.5 * (az + 32.174) * t**2, 3),
            'pfx_x': round(0.5 * ax_ * t**2, 3),
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
            st.markdown('<div class="pw-label">즐겨찾기 (이번 세션)</div>', unsafe_allow_html=True)
            for f in favs:
                # 구버전 세션엔 'label' 없이 {'id','name'}만 있을 수 있음 — 배포 중 세션이
                # 유지되면 즐겨찾기 스키마가 바뀌어도 session_state는 그대로 남는다.
                if st.button(f.get('label', f['name']), key=f"favbtn_{f['id']}", use_container_width=True):
                    st.session_state.pitcher_query = f['name']
                    st.rerun()

        query = st.text_input(
            "투수 이름 (영어 2글자 이상, 비우면 날짜로 검색)",
            placeholder="Gerrit Cole", key="pitcher_query",
        ).strip()

        if query and contains_korean(query):
            st.warning("영어로 입력해주세요 (예: Gerrit Cole).")
        elif len(query) >= 2:
            with st.spinner(f"'{query}' 검색 중…"):
                try:
                    candidates = search_pitchers(query)
                except requests.exceptions.RequestException as e:
                    candidates = None
                    st.error(f"선수 검색에 실패했습니다: {e}")

            if candidates is not None and not candidates:
                st.error("선수를 찾을 수 없습니다. 영문 철자를 확인하거나 성만 입력해 보세요.")
            elif candidates:
                options = {career_label(c): c for c in candidates}
                choice = st.selectbox("검색 결과", options=list(options.keys()))
                chosen = options[choice]
                st.session_state.p_throws = chosen['throws']

                is_fav = any(f['id'] == chosen['id'] for f in st.session_state.favorites)
                # 아이콘만 있는 버튼 대신 체크박스 — 이름·선택 상태가 화면 낭독기에도 전달된다.
                if st.checkbox("즐겨찾기", value=is_fav, key=f"fav_toggle_{chosen['id']}") != is_fav:
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
                                              min_value=min_d, max_value=max_d,
                                              key=f"pitcher_start_{chosen['id']}")
                    end_d = dc2.date_input("종료일", value=def_end,
                                            min_value=min_d, max_value=max_d,
                                            key=f"pitcher_end_{chosen['id']}")

                    if start_d > end_d:
                        st.error("시작일이 종료일보다 늦을 수 없습니다.")
                    else:
                        with st.spinner(
                            f"{chosen['name']} 선수의 {start_d}~{end_d} 투구 데이터를 가져오는 중…"
                        ):
                            df = fetch_pitcher_statcast(chosen['id'], str(start_d), str(end_d))

                        if df.empty:
                            st.warning("해당 기간 투구 기록이 없습니다. "
                                       "타자로만 뛴 기간일 수 있어요.")
                        elif 'game_pk' in df.columns:
                            games = games_from_pitches(df)
                            game_pk = st.selectbox(
                                f"등판 ({len(games)}경기)", options=games.index,
                                format_func=lambda g: games.loc[g, 'label'],
                            )
                            g, flt_sig = game_filters(df[df['game_pk'] == game_pk], game_pk)
                            if g.empty:
                                st.warning("필터 조건에 맞는 투구가 없습니다.")
                            else:
                                df_valid = prepare_pitches(g)
                                if df_valid is not None:
                                    select_and_predict(df_valid, key=f"pitch_{game_pk}_{flt_sig}")
                        else:
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
                _req = (str(start_d), str(end_d))
                st.caption("리그 전체 투구를 받아 오므로 시간이 걸릴 수 있어요.")
                if st.button("이 기간 조회", key="range_go"):
                    st.session_state["_range_req"] = _req
                if st.session_state.get("_range_req") == _req:
                    with st.spinner(f"{start_d}~{end_d} 전체 투구 데이터를 가져오는 중…"):
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
        render_summary(label, conf, inf_ms, st.session_state.get('actual_pitch'))


left, right = st.columns([2.2, 1], gap="large")

nav = st.session_state.get("_pitch_nav")

with left:
    if nav:
        with st.container(key="pitch_nav"):
            c_prev, c_mid, c_next = st.columns([1, 2.4, 1])
            c_prev.button("이전", icon=":material/chevron_left:", on_click=_step_pitch,
                          args=(-1, nav["n"], nav["tkey"]), disabled=nav["idx"] == 0,
                          use_container_width=True, key="pitch_prev")
            c_mid.markdown(f'<div class="pw-nav"><b>{nav["idx"] + 1} / {nav["n"]}구</b><br>{nav["info"]}</div>',
                           unsafe_allow_html=True)
            c_next.button("다음", icon=":material/chevron_right:", on_click=_step_pitch,
                          args=(1, nav["n"], nav["tkey"]), disabled=nav["idx"] == nav["n"] - 1,
                          use_container_width=True, key="pitch_next")
    has_sum = bool(nav) and 'pitch_type' in nav["df"].columns
    tab_names = (["대시보드", "궤적 분석"] + (["투구 목록"] if nav else []) + (["경기 요약"] if has_sum else [])
                 + ["모델 정보"])
    tabs = dict(zip(tab_names, st.tabs(tab_names)))
    tab_dash, tab_traj, tab_model = tabs["대시보드"], tabs["궤적 분석"], tabs["모델 정보"]

    with tab_dash:
        summary_card()
        if label and proba:
            render_prob_dist(proba, label)
            c_mv, c_zone = st.columns(2)
            with c_mv, st.container(border=True):
                render_movement(inp)
            with c_zone, st.container(border=True):
                render_strikezone(inp, clickable=(input_mode == "직접 조작"))
        else:
            with st.container(border=True):
                st.markdown("사이드바에서 값을 조절하거나 투수를 선택하면 "
                            "여기에 예측 결과가 표시됩니다.")

    with tab_traj:
        summary_card()
        with st.container(border=True):
            render_paths(inp, st.session_state.p_throws)

    if nav:
        with tabs["투구 목록"]:
            summary_card()
            st.dataframe(nav["table"], on_select="rerun", selection_mode="single-row", key=nav["tkey"],
                         hide_index=True, use_container_width=True,
                         column_config={"구속": st.column_config.NumberColumn(format="%.1f")},
                         height=min(36 * (nav["n"] + 1) + 3, 460))
            st.caption("행을 누르면 그 투구로 바뀝니다. 표는 투구 순서(타석, 구)대로입니다.")

    if has_sum:
        with tabs["경기 요약"]:
            summary_card()
            render_game_summary(nav["df"], nav["idx"])

    with tab_model:
        summary_card()
        with st.container(border=True):
            render_model_info()

with right:
    with st.container(border=True):
        render_shap_panel(attribution, conf, label)
