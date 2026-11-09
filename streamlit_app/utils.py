"""utils.py - 모델 직접 로드 예측 및 궤적 계산 유틸리티

backend/ 의 추론 코드(model_utils, explain)를 그대로 재사용한다. Streamlit Cloud는
레포 전체를 배포하므로 backend/model/ 아티팩트도 함께 올라오고, 경로는 각 모듈이
자기 파일 기준으로 잡으므로 import 위치와 무관하게 동작한다.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from model_utils import predict as _model_predict  # noqa: E402
from explain import generate_explanation  # noqa: E402

FEATURE_COLS = [
    'release_speed', 'release_spin_rate', 'release_extension',
    'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z',
    'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0',
    'ax', 'ay', 'az', 'effective_speed', 'spin_axis',
]

PITCH_NAMES = {
    'FF': '포심 패스트볼', 'SI': '싱커',    'SL': '슬라이더',
    'CU': '커브',          'CH': '체인지업', 'FC': '커터', 'FS': '스플리터',
}

# 라이트 테마 "Pitch Workbench" 카테고리 팔레트 (oklch → sRGB hex, plotly용)
PITCH_COLORS = {
    'FF': '#3383ad', 'SI': '#279ea4', 'SL': '#c53637',
    'CU': '#686bb9', 'CH': '#539156', 'FC': '#a172ac', 'FS': '#496b96',
}

PITCH_DIST = 60.5

# 기본 입력값 = 2026시즌 우완 싱커(SI) 실측 평균(79,502건, 슬라이더 step에 맞춰 반올림).
# SHAP 특성 카드의 "일반적인 수준" 기준값으로도 쓰인다.
SAMPLE_VALUES = {
    'release_speed': 94.5, 'release_spin_rate': 2210.0,
    'release_extension': 6.4, 'release_pos_x': -1.9,
    'release_pos_z': 5.6, 'pfx_x': -1.26, 'pfx_z': 0.63,
    'plate_x': -0.17, 'plate_z': 2.35, 'vx0': 7.34,
    'vy0': -137.4, 'vz0': -3.98, 'ax': -17.5, 'ay': 30.3,
    'az': -23.5, 'effective_speed': 94.7, 'spin_axis': 225.0,
}


def predict(input_dict):
    """모델을 직접 로드해 (예측 구종, 신뢰도, 확률 딕셔너리, 피처 기여도)를 반환한다.

    자연어 설명(Gemini API, 최대 15초 타임아웃)은 여기 포함하지 않는다 — 슬라이더나
    스트라이크존 클릭마다 자동으로 걸리면 그 인터랙션 자체가 최대 15초씩 막혀서,
    배포 환경처럼 왕복 지연이 조금만 있어도 "두 번 조작해야 반영되는" 것처럼 느껴진다.
    설명은 explain()으로 분리해 버튼으로 명시적으로만 호출한다.
    """
    return _model_predict(input_dict)


def explain(input_dict, label, confidence, proba, p_throws='R'):
    """자연어 설명을 생성한다 (느린 네트워크 호출 — 명시적으로 요청했을 때만 부른다)."""
    return generate_explanation(input_dict, label, confidence, proba, p_throws)


def compute_trajectory_side(inp, n=50):
    """측면뷰 궤적(거리, 높이) 좌표 배열을 계산한다."""
    vy0, vz0, az = inp['vy0'], inp['vz0'], inp['az']
    rz, ext = inp['release_pos_z'], inp['release_extension']
    dist = PITCH_DIST - ext
    t = np.linspace(0, dist / max(abs(vy0), 1), n)
    z = rz + vz0 * t + 0.5 * az * t ** 2
    x = ext + abs(vy0) * t
    return x, z


def compute_trajectory_top(inp, p_throws='R', n=50):
    """상단뷰 궤적(거리, 좌우) 좌표 배열을 계산한다.

    좌우 성분은 app.py 사이드바가 plate_x를 만들 때 쓰는 식과 동일하게 맞춘다
    (vx0 = 0.20*ax, 부호 반전 없음). 그래야 궤적 끝점이 로케이션·스트라이크존
    차트의 빨간 점(plate_x)과 일치한다.
    """
    vy0, ax_, rx = inp['vy0'], inp['ax'], inp['release_pos_x']
    ext = inp['release_extension']
    dist = PITCH_DIST - ext
    t = np.linspace(0, dist / max(abs(vy0), 1), n)
    x_pos = rx + 0.20 * ax_ * t + 0.5 * ax_ * t ** 2
    y_pos = ext + abs(vy0) * t
    return y_pos, x_pos


GAME_TYPE_TAG = {'S': ' (시범)', 'F': ' (PS)', 'D': ' (PS)', 'L': ' (PS)', 'W': ' (PS)'}


def games_from_pitches(df):
    """투수 한 명의 Statcast 투구 df → 등판 목록(game_pk 인덱스, 날짜 내림차순).

    컬럼: date, opp, n(투구 수 — 결측 제거 전 원본 기준), label("11/02 vs NYY · 98구").
    상대 팀: 투수가 초(Top)에 던지면 홈 팀이므로 상대는 away_team, 말(Bot)이면 home_team.
    """
    # 더블헤더 순서: Statcast엔 경기 시작 시각이 없고 game_pk 순서는 재편성 경기면 뒤집힌다
    # (2026-04-30 PHI·08-29 실측 모두 반대). 대신 2차전은 "직전 등판 후 0일"이므로
    # pitcher_days_since_prev_game이 큰 쪽이 1차전(시즌 첫 등판 NaN은 가장 앞).
    since = df.get('pitcher_days_since_prev_game', 0)
    g = df.assign(_since=since).groupby('game_pk').agg(
        date=('game_date', 'first'), topbot=('inning_topbot', 'first'),
        home=('home_team', 'first'), away=('away_team', 'first'),
        game_type=('game_type', 'first'), since=('_since', 'first'), n=('game_pk', 'size'),
    )
    g['date'] = pd.to_datetime(g['date'])
    g['opp'] = g['away'].where(g['topbot'] == 'Top', g['home'])
    g = g.assign(_k=-g['since'].fillna(1e9)).sort_values(['date', '_k'])
    nth = g.groupby('date').cumcount() + 1
    dh = g.groupby('date')['n'].transform('size') > 1
    g['label'] = (g['date'].dt.strftime('%m/%d')
                  + (' (' + nth.astype(str) + ')').where(dh, '')
                  + ' vs ' + g['opp'] + ' · ' + g['n'].astype(str) + '구'
                  + g['game_type'].map(GAME_TYPE_TAG).fillna(''))
    g = g.iloc[::-1]   # 최신 등판이 위 — 같은 날이면 (2)가 (1) 위
    return g[['date', 'opp', 'n', 'label']]


if __name__ == '__main__':
    # 홈(Top→away 상대)/원정(Bot→home 상대) 판정 + 더블헤더 (1)/(2) + 정렬·투구 수.
    # 더블헤더는 실제 재편성 경기처럼 game_pk가 큰 쪽(3)이 1차전(직전 등판 후 4일).
    rows = ([(1, '2026-05-01', 'Top', 'MIA', 'NYM', 'R', None)] * 3
            + [(3, '2026-06-02', 'Bot', 'ATL', 'MIA', 'S', 4)] * 2
            + [(2, '2026-06-02', 'Bot', 'ATL', 'MIA', 'R', 0)])
    t = pd.DataFrame(rows, columns=['game_pk', 'game_date', 'inning_topbot', 'home_team',
                                    'away_team', 'game_type', 'pitcher_days_since_prev_game'])
    got = games_from_pitches(t)
    assert list(got.index) == [2, 3, 1], got
    assert list(got['label']) == ['06/02 (2) vs ATL · 1구', '06/02 (1) vs ATL · 2구 (시범)',
                                  '05/01 vs NYM · 3구'], list(got['label'])
    print('games_from_pitches OK')
