"""utils.py - 모델 직접 로드 예측 및 궤적 계산 유틸리티

backend/ 의 추론 코드(model_utils, explain)를 그대로 재사용한다. Streamlit Cloud는
레포 전체를 배포하므로 backend/model/ 아티팩트도 함께 올라오고, 경로는 각 모듈이
자기 파일 기준으로 잡으므로 import 위치와 무관하게 동작한다.
"""

import sys
from pathlib import Path

import numpy as np

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

PITCH_COLORS = {
    'FF': '#C8443C', 'SI': '#E8843C', 'SL': '#8B5FBF',
    'CU': '#3D7DC9', 'CH': '#2D9D6F', 'FC': '#D9622B', 'FS': '#3FB8C4',
}

PITCH_DIST = 60.5

SAMPLE_VALUES = {
    'release_speed': 94.5, 'release_spin_rate': 2280.0,
    'release_extension': 6.2, 'release_pos_x': -1.5,
    'release_pos_z': 6.1, 'pfx_x': 0.8, 'pfx_z': 1.2,
    'plate_x': 0.3, 'plate_z': 2.8, 'vx0': 5.2,
    'vy0': -138.0, 'vz0': -5.1, 'ax': 8.3, 'ay': 28.5,
    'az': -14.2, 'effective_speed': 93.1, 'spin_axis': 210.0,
}


def predict(input_dict, p_throws='R'):
    """모델을 직접 로드해 (예측 구종, 신뢰도, 확률 딕셔너리, 피처 기여도, 설명)을 반환한다."""
    label, confidence, proba, attribution = _model_predict(input_dict)
    explanation = generate_explanation(input_dict, label, confidence, proba, p_throws)
    return label, confidence, proba, attribution, explanation


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
    """상단뷰 궤적(거리, 좌우) 좌표 배열을 계산한다. ax 부호는 포수 시점 기준이라 반전한다."""
    vy0, ax_, rx = inp['vy0'], inp['ax'], inp['release_pos_x']
    ext = inp['release_extension']
    dist = PITCH_DIST - ext
    t = np.linspace(0, dist / max(abs(vy0), 1), n)
    x_pos = rx + 0.5 * (-ax_) * t ** 2
    y_pos = ext + abs(vy0) * t
    return y_pos, x_pos
