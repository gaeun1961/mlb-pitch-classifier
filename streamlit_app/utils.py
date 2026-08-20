"""utils.py - FastAPI 백엔드 호출 및 궤적 계산 유틸리티"""

import os
import numpy as np
import requests
import streamlit as st

try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except Exception:
    BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

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
    """FastAPI 백엔드에 예측을 요청해 (예측 구종, 신뢰도, 확률 딕셔너리, 설명)을 반환한다."""
    payload = dict(input_dict, p_throws=p_throws)
    resp = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data['predicted_label'], data['confidence'], data['probabilities'], data['explanation']


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
