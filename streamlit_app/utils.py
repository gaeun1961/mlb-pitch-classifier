"""utils.py - 모델 로드 및 예측 유틸리티 (Streamlit 캐싱 적용)"""

import os
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf

MODEL_DIR    = os.path.join(os.path.dirname(__file__), 'model')
MODEL_PATH   = os.path.join(MODEL_DIR, 'pitch_model.h5')
SCALER_PATH  = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

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


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """모델, 스케일러, 레이블 인코더를 1회만 로드해 세션 내내 재사용한다."""
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    return model, scaler, le


def predict(input_dict):
    """입력 딕셔너리를 받아 (예측 구종, 신뢰도, 전체 확률 딕셔너리)를 반환한다."""
    model, scaler, le = load_artifacts()

    x = np.array([[input_dict[col] for col in FEATURE_COLS]], dtype=np.float32)
    x = scaler.transform(x)

    proba      = model.predict(x, verbose=0)[0]
    pred_idx   = int(np.argmax(proba))
    pred_label = le.classes_[pred_idx]
    confidence = float(proba[pred_idx])
    proba_dict = {le.classes_[i]: float(proba[i]) for i in range(len(le.classes_))}

    return pred_label, confidence, proba_dict


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
