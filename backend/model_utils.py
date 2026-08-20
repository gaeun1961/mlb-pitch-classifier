"""model_utils.py - 모델/스케일러/인코더 로드 및 예측 유틸리티"""

import os
import pickle
from functools import lru_cache

import numpy as np
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


@lru_cache(maxsize=1)
def load_artifacts():
    """모델, 스케일러, 레이블 인코더를 1회만 로드해 프로세스 내내 재사용한다."""
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
