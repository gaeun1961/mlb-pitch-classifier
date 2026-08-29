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


def _feature_attribution(model, x_scaled, pred_idx):
    """예측 클래스 확률을 표준화된 각 입력 피처로 편미분한 gradient×input 기여도.

    양수면 그 피처 값이 이 구종 예측을 밀어올린 방향, 음수면 끌어내린 방향이다.
    값은 스케일러 적용 후 공간 기준이라 절대 크기가 아니라 상대 비교·부호로 읽는다.
    """
    xt = tf.convert_to_tensor(np.asarray(x_scaled, dtype=np.float32))
    with tf.GradientTape() as tape:
        tape.watch(xt)
        p = model(xt, training=False)[:, pred_idx]
    grad = tape.gradient(p, xt).numpy()[0]
    contrib = grad * np.asarray(x_scaled, dtype=np.float32)[0]
    return {col: float(v) for col, v in zip(FEATURE_COLS, contrib)}


def predict(input_dict):
    """입력 딕셔너리를 받아 (예측 구종, 신뢰도, 확률 딕셔너리, 피처 기여도 딕셔너리)를 반환한다."""
    model, scaler, le = load_artifacts()

    x = np.array([[input_dict[col] for col in FEATURE_COLS]], dtype=np.float32)
    x = scaler.transform(x)

    proba      = model.predict(x, verbose=0)[0]
    pred_idx   = int(np.argmax(proba))
    pred_label = le.classes_[pred_idx]
    confidence = float(proba[pred_idx])
    proba_dict = {le.classes_[i]: float(proba[i]) for i in range(len(le.classes_))}
    attribution = _feature_attribution(model, x, pred_idx)

    return pred_label, confidence, proba_dict, attribution


if __name__ == "__main__":
    _sample = {
        'release_speed': 94.5, 'release_spin_rate': 2280.0, 'release_extension': 6.2,
        'release_pos_x': -1.5, 'release_pos_z': 6.1, 'pfx_x': 0.8, 'pfx_z': 1.2,
        'plate_x': 0.3, 'plate_z': 2.8, 'vx0': 5.2, 'vy0': -138.0, 'vz0': -5.1,
        'ax': 8.3, 'ay': 28.5, 'az': -14.2, 'effective_speed': 93.1, 'spin_axis': 210.0,
    }
    _label, _conf, _proba, _attr = predict(_sample)
    assert set(_attr) == set(FEATURE_COLS), _attr.keys()
    assert abs(sum(_proba.values()) - 1.0) < 1e-4, sum(_proba.values())
    assert any(v != 0.0 for v in _attr.values()), "기여도가 전부 0이면 gradient 경로가 끊긴 것"
    print(_label, f"{_conf:.3f}")
    for _k, _v in sorted(_attr.items(), key=lambda kv: -abs(kv[1]))[:7]:
        print(f"  {_k:20s} {_v:+.4f}")
