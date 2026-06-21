# predict.py - 학습된 모델로 구종 예측

import os
import pickle
import numpy as np

from model import load_model
from data_loader import FEATURE_COLS

MODEL_DIR    = os.path.join(os.path.dirname(__file__), '..', 'model')
SCALER_PATH  = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')


def load_artifacts():
    """저장된 스케일러와 레이블 인코더를 불러온다."""
    with open(SCALER_PATH,  'rb') as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)

    return scaler, le


def predict(input_dict):
    """
    Args:
        input_dict (dict): {특징명: 값} 형태의 입력 딕셔너리
    Returns:
        pred_label (str):  예측 구종명 (예: 'FF')
        confidence (float): 예측 확률 (0~1)
        proba_dict (dict): 전체 구종별 확률
    """
    model          = load_model()
    scaler, le     = load_artifacts()

    x = np.array([[input_dict[col] for col in FEATURE_COLS]], dtype=np.float32)
    x = scaler.transform(x)

    proba      = model.predict(x, verbose=0)[0]
    pred_idx   = int(np.argmax(proba))
    pred_label = le.classes_[pred_idx]
    confidence = float(proba[pred_idx])

    proba_dict = {le.classes_[i]: float(proba[i]) for i in range(len(le.classes_))}

    return pred_label, confidence, proba_dict


if __name__ == '__main__':
    sample = {
        'release_speed'   : 94.5,
        'release_spin_rate': 2280.0,
        'release_extension': 6.2,
        'release_pos_x'   : -1.5,
        'release_pos_z'   : 6.1,
        'pfx_x'           : 0.8,
        'pfx_z'           : 1.2,
        'plate_x'         : 0.3,
        'plate_z'         : 2.8,
        'vx0'             : 5.2,
        'vy0'             : -138.0,
        'vz0'             : -5.1,
        'ax'              : 8.3,
        'ay'              : 28.5,
        'az'              : -14.2,
        'effective_speed' : 93.1,
        'spin_axis'       : 210.0,
    }

    label, conf, proba = predict(sample)

    print(f"\n예측 구종: {label}  ({conf*100:.1f}%)")
    print("\n전체 확률:")
    for pitch, p in sorted(proba.items(), key=lambda x: -x[1]):
        print(f"  {pitch}: {p*100:.1f}%")
